from __future__ import annotations

from dataclasses import dataclass

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gita_llm.schemas import VerseRecord

_CLOSEST_VERSE_LABEL = "ClosestVerse:"

@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    closest_verse: str


def _looks_hindi(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def build_chat_prompt(question: str, verses: list[VerseRecord]) -> str:
    """
    Prompt for an instruction-tuned causal LM.
    We include top-k verses as evidence and require a simple, grounded answer.
    """
    lang = "Hindi" if _looks_hindi(question) else "English"
    evidence_blocks: list[str] = []
    for v in verses:
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{v.ref()}]",
                    f"Sanskrit: {v.sanskrit_devanagari}",
                    f"Hindi: {v.hindi}",
                    f"English: {v.english}",
                ]
            )
        )

    evidence = "\n\n".join(evidence_blocks)
    return "\n".join(
        [
            f"Language: {lang}",
            "You are a helpful tutor of the Bhagavad Gita.",
            "Answer the user's question using ONLY the evidence verses provided.",
            "Be simple and clear (2-6 sentences).",
            "Choose ONE closest verse from the evidence and cite it exactly as: ClosestVerse: BG x.y",
            "Do not repeat the evidence text. Do not use placeholders like <your answer>.",
            "",
            f"Question: {question}",
            "",
            "Evidence verses:",
            evidence,
            "",
            "Output (2 lines only):",
            "Answer: <write the answer here>",
            "ClosestVerse: BG x.y",
        ]
    ).strip()


def _resolve_base_model_name_or_path(model_name_or_path: str) -> tuple[bool, str]:
    """
    Returns (is_adapter_dir, base_model_name_or_path).
    Adapter dirs are PEFT outputs containing adapter_config.json.
    """
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return False, model_name_or_path
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path") or ""
    if not base:
        raise ValueError(
            f"Adapter dir {model_name_or_path!r} is missing base_model_name_or_path in adapter_config.json"
        )
    return True, base


def _load_tokenizer(model_name_or_path: str, base_model_name_or_path: str) -> AutoTokenizer:
    # Prefer tokenizer saved with adapters (if present), else fall back to base model tokenizer.
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)


def generate_chat(
    question: str,
    verses: list[VerseRecord],
    model_name_or_path: str,
    max_new_tokens: int = 220,
    offload_dir: str | None = None,
) -> ChatAnswer:
    """
    Runs an instruction causal LM. For Colab GPU, you can pass a 4-bit loaded model dir too.
    """
    prompt = build_chat_prompt(question, verses)

    # Support both HF models and PEFT adapter dirs (outputs of `train_qlora.py`).
    is_adapter_dir, base_model_name_or_path = _resolve_base_model_name_or_path(model_name_or_path)
    tokenizer = _load_tokenizer(model_name_or_path, base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    model_kwargs = {
        "torch_dtype": (torch.float16 if torch.cuda.is_available() else None),
        "device_map": device_map,
    }
    # If accelerate decides to offload some layers to CPU, it requires an explicit offload folder.
    if device_map is not None and offload_dir:
        model_kwargs["offload_folder"] = offload_dir

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **model_kwargs)
    if is_adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_name_or_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated continuation (avoid prompt echo)
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if "Answer:" not in text and "ClosestVerse:" not in text:
        # Some models omit labels; keep it simple and add them.
        text = f"Answer: {text}\nClosestVerse: {verses[0].ref() if verses else ''}".strip()

    closest = verses[0].ref() if verses else ""
    for line in text.splitlines():
        if line.strip().startswith(_CLOSEST_VERSE_LABEL):
            closest = line.split(_CLOSEST_VERSE_LABEL, 1)[1].strip()
            break

    return ChatAnswer(answer=text, closest_verse=closest)

