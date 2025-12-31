from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gita_llm.schemas import VerseRecord


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
            "You MUST cite the single closest verse exactly as 'ClosestVerse: BG x.y'.",
            "Do not repeat the evidence.",
            "",
            f"Question: {question}",
            "",
            "Evidence verses:",
            evidence,
            "",
            "Output format:",
            "Answer: <your answer>",
            "ClosestVerse: BG x.y",
        ]
    ).strip()


def generate_chat(
    question: str,
    verses: list[VerseRecord],
    model_name_or_path: str,
    max_new_tokens: int = 220,
) -> ChatAnswer:
    """
    Runs an instruction causal LM. For Colab GPU, you can pass a 4-bit loaded model dir too.
    """
    prompt = build_chat_prompt(question, verses)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

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

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Keep only the generated tail if the model echoed the prompt
    if "Output format:" in text and "Answer:" in text:
        text = text.split("Output format:", 1)[-1]
    if "Answer:" in text:
        text = "Answer:" + text.split("Answer:", 1)[-1]
    text = text.strip()

    closest = verses[0].ref() if verses else ""
    for line in text.splitlines():
        if line.strip().startswith("ClosestVerse:"):
            closest = line.split("ClosestVerse:", 1)[1].strip()
            break

    return ChatAnswer(answer=text, closest_verse=closest)


