from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gita_llm.schemas import VerseRecord


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    used_ref: str
    word_by_word: str
    word_by_word_is_generated: bool


def _looks_hindi(text: str) -> bool:
    # Basic heuristic: Devanagari range
    return any("\u0900" <= ch <= "\u097F" for ch in text)

def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return 0.0
    dev = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    return dev / total


def build_prompt(question: str, verse: VerseRecord, answer_lang: str, include_word_by_word: bool) -> str:
    """
    Minimal prompt designed to avoid instruction-echoing.
    The model must output ONLY the final fields.
    """
    if answer_lang == "hi":
        lang_inst = "भाषा: हिन्दी"
    else:
        lang_inst = "Language: English"

    lines = [
        lang_inst,
        "You are a helpful tutor. Answer the user's question using the evidence verse.",
        "Do NOT repeat the question or the instructions. Output ONLY the fields below.",
        "",
        f"Question: {question}",
        "",
        f"ClosestVerse: {verse.ref()}",
        f"Sanskrit: {verse.sanskrit_devanagari}",
        f"Hindi: {verse.hindi}",
        f"English: {verse.english}",
        "",
        "Return exactly in this format:",
        "Answer: <2-5 simple sentences>",
        f"ClosestVerse: {verse.ref()}",
    ]
    if include_word_by_word:
        lines += [
            "WordByWord:",
            "- <sa> : <meaning>",
        ]
    return "\n".join(lines).strip()


def _clean_model_output(text: str) -> str:
    """
    Remove common prompt-echo artifacts and keep only the first well-formed block.
    """
    text = text.strip()
    if not text:
        return text
    # Drop everything after a repeated instruction section if it appears.
    for bad in ["Task:", "Output format:", "Instruction:", "Question:", "Sanskrit:", "Hindi:", "English:"]:
        if bad in text and not text.lstrip().startswith("Answer:"):
            # If the model didn't start with Answer:, keep from the first Answer:
            if "Answer:" in text:
                text = text.split("Answer:", 1)[1]
                text = "Answer:" + text
                break
    # If it repeats "Answer:" many times, keep the first chunk.
    if text.count("Answer:") > 1:
        first = text.split("Answer:", 2)[1]
        text = "Answer:" + first
    # Hard truncate at very long length (avoid runaway repetition)
    return text.strip()[:4000]


def format_gold_word_by_word(verse: VerseRecord, lang: str) -> str | None:
    if not verse.word_by_word:
        return None
    lines: list[str] = []
    for item in verse.word_by_word:
        meaning = item.hi if lang == "hi" else item.en
        if not meaning:
            # fall back to whichever exists
            meaning = item.en or item.hi or ""
        lines.append(f"- {item.sa} : {meaning}")
    return "\n".join(lines).strip() if lines else None


def generate(
    question: str,
    verse: VerseRecord,
    model_name: str = "google/flan-t5-base",
    device: str | None = None,
    max_new_tokens: int = 280,
    mode: str = "baseline",
) -> GeneratedAnswer:
    answer_lang = "hi" if _looks_hindi(question) else "en"
    gold_wbw = format_gold_word_by_word(verse, lang=answer_lang)

    # Minimal baselines (before fine-tuning):
    # - Always cite the retrieved verse
    # - Keep language simple and deterministic
    if mode == "baseline" and answer_lang == "en":
        answer = (
            f"In the Bhagavad Gita, **karma** broadly means action/duty. "
            f"{verse.ref()} supports this idea: {verse.english}"
        ).strip()
        return GeneratedAnswer(
            answer=answer,
            used_ref=verse.ref(),
            word_by_word=gold_wbw or "(word-by-word not available in dataset)",
            word_by_word_is_generated=(gold_wbw is None),
        )

    # Minimal baseline for Hindi:
    # flan-t5-base is not reliable for EN->HI translation here, so we answer using the provided Hindi translation.
    if mode == "baseline" and answer_lang == "hi":
        answer = (
            f"गीता के अनुसार कर्म का अर्थ है अपना कर्तव्य/कार्य करना और फल की आसक्ति छोड़ना। "
            f"{verse.ref()} में कहा गया है: {verse.hindi}"
        ).strip()
        return GeneratedAnswer(
            answer=answer,
            used_ref=verse.ref(),
            word_by_word=gold_wbw or "(word-by-word not available in dataset)",
            word_by_word_is_generated=(gold_wbw is None),
        )
    prompt = build_prompt(
        question=question,
        verse=verse,
        answer_lang=answer_lang,
        include_word_by_word=(gold_wbw is None),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=2,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # Fallback: if output is empty or whitespace-only, try a deterministic decode.
    if not text:
        out2 = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        text = tokenizer.decode(out2[0], skip_special_tokens=True).strip()

    text = _clean_model_output(text)

    if gold_wbw:
        # Keep model answer but replace/append WBW with gold.
        # Minimal: just return gold WBW separately.
        return GeneratedAnswer(
            answer=text,
            used_ref=verse.ref(),
            word_by_word=gold_wbw,
            word_by_word_is_generated=False,
        )

    # If we don't have gold WBW, extract from model output if possible.
    wbw = ""
    if "WordByWord:" in text:
        wbw = text.split("WordByWord:", 1)[1].strip()
    else:
        wbw = "(auto-generated) " + text

    return GeneratedAnswer(
        answer=text,
        used_ref=verse.ref(),
        word_by_word=wbw.strip(),
        word_by_word_is_generated=True,
    )


