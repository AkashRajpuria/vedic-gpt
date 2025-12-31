from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from gita_llm.io import read_jsonl, write_jsonl
from gita_llm.schemas import VerseRecord


def _clean_en(s: str) -> str:
    s = s.strip()
    # remove leading "10.18 " patterns
    s = re.sub(r"^\d+\.\d+\s*", "", s)
    return s.strip()


def _clean_hi(s: str) -> str:
    s = s.strip()
    # remove leading "।।10.18।।" patterns
    s = re.sub(r"^।।\s*\d+\.\d+\s*।।\s*", "", s)
    return s.strip()


def _wbw_lines(verse: VerseRecord, lang: str) -> str:
    if not verse.word_by_word:
        return "(word-by-word not available)"
    lines: list[str] = []
    for item in verse.word_by_word:
        meaning = item.hi if lang == "hi" else item.en
        meaning = meaning or item.en or item.hi or ""
        lines.append(f"- {item.sa} : {meaning}")
    return "\n".join(lines).strip()


def _make_pair(verse: VerseRecord, lang: str) -> dict:
    """
    Create one instruction-style example.
    We keep the question "light" because retrieval will provide the verse at inference time.
    """
    if lang == "hi":
        question = random.choice(
            [
                "इस श्लोक का बहुत सरल अर्थ बताइए।",
                "यह श्लोक क्या सिखाता है? सरल भाषा में समझाइए।",
                "इस श्लोक का सार बताइए।",
            ]
        )
        answer = (
            f"Answer: यह श्लोक सरल शब्दों में यह बताता है: {_clean_hi(verse.hindi)}\n"
            f"ClosestVerse: {verse.ref()}\n"
            f"WordByWord:\n{_wbw_lines(verse, 'hi')}\n"
        )
    else:
        question = random.choice(
            [
                "Explain this verse in very simple words.",
                "What is the simple meaning of this verse?",
                "Summarize what this verse teaches.",
            ]
        )
        answer = (
            f"Answer: In simple words, this verse teaches: {_clean_en(verse.english)}\n"
            f"ClosestVerse: {verse.ref()}\n"
            f"WordByWord:\n{_wbw_lines(verse, 'en')}\n"
        )

    prompt = "\n".join(
        [
            "You are a helpful tutor. Answer using the verse below. Keep it simple.",
            "Do NOT repeat instructions. Output ONLY the fields.",
            "",
            f"Question: {question}",
            f"ClosestVerse: {verse.ref()}",
            f"Sanskrit: {verse.sanskrit_devanagari}",
            f"Hindi: {verse.hindi}",
            f"English: {verse.english}",
            "",
            "Return exactly:",
            "Answer: ...",
            f"ClosestVerse: {verse.ref()}",
            "WordByWord:",
            "- <sa> : <meaning>",
        ]
    )

    return {
        "lang": lang,
        "input_text": prompt,
        "target_text": answer.strip(),
        "chapter": verse.chapter,
        "verse": verse.verse,
        "ref": verse.ref(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Create a small SFT dataset from gita_full.jsonl.")
    p.add_argument("--input", type=str, default="data/gita_full.jsonl", help="Input verse JSONL")
    p.add_argument("--out_dir", type=str, default="data/sft", help="Output directory")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_verses", type=int, default=700, help="Limit number of verses used (for Colab).")
    p.add_argument("--train_frac", type=float, default=0.98)
    p.add_argument(
        "--add_concept_questions",
        action="store_true",
        help="Also add question->answer examples like 'What is karma?' based on keywords in the verse text.",
    )
    p.add_argument("--max_concepts_per_verse", type=int, default=1)
    args = p.parse_args()

    random.seed(args.seed)
    verses = read_jsonl(args.input)
    verses = verses[: min(len(verses), args.max_verses)]

    examples: list[dict] = []
    for v in verses:
        examples.append(_make_pair(v, "en"))
        examples.append(_make_pair(v, "hi"))

        if args.add_concept_questions:
            # Very small heuristic keyword list for Phase 0 (helps the model answer real questions).
            concepts = [
                ("karma", "कर्म"),
                ("duty", "कर्तव्य"),
                ("desire", "इच्छा"),
                ("attachment", "आसक्ति"),
                ("mind", "मन"),
                ("yoga", "योग"),
                ("self", "आत्म"),
            ]
            added = 0
            en_low = v.english.lower()
            hi_txt = v.hindi
            for en_kw, hi_kw in concepts:
                if added >= args.max_concepts_per_verse:
                    break
                if en_kw in en_low or hi_kw in hi_txt:
                    # EN concept QA
                    examples.append(
                        {
                            "lang": "en",
                            "input_text": "\n".join(
                                [
                                    "You are a helpful tutor. Answer using the verse below. Keep it simple.",
                                    "Do NOT repeat instructions. Output ONLY the fields.",
                                    "",
                                    f"Question: What does the Bhagavad Gita say about {en_kw}?",
                                    f"ClosestVerse: {v.ref()}",
                                    f"Sanskrit: {v.sanskrit_devanagari}",
                                    f"Hindi: {v.hindi}",
                                    f"English: {v.english}",
                                    "",
                                    "Return exactly:",
                                    "Answer: ...",
                                    f"ClosestVerse: {v.ref()}",
                                    "WordByWord:",
                                    "- <sa> : <meaning>",
                                ]
                            ),
                            "target_text": "\n".join(
                                [
                                    f"Answer: In simple words, this verse teaches about {en_kw}: {_clean_en(v.english)}",
                                    f"ClosestVerse: {v.ref()}",
                                    f"WordByWord:\n{_wbw_lines(v, 'en')}",
                                ]
                            ),
                            "chapter": v.chapter,
                            "verse": v.verse,
                            "ref": v.ref(),
                        }
                    )
                    # HI concept QA
                    examples.append(
                        {
                            "lang": "hi",
                            "input_text": "\n".join(
                                [
                                    "You are a helpful tutor. Answer using the verse below. Keep it simple.",
                                    "Do NOT repeat instructions. Output ONLY the fields.",
                                    "",
                                    f"Question: गीता के अनुसार {hi_kw} क्या है?",
                                    f"ClosestVerse: {v.ref()}",
                                    f"Sanskrit: {v.sanskrit_devanagari}",
                                    f"Hindi: {v.hindi}",
                                    f"English: {v.english}",
                                    "",
                                    "Return exactly:",
                                    "Answer: ...",
                                    f"ClosestVerse: {v.ref()}",
                                    "WordByWord:",
                                    "- <sa> : <meaning>",
                                ]
                            ),
                            "target_text": "\n".join(
                                [
                                    f"Answer: सरल शब्दों में, यह श्लोक {hi_kw} के बारे में यह सिखाता है: {_clean_hi(v.hindi)}",
                                    f"ClosestVerse: {v.ref()}",
                                    f"WordByWord:\n{_wbw_lines(v, 'hi')}",
                                ]
                            ),
                            "chapter": v.chapter,
                            "verse": v.verse,
                            "ref": v.ref(),
                        }
                    )
                    added += 1

    random.shuffle(examples)
    split = int(len(examples) * args.train_frac)
    train = examples[:split]
    val = examples[split:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "seed": args.seed,
                "verses_used": len(verses),
                "examples_total": len(examples),
                "train": len(train),
                "val": len(val),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote SFT dataset to {out_dir} (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()


