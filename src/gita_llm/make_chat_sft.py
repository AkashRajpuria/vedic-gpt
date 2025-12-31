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
    s = re.sub(r"^\d+\.\d+\s*", "", s).strip()
    return s


def _clean_hi(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^।।\s*\d+\.\d+\s*।।\s*", "", s).strip()
    return s


def _wbw_block(v: VerseRecord, lang: str) -> str:
    if not v.word_by_word:
        return "(word-by-word not available)"
    lines: list[str] = []
    for item in v.word_by_word:
        meaning = item.hi if lang == "hi" else item.en
        meaning = meaning or item.en or item.hi or ""
        lines.append(f"- {item.sa} : {meaning}")
    return "\n".join(lines).strip()


def _inst_prompt(question: str, v: VerseRecord, lang: str) -> str:
    """
    Mistral/Llama style instruction prompt using [INST]...[/INST].
    We include one verse as evidence; retrieval will provide top-k at inference time.
    """
    if lang == "hi":
        sys = "आप भगवद्गीता के सहायक शिक्षक हैं। उत्तर बहुत सरल हिन्दी में दें।"
    else:
        sys = "You are a helpful Bhagavad Gita tutor. Answer in simple English."

    user = "\n".join(
        [
            f"Question: {question}",
            "",
            f"ClosestVerse: {v.ref()}",
            f"Sanskrit: {v.sanskrit_devanagari}",
            f"Hindi: {v.hindi}",
            f"English: {v.english}",
            "",
            "Rules:",
            "- Use ONLY the verse above as evidence.",
            "- Do NOT copy the verse as the answer; explain it simply.",
            "- Output ONLY the fields in the format below.",
            "",
            "Output format:",
            "Answer: <2-6 simple sentences>",
            f"ClosestVerse: {v.ref()}",
            "WordByWord:",
            "- <sa> : <meaning>",
        ]
    )
    return f"<s>[INST] {sys}\n\n{user} [/INST]\n"


def _target(v: VerseRecord, lang: str) -> str:
    if lang == "hi":
        ans = f"Answer: सरल अर्थ: {_clean_hi(v.hindi)}"
        wbw = _wbw_block(v, "hi")
    else:
        ans = f"Answer: Simple meaning: {_clean_en(v.english)}"
        wbw = _wbw_block(v, "en")
    return "\n".join([ans, f"ClosestVerse: {v.ref()}", "WordByWord:", wbw]).strip() + "\n</s>"


def main() -> None:
    p = argparse.ArgumentParser(description="Create chat-style SFT dataset for a causal instruct model (Mistral/Llama).")
    p.add_argument("--input", type=str, default="data/gita_full.jsonl")
    p.add_argument("--out_dir", type=str, default="data/chat_sft")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_verses", type=int, default=700)
    p.add_argument("--train_frac", type=float, default=0.98)
    p.add_argument("--max_concepts_per_verse", type=int, default=1)
    args = p.parse_args()

    random.seed(args.seed)
    verses = read_jsonl(args.input)[: min(args.max_verses, 10_000)]

    concepts = [
        ("karma", "कर्म"),
        ("dharma", "धर्म"),
        ("yoga", "योग"),
        ("mind", "मन"),
        ("self", "आत्म"),
        ("desire", "इच्छा"),
        ("attachment", "आसक्ति"),
    ]

    rows: list[dict] = []
    for v in verses:
        # verse-explain style
        rows.append(
            {
                "lang": "en",
                "text": _inst_prompt("What does this verse teach? Explain simply.", v, "en") + _target(v, "en"),
                "ref": v.ref(),
            }
        )
        rows.append(
            {
                "lang": "hi",
                "text": _inst_prompt("इस श्लोक का सरल अर्थ बताइए।", v, "hi") + _target(v, "hi"),
                "ref": v.ref(),
            }
        )

        # concept style (teaches the model to answer real questions)
        added = 0
        en_low = v.english.lower()
        hi_txt = v.hindi
        sa_txt = v.sanskrit_devanagari
        for en_kw, hi_kw in concepts:
            if added >= args.max_concepts_per_verse:
                break
            if en_kw in en_low or hi_kw in hi_txt or hi_kw in sa_txt:
                rows.append(
                    {
                        "lang": "en",
                        "text": _inst_prompt(f"What does the Gita say about {en_kw}?", v, "en") + _target(v, "en"),
                        "ref": v.ref(),
                    }
                )
                rows.append(
                    {
                        "lang": "hi",
                        "text": _inst_prompt(f"गीता के अनुसार {hi_kw} क्या है?", v, "hi") + _target(v, "hi"),
                        "ref": v.ref(),
                    }
                )
                added += 1

    random.shuffle(rows)
    split = int(len(rows) * args.train_frac)
    train = rows[:split]
    val = rows[split:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "input": args.input,
                "seed": args.seed,
                "verses_used": len(verses),
                "examples_total": len(rows),
                "train": len(train),
                "val": len(val),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote chat SFT dataset to {out_dir} (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()


