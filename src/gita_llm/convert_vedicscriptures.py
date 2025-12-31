from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gita_llm.io import write_jsonl


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def clean_sanskrit_slok(s: str) -> str:
    """
    Remove embedded verse markers like ||२-४७|| or ||2-47||, normalize whitespace.
    Keep Devanagari punctuation like । and ॥
    """
    s = s.replace("\\n", "\n")
    s = re.sub(r"\|\|\s*[\d\u0966-\u096F]+\s*-\s*[\d\u0966-\u096F]+\s*\|\|", "", s)
    s = s.replace("|", " ").replace("||", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s).strip()
    return s


def pick_translation(obj: dict, candidates: list[tuple[str, str]]) -> str | None:
    for block, field in candidates:
        b = obj.get(block)
        if isinstance(b, dict):
            v = b.get(field)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def extract_word_by_word_from_siva_ec(obj: dict) -> list[dict] | None:
    """
    Heuristic extraction from Swami Sivananda `ec` field, which often starts like:
    "2.47 कर्मणि in work? एव only? ... Commentary ..."
    Returns list of {"sa": <devanagari>, "en": <gloss>} when found.
    """
    siva = obj.get("siva")
    if not isinstance(siva, dict):
        return None
    ec = siva.get("ec")
    if not isinstance(ec, str) or not ec.strip():
        return None

    head = ec.split("Commentary", 1)[0]
    # Strip initial verse number like "2.47"
    head = re.sub(r"^\s*\d+\.\d+\s*", "", head.strip())

    pairs: list[dict] = []
    for m in re.finditer(r"([\u0900-\u097F]+)\s+([^?]+)\?", head):
        sa = m.group(1).strip()
        en = m.group(2).strip()
        if not sa or not DEVANAGARI_RE.search(sa):
            continue
        # Avoid huge commentary-like segments
        if len(en) > 120:
            continue
        pairs.append({"sa": sa, "en": en})

    return pairs or None


def main() -> None:
    p = argparse.ArgumentParser(description="Convert vedicscriptures/bhagavad-gita-data into our JSONL schema.")
    p.add_argument(
        "--repo_dir",
        type=str,
        default="data/raw/bhagavad-gita-data",
        help="Path to cloned bhagavad-gita-data repo",
    )
    p.add_argument("--out", type=str, default="data/gita_full.jsonl", help="Output JSONL path")
    p.add_argument(
        "--include_word_by_word",
        action="store_true",
        help="Try to extract English word-by-word from Swami Sivananda `ec` field when available",
    )
    args = p.parse_args()

    repo_dir = Path(args.repo_dir)
    slok_dir = repo_dir / "slok"
    if not slok_dir.exists():
        raise SystemExit(f"Expected slok/ directory not found in: {repo_dir}")

    # Defaults chosen for "simple" output.
    # Hindi: prefer short `ht` (translation) from Ramsukhdas -> Tejomayananda -> Shankaracharya.
    hindi_candidates = [
        ("rams", "ht"),
        ("tej", "ht"),
        ("sankar", "ht"),
    ]
    # English: prefer `et` (translation) from Sivananda -> Prabhupada -> Gambirananda.
    english_candidates = [
        ("siva", "et"),
        ("prabhu", "et"),
        ("gambir", "et"),
        ("adi", "et"),
    ]

    records: list[dict] = []
    for path in sorted(slok_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        chapter = int(obj["chapter"])
        verse = int(obj["verse"])
        slok = clean_sanskrit_slok(obj.get("slok", ""))
        if not slok:
            continue

        hindi = pick_translation(obj, hindi_candidates)
        english = pick_translation(obj, english_candidates)
        if not hindi or not english:
            # Skip verses without required translations for this minimal run.
            continue

        rec: dict = {
            "source": "bhagavad_gita",
            "chapter": chapter,
            "verse": verse,
            "sanskrit_devanagari": slok,
            "english": english,
            "hindi": hindi,
        }
        if args.include_word_by_word:
            wbw = extract_word_by_word_from_siva_ec(obj)
            if wbw:
                rec["word_by_word"] = wbw

        records.append(rec)

    if not records:
        raise SystemExit("No records produced. Check translation candidates and repo contents.")

    write_jsonl(args.out, records)
    print(f"Wrote {len(records)} verses to {args.out}")


if __name__ == "__main__":
    main()


