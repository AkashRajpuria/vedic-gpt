from __future__ import annotations

import argparse

from gita_llm.chat_generation import generate_chat
from gita_llm.retrieval import load_index, search


def _looks_hindi(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _format_word_by_word(verse, lang: str) -> str:
    if not verse.word_by_word:
        return "(word-by-word not available for this verse in dataset)"
    lines = []
    for item in verse.word_by_word:
        meaning = item.hi if lang == "hi" else item.en
        meaning = meaning or item.en or item.hi or ""
        lines.append(f"- {item.sa} : {meaning}")
    return "\n".join(lines).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Ask Bhagavad Gita questions using a larger chat/instruct model + retrieval.")
    p.add_argument("--index", type=str, required=True, help="Index directory created by build_index (e.g. data/index_full)")
    p.add_argument("--question", type=str, required=True, help="Question in English or Hindi")
    p.add_argument("--k", type=int, default=3, help="Number of evidence verses to retrieve (top-k)")
    p.add_argument(
        "--chat_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Causal/instruct model name or local path (recommended on Colab GPU).",
    )
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument(
        "--offload_dir",
        type=str,
        default="",
        help="Optional folder for CPU offload when device_map=auto (useful on small GPUs). Example: /content/offload",
    )
    args = p.parse_args()

    index, embedder, verses = load_index(args.index)
    hits = search(index=index, model=embedder, verses=verses, query=args.question, k=args.k)
    if not hits:
        raise SystemExit("No retrieval hits found.")

    evidence = [h.verse for h in hits]
    out = generate_chat(
        question=args.question,
        verses=evidence,
        model_name_or_path=args.chat_model,
        max_new_tokens=args.max_new_tokens,
        offload_dir=(args.offload_dir or None),
    )

    best = evidence[0]
    lang = "hi" if _looks_hindi(args.question) else "en"
    print("\n=== Question ===")
    print(args.question)

    print("\n=== Retrieved evidence (top hits) ===")
    for h in hits:
        print(f"- {h.verse.ref()}  score={h.score:.4f}")

    print("\n=== Closest Verse ===")
    print(best.ref())

    print("\n=== Answer (chat model) ===")
    print(out.answer)

    print("\n=== Word-by-word (from dataset) ===")
    print(_format_word_by_word(best, lang=lang))


if __name__ == "__main__":
    main()
