from __future__ import annotations

import argparse

from gita_llm.generation import generate
from gita_llm.retrieval import load_index, search


def _rerank_with_keyword_bonus(question: str, hits):
    q = question.lower()
    q_has_karma = ("karma" in q) or ("कर्म" in question)
    q_has_dharma = ("dharma" in q) or ("धर्म" in question)

    def bonus(v):
        b = 0.0
        en = (v.english or "").lower()
        hi = v.hindi or ""
        sa = v.sanskrit_devanagari or ""
        if q_has_karma and ("karma" in en or "कर्म" in hi or "कर्म" in sa):
            b += 0.15
        if q_has_dharma and ("dharma" in en or "धर्म" in hi or "धर्म" in sa):
            b += 0.15
        return b

    return sorted(hits, key=lambda h: (h.score + bonus(h.verse)), reverse=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Ask Bhagavad Gita questions (Hindi/English) with verse citation.")
    p.add_argument("--index", type=str, required=True, help="Index directory created by build_index")
    p.add_argument("--question", type=str, required=True, help="Question in English or Hindi")
    p.add_argument("--k", type=int, default=3, help="Number of candidate verses to retrieve")
    p.add_argument("--gen_model", type=str, default="google/flan-t5-base", help="Generator model")
    p.add_argument(
        "--gen_mode",
        type=str,
        default="baseline",
        choices=["baseline", "model"],
        help="baseline=deterministic template answer, model=use seq2seq generator (fine-tuned or base).",
    )
    args = p.parse_args()

    index, embedder, verses = load_index(args.index)
    hits = search(index=index, model=embedder, verses=verses, query=args.question, k=args.k)
    if not hits:
        raise SystemExit("No retrieval hits found.")

    hits = _rerank_with_keyword_bonus(args.question, hits)
    best = hits[0].verse
    gen = generate(question=args.question, verse=best, model_name=args.gen_model, mode=args.gen_mode)

    print("\n=== Question ===")
    print(args.question)

    print("\n=== Closest Verse ===")
    print(best.ref())
    print("\nSanskrit:")
    print(best.sanskrit_devanagari)
    print("\nHindi:")
    print(best.hindi)
    print("\nEnglish:")
    print(best.english)

    print("\n=== Answer ===")
    print(gen.answer)

    print("\n=== Word-by-word ===")
    if gen.word_by_word_is_generated:
        print("(auto-generated; add `word_by_word` to dataset for gold output)")
    print(gen.word_by_word)

    print("\n=== Retrieval (top hits) ===")
    for h in hits:
        print(f"- {h.verse.ref()}  score={h.score:.4f}")


if __name__ == "__main__":
    main()


