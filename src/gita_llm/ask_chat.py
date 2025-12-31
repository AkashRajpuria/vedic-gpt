from __future__ import annotations

import argparse

from gita_llm.chat_generation import generate_chat
from gita_llm.retrieval import load_index, search


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
    args = p.parse_args()

    index, embedder, verses = load_index(args.index)
    hits = search(index=index, model=embedder, verses=verses, query=args.question, k=args.k)
    if not hits:
        raise SystemExit("No retrieval hits found.")

    evidence = [h.verse for h in hits]
    out = generate_chat(question=args.question, verses=evidence, model_name_or_path=args.chat_model)

    best = evidence[0]
    print("\n=== Question ===")
    print(args.question)

    print("\n=== Retrieved evidence (top hits) ===")
    for h in hits:
        print(f"- {h.verse.ref()}  score={h.score:.4f}")

    print("\n=== Closest Verse ===")
    print(best.ref())

    print("\n=== Answer (chat model) ===")
    print(out.answer)


if __name__ == "__main__":
    main()


