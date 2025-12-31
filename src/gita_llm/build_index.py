from __future__ import annotations

import argparse
from pathlib import Path

from gita_llm.io import read_jsonl
from gita_llm.retrieval import build_faiss_index, save_index


def main() -> None:
    p = argparse.ArgumentParser(description="Build FAISS retrieval index for Bhagavad Gita verses.")
    p.add_argument("--input", type=str, required=True, help="Path to JSONL verse file")
    p.add_argument("--out", type=str, required=True, help="Output directory for index artifacts")
    p.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for multilingual retrieval (Hindi+English).",
    )
    args = p.parse_args()

    verses = read_jsonl(args.input)
    index, model_name, _texts = build_faiss_index(verses, model_name=args.embed_model)
    save_index(Path(args.out), index=index, verses=verses, model_name=model_name)
    print(f"Saved index with {len(verses)} verses to: {args.out}")


if __name__ == "__main__":
    main()


