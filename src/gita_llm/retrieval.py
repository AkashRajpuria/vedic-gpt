from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from gita_llm.schemas import VerseRecord


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    verse: VerseRecord


def _normalize(v: np.ndarray) -> np.ndarray:
    # cosine similarity via inner product on L2-normalized vectors
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


def verse_to_index_text(v: VerseRecord) -> str:
    # Multilingual index text: favor Hindi + English for retrieval, keep Sanskrit for anchoring.
    return (
        f"{v.ref()}\n"
        f"Sanskrit: {v.sanskrit_devanagari}\n"
        f"Hindi: {v.hindi}\n"
        f"English: {v.english}\n"
    )


def build_faiss_index(
    verses: list[VerseRecord],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    model = SentenceTransformer(model_name)
    texts = [verse_to_index_text(v) for v in verses]
    emb = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    emb = emb.astype("float32")
    emb = _normalize(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, model_name, texts


def save_index(out_dir: str | Path, index, verses: list[VerseRecord], model_name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    meta = {
        "model_name": model_name,
        "count": len(verses),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save verse records so the index can be mapped back to content
    payload = [v.model_dump(mode="json") for v in verses]
    (out_dir / "verses.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_dir: str | Path):
    index_dir = Path(index_dir)
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.json"
    verses_path = index_dir / "verses.json"

    if not (idx_path.exists() and meta_path.exists() and verses_path.exists()):
        raise FileNotFoundError(
            f"Index directory missing required files. Expected: {idx_path}, {meta_path}, {verses_path}"
        )

    index = faiss.read_index(str(idx_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    verses_raw = json.loads(verses_path.read_text(encoding="utf-8"))
    verses = [VerseRecord.model_validate(v) for v in verses_raw]

    model_name = meta["model_name"]
    model = SentenceTransformer(model_name)
    return index, model, verses


def search(
    index,
    model: SentenceTransformer,
    verses: list[VerseRecord],
    query: str,
    k: int = 3,
) -> list[RetrievalHit]:
    q = model.encode([query], convert_to_numpy=True)
    q = q.astype("float32")
    q = _normalize(q)
    scores, ids = index.search(q, k)

    hits: list[RetrievalHit] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(verses):
            continue
        hits.append(RetrievalHit(score=float(score), verse=verses[idx]))
    return hits


