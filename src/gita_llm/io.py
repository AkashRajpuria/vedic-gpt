from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from gita_llm.schemas import VerseRecord


def read_jsonl(path: str | Path) -> list[VerseRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[VerseRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e

            rec = VerseRecord.model_validate(obj)
            records.append(rec)

    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: str | Path, items: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


