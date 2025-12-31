from __future__ import annotations

import argparse
import json
import re

import torch
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _format_ok(text: str) -> bool:
    return ("Answer:" in text) and ("ClosestVerse:" in text) and ("WordByWord" in text or "WordByWord:" in text)


def _extract_ref(text: str) -> str | None:
    m = re.search(r"ClosestVerse:\s*(BG\s+\d+\.\d+)", text)
    return m.group(1) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned seq2seq model on the SFT validation set.")
    p.add_argument("--val_file", type=str, default="data/sft/val.jsonl")
    p.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model dir (e.g. outputs/...)")
    p.add_argument("--max_source_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--limit", type=int, default=200, help="Max validation examples to evaluate")
    args = p.parse_args()

    ds = load_dataset("json", data_files={"validation": args.val_file})["validation"]
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    rouge = evaluate.load("rouge")

    preds: list[str] = []
    refs: list[str] = []
    format_ok = 0
    ref_ok = 0

    for ex in ds:
        inp = tok(ex["input_text"], return_tensors="pt", truncation=True, max_length=args.max_source_len).to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = tok.decode(out[0], skip_special_tokens=True).strip()

        preds.append(pred)
        refs.append(ex["target_text"])

        if _format_ok(pred):
            format_ok += 1
        pred_ref = _extract_ref(pred)
        if pred_ref and pred_ref == ex.get("ref"):
            ref_ok += 1

    rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    report = {
        "count": len(ds),
        "device": device,
        "format_compliance": format_ok / max(1, len(ds)),
        "closest_verse_accuracy": ref_ok / max(1, len(ds)),
        "rouge": rouge_scores,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


