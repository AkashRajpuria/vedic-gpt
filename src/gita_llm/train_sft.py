from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a small seq2seq model on Gita SFT JSONL data.")
    p.add_argument("--train_file", type=str, default="data/sft/train.jsonl")
    p.add_argument("--val_file", type=str, default="data/sft/val.jsonl")
    p.add_argument("--model", type=str, default="google/flan-t5-small")
    p.add_argument("--output_dir", type=str, default="outputs/gita_flan_t5_small")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--max_source_len", type=int, default=512)
    p.add_argument("--max_target_len", type=int, default=256)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess(batch):
        inputs = tokenizer(batch["input_text"], max_length=args.max_source_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target_text"], max_length=args.max_target_len, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tok_ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()


