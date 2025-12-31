from datasets import load_dataset

def main() -> None:
    """
    QLoRA fine-tuning for an instruction causal LLM.

    Notes:
    - This is intended for Colab / GPU. On CPU this will be extremely slow and often infeasible.
    - Requires: transformers, datasets, accelerate, peft, bitsandbytes, trl
    """
    p = argparse.ArgumentParser(description="QLoRA fine-tune an instruct causal LLM on chat_sft JSONL.")
    p.add_argument("--train_file", type=str, default="data/chat_sft/train.jsonl")
    p.add_argument("--val_file", type=str, default="data/chat_sft/val.jsonl")
    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--output_dir", type=str, default="outputs/gita_mistral_qlora")
    p.add_argument(
        "--quant",
        type=str,
        default="4bit",
        choices=["4bit", "none"],
        help="4bit=QLoRA via bitsandbytes, none=LoRA in fp16 (recommended on Kaggle if bitsandbytes/triton breaks).",
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--train_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Reduce VRAM usage (recommended on Colab T4).",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of modules to LoRA-tune.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. QLoRA on a 7B model needs a GPU (Colab/T4/A100). "
            "If you want to test locally, use ask_chat with the base model (no fine-tune) or a much smaller model."
        )

    # Lazy imports so the script gives a clear error if deps missing.
    # Colab sometimes has a broken torchvision build; transformers may import it indirectly.
    # Force-disable torchvision usage in transformers (we are text-only).
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    # Helps avoid fragmentation OOMs on long runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig
    from trl import SFTTrainer

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    bnb_config = None
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL warning: for fp16 training, right padding is safer.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA config (standard for Mistral/Llama)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    optim = "paged_adamw_8bit" if args.quant == "4bit" else "adamw_torch"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=optim,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
    )

    # SFT on the "text" field (already includes [INST]...[/INST] format)
    # TRL has changed its SFTTrainer API across versions; adapt at runtime.
    sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
        "peft_config": peft_config,
    }
    if "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = args.max_seq_len
    if "packing" in sig.parameters:
        trainer_kwargs["packing"] = False
    if "dataset_text_field" in sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sig.parameters:
        trainer_kwargs["formatting_func"] = lambda ex: ex["text"]

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved QLoRA adapters to: {args.output_dir}")


if __name__ == "__main__":
    # Avoid tokenizer parallelism warnings in Colab
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
