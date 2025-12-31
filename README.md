## Minimal Bhagavad Gita QA (English + Hindi) — VedicGPT Phase 0

This is a **minimal, end-to-end starter** to validate the workflow:

- ingest Bhagavad Gita verses (Sanskrit + Hindi + English)
- build a **verse retrieval index** (for “closest verse” citation)
- answer a question in **Hindi or English**
- return:
  - an **answer**
  - a **verse citation** (chapter.verse)
  - a **word-by-word explanation** (uses dataset field if available; otherwise model-generated fallback)

Once this works with a small sample, we can scale data and add **TPU fine-tuning**.

### What you put in (data)

Create a JSONL file where each line is one verse record:

```json
{
  "source": "bhagavad_gita",
  "chapter": 2,
  "verse": 47,
  "sanskrit_devanagari": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ...",
  "english": "You have the right to perform your duty...",
  "hindi": "तुम्हारा अधिकार केवल कर्म करने में है...",
  "word_by_word": [
    {"sa":"कर्मणि", "en":"in action", "hi":"कर्म में"},
    {"sa":"एव", "en":"only", "hi":"केवल"}
  ],
  "notes": "Optional"
}
```

Required fields: `source`, `chapter`, `verse`, `sanskrit_devanagari`, `english`, `hindi`.

Optional: `word_by_word` (recommended if you want reliable “each word” output).

### Quickstart (local)

1) Create a venv and install:

```bash
cd "Project/vedicgpt-gita"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2) Build an index (uses sample data by default):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python -m gita_llm.build_index --input data/sample/gita_sample.jsonl --out data/index
```

### Collect full Bhagavad Gita data (Option 1: vedicscriptures repo)

This project can auto-convert the `vedicscriptures/bhagavad-gita-data` repo into our JSONL schema.

1) Clone the dataset:

```bash
mkdir -p data/raw
cd data/raw
git clone --depth 1 https://github.com/vedicscriptures/bhagavad-gita-data.git
cd ../..
```

2) Convert to a single JSONL file (719 verses):

```bash
python -m gita_llm.convert_vedicscriptures \
  --repo_dir data/raw/bhagavad-gita-data \
  --out data/gita_full.jsonl \
  --include_word_by_word
```

3) Build an index on the full corpus:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python -m gita_llm.build_index --input data/gita_full.jsonl --out data/index_full
```

### Create a tiny fine-tuning dataset (SFT)

This generates simple bilingual examples (Hindi + English) to teach a small model the **output format**.

```bash
python -m gita_llm.make_sft --input data/gita_full.jsonl --out_dir data/sft --max_verses 400
```

### Train (script)

This fine-tunes a small model (default: `google/flan-t5-small`) on the SFT dataset:

```bash
python -m gita_llm.train_sft \
  --train_file data/sft/train.jsonl \
  --val_file data/sft/val.jsonl \
  --model google/flan-t5-small \
  --output_dir outputs/gita_flan_t5_small \
  --epochs 1 \
  --train_bs 8 \
  --eval_bs 8
```

### Test your fine-tuned model (ask questions)

Use the full-corpus index (recommended) and run with `--gen_mode model`:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python -m gita_llm.ask --index data/index_full --question "What is karma?" --gen_model outputs/gita_flan_t5_small --gen_mode model
python -m gita_llm.ask --index data/index_full --question "कर्म क्या है?" --gen_model outputs/gita_flan_t5_small --gen_mode model
```

### Use a bigger “real LLM” (recommended): Mistral / similar

This uses retrieval for citation, but generation is done by a **chat/instruct causal model**.

On Colab GPU, run:

```bash
python -m gita_llm.ask_chat --index data/index_full --question "Who is Krishna?" --k 3 --chat_model mistralai/Mistral-7B-Instruct-v0.3
python -m gita_llm.ask_chat --index data/index_full --question "कर्म क्या है?" --k 3 --chat_model mistralai/Mistral-7B-Instruct-v0.3
```

### “Best” fine-tuning: QLoRA on an instruct LLM (GPU recommended)

1) Build chat SFT data:

```bash
python -m gita_llm.make_chat_sft --input data/gita_full.jsonl --out_dir data/chat_sft --max_verses 700
```

2) QLoRA fine-tune (run on Colab GPU / any CUDA machine):

```bash
python -m gita_llm.train_qlora \
  --train_file data/chat_sft/train.jsonl \
  --val_file data/chat_sft/val.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output_dir outputs/gita_mistral_qlora \
  --epochs 1 \
  --train_bs 1 \
  --grad_accum 8
```

3) Use the fine-tuned adapters for generation (point `--chat_model` to the output dir):

```bash
python -m gita_llm.ask_chat --index data/index_full --question "Who is Krishna?" --k 3 --chat_model outputs/gita_mistral_qlora
```

### Validation report / score

This runs generation on `data/sft/val.jsonl` and prints a JSON report (ROUGE + format compliance):

```bash
python -m gita_llm.eval_sft --val_file data/sft/val.jsonl --model_dir outputs/gita_flan_t5_small
```

### Colab notebook (free GPU)

See: `notebooks/colab_gita_finetune.ipynb`

### Colab install (fix dependency conflicts)

Colab already has a CUDA-enabled `torch` installed. Don’t replace it with our pinned CPU torch.

In a fresh Colab runtime (GPU ON):

```bash
pip -q install -r requirements-colab.txt
pip -q install -e . --no-deps
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

3) Ask a question:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python -m gita_llm.ask --index data/index --question "What is karma?"
python -m gita_llm.ask --index data/index --question "कर्म क्या है?"
```

### Output format

- **Answer**: model-generated, conditioned on retrieved verse(s)
- **Closest verse**: e.g. `BG 2.47`
- **Verse text**: Sanskrit + Hindi/English translations
- **Word-by-word**:
  - if verse has `word_by_word`, we show that (best)
  - else we generate it and mark it as **auto-generated**

### Next step (TPU fine-tune)

After this pipeline works, we’ll:

- expand the corpus to full Gita
- add gold word-by-word gloss where possible
- build instruction SFT data (Hindi + English)
- fine-tune on TPU (LoRA/QLoRA or full fine-tune, depending on model choice)


