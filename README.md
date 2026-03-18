# NLP Project: E-commerce Review Understanding

This project processes Amazon gift-card reviews with three NLP tasks:

- sentiment classification (`Negative`, `Positive`, `Uncertain`)
- multi-label issue extraction
- optional transformer-based sentiment extension

## Requirements

- Python 3.11+
- pip
- Git LFS (required if large model/data files are tracked via LFS)

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

If `torch` import fails on Windows CPU:

```bash
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

## Data and Model Download (If Applicable)

Clone repository and pull LFS files:

```bash
git clone <your-repo-url>
cd NLP_code
git lfs install
git lfs pull
```

Expected data files:

- `data/Gift_Cards.jsonl`
- `data/issue_labels.csv`

Expected model folders:

- `models/issue_classifier/`
- `models/transformer_model/` (optional for transformer demo)

## Deploy and Run the System

Start the web system (FastAPI + UI):

```bash
python -m uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000`

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Quick Run (CLI)

Classic sentiment pipeline:

```bash
python -m src.run_all --data_path data/Gift_Cards.jsonl --enable_abbrev_norm --enable_negation_tagging --enable_char_ngrams
```

Issue extraction training:

```bash
python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl --output_dir results/issue_steps --model_dir models/issue_classifier --max_performance
```

Transformer demo (optional):

```bash
python demo_transformer.py "not bad at all"
```
