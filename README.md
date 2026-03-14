# NLP Project: E-commerce Review Understanding

This project studies Amazon gift-card reviews with three connected NLP tasks:

- sentiment classification with `Negative`, `Positive`, and `Uncertain` decisions
- multi-label issue extraction for actionable complaint categories
- transformer-based extension for deeper contextual analysis

## Project Goals

- Build a practical review-understanding pipeline instead of a single isolated classifier.
- Detect risky negative reviews with high recall so important complaints are not missed.
- Convert free-form reviews into structured issue tags such as delivery, redemption, refund, and fraud-related complaints.
- Compare classic sparse NLP methods with transformer-based methods on the same domain.

## Target Users

- Customer-support teams that need to identify urgent complaints quickly.
- Operations teams that need structured issue categories for routing and escalation.
- Product analysts who want a summary of user pain points from large review collections.
- Students and instructors who want a complete applied NLP case study.

## Repository Contents

- `src/`: training, evaluation, inference, and extension modules
- `data/`: input datasets used by the project
- `models/`: trained artifacts used by demos and inference
- `results/`: experiment outputs, reports, and comparison tables
- `docs/`: supporting project documents
- `tests/`: automated tests

## Data and Models

Data already included in this repository:

- main review corpus: `data/Gift_Cards.jsonl`
- issue labels: `data/issue_labels.csv`

Models included in this repository:

- classic issue model under `models/issue_classifier/`
- transformer sentiment model under `models/transformer_model/`

If the transformer model is missing after clone, run:

```bash
git lfs install
git lfs pull
```

If you prefer to retrain models instead of using saved artifacts, see the commands in the reproduction section below.

## Environment Setup

Install the main dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

For Windows CPU environments, if `torch` fails to import, install the CPU wheel explicitly:

```bash
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

## Quick Start

Run the strongest classic sentiment pipeline:

```bash
python -m src.run_all --data_path data/Gift_Cards.jsonl --enable_abbrev_norm --enable_negation_tagging --enable_char_ngrams
```

Train the strongest issue-extraction pipeline:

```bash
python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl --output_dir results/issue_steps --model_dir models/issue_classifier --max_performance
```

Build the summary scoreboard:

```bash
python scripts/build_scoreboard.py
```

## Reproduce Main Components

Classic sentiment:

```bash
python -m src.run_all --data_path data/Gift_Cards.jsonl --output_dir results/dm2_steps --enable_abbrev_norm --enable_negation_tagging --enable_char_ngrams
```

Issue extraction:

```bash
python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl --output_dir results/issue_steps --model_dir models/issue_classifier --max_performance
```

Transformer sentiment extension:

```bash
python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext
```

Syllabus-upgrade package:

```bash
python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade
```

Fair comparison for issue models:

```bash
python scripts/build_issue_fair_comparison.py
python scripts/build_rubric_syllabus_assessment.py
```

## Run the System

CLI demo:

```bash
python demo.py "great product and fast shipping" "terrible experience, support never replied"
```

Issue prediction:

```bash
python -m src.issue_steps predict --text "good but slow delivery"
```

Transformer demo:

```bash
python demo_transformer.py "not bad at all"
```

## Deploy the Demo

Streamlit UI:

```bash
streamlit run demo_app.py
```

FastAPI app:

```bash
python -m uvicorn webapp.main:app --reload
```

Then open `http://127.0.0.1:8000`.

Available API endpoints:

- `GET /api/health`
- `GET /api/status?include_transformer=false`
- `POST /api/predict`

Example request body:

```json
{
  "texts": [
    "great product and fast shipping",
    "terrible experience, support never replied"
  ],
  "include_transformer": false
}
```

## Main Submission Outputs

- report PDF: `results/reports/NLP_project_report_20260306.pdf`
- scoreboard tables: `results/scoreboard/`
- classic sentiment outputs: `results/dm2_steps/`
- issue extraction outputs: `results/issue_steps/`
- transformer and extension outputs: `results/nlp_ext/`

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

## Notes

- Full transformer fine-tuning is much slower than the classic pipeline on CPU.
- If you only need the main project deliverables, the most important files are the report PDF, this README, the GitHub repository URL, and the generated results under `results/`.
