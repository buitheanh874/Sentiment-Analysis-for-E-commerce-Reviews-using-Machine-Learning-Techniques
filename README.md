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

Runtime note:
- On CPU-only environments, use `--fast_mode` for transformer training.
- Full transformer fine-tuning can exceed 20 minutes depending on machine.

## Demo

```bash
# CLI sentiment demo
python demo.py "great product!" "terrible experience"

# Issue inference demo
python -m src.issue_steps predict --text "good but slow delivery"

# Transformer demo (optional model/deps)
python demo_transformer.py "not bad at all"

# UI demo (optional)
streamlit run demo_app.py

# FastAPI + Web UI (recommended)
python -m uvicorn webapp.main:app --reload
# then open http://127.0.0.1:8000
```

Demo smoke inputs and expected behavior:

- `docs/demo_inputs.txt`
- `docs/expected_outputs.md`

FastAPI endpoints:

- `GET /api/health`
- `GET /api/status?include_transformer=false`
- `POST /api/predict`

Example request body for `POST /api/predict`:

```json
{
  "texts": [
    "great product and fast shipping",
    "terrible experience, support never replied"
  ],
  "include_transformer": false
}
```

## Rubric Alignment Artifacts

- Rubric-to-evidence map: `docs/rubric_checklist.md`
- Team process and cross-review matrix: `docs/contribution_matrix.md`
- 10-minute presentation script: `docs/presentation_10min_script.md`
- Demo runbook: `docs/demo_runbook.md`
- PR quality gate template: `.github/pull_request_template.md`
- Unified metrics table:
  - `results/scoreboard/model_scoreboard.csv`
  - `results/scoreboard/model_scoreboard.md`

## Main Outputs

- `results/dm2_steps/`: classic sentiment artifacts.
- `results/issue_steps/`: multi-label issue metrics and plots.
- `results/nlp_ext/`: transformer and syllabus-upgrade outputs.
- `results/nlp_ext/syllabus_upgrade/nlp_ablation.csv`: ablation results for classic variants.
- `results/reports/`: project reports (EN + VI).
- `models/`: trained artifacts used by demos.
- `*/_run_metadata/`: per-command run metadata JSON (args, git commit, status, duration).

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

## Notes

- Full transformer fine-tuning is much slower than the classic pipeline on CPU.
- If you only need the main project deliverables, the most important files are the report PDF, this README, the GitHub repository URL, and the generated results under `results/`.
