# NLP Project: E-commerce Review Understanding

This repository contains a reproducible NLP minor-project pipeline aligned with course topics and rubric expectations.

## Scope

- Task A: Sentiment classification (`Negative`, `Positive`, `Uncertain`) with TF-IDF, Chi-square selection, and logistic regression.
- Task B: Multi-label issue extraction (classic ML one-vs-rest).
- Task C: Transformer and syllabus-alignment extensions (N-gram LM, classic benchmark suite, course-fit matrix).

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

For Windows CPU environments, if `torch` import fails with DLL initialization errors, install the CPU wheel explicitly:

```bash
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

## Data

- Main corpus: `data/Gift_Cards.jsonl` (requires `text`, `rating`).
- Manual issue labels: `data/issue_labels.csv`.

## Reproducible Commands

```bash
# 1) Classic sentiment end-to-end
python -m src.run_all --data_path data/Gift_Cards.jsonl

# 2) One sentiment step (legacy module name: dm2_steps)
python -m src.dm2_steps 06b --data_path data/Gift_Cards.jsonl --enable_negation_tagging --enable_char_ngrams

# 3) Issue classifier train/eval
python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl

# 4) Transformer fine-tuning (optional)
python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl

# 5) Syllabus upgrade benchmark package (optional)
python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade

# 6) RNN/LSTM baseline (optional)
python -m src.nlp_ext rnn_lstm_baseline --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade --lstm_epochs 2

# 7) Unified cross-task scoreboard
python scripts/build_scoreboard.py
```

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
```

Demo smoke inputs and expected behavior:

- `docs/demo_inputs.txt`
- `docs/expected_outputs.md`

## Rubric Alignment Artifacts

- Rubric-to-evidence map: `docs/rubric_checklist.md`
- Team process and cross-review matrix: `docs/contribution_matrix.md`
- PR quality gate template: `.github/pull_request_template.md`
- Unified metrics table:
  - `results/scoreboard/model_scoreboard.csv`
  - `results/scoreboard/model_scoreboard.md`

## Main Outputs

- `results/dm2_steps/`: classic sentiment artifacts.
- `results/issue_steps/`: multi-label issue metrics and plots.
- `results/nlp_ext/`: transformer and syllabus-upgrade outputs.
- `results/reports/`: project reports (EN + VI).
- `models/`: trained artifacts used by demos.

## Testing

```bash
python -m pytest -q
```
