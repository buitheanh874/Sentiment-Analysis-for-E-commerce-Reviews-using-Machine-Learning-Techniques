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

Recommended for contributors:

```bash
pip install pre-commit
pre-commit install
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

# 3b) Level-3 issue training: per-label class-weight search + calibration + threshold tuning
python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl --enable_char_ngrams --enable_chi2_topk --tune_thresholds --class_weight_search --calibrate_probs --calibration_method sigmoid --include_svm_baseline

# 4) Transformer fine-tuning (optional)
python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl

# 5) Transformer fast-mode for CPU (recommended on non-GPU machines)
python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext_fast --fast_mode --skip_hard_cases --skip_model_save

# 6) Level-4 issue transformer (multi-label + optional hybrid routing)
python -m src.nlp_ext issue_transformer_multilabel --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/issue_transformer --fast_mode --skip_model_save --hybrid_max_route_rate 0.25

# 7) Syllabus upgrade benchmark package (optional)
python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade

# 7b) Full package including MLM + LLM-prompt artifacts
python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade --include_mlm_probe --include_llm_prompt

# 8) RNN/LSTM baseline (optional)
python -m src.nlp_ext rnn_lstm_baseline --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade --lstm_epochs 2

# 9) MLM probe (masked language model topic)
python -m src.nlp_ext mlm_probe --output_dir results/nlp_ext/syllabus_upgrade

# 10) LLM application proxy (prompt-style semantic baseline)
python -m src.nlp_ext llm_prompt_baseline --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade

# 11) Classic ablation study (negation/char/lexicon toggles)
python -m src.nlp_ext classic_ablation --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade --threshold_low 0.40 --threshold_high 0.60

# 12) Level-5 evaluation rigor package (bootstrap CI, significance test, error taxonomy)
python -m src.nlp_ext eval_rigor --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade --bootstrap_iters 1000

# 13) Unified cross-task scoreboard
python scripts/build_scoreboard.py
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
- `results/nlp_ext/syllabus_upgrade/nlp_ablation.csv`: ablation results for classic variants.
- `results/reports/`: project reports (EN + VI).
- `models/`: trained artifacts used by demos.
- `*/_run_metadata/`: per-command run metadata JSON (args, git commit, status, duration).

## Testing

```bash
python -m pytest -q
```

## Level Upgrade Plan

- Roadmap file: `docs/level_upgrade_roadmap.md`
