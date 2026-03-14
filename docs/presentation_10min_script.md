# 10-Minute Presentation Script (NLP Minor Project)

## Slide 1 - Problem & Goal (0:00-0:50)
- Problem: e-commerce reviews need fast triage of negative/risky feedback.
- Goal:
  - Sentiment with negative-first safety (`Negative`, `Positive`, `Uncertain`).
  - Multi-label issue extraction for actionable routing.

## Slide 2 - Data & Split (0:50-1:40)
- Data source: `data/Gift_Cards.jsonl`, labels: `data/issue_labels.csv`.
- Sentiment split: train/val/test + 3-star uncertainty pool.
- Issue split: multi-label split (train/val/test), with schema validation.

## Slide 3 - Classic NLP Pipeline (1:40-2:50)
- TF-IDF word/char, negation tagging, contrast clause handling, lexicon features.
- Chi2 feature selection + class-weight search + threshold tuning.
- Why: strong controllability and high recall on negative class.

## Slide 4 - Sentiment Results vs Baselines (2:50-4:00)
- Main result: `logreg_best_variant (V6)` has best `f2_0`.
- Compare with DT/RF/transformer/classic bench in:
  - `results/scoreboard/model_comparison_baselines.md`
  - `results/scoreboard/model_scoreboard.md`

## Slide 5 - Issue Multi-label Results (4:00-5:10)
- Main model: `ovr_logreg` selected by validation micro-F1.
- Baselines: `ovr_linearsvm`, `ovr_blend_lr_svm`, dummy baselines.
- Fair comparison (same split) with transformer/hybrid:
  - `results/scoreboard/issue_fair_comparison.md`

## Slide 6 - Transformer / Deep Coverage (5:10-6:20)
- DistilBERT sentiment fine-tune.
- Transformer multi-label issue + hybrid route.
- LSTM baseline, MLM probe, LLM prompt baseline.

## Slide 7 - Evaluation Rigor (6:20-7:20)
- Metrics: recall_0, precision_0, f2_0, micro/macro-F1, subset accuracy, hamming loss.
- Statistical confidence:
  - bootstrap CI: `nlp_eval_ci_bootstrap.csv`
  - significance test: `nlp_eval_significance.csv`

## Slide 8 - Error Analysis & Safety (7:20-8:20)
- Hard-case comparisons and error taxonomy artifacts.
- Uncertainty/fallback for short/sparse/ambiguous input.
- Operational benefit: avoid over-confident wrong routing.

## Slide 9 - Demo (8:20-9:20)
- Live demo order:
  1. `python -m pytest tests/test_smoke_cli.py -q`
  2. Show one failing/edge case if available, then rerun to green.

## Slide 10 - Conclusion & Q/A (9:20-10:00)
- Project achieves strong practical NLP coverage and reproducibility.
- Classic models remain competitive/strong in negative-first objective.
- Transformer is included and compared; hybrid path is available for deployment tuning.
