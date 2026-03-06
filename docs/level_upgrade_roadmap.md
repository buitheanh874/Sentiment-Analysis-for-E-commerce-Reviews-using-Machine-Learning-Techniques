# nlp project upgrade roadmap

This roadmap proposes incremental upgrades so the team can improve quality without breaking reproducibility.

## level 1 - stability baseline

Goal: make daily development deterministic and safer for `main`.

Tasks:
- standardize run metadata logging for CLI entrypoints (arguments, commit hash, runtime, status).
- add artifact regression tests for sentiment, issue extraction, and scoreboard outputs.
- add pre-commit hooks and CI checks (`lint + pytest + scoreboard build`).

Done criteria:
- core CLI commands write run metadata JSON under output-specific `_run_metadata` directories.
- tests detect missing/invalid key artifacts early.
- pull requests must pass CI before merge.

## level 2 - data and annotation quality

Goal: reduce label noise and make annotation workflow auditable.

Tasks:
- add stronger consistency checks for multi-label CSV inputs.
- add batch-level annotation quality reports (conflicts, sparsity, drift).
- prioritize queue sampling for uncertain/hard-case records.

## level 3 - classic model improvements

Goal: improve classic pipeline quality with controlled complexity.

Tasks:
- systematic threshold and class-weight tuning per label.
- probability calibration for threshold stability.
- commandized ablation study (negation/char/lexicon toggles).

## level 4 - deep model improvements

Goal: extend contextual modeling while preserving demo runtime.

Tasks:
- multi-label transformer baseline for issue extraction.
- hybrid routing (classic for recall, transformer for uncertain refinement).
- optional model compression for CPU-friendly inference.

## level 5 - evaluation rigor

Goal: strengthen scientific validity of reported gains.

Tasks:
- confidence intervals via bootstrap for primary metrics.
- significance testing between main models and baselines.
- structured error taxonomy reports.

## level 6 - demo and productization

Goal: deliver a robust defense/demo experience.

Tasks:
- interactive decision cockpit with confidence and rationale.
- live threshold controls for precision/recall trade-off.
- one-command demo script for end-to-end defense flow.

## recommended order

1. level 1
2. level 2
3. level 3
4. level 5
5. level 4
6. level 6
