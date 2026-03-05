# AI Agent Handoff: Current Status and Next Steps

Last updated: 2026-03-05

## 1) Current Project State

The project has been upgraded to align with NLP course requirements and rubric-driven evaluation.

Completed upgrades include:

- Rubric/process docs:
  - `docs/rubric_checklist.md`
  - `docs/contribution_matrix.md`
  - `.github/pull_request_template.md`
- Reproducibility and reporting:
  - `README.md` updated with reproducible commands
  - `scripts/build_scoreboard.py`
  - `results/scoreboard/model_scoreboard.csv`
  - `results/scoreboard/model_scoreboard.md`
- New NLP baseline:
  - LSTM sentiment baseline integrated in `src/nlp_ext/syllabus_upgrades.py`
  - CLI command added in `src/nlp_ext/__main__.py`
  - Outputs:
    - `results/nlp_ext/syllabus_upgrade/nlp_rnn_lstm_metrics.csv`
    - `results/nlp_ext/syllabus_upgrade/nlp_rnn_lstm_summary.md`
- Syllabus fit recomputed:
  - `results/nlp_ext/syllabus_upgrade/nlp_course_fit_matrix.md`
  - Current estimated coverage: 94.3%
- Tests:
  - `tests/test_smoke_cli.py`
  - `tests/test_scoreboard_builder.py`
  - Current status: `python -m pytest -q` => 4 passed

## 2) Known Constraints / Risks

1. Full retraining for `transformer_finetune` is too heavy on CPU in this environment and may timeout.
2. `transformers` API compatibility required code updates (`evaluation_strategy`, `tokenizer` argument).
3. Dependency versions must remain pinned for stability.

Pinned optional dependency set:

- `torch==2.5.1`
- `transformers==4.36.2`
- `accelerate==0.25.0`

## 3) High-Priority Backlog (Do Next)

Status update:

- `transformer_finetune` now supports `--fast_mode` and `--skip_hard_cases`.
- New syllabus commands added:
  - `mlm_probe`
  - `llm_prompt_baseline`
- Course-fit scoring now checks MLM and LLM prompt artifacts.
- Scoreboard now supports:
  - `llm_prompt_sentiment`
  - `mlm_probe` (`aux_score = hit@k`)

### P0 - Finish Transformer Fast Path (for reliable reproducibility)

Goal:
- Ensure transformer pipeline can run successfully on CPU within practical time.

Tasks completed:
1. Added `--fast_mode`, `--skip_hard_cases`, `--fast_eval_max_samples`, and `--skip_model_save`.
2. Added CPU-friendly fast run path and validated successful execution.
3. Added runtime note and command examples in `README.md`.
4. Fast-run artifacts can be written to dedicated output directories (for example `results/nlp_ext_fast`).

Done criteria:
- Command completes on CPU without timeout.
- Artifacts are regenerated and referenced in scoreboard.

### P1 - Add Masked Language Model (MLM) coverage

Goal:
- Cover syllabus topic: Masked Language Models.

Tasks completed:
1. Added `mlm_probe` command in `src/nlp_ext/__main__.py`.
2. Implemented fill-mask probe evaluation in `src/nlp_ext/syllabus_upgrades.py`.
3. Added outputs:
   - `results/nlp_ext/syllabus_upgrade/nlp_mlm_probe.csv`
   - `results/nlp_ext/syllabus_upgrade/nlp_mlm_probe.md`
4. Updated `build_course_fit_matrix` to account for MLM artifacts.

Done criteria:
- New command runs end-to-end and produces both CSV + markdown summary.

### P2 - Add LLM Application mini-task

Goal:
- Cover syllabus topic: Applications of LLMs.

Tasks completed:
1. Added `llm_prompt_baseline` command with deterministic prompt-style semantic classification.
2. Added outputs:
   - `results/nlp_ext/syllabus_upgrade/nlp_llm_prompt_metrics.csv`
   - `results/nlp_ext/syllabus_upgrade/nlp_llm_prompt_summary.md`
3. Updated course-fit scoring and scoreboard ingestion for LLM prompt artifacts.

Done criteria:
- Clear comparison table exists and command is reproducible.

## 4) Secondary Backlog (Improve Toward 95-100%)

1. Add ablation study command:
   - Toggle negation/char n-gram/lexicon features and compare metrics.
   - Output: `results/nlp_ext/syllabus_upgrade/nlp_ablation.csv` and summary markdown.
2. Add error analysis artifact:
   - Top false negatives and representative hard cases with short explanations.
3. Add one end-to-end script:
   - Run all key commands and refresh scoreboard in one shot.

## 5) Required Commands for Next Agent Session

Environment:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

Core checks:

```bash
python -m pytest -q
python scripts/build_scoreboard.py
python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade --include_mlm_probe --include_llm_prompt
```

## 6) Acceptance Checklist for Next Delivery

- [x] Transformer fast path runs on CPU without timeout.
- [x] MLM artifact exists and is linked in rubric/checklist docs.
- [x] LLM application artifact exists and is linked in rubric/checklist docs.
- [x] Course-fit matrix updated and coverage increased from 84.3%.
- [x] Scoreboard refreshed and includes new baselines.
- [x] Tests pass (`python -m pytest -q`).

