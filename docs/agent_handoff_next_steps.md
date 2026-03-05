# AI Agent Handoff: Current Status and Next Steps

Last updated: 2026-03-03

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
  - Current estimated coverage: 84.3%
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

### P0 - Finish Transformer Fast Path (for reliable reproducibility)

Goal:
- Ensure transformer pipeline can run successfully on CPU within practical time.

Tasks:
1. Add `--fast_mode` to `transformer_finetune`:
   - Reduce train samples, max length, and epochs automatically.
   - Example targets: `max_train_samples<=1000`, `max_length<=96`, `epochs<=0.2`.
2. Add `--skip_hard_cases` option to shorten post-processing.
3. Add a clear runtime note in README with expected CPU duration.
4. Generate fresh artifacts:
   - `results/nlp_ext/nlp_metrics.csv`
   - `results/nlp_ext/nlp_threshold_sweep.csv`
   - `results/nlp_ext/nlp_threshold_tradeoff.png`

Done criteria:
- Command completes on CPU without timeout.
- Artifacts are regenerated and referenced in scoreboard.

### P1 - Add Masked Language Model (MLM) coverage

Goal:
- Cover syllabus topic: Masked Language Models.

Tasks:
1. Add command in `src/nlp_ext/__main__.py`:
   - `mlm_probe` (or similar).
2. Implement minimal MLM experiment (HuggingFace fill-mask):
   - Evaluate top-k token prediction on a curated sentence set.
3. Save outputs:
   - `results/nlp_ext/syllabus_upgrade/nlp_mlm_probe.csv`
   - `results/nlp_ext/syllabus_upgrade/nlp_mlm_probe.md`
4. Update `build_course_fit_matrix` to raise MLM topic score when artifact exists.

Done criteria:
- New command runs end-to-end and produces both CSV + markdown summary.

### P2 - Add LLM Application mini-task

Goal:
- Cover syllabus topic: Applications of LLMs.

Tasks:
1. Add `llm_prompt_baseline` command with a small, reproducible prompt-based classification task.
2. Use deterministic settings and fixed prompt template.
3. Compare with classic baseline on a small fixed subset.
4. Save outputs:
   - `results/nlp_ext/syllabus_upgrade/nlp_llm_app_metrics.csv`
   - `results/nlp_ext/syllabus_upgrade/nlp_llm_app_summary.md`
5. Update course-fit matrix scoring.

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
python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl --output_dir results/nlp_ext/syllabus_upgrade
```

## 6) Acceptance Checklist for Next Delivery

- [ ] Transformer fast path runs on CPU without timeout.
- [ ] MLM artifact exists and is linked in rubric/checklist docs.
- [ ] LLM application artifact exists and is linked in rubric/checklist docs.
- [ ] Course-fit matrix updated and coverage increased from 84.3%.
- [ ] Scoreboard refreshed and includes new baselines.
- [ ] Tests pass (`python -m pytest -q`).

