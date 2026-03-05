# Rubric Checklist (Code and Artifacts)

This file maps grading criteria to concrete, reproducible project evidence.

## 1) Content Representation

- Reproducible commands: [`README.md`](/d:/Code/NLP_code/README.md)
- End-to-end runner: [`src/run_all.py`](/d:/Code/NLP_code/src/run_all.py)
- Demo commands: [`demo.py`](/d:/Code/NLP_code/demo.py), [`demo_transformer.py`](/d:/Code/NLP_code/demo_transformer.py), [`demo_app.py`](/d:/Code/NLP_code/demo_app.py)
- Unified metrics table: `python scripts/build_scoreboard.py`

## 2) Project Significance

- Real-world task framing (customer review risk handling): [`README.md`](/d:/Code/NLP_code/README.md)
- Negative-first model selection logic (risk-aware): [`src/dm2_steps/common.py`](/d:/Code/NLP_code/src/dm2_steps/common.py)
- Error-focused artifacts:
  - [`results/dm2_steps/09_uncertainty_summary.md`](/d:/Code/NLP_code/results/dm2_steps/09_uncertainty_summary.md)
  - [`results/nlp_ext/hard_cases_comparison.csv`](/d:/Code/NLP_code/results/nlp_ext/hard_cases_comparison.csv)

## 3) Working Process

- Module ownership and cross-review matrix: [`docs/contribution_matrix.md`](/d:/Code/NLP_code/docs/contribution_matrix.md)
- Pull-request quality gate: [`.github/pull_request_template.md`](/d:/Code/NLP_code/.github/pull_request_template.md)
- Smoke tests:
  - [`tests/test_smoke_cli.py`](/d:/Code/NLP_code/tests/test_smoke_cli.py)
  - [`tests/test_scoreboard_builder.py`](/d:/Code/NLP_code/tests/test_scoreboard_builder.py)

## 4) Results Quality

- Multi-model benchmark:
  - [`results/nlp_ext/syllabus_upgrade/nlp_syllabus_bench_metrics.csv`](/d:/Code/NLP_code/results/nlp_ext/syllabus_upgrade/nlp_syllabus_bench_metrics.csv)
  - [`results/nlp_ext/syllabus_upgrade/nlp_syllabus_bench_test_summary.csv`](/d:/Code/NLP_code/results/nlp_ext/syllabus_upgrade/nlp_syllabus_bench_test_summary.csv)
  - [`results/nlp_ext/syllabus_upgrade/nlp_rnn_lstm_metrics.csv`](/d:/Code/NLP_code/results/nlp_ext/syllabus_upgrade/nlp_rnn_lstm_metrics.csv)
- Task-specific evaluations:
  - [`results/dm2_steps/`](/d:/Code/NLP_code/results/dm2_steps)
  - [`results/issue_steps/`](/d:/Code/NLP_code/results/issue_steps)
  - [`results/nlp_ext/`](/d:/Code/NLP_code/results/nlp_ext)
- Unified scoreboard output:
  - [`results/scoreboard/model_scoreboard.csv`](/d:/Code/NLP_code/results/scoreboard/model_scoreboard.csv)
  - [`results/scoreboard/model_scoreboard.md`](/d:/Code/NLP_code/results/scoreboard/model_scoreboard.md)

## 5) Demo Quality

- CLI demo inputs and expected behavior:
  - [`docs/demo_inputs.txt`](/d:/Code/NLP_code/docs/demo_inputs.txt)
  - [`docs/expected_outputs.md`](/d:/Code/NLP_code/docs/expected_outputs.md)
- Interactive demo app:
  - [`demo_app.py`](/d:/Code/NLP_code/demo_app.py)
