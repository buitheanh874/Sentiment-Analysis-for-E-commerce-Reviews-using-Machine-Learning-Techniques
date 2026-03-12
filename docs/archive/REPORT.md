# NLP Project Report Notes

Tai lieu nay tom tat nhanh cac thanh phan can giu cho mon NLP sau khi don repo.

## Kept Components
- Sentiment pipeline: `src/dm2_steps/`, `src/run_all.py`, `demo.py`.
- Issue extraction: `src/issue_steps/`.
- Transformer and syllabus extension: `src/nlp_ext/`, `demo_transformer.py`.
- Main reports: `results/reports/NLP_project_report.tex` va `NLP_project_report_vi.tex`.

## Core Run Commands
```bash
python -m src.run_all --data_path data/Gift_Cards.jsonl
python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl
python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl
```

## Report Artifacts
- Sentiment figures: `results/dm2_steps/`
- Issue figures: `results/issue_steps/`, `results/issue_steps_char_demo/`
- Transformer figures: `results/nlp_ext/`

## Cleanup Policy
- Loai bo artifact tam, cache, venv local, draft labels, UI annotator, va tai lieu khong thuoc NLP scope.
- Giu lai du lieu va model can cho train/eval/demo va compile report NLP.
