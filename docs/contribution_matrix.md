# Contribution Matrix

This matrix defines clear module ownership and required cross-review.

## Ownership

| Member | Primary Module | Required Deliverables |
|---|---|---|
| BTA | Classic sentiment pipeline (`src/dm2_steps`) | Training logic, metrics update, reproducible command |
| An | Feature engineering (`src/text_features.py`) | Variant definitions, ablation notes, metric impact |
| Bao | Data and labeling (`data/`, issue templates) | Data quality checks, schema consistency |
| Chuong | Multi-label issue extraction (`src/issue_steps`) | Training/inference updates, per-label metrics |
| Duc | Transformer extension (`src/nlp_ext`, `demo_transformer.py`) | Fine-tuning pipeline, threshold/coverage analysis |
| Tuan | Reports and benchmark synthesis (`results/reports`, syllabus outputs) | Final tables, citation consistency, summary integrity |
| Tung | Product demo integration (`demo.py`, `demo_app.py`) | User-facing flow, fallback behavior, demo readiness |

## Cross-Review Rules

1. A pull request must be reviewed by at least one member outside the owner row.
2. Any metric-regressing change must include an explicit justification in the pull request.
3. Any change under `src/` must include at least one smoke test update when behavior changes.
4. Any new result under `results/` must include the exact command used to generate it.
