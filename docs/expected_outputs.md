# Expected Demo Outputs

The exact probabilities can vary by retraining. Expected behavior focuses on label direction and fallback rules.

## `demo.py`

- Input with strongly positive sentiment should return `Positive` with high confidence.
- Input with strongly negative sentiment should return `Negative` or `Needs Attention`.
- Very short or sparse inputs should return `Uncertain`.

## `demo_transformer.py`

- Inputs near neutral should often be in the threshold band and may return `Uncertain`.
- Clear polarity inputs should map to `Positive` or `Negative`.
- If optional dependencies are missing, the script should show install guidance.

## `demo_app.py`

- The app should load with no runtime error after installing optional requirements.
- Batch/interactive predictions should match CLI logic directionally.
