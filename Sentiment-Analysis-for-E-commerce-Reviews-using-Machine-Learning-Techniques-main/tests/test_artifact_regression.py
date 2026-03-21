from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_sentiment_artifact_schema_regression():
    path = ROOT / "results" / "dm2_steps" / "08_ensemble_metrics.csv"
    assert path.exists(), f"Missing artifact: {path}"
    df = pd.read_csv(path)
    required = {"model", "class_weight", "k", "recall_0", "precision_0", "f2_0"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    assert len(df) > 0


def test_issue_artifact_schema_regression():
    path = ROOT / "results" / "issue_steps" / "02_metrics_overall.csv"
    assert path.exists(), f"Missing artifact: {path}"
    df = pd.read_csv(path)
    required = {"split", "model", "micro_f1", "macro_f1"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    assert "test" in set(df["split"].astype(str))


def test_scoreboard_artifact_schema_regression():
    path = ROOT / "results" / "scoreboard" / "model_scoreboard.csv"
    assert path.exists(), f"Missing artifact: {path}"
    df = pd.read_csv(path)
    required = {"task", "model", "split", "source_file"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    tasks = set(df["task"].astype(str))
    assert "sentiment" in tasks
    assert "issue_multilabel" in tasks
