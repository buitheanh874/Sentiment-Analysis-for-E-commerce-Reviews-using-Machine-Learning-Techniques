from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "scoreboard"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_float(row: Dict, key: str):
    val = row.get(key)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _row(
    task: str,
    model: str,
    source_file: str,
    split: str | None = None,
    recall_0: float | None = None,
    precision_0: float | None = None,
    f2_0: float | None = None,
    micro_f1: float | None = None,
    macro_f1: float | None = None,
    coverage: float | None = None,
) -> Dict:
    missed_negative_rate = None if recall_0 is None else 1.0 - recall_0
    return {
        "task": task,
        "model": model,
        "split": split,
        "recall_0": recall_0,
        "precision_0": precision_0,
        "f2_0": f2_0,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "coverage": coverage,
        "missed_negative_rate": missed_negative_rate,
        "source_file": source_file,
    }


def build_scoreboard() -> pd.DataFrame:
    rows: List[Dict] = []

    dm2_path = RESULTS_DIR / "dm2_steps" / "08_ensemble_metrics.csv"
    dm2_df = _safe_read_csv(dm2_path)
    if not dm2_df.empty:
        for rec in dm2_df.to_dict(orient="records"):
            rows.append(
                _row(
                    task="sentiment",
                    model=f"{rec.get('model')}|cw={rec.get('class_weight')}|k={rec.get('k')}",
                    split="test",
                    recall_0=_as_float(rec, "recall_0"),
                    precision_0=_as_float(rec, "precision_0"),
                    f2_0=_as_float(rec, "f2_0"),
                    source_file=str(dm2_path.relative_to(ROOT)),
                )
            )

    issue_path = RESULTS_DIR / "issue_steps" / "02_metrics_overall.csv"
    issue_df = _safe_read_csv(issue_path)
    if not issue_df.empty:
        issue_test = issue_df[issue_df["split"] == "test"] if "split" in issue_df.columns else issue_df
        for rec in issue_test.to_dict(orient="records"):
            rows.append(
                _row(
                    task="issue_multilabel",
                    model=str(rec.get("model", "ovr_logreg")),
                    split=str(rec.get("split", "test")),
                    micro_f1=_as_float(rec, "micro_f1"),
                    macro_f1=_as_float(rec, "macro_f1"),
                    source_file=str(issue_path.relative_to(ROOT)),
                )
            )

    tf_path = RESULTS_DIR / "nlp_ext" / "nlp_metrics.csv"
    tf_df = _safe_read_csv(tf_path)
    if not tf_df.empty:
        tf_test = tf_df[tf_df["split"] == "test"] if "split" in tf_df.columns else tf_df
        for rec in tf_test.to_dict(orient="records"):
            rows.append(
                _row(
                    task="transformer_sentiment",
                    model="distilbert_finetune",
                    split=str(rec.get("split", "test")),
                    recall_0=_as_float(rec, "recall_0"),
                    precision_0=_as_float(rec, "precision_0"),
                    f2_0=_as_float(rec, "f2_0"),
                    coverage=_as_float(rec, "coverage"),
                    source_file=str(tf_path.relative_to(ROOT)),
                )
            )

    bench_path = RESULTS_DIR / "nlp_ext" / "syllabus_upgrade" / "nlp_syllabus_bench_test_summary.csv"
    bench_df = _safe_read_csv(bench_path)
    if not bench_df.empty:
        for rec in bench_df.to_dict(orient="records"):
            rows.append(
                _row(
                    task="syllabus_bench_sentiment",
                    model=str(rec.get("model")),
                    split="test",
                    recall_0=_as_float(rec, "recall_0"),
                    precision_0=_as_float(rec, "precision_0"),
                    f2_0=_as_float(rec, "f2_0"),
                    source_file=str(bench_path.relative_to(ROOT)),
                )
            )

    rnn_path = RESULTS_DIR / "nlp_ext" / "syllabus_upgrade" / "nlp_rnn_lstm_metrics.csv"
    rnn_df = _safe_read_csv(rnn_path)
    if not rnn_df.empty:
        if "split" in rnn_df.columns:
            rnn_test = rnn_df[rnn_df["split"] == "test"]
        else:
            rnn_test = rnn_df
        if "model" in rnn_test.columns:
            lstm_only = rnn_test[rnn_test["model"] == "lstm_text"]
            if not lstm_only.empty:
                rnn_test = lstm_only
        for rec in rnn_test.to_dict(orient="records"):
            rows.append(
                _row(
                    task="rnn_lstm_sentiment",
                    model=str(rec.get("model", "lstm_text")),
                    split=str(rec.get("split", "test")),
                    recall_0=_as_float(rec, "recall_0"),
                    precision_0=_as_float(rec, "precision_0"),
                    f2_0=_as_float(rec, "f2_0"),
                    source_file=str(rnn_path.relative_to(ROOT)),
                )
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_cols = ["task", "f2_0", "recall_0", "micro_f1", "macro_f1"]
    for col in sort_cols:
        if col not in df.columns:
            df[col] = None
    df = df.sort_values(by=sort_cols, ascending=[True, False, False, False, False], na_position="last")
    return df


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "task",
        "model",
        "split",
        "recall_0",
        "precision_0",
        "f2_0",
        "micro_f1",
        "macro_f1",
        "coverage",
        "missed_negative_rate",
        "source_file",
    ]
    view = df[cols].copy()
    return view.to_markdown(index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_scoreboard()
    if df.empty:
        raise SystemExit("No scoreboard rows were built. Check result artifact paths.")

    csv_path = OUT_DIR / "model_scoreboard.csv"
    md_path = OUT_DIR / "model_scoreboard.md"
    df.to_csv(csv_path, index=False)
    md_lines = [
        "# Model Scoreboard",
        "",
        "Generated from existing experiment artifacts.",
        "",
        _to_markdown_table(df),
        "",
        "Notes:",
        "- `missed_negative_rate = 1 - recall_0`.",
        "- Multi-label rows use micro/macro F1 when class-0 metrics are not defined.",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
