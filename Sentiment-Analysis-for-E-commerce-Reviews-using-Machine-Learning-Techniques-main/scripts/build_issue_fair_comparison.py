from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_metadata import begin_run, end_run

RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "scoreboard"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_issue_fair_table() -> Dict[str, pd.DataFrame]:
    fair_path = RESULTS_DIR / "nlp_ext" / "issue_transformer" / "nlp_issue_hybrid_metrics.csv"
    classic_path = RESULTS_DIR / "issue_steps" / "02_metrics_overall.csv"

    fair_df = _safe_read_csv(fair_path)
    classic_df = _safe_read_csv(classic_path)

    if not fair_df.empty and "split" in fair_df.columns:
        fair_df = fair_df[fair_df["split"].astype(str) == "test"].copy()
    if not classic_df.empty and "split" in classic_df.columns:
        classic_df = classic_df[classic_df["split"].astype(str) == "test"].copy()

    if not fair_df.empty:
        fair_df = fair_df.sort_values(
            by=["micro_f1", "macro_f1", "subset_accuracy"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

    if not classic_df.empty:
        classic_df = classic_df.sort_values(
            by=["micro_f1", "macro_f1", "subset_accuracy"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

    return {"fair": fair_df, "classic": classic_df}


def _build_markdown(fair_df: pd.DataFrame, classic_df: pd.DataFrame) -> str:
    lines: List[str] = [
        "# Issue Fair Comparison (Classic vs Transformer)",
        "",
        "This report separates:",
        "- Fair-split comparison: models evaluated on the same split in `nlp_issue_hybrid_metrics.csv`.",
        "- Reference classic split: metrics from `results/issue_steps/02_metrics_overall.csv`.",
        "",
        "## A) Fair split comparison (same test split)",
        "",
    ]

    if fair_df.empty:
        lines.extend(["No fair comparison artifact found.", ""])
    else:
        cols = [
            "model",
            "split",
            "micro_f1",
            "macro_f1",
            "subset_accuracy",
            "hamming_loss",
        ]
        cols = [c for c in cols if c in fair_df.columns]
        lines.append(fair_df[cols].to_markdown(index=False))
        lines.append("")

        best_row = fair_df.iloc[0].to_dict()
        lines.append(
            f"- Best fair-split model: **{best_row.get('model')}** "
            f"(micro_f1={_num(best_row.get('micro_f1')):.4f})"
        )

        fair_by_name = {str(r["model"]): r for r in fair_df.to_dict(orient="records")}
        best_micro = _num(best_row.get("micro_f1"))
        for ref_name in ["transformer_multilabel", "hybrid_route", "classic_issue_model"]:
            if ref_name in fair_by_name:
                ref_micro = _num(fair_by_name[ref_name].get("micro_f1"))
                delta = best_micro - ref_micro
                lines.append(f"- micro_f1 delta vs {ref_name}: {delta:+.4f}")
        lines.append("")

    lines.extend(["## B) Reference classic split (issue_steps)", ""])
    if classic_df.empty:
        lines.extend(["No classic metrics artifact found.", ""])
    else:
        cols = [
            "model",
            "split",
            "micro_f1",
            "macro_f1",
            "subset_accuracy",
            "hamming_loss",
        ]
        cols = [c for c in cols if c in classic_df.columns]
        lines.append(classic_df[cols].to_markdown(index=False))
        lines.append("")
        lines.append(
            "- Note: section B may use a different split protocol than section A and is shown for reference only."
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = begin_run(
        command_name="scripts.build_issue_fair_comparison",
        args={"output_dir": OUT_DIR},
        metadata_dir=OUT_DIR / "_run_metadata",
    )
    try:
        payload = build_issue_fair_table()
        fair_df = payload["fair"]
        classic_df = payload["classic"]

        fair_csv = OUT_DIR / "issue_fair_comparison.csv"
        fair_md = OUT_DIR / "issue_fair_comparison.md"

        merged_rows: List[pd.DataFrame] = []
        if not fair_df.empty:
            f = fair_df.copy()
            f["source"] = "fair_split"
            merged_rows.append(f)
        if not classic_df.empty:
            c = classic_df.copy()
            c["source"] = "classic_split_reference"
            merged_rows.append(c)

        merged_df = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
        merged_df.to_csv(fair_csv, index=False)
        fair_md.write_text(_build_markdown(fair_df, classic_df), encoding="utf-8")

        print(f"Wrote: {fair_csv}")
        print(f"Wrote: {fair_md}")
        end_run(
            record,
            status="success",
            extra={
                "fair_rows": int(len(fair_df)),
                "classic_rows": int(len(classic_df)),
            },
        )
    except Exception as exc:
        end_run(record, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
