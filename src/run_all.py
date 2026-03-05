"""
Optional orchestrator for NLP sentiment lab steps.

This simply runs step01â€“step10 in sequence so that every artifact
is still produced per-step (CSV + figure + markdown). The per-step
CLI remains the primary entrypoint:

    python -m src.dm2_steps <step> --data_path data/Gift_Cards.jsonl
"""

import argparse
import re
from pathlib import Path

from src.dm2_steps import (
    DM2Config,
    DEFAULT_DATA_PATH,
    DEFAULT_THRESHOLDS,
    step01_data_overview,
    step02_cleaning_preview,
    step03_split_summary,
    step04_tfidf_stats,
    step05_baseline_lr,
    step06_feature_selection,
    step06b_context_feature_variants_sweep,
    step07_embedded_l1,
    step08_ensemble,
    step09_uncertainty_eval,
    step10_threshold_sweep,
)

STEP_SEQUENCE = [
    ("01_data_overview", step01_data_overview),
    ("02_cleaning_preview", step02_cleaning_preview),
    ("03_split_summary", step03_split_summary),
    ("04_tfidf_stats", step04_tfidf_stats),
    ("05_baseline_lr", step05_baseline_lr),
    ("06_feature_selection_chi2", step06_feature_selection),
    ("06b_context_feature_variants", step06b_context_feature_variants_sweep),
    ("07_embedded_l1", step07_embedded_l1),
    ("08_ensemble_dt_rf", step08_ensemble),
    ("09_uncertainty_eval", step09_uncertainty_eval),
    ("10_threshold_sweep", step10_threshold_sweep),
]


def main():
    parser = argparse.ArgumentParser(description="Run all NLP sentiment lab steps sequentially.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(DEFAULT_DATA_PATH),
        help="Path to Gift_Cards.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/dm2_steps"),
        help="Directory for step outputs",
    )
    parser.add_argument(
        "--enable_abbrev_norm",
        action="store_true",
        help="Enable abbreviation normalization before cleaning",
    )
    parser.add_argument(
        "--enable_negation_tagging",
        action="store_true",
        help="Enable negation tagging (NOT_ prefixes)",
    )
    parser.add_argument(
        "--enable_clause_split",
        action="store_true",
        help="Enable clause split handling in preview/stats",
    )
    parser.add_argument(
        "--enable_char_ngrams",
        action="store_true",
        help="Enable character 3-5gram TF-IDF in preview/stats",
    )
    parser.add_argument(
        "--negation_window",
        type=int,
        default=3,
        help="Window size for negation tagging",
    )
    parser.add_argument(
        "--min_nnz",
        type=int,
        default=2,
        help="Minimum TF-IDF nnz before fallback to Uncertain",
    )
    parser.add_argument(
        "--threshold_low",
        type=float,
        default=DEFAULT_THRESHOLDS[0],
        help="Lower probability threshold for Negative",
    )
    parser.add_argument(
        "--threshold_high",
        type=float,
        default=DEFAULT_THRESHOLDS[1],
        help="Upper probability threshold for Positive",
    )
    parser.add_argument(
        "--until_step",
        type=str,
        default="10",
        help="Run through this step number (01-10).",
    )
    args = parser.parse_args()

    config = DM2Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation_tagging=args.enable_negation_tagging,
        enable_clause_split=args.enable_clause_split,
        enable_char_ngrams=args.enable_char_ngrams,
        negation_window=args.negation_window,
        min_nnz=args.min_nnz,
        thresholds=(args.threshold_low, args.threshold_high),
    )

    target = int(args.until_step)
    for name, func in STEP_SEQUENCE:
        token = name.split("_")[0]
        match = re.match(r"(\d+)", token)
        step_num = int(match.group(1)) if match else 0
        if step_num > target:
            break
        print(f"[RUN_ALL] Running {name} ...")
        result = func(config)
        if result is not None:
            # Some steps return (best_k, best_cw); ignore otherwise
            pass
    print(f"Done. Artifacts saved under {args.output_dir}")


if __name__ == "__main__":
    main()

