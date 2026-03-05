import argparse
from pathlib import Path

from . import (
    DM2Config,
    DEFAULT_DATA_PATH,
    DEFAULT_THRESHOLDS,
    MIN_NNZ_DEFAULT,
    DEFAULT_NEGATION_WINDOW,
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
    step11_demo_one_review,
)


STEP_MAP = {
    "01": step01_data_overview,
    "step01": step01_data_overview,
    "data_overview": step01_data_overview,
    "02": step02_cleaning_preview,
    "step02": step02_cleaning_preview,
    "cleaning_preview": step02_cleaning_preview,
    "03": step03_split_summary,
    "step03": step03_split_summary,
    "split_summary": step03_split_summary,
    "04": step04_tfidf_stats,
    "step04": step04_tfidf_stats,
    "tfidf_stats": step04_tfidf_stats,
    "05": step05_baseline_lr,
    "step05": step05_baseline_lr,
    "baseline_lr": step05_baseline_lr,
    "06": step06_feature_selection,
    "step06": step06_feature_selection,
    "feature_selection_chi2": step06_feature_selection,
    "06b": step06b_context_feature_variants_sweep,
    "step06b": step06b_context_feature_variants_sweep,
    "context_feature_variants": step06b_context_feature_variants_sweep,
    "07": step07_embedded_l1,
    "step07": step07_embedded_l1,
    "embedded_l1": step07_embedded_l1,
    "08": step08_ensemble,
    "step08": step08_ensemble,
    "ensemble_dt_rf": step08_ensemble,
    "09": step09_uncertainty_eval,
    "step09": step09_uncertainty_eval,
    "uncertainty_eval": step09_uncertainty_eval,
    "10": step10_threshold_sweep,
    "step10": step10_threshold_sweep,
    "threshold_sweep": step10_threshold_sweep,
    "11": step11_demo_one_review,
    "step11": step11_demo_one_review,
    "demo_one_review": step11_demo_one_review,
}


def main():
    parser = argparse.ArgumentParser(description="NLP sentiment lab-style step runner")
    parser.add_argument("step", choices=STEP_MAP.keys(), help="Which step to run (01..11 or alias).")
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
        help="Where to write step artifacts",
    )
    parser.add_argument(
        "--enable_abbrev_norm",
        action="store_true",
        help="Enable abbreviation normalization before cleaning",
    )
    parser.add_argument(
        "--enable_negation_tagging",
        action="store_true",
        help="Enable negation tagging (NOT_ prefixes within small window)",
    )
    parser.add_argument(
        "--enable_clause_split",
        action="store_true",
        help="Enable clause split handling for preview/stats steps",
    )
    parser.add_argument(
        "--enable_char_ngrams",
        action="store_true",
        help="Enable character 3-5gram TF-IDF for preview/stats steps",
    )
    parser.add_argument(
        "--negation_window",
        type=int,
        default=DEFAULT_NEGATION_WINDOW,
        help="Window size for negation tagging",
    )
    parser.add_argument(
        "--min_nnz",
        type=int,
        default=MIN_NNZ_DEFAULT,
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
        "--text",
        type=str,
        default=None,
        help="Only for step11: text to score",
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

    func = STEP_MAP[args.step]
    if func is step11_demo_one_review:
        if not args.text:
            raise SystemExit("Provide --text \"review\" for step11_demo_one_review.")
        func(config, args.text)
    else:
        func(config)


if __name__ == "__main__":
    main()

