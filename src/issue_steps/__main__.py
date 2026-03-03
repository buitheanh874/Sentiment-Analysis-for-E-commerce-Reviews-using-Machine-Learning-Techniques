import argparse
from pathlib import Path

from .common import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_RESULTS_DIR,
    SEED,
)
from .steps import (
    cmd_make_template,
    cmd_merge_batches,
    cmd_predict,
    cmd_train,
    cmd_validate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Issue/Aspect Multi-Label Classification (classic ML only)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_template = subparsers.add_parser(
        "make_template",
        help="Create labeling CSV template from JSONL reviews.",
    )
    make_template.add_argument(
        "--data_path",
        type=Path,
        default=Path(DEFAULT_DATA_PATH),
        help="Path to Gift_Cards.jsonl",
    )
    make_template.add_argument(
        "--out",
        type=Path,
        default=Path("data/issue_labels_template.csv"),
        help="Output labeling template CSV path",
    )
    make_template.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of rows to sample for annotation",
    )
    make_template.add_argument(
        "--only_queue",
        action="store_true",
        help="Keep only Stage1 queue rows (NEGATIVE/NEEDS_ATTENTION)",
    )
    make_template.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for sampling",
    )
    make_template.add_argument(
        "--init_zero",
        action="store_true",
        help="Initialize label columns with 0 instead of blank",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate human-labeled multi-label CSV.",
    )
    validate.add_argument(
        "--labels_path",
        type=Path,
        required=True,
        help="Path to labeled CSV",
    )
    validate.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="Where to save validation outputs",
    )

    merge_batches = subparsers.add_parser(
        "merge_batches",
        help="Merge many batch CSV files into one deduplicated labels CSV.",
    )
    merge_batches.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/issue_labels_batches_29k_team7"),
        help="Directory containing batch CSV files",
    )
    merge_batches.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Filename pattern for batch files",
    )
    merge_batches.add_argument(
        "--output",
        type=Path,
        default=Path("data/issue_labels_29k_merged.csv"),
        help="Output merged CSV path",
    )
    merge_batches.add_argument(
        "--summary_out",
        type=Path,
        default=None,
        help="Optional markdown summary output path",
    )
    merge_batches.add_argument(
        "--conflict_out",
        type=Path,
        default=None,
        help="Optional CSV path for duplicate-id label conflicts",
    )
    merge_batches.add_argument(
        "--include_pending",
        action="store_true",
        help="Include rows with pending annotation_status if present",
    )
    merge_batches.add_argument(
        "--keep_incomplete",
        action="store_true",
        help="Keep rows where label values are not fully binary 0/1",
    )
    merge_batches.add_argument(
        "--fail_on_conflict",
        action="store_true",
        help="Stop with error if conflicting labels are found for same id",
    )

    train = subparsers.add_parser(
        "train",
        help="Train/evaluate issue multi-label classifier.",
    )
    train.add_argument(
        "--labels_path",
        type=Path,
        required=True,
        help="Path to labeled CSV",
    )
    train.add_argument(
        "--data_path",
        type=Path,
        default=Path(DEFAULT_DATA_PATH),
        help="JSONL source data path (for text/rating fallback by id)",
    )
    train.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="Where to write metrics and plots",
    )
    train.add_argument(
        "--model_dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help="Where to save issue model artifacts",
    )
    train.add_argument(
        "--enable_char_ngrams",
        action="store_true",
        help="Enable char 3-5gram TF-IDF in feature union",
    )
    train.add_argument(
        "--enable_chi2_topk",
        action="store_true",
        help="Enable train-only multi-label Chi2 top-k selection",
    )
    train.add_argument(
        "--tune_thresholds",
        action="store_true",
        help="Tune per-label decision thresholds on validation split",
    )
    train.add_argument(
        "--include_svm_baseline",
        action="store_true",
        help="Also train/evaluate OVR LinearSVM baseline",
    )
    train.add_argument(
        "--class_weight",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="Per-label class weighting strategy",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for splits and model training",
    )

    predict = subparsers.add_parser(
        "predict",
        help="Run inference with trained issue classifier.",
    )
    predict.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input review text",
    )
    predict.add_argument(
        "--model_dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help="Directory containing trained issue artifacts",
    )

    args = parser.parse_args()
    if args.command == "make_template":
        cmd_make_template(args)
        return
    if args.command == "validate":
        cmd_validate(args)
        return
    if args.command == "merge_batches":
        cmd_merge_batches(args)
        return
    if args.command == "train":
        cmd_train(args)
        return
    if args.command == "predict":
        cmd_predict(args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
