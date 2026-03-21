import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from .common import (
    BlendedOVRModel,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_RESULTS_DIR,
    ISSUE_LABELS,
    MultiLabelChi2Selector,
    SEED,
    clean_with_stage1,
    has_complaint_signal,
    keyword_suggested_labels,
    labels_to_pipe,
    load_issue_bundle,
    load_stage1_cleaning_config,
    predict_issue_labels,
    train_per_label_ovr,
)

matplotlib.use("Agg")


def _load_jsonl_reviews(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    df = pd.read_json(data_path, lines=True)
    required = {"text", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, found {set(df.columns)}")
    if "id" not in df.columns:
        df = df.copy()
        df.insert(0, "id", np.arange(len(df), dtype=int))
    return df


def _init_issue_vectorizer(enable_char_ngrams: bool, min_df: int = 2):
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
        max_features=50000,
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
    )
    if not enable_char_ngrams:
        return word_tfidf
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=min_df,
        max_features=50000,
        lowercase=False,
    )
    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ]
    )


def _fit_vectorizer_with_fallback(train_texts: List[str], enable_char_ngrams: bool):
    vectorizer = _init_issue_vectorizer(enable_char_ngrams=enable_char_ngrams, min_df=2)
    fallback_min_df = False
    try:
        X_train = vectorizer.fit_transform(train_texts)
        return vectorizer, X_train, fallback_min_df
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
    vectorizer = _init_issue_vectorizer(enable_char_ngrams=enable_char_ngrams, min_df=1)
    X_train = vectorizer.fit_transform(train_texts)
    fallback_min_df = True
    return vectorizer, X_train, fallback_min_df


def _write_labeling_guidelines(path: Path) -> None:
    content = """# Labeling Guidelines: Multi-Label Issue Classification (Level B)

## General Rules
- Assign one or more issue labels per review.
- If no listed issue applies, set `other=1`.
- Multi-label is expected: one review can mention multiple problems.
- Keep labels tied to explicit text evidence (not assumptions).

## Label Definitions + Examples
1. `delivery_shipping`
- Shipping or delivery speed/tracking/package arrival problems.
- Example: "Arrived two weeks late and tracking never updated."
- Example: "Package was marked delivered but I never received it."

2. `redemption_activation`
- Gift card redeem/activation/code/PIN problems.
- Example: "The code says invalid when I try to redeem."
- Example: "Card was never activated at checkout."

3. `product_quality`
- Physical card quality or condition defects.
- Example: "Card arrived bent and damaged."
- Example: "Printing quality was poor and hard to read."

4. `customer_service`
- Support/helpdesk response quality or behavior.
- Example: "Customer service was rude and unhelpful."
- Example: "Support never replied to my request."

5. `refund_return`
- Refund/return/reimbursement process issues.
- Example: "They refused my refund request."
- Example: "Still waiting for money back after return."

6. `usability`
- UX/workflow friction that blocks normal use.
- Example: "Redemption flow is confusing and fails repeatedly."
- Example: "Website throws errors when applying balance."

7. `value_price`
- Price/value dissatisfaction (too expensive, not worth it).
- Example: "Overpriced for what it offers."
- Example: "Not worth the money."

8. `fraud_scam`
- Scam/fraud/unauthorized-charge/security concerns.
- Example: "Looks like a scam and balance vanished."
- Example: "Unauthorized redemption happened immediately."

9. `other`
- Issue exists but none of the above labels fit.
- Example: "Problem not covered by taxonomy categories."
- Example: "General complaint with no specific issue type."

## Consistency Checks
- Avoid contradictions: if `other=1`, other labels should usually be 0.
- Prefer specific labels over `other` when evidence is clear.
- Keep annotation decisions concise in `notes` when uncertain.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _queue_priority_score(clean_text: str, stage1_label: str, stage1_prob: float) -> float:
    text = str(clean_text)
    tokens = len(text.split())
    margin = abs(float(stage1_prob) - 0.50)
    uncertainty_score = 1.0 - min(1.0, margin * 2.0)
    lower = f" {text.lower()} "
    has_contrast = any(marker in lower for marker in [" but ", " however ", " although ", " though ", " yet "])

    label_bonus = 0.15 if str(stage1_label) == "NEEDS_ATTENTION" else 0.05
    contrast_bonus = 0.10 if has_contrast else 0.0
    length_penalty = -0.20 if tokens < 3 else 0.0
    score = uncertainty_score + label_bonus + contrast_bonus + length_penalty
    return float(max(0.0, round(score, 6)))


def _infer_queue_stage1_labels(df: pd.DataFrame, base_dir: Path) -> Tuple[pd.DataFrame, str]:
    texts = df["text"].fillna("").astype(str).tolist()
    complaint_mask = np.array([has_complaint_signal(t) for t in texts], dtype=bool)

    models_dir = base_dir / "models"
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    selector_path = models_dir / "chi2_selector.joblib"
    model_path = models_dir / "best_lr_model.joblib"
    if vectorizer_path.exists() and selector_path.exists() and model_path.exists():
        vectorizer = joblib.load(vectorizer_path)
        selector = joblib.load(selector_path)
        model = joblib.load(model_path)
        probs = np.zeros(len(texts), dtype=float)
        batch_size = 5000
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            X_batch = vectorizer.transform(texts[start:end])
            X_batch = selector.transform(X_batch)
            probs[start:end] = model.predict_proba(X_batch)[:, 1]
        labels = np.full(len(df), "UNCERTAIN", dtype=object)
        labels[probs <= 0.40] = "NEGATIVE"
        labels[(probs > 0.40) & complaint_mask] = "NEEDS_ATTENTION"
        labels[(labels == "UNCERTAIN") & (probs >= 0.60)] = "POSITIVE"
        return (
            pd.DataFrame(
                {
                    "stage1_label": labels,
                    "stage1_prob": probs,
                },
                index=df.index,
            ),
            "stage1_model_policy",
        )

    rating = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0).to_numpy()
    probs = np.where(
        rating <= 2.0,
        0.20,
        np.where(complaint_mask, 0.45, np.where(rating >= 4.0, 0.80, 0.50)),
    ).astype(float)
    labels = np.where(
        probs <= 0.40,
        "NEGATIVE",
        np.where(complaint_mask, "NEEDS_ATTENTION", np.where(probs >= 0.60, "POSITIVE", "UNCERTAIN")),
    )
    return (
        pd.DataFrame(
            {
                "stage1_label": labels,
                "stage1_prob": probs,
            },
            index=df.index,
        ),
        "rating_keyword_fallback",
    )


def cmd_make_template(args) -> None:
    df = _load_jsonl_reviews(args.data_path).copy()
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    cleaning_cfg = load_stage1_cleaning_config(Path("."))
    df["clean_text"] = df["text"].apply(lambda text: clean_with_stage1(text, cleaning_cfg))
    df["suggested_tags"] = df["clean_text"].apply(
        lambda text: labels_to_pipe(keyword_suggested_labels(text))
    )

    if args.only_queue:
        stage1_frame, source = _infer_queue_stage1_labels(df, base_dir=Path("."))
        df = df.join(stage1_frame)
        df = df[df["stage1_label"].isin(["NEGATIVE", "NEEDS_ATTENTION"])].copy()
        df["queue_priority"] = df.apply(
            lambda row: _queue_priority_score(
                clean_text=row["clean_text"],
                stage1_label=row["stage1_label"],
                stage1_prob=row["stage1_prob"],
            ),
            axis=1,
        )
        print(f"[make_template] only_queue enabled using {source}: retained {len(df)} rows.")

    if args.sample_size is not None and args.sample_size > 0 and args.sample_size < len(df):
        queue_strategy = str(getattr(args, "queue_strategy", "priority"))
        use_priority = bool(args.only_queue and queue_strategy == "priority" and "queue_priority" in df.columns)
        if use_priority:
            df = (
                df.sort_values(
                    ["queue_priority", "stage1_prob", "id"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )
                .head(args.sample_size)
                .copy()
            )
        else:
            df = df.sample(n=args.sample_size, random_state=args.seed).copy()

    df["__id_sort_key"] = df["id"].map(_template_id_sort_key)
    df = (
        df.sort_values("__id_sort_key", kind="mergesort")
        .drop(columns=["__id_sort_key"])
        .reset_index(drop=True)
    )

    for label in ISSUE_LABELS:
        df[label] = 0 if args.init_zero else ""
    df["notes"] = ""

    output_cols = ["id", "rating", "text", "clean_text", "suggested_tags"]
    if "stage1_label" in df.columns:
        output_cols += ["stage1_label", "stage1_prob"]
    if "queue_priority" in df.columns:
        output_cols += ["queue_priority"]
    output_cols += ISSUE_LABELS + ["notes"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_csv(args.out, index=False, encoding="utf-8")

    guidelines_path = args.out.parent / "labeling_guidelines.md"
    _write_labeling_guidelines(guidelines_path)

    print(f"[make_template] wrote {len(df)} rows to {args.out}")
    print(f"[make_template] guidelines: {guidelines_path}")


def _normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return text


def _template_id_sort_key(value) -> Tuple[int, int, str]:
    normalized = _normalize_id(value)
    if normalized == "":
        return (2, 0, "")
    try:
        return (0, int(normalized), normalized)
    except ValueError:
        return (1, 0, normalized.casefold())


def _build_group_quality_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    group_values = (
        df[group_col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    label_df, valid_mask = _coerce_label_frame(df)
    label_binary = label_df.where(label_df.isin([0, 1]), 0).fillna(0).astype(int)
    row_sums = label_binary.sum(axis=1).astype(float)
    contradiction_mask = (label_binary["other"] == 1) & (label_binary.drop(columns=["other"]).sum(axis=1) > 0)
    zero_label_mask = row_sums == 0

    work = pd.DataFrame(
        {
            "group_key": group_values.values,
            "rows": np.ones(len(df), dtype=int),
            "valid_rows": valid_mask.astype(int).values,
            "zero_label_rows": zero_label_mask.astype(int).values,
            "contradiction_rows": contradiction_mask.astype(int).values,
            "avg_label_cardinality": row_sums.values,
        }
    )
    summary = (
        work.groupby("group_key", dropna=False)
        .agg(
            rows=("rows", "sum"),
            valid_rows=("valid_rows", "sum"),
            zero_label_rows=("zero_label_rows", "sum"),
            contradiction_rows=("contradiction_rows", "sum"),
            avg_label_cardinality=("avg_label_cardinality", "mean"),
        )
        .reset_index()
        .rename(columns={"group_key": group_col})
    )

    global_prevalence = label_binary.mean(axis=0)
    by_group_prevalence = label_binary.groupby(group_values, dropna=False).mean()
    prevalence_drift = (
        by_group_prevalence.sub(global_prevalence, axis=1).abs().mean(axis=1).rename("label_prevalence_drift")
    )
    summary = summary.merge(
        prevalence_drift.reset_index().rename(columns={"index": group_col}),
        on=group_col,
        how="left",
    )

    summary["valid_rate"] = summary["valid_rows"] / summary["rows"].clip(lower=1)
    summary["zero_label_rate"] = summary["zero_label_rows"] / summary["rows"].clip(lower=1)
    summary["contradiction_rate"] = summary["contradiction_rows"] / summary["rows"].clip(lower=1)
    summary = summary.sort_values(["rows", group_col], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return summary[
        [
            group_col,
            "rows",
            "valid_rows",
            "valid_rate",
            "zero_label_rows",
            "zero_label_rate",
            "contradiction_rows",
            "contradiction_rate",
            "avg_label_cardinality",
            "label_prevalence_drift",
        ]
    ]


def _write_quality_table_md(path: Path, title: str, table: pd.DataFrame) -> None:
    lines = [f"# {title}", "", f"- groups: {len(table)}", "", table.to_markdown(index=False), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_merge_batches(args) -> None:
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"[merge_batches] input_dir not found: {input_dir}")

    files = sorted(
        path
        for path in input_dir.glob(args.pattern)
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    if not files:
        raise SystemExit(
            f"[merge_batches] no CSV files found in {input_dir} with pattern {args.pattern}"
        )

    frames: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "id" not in df.columns:
            print(f"[merge_batches] skip {file_path.name}: missing id column")
            continue
        work = df.copy()
        work["__source_file"] = file_path.name
        work["__source_mtime"] = file_path.stat().st_mtime
        work["__row_order"] = np.arange(len(work), dtype=int)
        frames.append(work)

    if not frames:
        raise SystemExit("[merge_batches] no valid batch files with id column were loaded.")

    merged = pd.concat(frames, ignore_index=True)
    merged["id"] = merged["id"].apply(_normalize_id)
    merged = merged[merged["id"] != ""].copy()

    if "annotation_status" in merged.columns:
        status_norm = merged["annotation_status"].fillna("").astype(str).str.lower().str.strip()
        status_counts = status_norm.value_counts().to_dict()
        print(f"[merge_batches] status counts: {status_counts}")
        if not args.include_pending:
            done_status = {"done", "reviewed", "completed", "annotated", "final"}
            keep_mask = status_norm.isin(done_status)
            kept = int(keep_mask.sum())
            print(
                f"[merge_batches] keeping done/reviewed rows only: {kept}/{len(merged)}"
            )
            merged = merged.loc[keep_mask].copy()

    for label in ISSUE_LABELS:
        if label not in merged.columns:
            merged[label] = np.nan
    label_df, valid_mask = _coerce_label_frame(merged)
    merged.loc[:, ISSUE_LABELS] = label_df
    merged["__labels_complete"] = valid_mask

    if not args.keep_incomplete:
        before = len(merged)
        merged = merged[merged["__labels_complete"]].copy()
        print(
            f"[merge_batches] dropping incomplete label rows: {before - len(merged)} removed"
        )

    if merged.empty:
        raise SystemExit(
            "[merge_batches] no rows left after status/completeness filtering. "
            "Use --include_pending or --keep_incomplete to inspect intermediate state."
        )

    conflict_ids = (
        merged.groupby("id")[ISSUE_LABELS]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    conflict_id_list = conflict_ids[conflict_ids].index.tolist()

    conflict_out: Optional[Path] = args.conflict_out
    if conflict_out is None:
        conflict_out = args.output.with_name(args.output.stem + "_conflicts.csv")
    if conflict_id_list:
        conflict_rows = (
            merged[merged["id"].isin(conflict_id_list)]
            .sort_values(["id", "__source_mtime", "__row_order"], ascending=[True, False, False])
        )
        conflict_cols = ["id", "__source_file"] + ISSUE_LABELS + ["notes"]
        conflict_cols = [col for col in conflict_cols if col in conflict_rows.columns]
        conflict_out.parent.mkdir(parents=True, exist_ok=True)
        conflict_rows[conflict_cols].to_csv(conflict_out, index=False, encoding="utf-8")
        print(f"[merge_batches] conflict ids: {len(conflict_id_list)} -> {conflict_out}")
        if args.fail_on_conflict:
            raise SystemExit(
                "[merge_batches] conflicts found. Resolve conflicts first, then rerun."
            )

    source_quality_csv: Optional[Path] = None
    source_quality_md: Optional[Path] = None
    source_quality_df = _build_group_quality_table(merged, "__source_file")
    if not source_quality_df.empty:
        source_quality_csv = args.output.with_name(args.output.stem + "_quality_by_source.csv")
        source_quality_md = args.output.with_name(args.output.stem + "_quality_by_source.md")
        source_quality_csv.parent.mkdir(parents=True, exist_ok=True)
        source_quality_df.to_csv(source_quality_csv, index=False, encoding="utf-8")
        _write_quality_table_md(source_quality_md, "Batch Quality by Source File", source_quality_df)

    annotator_quality_csv: Optional[Path] = None
    annotator_quality_md: Optional[Path] = None
    annotator_quality_df = _build_group_quality_table(merged, "annotator")
    if not annotator_quality_df.empty:
        annotator_quality_csv = args.output.with_name(args.output.stem + "_quality_by_annotator.csv")
        annotator_quality_md = args.output.with_name(args.output.stem + "_quality_by_annotator.md")
        annotator_quality_csv.parent.mkdir(parents=True, exist_ok=True)
        annotator_quality_df.to_csv(annotator_quality_csv, index=False, encoding="utf-8")
        _write_quality_table_md(annotator_quality_md, "Batch Quality by Annotator", annotator_quality_df)

    dedup = (
        merged.sort_values(["id", "__source_mtime", "__row_order"], ascending=[True, False, False])
        .groupby("id", as_index=False)
        .first()
    )

    ordered_cols = (
        ["id", "rating", "text", "clean_text", "suggested_tags"]
        + ISSUE_LABELS
        + ["notes", "annotation_status", "annotator", "annotated_at"]
    )
    output_cols = [col for col in ordered_cols if col in dedup.columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dedup[output_cols].to_csv(args.output, index=False, encoding="utf-8")

    summary_out: Optional[Path] = args.summary_out
    if summary_out is None:
        summary_out = args.output.with_name(args.output.stem + "_merge_summary.md")
    summary_lines = [
        "# Merge Batches Summary",
        "",
        f"- input_dir: {input_dir}",
        f"- csv_files_loaded: {len(frames)}",
        f"- rows_after_filters: {len(merged)}",
        f"- unique_ids_output: {len(dedup)}",
        f"- conflict_ids: {len(conflict_id_list)}",
        f"- output_csv: {args.output}",
    ]
    if conflict_id_list:
        summary_lines.append(f"- conflict_csv: {conflict_out}")
    if source_quality_csv is not None:
        summary_lines.append(f"- source_quality_csv: {source_quality_csv}")
    if source_quality_md is not None:
        summary_lines.append(f"- source_quality_md: {source_quality_md}")
    if annotator_quality_csv is not None:
        summary_lines.append(f"- annotator_quality_csv: {annotator_quality_csv}")
    if annotator_quality_md is not None:
        summary_lines.append(f"- annotator_quality_md: {annotator_quality_md}")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[merge_batches] wrote merged file: {args.output}")
    print(f"[merge_batches] summary: {summary_out}")


def _coerce_label_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    label_df = pd.DataFrame(index=df.index)
    valid_mask = pd.Series(True, index=df.index)
    for label in ISSUE_LABELS:
        if label not in df.columns:
            label_df[label] = np.nan
            valid_mask &= False
            continue
        col = pd.to_numeric(df[label], errors="coerce")
        ok = col.isin([0, 1])
        valid_mask &= ok
        label_df[label] = col
    return label_df, valid_mask


def _schema_errors(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required = {"id", "rating", "text"}
    missing_base = [col for col in required if col not in df.columns]
    if missing_base:
        errors.append(f"Missing required columns: {missing_base}")
    missing_labels = [label for label in ISSUE_LABELS if label not in df.columns]
    if missing_labels:
        errors.append(f"Missing label columns: {missing_labels}")
    return errors


def _label_stats_markdown(
    stats_df: pd.DataFrame,
    total_rows: int,
    cardinality: float,
    contradiction_count: int,
    duplicate_id_count: int,
    duplicate_row_count: int,
    duplicate_conflict_id_count: int,
) -> str:
    lines = [
        "# Stage 2 Label Validation Summary",
        "",
        f"- Rows: {total_rows}",
        f"- Label cardinality (avg labels/review): {cardinality:.3f}",
        f"- Contradictions (`other=1` with other labels): {contradiction_count}",
        f"- Duplicate ids: {duplicate_id_count}",
        f"- Duplicate rows: {duplicate_row_count}",
        f"- Duplicate ids with conflicting labels: {duplicate_conflict_id_count}",
        "",
        "## Label Frequencies",
        "",
        "| label | count | prevalence |",
        "|---|---:|---:|",
    ]
    for _, row in stats_df.iterrows():
        lines.append(f"| {row['label']} | {int(row['count'])} | {row['prevalence']:.4f} |")
    return "\n".join(lines) + "\n"


def _find_duplicate_label_conflicts(ids: pd.Series, label_df: pd.DataFrame) -> pd.DataFrame:
    work = label_df.copy()
    work.insert(0, "id", ids.astype(str).values)
    duplicated_ids = work["id"].value_counts()
    duplicated_ids = duplicated_ids[duplicated_ids > 1].index.tolist()
    if not duplicated_ids:
        return pd.DataFrame(columns=["id"] + ISSUE_LABELS)

    dup_rows = work[work["id"].isin(duplicated_ids)].copy()
    conflict_mask = (
        dup_rows.groupby("id")[ISSUE_LABELS]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    conflict_ids = conflict_mask[conflict_mask].index.tolist()
    if not conflict_ids:
        return pd.DataFrame(columns=["id"] + ISSUE_LABELS)
    return dup_rows[dup_rows["id"].isin(conflict_ids)].copy()


def cmd_validate(args) -> None:
    df = pd.read_csv(args.labels_path)
    errors = _schema_errors(df)
    if errors:
        raise SystemExit("[validate] schema errors:\n- " + "\n- ".join(errors))
    df = df.copy()
    df["id"] = df["id"].apply(_normalize_id)
    missing_id_rows = int((df["id"] == "").sum())
    if missing_id_rows > 0:
        raise SystemExit(
            f"[validate] Found {missing_id_rows} rows with missing/invalid `id` values."
        )

    label_df, valid_mask = _coerce_label_frame(df)
    invalid_rows = (~valid_mask).sum()
    if invalid_rows > 0:
        raise SystemExit(
            f"[validate] Found {int(invalid_rows)} rows with non-binary label values; labels must be 0/1."
        )
    label_df = label_df.astype(int)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row_sums = label_df.sum(axis=1)
    zero_label_rows = int((row_sums == 0).sum())
    if zero_label_rows > 0:
        raise SystemExit(
            f"[validate] Found {zero_label_rows} rows with no active label. "
            "Use at least one label; set `other=1` if needed."
        )

    contradiction_mask = (label_df["other"] == 1) & (label_df.drop(columns=["other"]).sum(axis=1) > 0)
    contradiction_count = int(contradiction_mask.sum())
    if contradiction_count > 0:
        if bool(getattr(args, "strict_other", False)):
            raise SystemExit(
                f"[validate] Found {contradiction_count} rows with `other=1` and additional labels."
            )
        print(f"[validate] warning: {contradiction_count} rows have `other=1` with additional labels.")

    duplicate_counts = df["id"].value_counts()
    duplicate_id_count = int((duplicate_counts > 1).sum())
    duplicate_row_count = int(duplicate_counts[duplicate_counts > 1].sum()) if duplicate_id_count > 0 else 0

    duplicate_conflicts = _find_duplicate_label_conflicts(df["id"], label_df)
    duplicate_conflict_id_count = int(duplicate_conflicts["id"].nunique()) if not duplicate_conflicts.empty else 0

    duplicate_summary = pd.DataFrame(
        {
            "metric": [
                "duplicate_id_count",
                "duplicate_row_count",
                "duplicate_conflict_id_count",
            ],
            "value": [
                duplicate_id_count,
                duplicate_row_count,
                duplicate_conflict_id_count,
            ],
        }
    )
    duplicate_summary_csv = out_dir / "01_duplicate_id_summary.csv"
    duplicate_summary.to_csv(duplicate_summary_csv, index=False, encoding="utf-8")

    duplicate_conflict_csv: Optional[Path] = None
    if duplicate_conflict_id_count > 0:
        duplicate_conflict_csv = out_dir / "01_duplicate_id_conflicts.csv"
        duplicate_conflict_view = duplicate_conflicts.copy()
        duplicate_conflict_view.insert(1, "__row_index", duplicate_conflict_view.index.astype(int))
        duplicate_conflict_view = duplicate_conflict_view.sort_values(["id", "__row_index"], kind="mergesort")
        duplicate_conflict_view.to_csv(duplicate_conflict_csv, index=False, encoding="utf-8")
        if bool(getattr(args, "fail_on_duplicate_conflicts", False)):
            raise SystemExit(
                f"[validate] Found {duplicate_conflict_id_count} duplicate ids with conflicting labels. "
                f"Details: {duplicate_conflict_csv}"
            )
        print(
            f"[validate] warning: {duplicate_conflict_id_count} duplicate ids have conflicting labels. "
            f"Details: {duplicate_conflict_csv}"
        )

    counts = label_df.sum(axis=0).astype(int)
    stats_df = pd.DataFrame(
        {
            "label": ISSUE_LABELS,
            "count": [int(counts[label]) for label in ISSUE_LABELS],
            "prevalence": [float(counts[label]) / max(len(label_df), 1) for label in ISSUE_LABELS],
        }
    )
    stats_csv = out_dir / "01_label_stats.csv"
    stats_md = out_dir / "01_label_stats.md"
    stats_png = out_dir / "01_label_distribution.png"

    stats_df.to_csv(stats_csv, index=False)
    cardinality = float(row_sums.mean())
    stats_md.write_text(
        _label_stats_markdown(
            stats_df=stats_df,
            total_rows=len(label_df),
            cardinality=cardinality,
            contradiction_count=contradiction_count,
            duplicate_id_count=duplicate_id_count,
            duplicate_row_count=duplicate_row_count,
            duplicate_conflict_id_count=duplicate_conflict_id_count,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(10, 4))
    plt.bar(stats_df["label"], stats_df["count"], color="steelblue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Issue Label Distribution")
    plt.tight_layout()
    plt.savefig(stats_png, dpi=220)
    plt.close()

    print(f"[validate] wrote {stats_csv}")
    print(f"[validate] wrote {stats_md}")
    print(f"[validate] wrote {stats_png}")
    print(f"[validate] wrote {duplicate_summary_csv}")
    if duplicate_conflict_csv is not None:
        print(f"[validate] wrote {duplicate_conflict_csv}")


def _labelset_codes(y: np.ndarray) -> np.ndarray:
    codes: List[str] = []
    for row in y:
        active = [ISSUE_LABELS[i] for i, value in enumerate(row) if int(value) == 1]
        code = "|".join(active) if active else "none"
        codes.append(code)
    return np.array(codes, dtype=object)


def _can_stratify(codes: np.ndarray) -> bool:
    if len(codes) == 0:
        return False
    vc = pd.Series(codes).value_counts()
    return bool(len(vc) > 1 and vc.min() >= 2)


def _split_multilabel_indices(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    idx = np.arange(len(y))
    if len(idx) < 5:
        raise ValueError("Need at least 5 labeled rows to split train/val/test.")

    split_method = "random"
    codes = _labelset_codes(y)
    try:
        if not _can_stratify(codes):
            raise ValueError("Insufficient labelset support for stratified split.")
        idx_train, idx_temp = train_test_split(
            idx,
            test_size=0.30,
            random_state=seed,
            stratify=codes,
        )
        temp_codes = codes[idx_temp]
        if _can_stratify(temp_codes):
            idx_val, idx_test = train_test_split(
                idx_temp,
                test_size=2.0 / 3.0,
                random_state=seed,
                stratify=temp_codes,
            )
            split_method = "stratified_labelset"
        else:
            idx_val, idx_test = train_test_split(
                idx_temp,
                test_size=2.0 / 3.0,
                random_state=seed,
                shuffle=True,
            )
            split_method = "partially_stratified"
        return idx_train, idx_val, idx_test, split_method
    except Exception:
        idx_train, idx_temp = train_test_split(
            idx,
            test_size=0.30,
            random_state=seed,
            shuffle=True,
        )
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=2.0 / 3.0,
            random_state=seed,
            shuffle=True,
        )
        split_method = "random"
        return idx_train, idx_val, idx_test, split_method


def _apply_thresholds(scores: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    threshold_vec = np.array([float(thresholds[label]) for label in ISSUE_LABELS], dtype=float)
    return (scores >= threshold_vec).astype(int)


def _tune_thresholds(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    grid = np.arange(0.10, 0.91, 0.05)
    tuned: Dict[str, float] = {}
    for idx, label in enumerate(ISSUE_LABELS):
        y_col = y_true[:, idx]
        if np.unique(y_col).size < 2:
            tuned[label] = 0.50
            continue
        best_f1 = -1.0
        best_thr = 0.50
        best_dist = 10.0
        for thr in grid:
            pred = (scores[:, idx] >= thr).astype(int)
            f1 = f1_score(y_col, pred, zero_division=0)
            dist = abs(thr - 0.50)
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and dist < best_dist):
                best_f1 = f1
                best_thr = float(thr)
                best_dist = dist
        tuned[label] = float(best_thr)
    return tuned


CLASS_WEIGHT_OPTIONS: Dict[str, Optional[object]] = {
    "none": None,
    "balanced": "balanced",
    "w2": {0: 2, 1: 1},
    "w5": {0: 5, 1: 1},
    "w10": {0: 10, 1: 1},
}


def _class_weight_candidates(args) -> List[Tuple[str, Optional[object]]]:
    if bool(getattr(args, "class_weight_search", False)):
        return list(CLASS_WEIGHT_OPTIONS.items())
    selected = str(getattr(args, "class_weight", "balanced"))
    if selected not in CLASS_WEIGHT_OPTIONS:
        selected = "balanced"
    return [(selected, CLASS_WEIGHT_OPTIONS[selected])]


def _binary_positive_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (scores >= float(threshold)).astype(int)
    precision, recall, f1_vals, _ = precision_recall_fscore_support(
        y_true,
        preds,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "precision_1": float(precision[1]),
        "recall_1": float(recall[1]),
        "f1_1": float(f1_vals[1]),
        "f2_1": float(fbeta_score(y_true, preds, beta=2, pos_label=1, zero_division=0)),
    }


def _select_labelwise_class_weights(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    seed: int,
    candidate_weights: List[Tuple[str, Optional[object]]],
) -> Tuple[Dict[str, Optional[object]], Dict[str, str], pd.DataFrame]:
    selected_value_map: Dict[str, Optional[object]] = {}
    selected_label_map: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []

    for idx, label in enumerate(ISSUE_LABELS):
        best_key = None
        best_label = "balanced"
        best_value: Optional[object] = "balanced"
        y_train_col = y_train[:, idx]
        y_val_col = y_val[:, idx]

        if np.unique(y_train_col).size < 2 or np.unique(y_val_col).size < 2:
            selected_value_map[label] = CLASS_WEIGHT_OPTIONS["balanced"]
            selected_label_map[label] = "balanced"
            rows.append(
                {
                    "label": label,
                    "class_weight": "balanced",
                    "precision_1": np.nan,
                    "recall_1": np.nan,
                    "f1_1": np.nan,
                    "f2_1": np.nan,
                    "selected": 1,
                    "note": "insufficient_label_support",
                }
            )
            continue

        for weight_name, weight_value in candidate_weights:
            model = train_per_label_ovr(
                X_train,
                y_train_col.reshape(-1, 1),
                [label],
                model_kind="logreg",
                class_weight=weight_value,
                class_weight_map=None,
                calibrate_probs=False,
                random_state=seed,
            )
            val_scores = model.predict_scores(X_val)[:, 0]
            metrics = _binary_positive_metrics(y_val_col, val_scores, threshold=0.5)
            rows.append(
                {
                    "label": label,
                    "class_weight": weight_name,
                    **metrics,
                    "selected": 0,
                    "note": "",
                }
            )
            cand_key = (
                metrics["f2_1"],
                metrics["recall_1"],
                metrics["precision_1"],
            )
            if best_key is None or cand_key > best_key:
                best_key = cand_key
                best_label = weight_name
                best_value = weight_value

        selected_value_map[label] = best_value
        selected_label_map[label] = best_label

    tuning_df = pd.DataFrame(rows)
    if not tuning_df.empty:
        selected_keys = {(label, weight) for label, weight in selected_label_map.items()}
        tuning_df["selected"] = tuning_df.apply(
            lambda row: 1 if (str(row["label"]), str(row["class_weight"])) in selected_keys else 0,
            axis=1,
        )
    return selected_value_map, selected_label_map, tuning_df


def _tune_blend_weights_per_label(
    y_true: np.ndarray,
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    step: float = 0.1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if primary_scores.shape != secondary_scores.shape:
        raise ValueError(
            "Cannot tune blend weights with mismatched score matrices: "
            f"{primary_scores.shape} vs {secondary_scores.shape}"
        )

    grid = np.arange(0.0, 1.0 + 1e-9, float(step))
    rows: List[Dict[str, object]] = []
    selected: Dict[str, float] = {}

    for idx, label in enumerate(ISSUE_LABELS):
        y_col = y_true[:, idx]
        if np.unique(y_col).size < 2:
            selected[label] = 1.0
            rows.append(
                {
                    "label": label,
                    "alpha_lr": 1.0,
                    "precision_1": np.nan,
                    "recall_1": np.nan,
                    "f1_1": np.nan,
                    "f2_1": np.nan,
                    "selected": 1,
                    "note": "insufficient_label_support",
                }
            )
            continue

        best_key = None
        best_alpha = 0.5
        for alpha in grid:
            blended = alpha * primary_scores[:, idx] + (1.0 - alpha) * secondary_scores[:, idx]
            metrics = _binary_positive_metrics(y_col, blended, threshold=0.5)
            cand_key = (
                metrics["f2_1"],
                metrics["recall_1"],
                metrics["precision_1"],
                -abs(float(alpha) - 0.5),
            )
            rows.append(
                {
                    "label": label,
                    "alpha_lr": float(alpha),
                    **metrics,
                    "selected": 0,
                    "note": "",
                }
            )
            if best_key is None or cand_key > best_key:
                best_key = cand_key
                best_alpha = float(alpha)
        selected[label] = best_alpha

    tuning_df = pd.DataFrame(rows)
    if not tuning_df.empty:
        selected_keys = {(label, float(alpha)) for label, alpha in selected.items()}
        tuning_df["selected"] = tuning_df.apply(
            lambda row: 1 if (str(row["label"]), float(row["alpha_lr"])) in selected_keys else 0,
            axis=1,
        )
    return selected, tuning_df


def _blend_scores(
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    alpha_map: Dict[str, float],
) -> np.ndarray:
    alpha_vec = np.array([float(alpha_map.get(label, 0.5)) for label in ISSUE_LABELS], dtype=float)
    return alpha_vec.reshape(1, -1) * primary_scores + (1.0 - alpha_vec.reshape(1, -1)) * secondary_scores


def _dummy_labelset_majority_predict(y_train: np.ndarray, n_samples: int) -> np.ndarray:
    if y_train.ndim != 2 or y_train.shape[1] != len(ISSUE_LABELS):
        raise ValueError("Expected y_train shape (n_samples, n_labels) for dummy baseline.")
    codes = pd.Series(_labelset_codes(y_train))
    if codes.empty:
        return np.zeros((n_samples, len(ISSUE_LABELS)), dtype=int)
    top_code = str(codes.value_counts().idxmax())
    if top_code == "none":
        vector = np.zeros(len(ISSUE_LABELS), dtype=int)
    else:
        active = set(top_code.split("|"))
        vector = np.array([1 if label in active else 0 for label in ISSUE_LABELS], dtype=int)
    return np.tile(vector.reshape(1, -1), (int(n_samples), 1))


def _dummy_label_prior_predict(y_train: np.ndarray, n_samples: int) -> np.ndarray:
    if y_train.ndim != 2 or y_train.shape[1] != len(ISSUE_LABELS):
        raise ValueError("Expected y_train shape (n_samples, n_labels) for dummy baseline.")
    priors = y_train.mean(axis=0)
    vector = (priors >= 0.5).astype(int)
    if int(vector.sum()) == 0 and len(vector) > 0:
        vector[int(np.argmax(priors))] = 1
    return np.tile(vector.reshape(1, -1), (int(n_samples), 1))


def _select_best_model_name(overall_rows: List[Dict[str, object]]) -> str:
    if not overall_rows:
        return "ovr_logreg"
    overall_df = pd.DataFrame(overall_rows)
    if overall_df.empty or "split" not in overall_df.columns:
        return "ovr_logreg"
    val_df = overall_df[overall_df["split"] == "val"].copy()
    if val_df.empty:
        return "ovr_logreg"

    priority = {
        "ovr_blend_lr_svm": 0,
        "ovr_logreg": 1,
        "ovr_linearsvm": 2,
    }
    val_df = val_df[val_df["model"].isin(priority.keys())].copy()
    if val_df.empty:
        return "ovr_logreg"
    val_df["model_priority"] = val_df["model"].map(priority).fillna(99).astype(int)
    best_row = val_df.sort_values(
        by=["micro_f1", "macro_f1", "subset_accuracy", "model_priority"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    return str(best_row["model"])


def _threshold_stability_rows(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Dict[str, float],
    model_variant: str,
    split: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    deltas = [-0.10, -0.05, 0.00, 0.05, 0.10]
    for delta in deltas:
        adjusted = {
            label: float(np.clip(float(thresholds[label]) + delta, 0.05, 0.95))
            for label in ISSUE_LABELS
        }
        pred = _apply_thresholds(scores, adjusted)
        metrics = _overall_metrics(y_true, pred)
        rows.append(
            {
                "model_variant": model_variant,
                "split": split,
                "threshold_delta": float(delta),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "subset_accuracy": metrics["subset_accuracy"],
                "hamming_loss": metrics["hamming_loss"],
            }
        )
    return pd.DataFrame(rows)


def _write_stability_summary(path: Path, stability_df: pd.DataFrame) -> None:
    if stability_df.empty:
        return
    agg = (
        stability_df.groupby(["model_variant", "split"])[["micro_f1", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    lines = [
        "# Threshold Stability Summary",
        "",
        "Metric stability is measured under threshold perturbations "
        "delta in {-0.10, -0.05, 0.00, 0.05, 0.10}.",
        "",
        agg.to_markdown(index=False),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "label_cardinality_true": float(np.mean(y_true.sum(axis=1))),
        "label_cardinality_pred": float(np.mean(y_pred.sum(axis=1))),
    }


def _per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    predicted_count = y_pred.sum(axis=0)
    rows = []
    for idx, label in enumerate(ISSUE_LABELS):
        rows.append(
            {
                "model": model_name,
                "split": split,
                "label": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1_vals[idx]),
                "support_true": int(support[idx]),
                "predicted_positive": int(predicted_count[idx]),
                "frequency_true": float(np.mean(y_true[:, idx])),
            }
        )
    return pd.DataFrame(rows)


def _snippet(text: str, limit: int = 180) -> str:
    txt = str(text).replace("\n", " ").strip()
    if len(txt) <= limit:
        return txt
    return txt[: limit - 3] + "..."


def _build_confusion_like_summary(
    df_split: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> str:
    lines: List[str] = [
        "# Confusion-like Summary (Top FP/FN Examples Per Label)",
        "",
        "Source split: test",
        "",
    ]

    for idx, label in enumerate(ISSUE_LABELS):
        lines.append(f"## {label}")
        fp_idx = np.where((y_pred[:, idx] == 1) & (y_true[:, idx] == 0))[0]
        fn_idx = np.where((y_pred[:, idx] == 0) & (y_true[:, idx] == 1))[0]

        if len(fp_idx) > 0:
            ranked = fp_idx[np.argsort(-scores[fp_idx, idx])]
            lines.append("### False Positives (top 3 by confidence)")
            for pos in ranked[:3]:
                row = df_split.iloc[int(pos)]
                lines.append(
                    f"- id={row['id']} | conf={scores[pos, idx]:.3f} | text=\"{_snippet(row['text'])}\""
                )
        else:
            lines.append("### False Positives (top 3 by confidence)")
            lines.append("- none")

        if len(fn_idx) > 0:
            ranked = fn_idx[np.argsort(scores[fn_idx, idx])]
            lines.append("### False Negatives (top 3 by lowest confidence)")
            for pos in ranked[:3]:
                row = df_split.iloc[int(pos)]
                lines.append(
                    f"- id={row['id']} | conf={scores[pos, idx]:.3f} | text=\"{_snippet(row['text'])}\""
                )
        else:
            lines.append("### False Negatives (top 3 by lowest confidence)")
            lines.append("- none")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _prepare_labeled_dataframe(labels_path: Path, data_path: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path)
    if "id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df.insert(0, "id", np.arange(len(labels_df), dtype=int).astype(str))
    labels_df["id"] = labels_df["id"].astype(str)

    base_df = _load_jsonl_reviews(data_path)[["id", "text", "rating"]].copy()
    base_df["id"] = base_df["id"].astype(str)

    merged = labels_df.merge(
        base_df,
        on="id",
        how="left",
        suffixes=("", "_base"),
    )

    if "text" not in merged.columns:
        merged["text"] = merged["text_base"]
    else:
        merged["text"] = merged["text"].fillna(merged["text_base"])
    if "rating" not in merged.columns:
        merged["rating"] = merged["rating_base"]
    else:
        merged["rating"] = merged["rating"].fillna(merged["rating_base"])

    merged["text"] = merged["text"].fillna("").astype(str)
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
    return merged


def cmd_train(args) -> None:
    if bool(getattr(args, "max_performance", False)):
        args.enable_char_ngrams = True
        args.enable_chi2_topk = True
        args.tune_thresholds = True
        args.class_weight_search = True
        args.calibrate_probs = True
        args.include_svm_baseline = True
        args.enable_model_blend = True
        args.auto_select_best_model = True
        if str(getattr(args, "class_weight", "balanced")) not in CLASS_WEIGHT_OPTIONS:
            args.class_weight = "balanced"
        print(
            "[train] max_performance preset enabled: "
            "char_ngrams+chi2+class_weight_search+calibration+threshold_tuning+svm+blend+auto_select"
        )

    df = _prepare_labeled_dataframe(args.labels_path, args.data_path)
    schema_errors = _schema_errors(df)
    if schema_errors:
        raise SystemExit("[train] schema errors:\n- " + "\n- ".join(schema_errors))

    raw_label_df, valid_mask = _coerce_label_frame(df)
    invalid_rows = int((~valid_mask).sum())
    if invalid_rows > 0:
        print(f"[train] dropping {invalid_rows} rows with missing/non-binary labels.")
    df = df.loc[valid_mask].copy()
    raw_label_df = raw_label_df.loc[valid_mask].astype(int)

    row_sums = raw_label_df.sum(axis=1)
    zero_rows = int((row_sums == 0).sum())
    if zero_rows > 0:
        print(f"[train] dropping {zero_rows} rows with no active label.")
        keep_mask = row_sums > 0
        df = df.loc[keep_mask].copy()
        raw_label_df = raw_label_df.loc[keep_mask].copy()

    contradiction_mask = (raw_label_df["other"] == 1) & (raw_label_df.drop(columns=["other"]).sum(axis=1) > 0)
    contradiction_count = int(contradiction_mask.sum())
    if contradiction_count > 0:
        print(f"[train] warning: {contradiction_count} rows have `other=1` with additional labels.")

    if len(df) < 5:
        raise SystemExit("[train] Need at least 5 fully labeled rows after filtering.")

    cleaning_cfg = load_stage1_cleaning_config(Path("."))
    df["clean_text"] = df["text"].apply(lambda text: clean_with_stage1(text, cleaning_cfg))

    y = raw_label_df[ISSUE_LABELS].to_numpy(dtype=int)
    idx_train, idx_val, idx_test, split_method = _split_multilabel_indices(y, seed=args.seed)

    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    vectorizer, X_train, min_df_fallback = _fit_vectorizer_with_fallback(
        train_df["clean_text"].tolist(),
        enable_char_ngrams=args.enable_char_ngrams,
    )
    X_val = vectorizer.transform(val_df["clean_text"].tolist())
    X_test = vectorizer.transform(test_df["clean_text"].tolist())

    selector = None
    selected_k = None
    chi2_val_rows = []

    X_train_model = X_train
    X_val_model = X_val
    X_test_model = X_test

    if args.enable_chi2_topk:
        best_k = None
        best_val_micro = -1.0
        candidate_weights = _class_weight_candidates(args)
        probe_weight_label, probe_weight_value = candidate_weights[0]
        for k in [2000, 5000, 10000]:
            sel = MultiLabelChi2Selector(k=k).fit(X_train, y_train)
            Xtr_sel = sel.transform(X_train)
            Xva_sel = sel.transform(X_val)
            probe = train_per_label_ovr(
                Xtr_sel,
                y_train,
                ISSUE_LABELS,
                model_kind="logreg",
                class_weight=probe_weight_value,
                class_weight_map=None,
                calibrate_probs=False,
                random_state=args.seed,
            )
            val_scores_probe = probe.predict_scores(Xva_sel)
            val_preds_probe = (val_scores_probe >= 0.5).astype(int)
            val_micro = float(f1_score(y_val, val_preds_probe, average="micro", zero_division=0))
            chi2_val_rows.append(
                {
                    "k": int(sel.k_),
                    "class_weight_probe": probe_weight_label,
                    "val_micro_f1": val_micro,
                }
            )
            if val_micro > best_val_micro:
                best_val_micro = val_micro
                best_k = k
        selected_k = int(best_k) if best_k is not None else 2000
        selector = MultiLabelChi2Selector(k=selected_k).fit(X_train, y_train)
        X_train_model = selector.transform(X_train)
        X_val_model = selector.transform(X_val)
        X_test_model = selector.transform(X_test)
        print(f"[train] selected chi2 k={selected_k} by validation micro-F1.")

    selected_class_weight_map = None
    selected_class_weight_labels = {}
    class_weight_tuning_df = pd.DataFrame()
    class_weight_mode = "global"
    if bool(getattr(args, "class_weight_search", False)):
        candidate_weights = _class_weight_candidates(args)
        (
            selected_class_weight_map,
            selected_class_weight_labels,
            class_weight_tuning_df,
        ) = _select_labelwise_class_weights(
            X_train_model,
            y_train,
            X_val_model,
            y_val,
            seed=args.seed,
            candidate_weights=candidate_weights,
        )
        class_weight_mode = "per_label_search"

    default_weight_value = CLASS_WEIGHT_OPTIONS.get(str(args.class_weight), "balanced")
    lr_model = train_per_label_ovr(
        X_train_model,
        y_train,
        ISSUE_LABELS,
        model_kind="logreg",
        class_weight=default_weight_value,
        class_weight_map=selected_class_weight_map,
        calibrate_probs=bool(getattr(args, "calibrate_probs", False)),
        calibration_method=str(getattr(args, "calibration_method", "sigmoid")),
        calibration_cv=int(getattr(args, "calibration_cv", 3)),
        random_state=args.seed,
    )
    lr_val_scores = lr_model.predict_scores(X_val_model)
    lr_test_scores = lr_model.predict_scores(X_test_model)

    thresholds_lr = {label: 0.5 for label in ISSUE_LABELS}
    if args.tune_thresholds:
        thresholds_lr = _tune_thresholds(y_val, lr_val_scores)

    lr_val_pred = _apply_thresholds(lr_val_scores, thresholds_lr)
    lr_test_pred = _apply_thresholds(lr_test_scores, thresholds_lr)

    stability_frames: List[pd.DataFrame] = []
    stability_frames.append(
        _threshold_stability_rows(
            y_true=y_val,
            scores=lr_val_scores,
            thresholds=thresholds_lr,
            model_variant="logreg_calibrated" if bool(getattr(args, "calibrate_probs", False)) else "logreg",
            split="val",
        )
    )
    stability_frames.append(
        _threshold_stability_rows(
            y_true=y_test,
            scores=lr_test_scores,
            thresholds=thresholds_lr,
            model_variant="logreg_calibrated" if bool(getattr(args, "calibrate_probs", False)) else "logreg",
            split="test",
        )
    )

    if bool(getattr(args, "calibrate_probs", False)):
        uncal_model = train_per_label_ovr(
            X_train_model,
            y_train,
            ISSUE_LABELS,
            model_kind="logreg",
            class_weight=default_weight_value,
            class_weight_map=selected_class_weight_map,
            calibrate_probs=False,
            random_state=args.seed,
        )
        uncal_val_scores = uncal_model.predict_scores(X_val_model)
        uncal_test_scores = uncal_model.predict_scores(X_test_model)
        uncal_thresholds = {label: 0.5 for label in ISSUE_LABELS}
        if args.tune_thresholds:
            uncal_thresholds = _tune_thresholds(y_val, uncal_val_scores)
        stability_frames.append(
            _threshold_stability_rows(
                y_true=y_val,
                scores=uncal_val_scores,
                thresholds=uncal_thresholds,
                model_variant="logreg_uncalibrated",
                split="val",
            )
        )
        stability_frames.append(
            _threshold_stability_rows(
                y_true=y_test,
                scores=uncal_test_scores,
                thresholds=uncal_thresholds,
                model_variant="logreg_uncalibrated",
                split="test",
            )
        )

    overall_rows: List[Dict[str, object]] = []
    per_label_frames: List[pd.DataFrame] = []
    thresholds_by_model: Dict[str, Dict[str, float]] = {
        "ovr_logreg": {label: float(thresholds_lr[label]) for label in ISSUE_LABELS}
    }
    predictions_by_model: Dict[str, Dict[str, np.ndarray]] = {
        "ovr_logreg": {"val": lr_val_pred, "test": lr_test_pred}
    }
    scores_by_model: Dict[str, Dict[str, np.ndarray]] = {
        "ovr_logreg": {"val": lr_val_scores, "test": lr_test_scores}
    }
    model_objects: Dict[str, object] = {"ovr_logreg": lr_model}
    model_notes: Dict[str, Dict[str, str]] = {"ovr_logreg": lr_model.train_notes}
    blend_tuning_df = pd.DataFrame()
    blend_weight_map: Dict[str, float] = {}

    for split_name, y_true_split, y_pred_split in [
        ("val", y_val, lr_val_pred),
        ("test", y_test, lr_test_pred),
    ]:
        row = {"model": "ovr_logreg", "split": split_name}
        row.update(_overall_metrics(y_true_split, y_pred_split))
        overall_rows.append(row)
        per_label_frames.append(_per_label_metrics(y_true_split, y_pred_split, "ovr_logreg", split_name))

    need_svm = bool(args.include_svm_baseline or getattr(args, "enable_model_blend", False))
    svm_model = None
    svm_val_scores = None
    svm_test_scores = None
    if need_svm:
        svm_model = train_per_label_ovr(
            X_train_model,
            y_train,
            ISSUE_LABELS,
            model_kind="linearsvm",
            class_weight=default_weight_value,
            class_weight_map=selected_class_weight_map,
            calibrate_probs=False,
            random_state=args.seed,
        )
        svm_val_scores = svm_model.predict_scores(X_val_model)
        svm_test_scores = svm_model.predict_scores(X_test_model)
        thresholds_svm = {label: 0.5 for label in ISSUE_LABELS}
        if args.tune_thresholds:
            thresholds_svm = _tune_thresholds(y_val, svm_val_scores)
        svm_val_pred = _apply_thresholds(svm_val_scores, thresholds_svm)
        svm_test_pred = _apply_thresholds(svm_test_scores, thresholds_svm)

        thresholds_by_model["ovr_linearsvm"] = {
            label: float(thresholds_svm[label]) for label in ISSUE_LABELS
        }
        predictions_by_model["ovr_linearsvm"] = {"val": svm_val_pred, "test": svm_test_pred}
        scores_by_model["ovr_linearsvm"] = {"val": svm_val_scores, "test": svm_test_scores}
        model_objects["ovr_linearsvm"] = svm_model
        model_notes["ovr_linearsvm"] = svm_model.train_notes

        for split_name, y_true_split, y_pred_split in [
            ("val", y_val, svm_val_pred),
            ("test", y_test, svm_test_pred),
        ]:
            row = {"model": "ovr_linearsvm", "split": split_name}
            row.update(_overall_metrics(y_true_split, y_pred_split))
            overall_rows.append(row)
            per_label_frames.append(_per_label_metrics(y_true_split, y_pred_split, "ovr_linearsvm", split_name))

    if bool(getattr(args, "enable_model_blend", False)) and svm_model is not None:
        blend_weight_map, blend_tuning_df = _tune_blend_weights_per_label(
            y_true=y_val,
            primary_scores=lr_val_scores,
            secondary_scores=svm_val_scores,
            step=0.1,
        )
        blend_val_scores = _blend_scores(lr_val_scores, svm_val_scores, blend_weight_map)
        blend_test_scores = _blend_scores(lr_test_scores, svm_test_scores, blend_weight_map)
        thresholds_blend = {label: 0.5 for label in ISSUE_LABELS}
        if args.tune_thresholds:
            thresholds_blend = _tune_thresholds(y_val, blend_val_scores)
        blend_val_pred = _apply_thresholds(blend_val_scores, thresholds_blend)
        blend_test_pred = _apply_thresholds(blend_test_scores, thresholds_blend)

        blend_note_map = {
            label: f"blend_alpha_lr={float(blend_weight_map.get(label, 0.5)):.2f}"
            for label in ISSUE_LABELS
        }
        blend_note_map.update(
            {
                "__primary_model": "ovr_logreg",
                "__secondary_model": "ovr_linearsvm",
            }
        )
        blend_model = BlendedOVRModel(
            primary_model=lr_model,
            secondary_model=svm_model,
            label_names=list(ISSUE_LABELS),
            blend_weights={label: float(blend_weight_map.get(label, 0.5)) for label in ISSUE_LABELS},
            train_notes=blend_note_map,
        )

        thresholds_by_model["ovr_blend_lr_svm"] = {
            label: float(thresholds_blend[label]) for label in ISSUE_LABELS
        }
        predictions_by_model["ovr_blend_lr_svm"] = {
            "val": blend_val_pred,
            "test": blend_test_pred,
        }
        scores_by_model["ovr_blend_lr_svm"] = {
            "val": blend_val_scores,
            "test": blend_test_scores,
        }
        model_objects["ovr_blend_lr_svm"] = blend_model
        model_notes["ovr_blend_lr_svm"] = blend_note_map

        stability_frames.append(
            _threshold_stability_rows(
                y_true=y_val,
                scores=blend_val_scores,
                thresholds=thresholds_blend,
                model_variant="blend_lr_svm",
                split="val",
            )
        )
        stability_frames.append(
            _threshold_stability_rows(
                y_true=y_test,
                scores=blend_test_scores,
                thresholds=thresholds_blend,
                model_variant="blend_lr_svm",
                split="test",
            )
        )

        for split_name, y_true_split, y_pred_split in [
            ("val", y_val, blend_val_pred),
            ("test", y_test, blend_test_pred),
        ]:
            row = {"model": "ovr_blend_lr_svm", "split": split_name}
            row.update(_overall_metrics(y_true_split, y_pred_split))
            overall_rows.append(row)
            per_label_frames.append(_per_label_metrics(y_true_split, y_pred_split, "ovr_blend_lr_svm", split_name))

    dummy_majority_val = _dummy_labelset_majority_predict(y_train, n_samples=len(y_val))
    dummy_majority_test = _dummy_labelset_majority_predict(y_train, n_samples=len(y_test))
    for split_name, y_true_split, y_pred_split in [
        ("val", y_val, dummy_majority_val),
        ("test", y_test, dummy_majority_test),
    ]:
        row = {"model": "dummy_labelset_majority", "split": split_name}
        row.update(_overall_metrics(y_true_split, y_pred_split))
        overall_rows.append(row)
        per_label_frames.append(
            _per_label_metrics(y_true_split, y_pred_split, "dummy_labelset_majority", split_name)
        )

    dummy_prior_val = _dummy_label_prior_predict(y_train, n_samples=len(y_val))
    dummy_prior_test = _dummy_label_prior_predict(y_train, n_samples=len(y_test))
    for split_name, y_true_split, y_pred_split in [
        ("val", y_val, dummy_prior_val),
        ("test", y_test, dummy_prior_test),
    ]:
        row = {"model": "dummy_label_prior", "split": split_name}
        row.update(_overall_metrics(y_true_split, y_pred_split))
        overall_rows.append(row)
        per_label_frames.append(
            _per_label_metrics(y_true_split, y_pred_split, "dummy_label_prior", split_name)
        )

    selected_model_name = "ovr_logreg"
    if bool(getattr(args, "auto_select_best_model", False)):
        selected_model_name = _select_best_model_name(overall_rows)
    if selected_model_name not in model_objects:
        selected_model_name = "ovr_logreg"

    selected_model = model_objects[selected_model_name]
    selected_thresholds = thresholds_by_model[selected_model_name]
    selected_test_pred = predictions_by_model[selected_model_name]["test"]
    selected_test_scores = scores_by_model[selected_model_name]["test"]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame(overall_rows)
    per_label_df = pd.concat(per_label_frames, ignore_index=True)
    overall_df.to_csv(out_dir / "02_metrics_overall.csv", index=False)
    per_label_df.to_csv(out_dir / "02_metrics_per_label.csv", index=False)

    if not class_weight_tuning_df.empty:
        class_weight_tuning_df.to_csv(out_dir / "02_class_weight_tuning.csv", index=False)
    if not blend_tuning_df.empty:
        blend_tuning_df.to_csv(out_dir / "02_blend_weight_tuning.csv", index=False)

    stability_df = pd.concat(stability_frames, ignore_index=True) if stability_frames else pd.DataFrame()
    if not stability_df.empty:
        stability_df.to_csv(out_dir / "02_threshold_stability.csv", index=False)
        _write_stability_summary(out_dir / "02_threshold_stability_summary.md", stability_df)

    confusion_md = _build_confusion_like_summary(test_df, y_test, selected_test_pred, selected_test_scores)
    (out_dir / "02_confusion_like_summary.md").write_text(confusion_md, encoding="utf-8")

    if args.tune_thresholds:
        threshold_rows: List[Dict[str, object]] = []
        for model_name, threshold_map in thresholds_by_model.items():
            for label in ISSUE_LABELS:
                threshold_rows.append(
                    {
                        "model": model_name,
                        "label": label,
                        "threshold": float(threshold_map[label]),
                    }
                )
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(out_dir / "02_label_thresholds.csv", index=False)

    plot_model_name = selected_model_name
    selected_label_frame = per_label_df[
        (per_label_df["model"] == plot_model_name) & (per_label_df["split"] == "test")
    ]
    if selected_label_frame.empty:
        plot_model_name = "ovr_logreg"
        selected_label_frame = per_label_df[
            (per_label_df["model"] == plot_model_name) & (per_label_df["split"] == "test")
        ]
    if selected_label_frame.empty:
        selected_label_frame = pd.DataFrame({"label": ISSUE_LABELS, "f1": [0.0] * len(ISSUE_LABELS)})
    test_label_f1 = selected_label_frame.set_index("label").loc[ISSUE_LABELS]["f1"]
    plt.figure(figsize=(10, 4))
    plt.bar(ISSUE_LABELS, test_label_f1.values, color="#1f77b4")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1")
    plt.title(f"Per-label F1 (test, {plot_model_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "02_per_label_f1.png", dpi=220)
    plt.close()

    full_counts = raw_label_df[ISSUE_LABELS].sum(axis=0).reindex(ISSUE_LABELS)
    plt.figure(figsize=(10, 4))
    plt.bar(ISSUE_LABELS, full_counts.values, color="#4c72b0")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Label Frequency (labeled rows)")
    plt.tight_layout()
    plt.savefig(out_dir / "02_label_frequency.png", dpi=220)
    plt.close()

    train_config = {
        "labels_path": str(args.labels_path),
        "data_path": str(args.data_path),
        "split_method": split_method,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "seed": int(args.seed),
        "enable_char_ngrams": bool(args.enable_char_ngrams),
        "enable_chi2_topk": bool(args.enable_chi2_topk),
        "selected_chi2_k": selected_k,
        "chi2_val_micro_f1": chi2_val_rows,
        "min_df_fallback_to_1": bool(min_df_fallback),
        "class_weight": args.class_weight,
        "class_weight_mode": class_weight_mode,
        "class_weight_per_label": selected_class_weight_labels,
        "calibrate_probs": bool(getattr(args, "calibrate_probs", False)),
        "calibration_method": str(getattr(args, "calibration_method", "sigmoid")),
        "calibration_cv": int(getattr(args, "calibration_cv", 3)),
        "tune_thresholds": bool(args.tune_thresholds),
        "include_svm_baseline": bool(args.include_svm_baseline),
        "include_dummy_baselines": True,
        "enable_model_blend": bool(getattr(args, "enable_model_blend", False)),
        "auto_select_best_model": bool(getattr(args, "auto_select_best_model", False)),
        "selected_model_name": selected_model_name,
        "selected_thresholds": selected_thresholds,
        "blend_weights": blend_weight_map,
        "label_list": ISSUE_LABELS,
        "cleaning": cleaning_cfg,
        "dropped_rows": {
            "invalid_label_rows": invalid_rows,
            "zero_label_rows": zero_rows,
        },
        "warnings": {
            "other_with_other_labels": contradiction_count,
        },
        "model_notes": model_notes,
    }
    (out_dir / "02_train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    selector_path = model_dir / "chi2_selector.joblib"
    if selector is not None:
        joblib.dump(selector, selector_path)
    elif selector_path.exists():
        selector_path.unlink()
    joblib.dump(selected_model, model_dir / "ovr_model.joblib")
    (model_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "thresholds": selected_thresholds,
                "cleaning": cleaning_cfg,
                "selected_model_name": selected_model_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "label_list.json").write_text(json.dumps(ISSUE_LABELS, indent=2), encoding="utf-8")

    print(f"[train] wrote metrics to {out_dir}")
    print(f"[train] saved artifacts to {model_dir}")


def cmd_predict(args) -> None:
    bundle = load_issue_bundle(args.model_dir)
    if bundle is None:
        print(f"[predict] issue model artifacts not found in {args.model_dir}")
        print(
            "Run: python -m src.issue_steps train --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl"
        )
        raise SystemExit(1)
    payload = predict_issue_labels(args.text, bundle)
    print(json.dumps(payload, ensure_ascii=False))
