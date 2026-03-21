import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from scipy.sparse import vstack

from src.text_features import CONTEXT_VARIANTS, VariantSpec, negation_sanity_tests

from .common import (
    CLASS_WEIGHT_GRID_LR,
    CLASS_WEIGHT_GRID_TREE,
    CS_THRESHOLD_PAIRS,
    DEFAULT_THRESHOLDS,
    K_GRID,
    SEED,
    DM2Config,
    MIN_NNZ_DEFAULT,
    VectorizerBundle,
    apply_uncertainty_rule,
    clean_text,
    decision_tree,
    fit_vectorizer,
    load_data,
    make_splits,
    metrics_from_probs,
    negative_first_better,
    persist_core_artifacts,
    plot_bar,
    plot_confusion,
    plot_hist,
    prob_hist,
    random_forest,
    selective_metrics,
    set_seed,
    simple_prob_hist,
    lr_model,
    save_json,
    DEFAULT_ABBREV_MAP,
)


def _variant_from_config(config: DM2Config) -> VariantSpec:
    """
    Lightweight mapper to pick a variant spec based on CLI flags.
    Used for preview/stats steps; full sweep uses explicit variants.
    """
    if config.enable_clause_split and config.enable_negation_tagging:
        return next(v for v in CONTEXT_VARIANTS if v.name == "V4")
    if config.enable_clause_split:
        return next(v for v in CONTEXT_VARIANTS if v.name == "V3")
    if config.enable_char_ngrams and config.enable_negation_tagging:
        return next(v for v in CONTEXT_VARIANTS if v.name == "V6")
    if config.enable_char_ngrams:
        return next(v for v in CONTEXT_VARIANTS if v.name == "V5")
    if config.enable_negation_tagging:
        return next(v for v in CONTEXT_VARIANTS if v.name == "V2")
    return CONTEXT_VARIANTS[0]


def _load_df_and_splits(config: DM2Config):
    set_seed()
    df = load_data(Path(config.data_path))
    splits = make_splits(
        df,
        enable_abbrev_norm=config.enable_abbrev_norm,
        enable_negation=config.enable_negation_tagging,
        negation_window=config.negation_window,
    )
    return df, splits


def _vectorize(config: DM2Config, variant: Optional[VariantSpec] = None):
    df, splits = _load_df_and_splits(config)
    spec = variant or _variant_from_config(config)
    vec_bundle = fit_vectorizer(
        splits,
        variant=spec,
        enable_abbrev_norm=config.enable_abbrev_norm,
        negation_window=config.negation_window,
    )
    return df, splits, vec_bundle


def step01_data_overview(config: DM2Config) -> None:
    df = load_data(Path(config.data_path))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    missing_text = df["text"].isna().sum()
    missing_rating = df["rating"].isna().sum()
    rating_counts = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

    strong_neg = df[df["rating"].isin([1, 2])]
    strong_pos = df[df["rating"].isin([4, 5])]
    three_star = df[df["rating"] == 3]
    imbalance = (
        strong_pos.shape[0] / strong_neg.shape[0] if strong_neg.shape[0] > 0 else np.nan
    )

    # CSVs
    rating_counts.to_csv(out_dir / "01_rating_distribution.csv", header=["count"])
    pd.DataFrame(
        {
            "label": ["negative", "positive", "three_star"],
            "count": [len(strong_neg), len(strong_pos), len(three_star)],
            "pos_neg_ratio": [np.nan, imbalance, np.nan],
        }
    ).to_csv(out_dir / "01_strong_label_counts.csv", index=False)

    # Figure
    plot_bar(
        {str(k): int(v) for k, v in rating_counts.items()},
        out_dir / "01_rating_distribution.png",
        "Rating Distribution (1-5)",
    )

    # Markdown
    lines = [
        "# Step 01 · Data Overview",
        f"- Total rows: {total_rows}",
        f"- Missing text: {missing_text}, missing rating: {missing_rating}",
        f"- Columns present: {list(df.columns)}",
        f"- Rating distribution saved to 01_rating_distribution.csv/ .png",
        f"- Strong labels -> Negative: {len(strong_neg)}, Positive: {len(strong_pos)}, 3-star: {len(three_star)}, pos/neg ratio: {imbalance:.2f}",
    ]
    (out_dir / "01_data_overview.md").write_text("\n".join(lines))


def step02_cleaning_preview(config: DM2Config) -> None:
    df = load_data(Path(config.data_path))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = df.sample(n=min(10, len(df)), random_state=SEED)

    preview_rows = []
    for _, row in sample.iterrows():
        raw = row.get("text", "")
        cleaned = clean_text(raw, enable_abbrev_norm=config.enable_abbrev_norm)
        cleaned_neg = clean_text(
            raw,
            enable_abbrev_norm=config.enable_abbrev_norm,
            enable_negation=config.enable_negation_tagging,
            negation_window=config.negation_window,
        )
        cleaned_no_abbrev = clean_text(raw, enable_abbrev_norm=False)
        preview_rows.append(
            {
                "rating": row.get("rating", np.nan),
                "raw_text": raw,
                "clean_text": cleaned,
                "clean_text_with_negation": cleaned_neg,
                "clean_text_no_abbrev": cleaned_no_abbrev,
            }
        )
    pd.DataFrame(preview_rows).to_csv(out_dir / "02_cleaning_examples.csv", index=False)

    # Abbreviation normalization table
    abbrev_examples = [
        "gr8 product",
        "thx for the gift card",
        "u are awesome",
        "idk about this card",
        "imo delivery was late",
        "redeemed w/ ease",
        "arrived w/o code",
        "cant redeem balance",
        "didnt like it",
        "u r gr8 but late",
    ]
    abbrev_rows = []
    for txt in abbrev_examples:
        norm = clean_text(txt, enable_abbrev_norm=True)
        base = clean_text(txt, enable_abbrev_norm=False)
        abbrev_rows.append(
            {"example": txt, "clean_no_abbrev": base, "clean_with_abbrev": norm}
        )
    pd.DataFrame(abbrev_rows).to_csv(
        out_dir / "02_abbrev_norm_comparison.csv", index=False
    )

    # Negation tagging sanity examples
    neg_rows = []
    for raw, tagged in negation_sanity_tests(window=config.negation_window):
        neg_rows.append({"example": raw, "with_negation_tag": tagged})
    pd.DataFrame(neg_rows).to_csv(out_dir / "02_negation_tagging_examples.csv", index=False)

    lines = [
        "# Step 02 · Cleaning Preview",
        "- Lowercase, remove URLs/punctuation, normalize whitespace.",
        f"- Abbreviation normalization enabled: {config.enable_abbrev_norm}.",
        f"- Negation tagging flag: {config.enable_negation_tagging} (window={config.negation_window}).",
        "- 10 sampled rows saved to 02_cleaning_examples.csv (raw vs cleaned).",
        "- Abbreviation test list saved to 02_abbrev_norm_comparison.csv.",
        "- Negation tagging examples saved to 02_negation_tagging_examples.csv.",
    ]
    (out_dir / "02_cleaning_preview.md").write_text("\n".join(lines))


def step03_split_summary(config: DM2Config) -> None:
    _, splits = _load_df_and_splits(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name, split_df in [
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ]:
        counts = split_df["label"].value_counts().reindex([0, 1], fill_value=0)
        rows.append(
            {
                "split": split_name,
                "negative": counts[0],
                "positive": counts[1],
                "total": len(split_df),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "03_split_summary.csv", index=False)

    lines = [
        "# Step 03 · Split Summary",
        "- Strong labels only (ratings 1,2,4,5).",
        "- Stratified split: Train 70%, Val 10%, Test 20%, seed=42.",
        f"- 3-star rows held out: {len(splits.three_star)}",
    ]
    (out_dir / "03_split_summary.md").write_text("\n".join(lines))


def step04_tfidf_stats(config: DM2Config) -> None:
    variant = _variant_from_config(config)
    _, splits, vec_bundle = _vectorize(config, variant=variant)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_features = vec_bundle.X_train.shape[1]
    component_dims = {}
    if hasattr(vec_bundle.vectorizer, "transformer_list"):
        for name, tr in vec_bundle.vectorizer.transformer_list:
            if hasattr(tr, "named_steps"):
                for key in ["tfidf", "tfidf_left", "tfidf_right", "tfidf_char"]:
                    if key in tr.named_steps and hasattr(tr.named_steps[key], "vocabulary_"):
                        component_dims[name] = len(tr.named_steps[key].vocabulary_)
            elif hasattr(tr, "vocabulary_"):
                component_dims[name] = len(tr.vocabulary_)
    vocab_size = component_dims.get("word_tfidf", total_features) if component_dims else total_features
    shapes = {
        "train_shape": vec_bundle.X_train.shape,
        "val_shape": vec_bundle.X_val.shape,
        "test_shape": vec_bundle.X_test.shape,
        "three_star_shape": vec_bundle.X_3star.shape if vec_bundle.X_3star is not None else (0, vocab_size),
    }
    nnz_train = np.diff(vec_bundle.X_train.indptr)
    nnz_test = np.diff(vec_bundle.X_test.indptr)
    avg_nnz = {
        "train_avg_nnz": float(np.mean(nnz_train)),
        "val_avg_nnz": float(np.mean(np.diff(vec_bundle.X_val.indptr))),
        "test_avg_nnz": float(np.mean(nnz_test)),
    }

    top_token_rows = []

    def _append_top_tokens(prefix: str, tfidf_obj, top_n: int = 20) -> None:
        if not hasattr(tfidf_obj, "get_feature_names_out"):
            return
        try:
            tokens = tfidf_obj.get_feature_names_out()
        except Exception:
            return
        if len(tokens) == 0:
            return
        idf_vals = np.asarray(
            getattr(tfidf_obj, "idf_", np.ones(len(tokens), dtype=float)),
            dtype=float,
        )
        if idf_vals.shape[0] != len(tokens):
            idf_vals = np.ones(len(tokens), dtype=float)
        top_k = min(int(top_n), len(tokens))
        ranked_idx = np.argsort(-idf_vals)[:top_k]
        for idx in ranked_idx:
            top_token_rows.append(
                {
                    "token": f"{prefix}{tokens[idx]}",
                    "idf": float(idf_vals[idx]),
                }
            )

    if hasattr(vec_bundle.vectorizer, "transformer_list"):
        for comp_name, transformer in vec_bundle.vectorizer.transformer_list:
            if hasattr(transformer, "named_steps"):
                for key in ["tfidf", "tfidf_left", "tfidf_right", "tfidf_char"]:
                    if key in transformer.named_steps:
                        _append_top_tokens(
                            prefix=f"{comp_name}/{key}::",
                            tfidf_obj=transformer.named_steps[key],
                        )
            else:
                _append_top_tokens(prefix=f"{comp_name}::", tfidf_obj=transformer)
    else:
        _append_top_tokens(prefix="", tfidf_obj=vec_bundle.vectorizer)

    top_tokens = (
        pd.DataFrame(top_token_rows)
        .sort_values(by="idf", ascending=False)
        .head(20)
        .to_dict(orient="records")
        if top_token_rows
        else []
    )

    payload = {
        "variant": variant.name,
        "total_features": total_features,
        "vocab_size": vocab_size,
        "component_dims": component_dims,
        **{k: list(v) if hasattr(v, "__len__") and len(v) == 2 else v for k, v in shapes.items()},
        **avg_nnz,
        "top_idf_tokens": top_tokens,
    }
    save_json(out_dir / "04_tfidf_stats.json", payload)

    # Sparsity histogram
    plt_path = out_dir / "04_tfidf_sparsity.png"
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    bins = np.arange(0, max(nnz_train.max(), nnz_test.max()) + 2, 2)
    plt.hist(nnz_train, bins=bins, alpha=0.6, label="train")
    plt.hist(nnz_test, bins=bins, alpha=0.6, label="test")
    plt.xlabel("Non-zero TF-IDF entries per sample")
    plt.ylabel("Count")
    plt.title("TF-IDF sparsity (train vs test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plt_path, dpi=200)
    plt.close()

    lines = [
        "# Step 04 - TF-IDF Stats",
        f"- Variant: {variant.name} (char={variant.use_char}, clause_split={variant.use_clause}, negation={variant.use_negation}).",
        f"- Feature dims: total {total_features}, primary vocab ~{vocab_size}; components: {component_dims}.",
        f"- Matrix shapes: train {shapes['train_shape']}, val {shapes['val_shape']}, test {shapes['test_shape']}",
        f"- Avg nnz per sample: train {avg_nnz['train_avg_nnz']:.1f}, test {avg_nnz['test_avg_nnz']:.1f}",
        "- Top 20 tokens by IDF saved in 04_tfidf_stats.json (when available).",
    ]
    (out_dir / "04_tfidf_stats.md").write_text("\n".join(lines))


def _chi2_sweep(
    splits,
    vec_bundle: VectorizerBundle,
    variant_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, int, str, SelectKBest]:
    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    records = []
    best_metrics = None
    best_k = None
    best_cw_label = None
    best_selector = None

    for k in K_GRID:
        selector = SelectKBest(chi2, k=k)
        selector.fit(vec_bundle.X_train, y_train)
        X_train_sel = selector.transform(vec_bundle.X_train)
        X_val_sel = selector.transform(vec_bundle.X_val)
        for cw_label, cw_value in CLASS_WEIGHT_GRID_LR:
            model = lr_model("l2", class_weight=cw_value)
            start = time.time()
            model.fit(X_train_sel, y_train)
            train_seconds = time.time() - start
            val_probs = model.predict_proba(X_val_sel)[:, 1]
            metrics = metrics_from_probs(y_val, val_probs, threshold=0.5)
            record = {
                "k": k,
                "class_weight": cw_label,
                "train_seconds": train_seconds,
                **metrics,
            }
            if variant_label:
                record["variant"] = variant_label
            records.append(record)
            if negative_first_better(metrics, best_metrics, k, best_k or k):
                best_metrics = metrics
                best_k = k
                best_cw_label = cw_label
                best_selector = selector
    sweep_df = pd.DataFrame(records)
    return sweep_df, best_k, best_cw_label, best_selector


def _cw_value(label: str):
    for name, value in CLASS_WEIGHT_GRID_LR:
        if name == label:
            return value
    return None if label in [None, "none"] else label


def _decision_label(decision: int) -> str:
    if decision == 1:
        return "Positive"
    if decision == 0:
        return "Negative"
    return "Uncertain"


def _fallback_statistics(clean_texts: pd.Series, matrix, min_nnz: int):
    nnz_counts = np.diff(matrix.indptr)
    reasons = []
    for text, nnz in zip(clean_texts, nnz_counts):
        tokens = len(str(text).split())
        if tokens < 2 or str(text).strip() == "":
            reasons.append("too_short")
        elif nnz < min_nnz:
            reasons.append("sparse_vector")
    total = len(clean_texts)
    too_short = reasons.count("too_short")
    sparse = reasons.count("sparse_vector")
    fallback_count = len(reasons)
    rate = fallback_count / total if total else np.nan
    return {
        "total": total,
        "fallback_count": fallback_count,
        "fallback_rate": rate,
        "too_short": too_short,
        "sparse_vector": sparse,
    }


def _parse_chosen(path: Path) -> Optional[Tuple[int, str]]:
    if not path.exists():
        return None
    text = path.read_text().strip()
    k_match = re.search(r"k=([0-9]+)", text)
    cw_match = re.search(r"class_weight=([A-Za-z0-9_]+)", text)
    if k_match and cw_match:
        return int(k_match.group(1)), cw_match.group(1)
    return None


def _variant_by_name(name: str) -> VariantSpec:
    for spec in CONTEXT_VARIANTS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown variant: {name}")


def _parse_best_variant(out_dir: Path) -> Optional[Tuple[VariantSpec, int, str]]:
    path = out_dir / "06b_best_variant.txt"
    if not path.exists():
        return None
    text = path.read_text().strip()
    v_match = re.search(r"variant=([A-Za-z0-9]+)", text)
    k_match = re.search(r"k=([0-9]+)", text)
    cw_match = re.search(r"class_weight=([A-Za-z0-9_]+)", text)
    if v_match and k_match and cw_match:
        spec = _variant_by_name(v_match.group(1))
        return spec, int(k_match.group(1)), cw_match.group(1)
    return None


def _choose_best_and_write(
    sweep_df: pd.DataFrame, out_dir: Path
) -> Tuple[int, str]:
    # pick best by negative-first tie-breaker already encoded in sweep_df order
    sweep_df = sweep_df.copy()
    best_row = sweep_df.sort_values(
        by=["recall_0", "f2_0", "precision_0", "k"],
        ascending=[False, False, False, True],
    ).iloc[0]
    best_k = int(best_row["k"])
    best_cw = str(best_row["class_weight"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "06_chosen_k.txt").write_text(f"k={best_k}, class_weight={best_cw}\n")
    return best_k, best_cw


def step05_baseline_lr(config: DM2Config) -> None:
    _, splits, vec_bundle = _vectorize(config, variant=CONTEXT_VARIANTS[0])
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    y_test = splits.test["label"].values

    model = lr_model("l2", class_weight=None)
    model.fit(vec_bundle.X_train, y_train)
    val_probs = model.predict_proba(vec_bundle.X_val)[:, 1]
    test_probs = model.predict_proba(vec_bundle.X_test)[:, 1]

    val_metrics = metrics_from_probs(y_val, val_probs, threshold=0.5)
    test_metrics = metrics_from_probs(y_test, test_probs, threshold=0.5)

    metrics_df = pd.DataFrame(
        [
            {"split": "val", **val_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_df.to_csv(out_dir / "05_baseline_lr_metrics.csv", index=False)

    y_pred_test = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    plot_confusion(cm, out_dir / "05_confusion_matrix_baseline.png", ["Negative", "Positive"])

    lines = [
        "# Step 05 · Baseline LR (L2, full TF-IDF)",
        f"- Validation recall_0: {val_metrics['recall_0']:.3f}, F2_0: {val_metrics['f2_0']:.3f}",
        f"- Test recall_0: {test_metrics['recall_0']:.3f}, precision_0: {test_metrics['precision_0']:.3f}, F2_0: {test_metrics['f2_0']:.3f}",
        "- Confusion matrix saved to 05_confusion_matrix_baseline.png",
    ]
    (out_dir / "05_baseline_lr.md").write_text("\n".join(lines))


def step06_feature_selection(config: DM2Config) -> Tuple[int, str]:
    _, splits, vec_bundle = _vectorize(config, variant=CONTEXT_VARIANTS[0])
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_df, best_k, best_cw, _selector = _chi2_sweep(splits, vec_bundle)
    sweep_df.to_csv(out_dir / "06_chi2_sweep_table.csv", index=False)
    (out_dir / "06_chosen_k.txt").write_text(f"k={best_k}, class_weight={best_cw}\n")

    best_by_k = (
        sweep_df.sort_values(
            by=["recall_0", "f2_0", "precision_0", "class_weight"],
            ascending=[False, False, False, True],
        )
        .groupby("k")
        .first()
        .reset_index()
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(best_by_k["k"], best_by_k["f1"], marker="o")
    plt.xlabel("Chi2 Top K")
    plt.ylabel("Validation F1")
    plt.title("F1 vs K (best class_weight per K)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "06_f1_vs_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(best_by_k["k"], best_by_k["recall_0"], marker="o", color="darkorange")
    plt.xlabel("Chi2 Top K")
    plt.ylabel("Validation recall_0")
    plt.title("Recall_0 vs K (negative-first)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "06_recall0_vs_k.png", dpi=200)
    plt.close()

    lines = [
        "# Step 06 · Chi-square Feature Selection",
        "- Grid K in {1000,2000,5000,10000} x class_weight experiments.",
        f"- Best (negative-first) -> K*={best_k}, class_weight*={best_cw}.",
        "- Full table saved to 06_chi2_sweep_table.csv; charts saved as 06_f1_vs_k.png and 06_recall0_vs_k.png.",
    ]
    (out_dir / "06_chi2_selection.md").write_text("\n".join(lines))
    return best_k, best_cw


def step06b_context_feature_variants_sweep(config: DM2Config):
    """
    Evaluate context-aware feature variants (V0-V6) on val/test with Chi2+LR grid.
    Saves per-variant tables and figures; selects best variant by val recall_0.
    """
    df, splits = _load_df_and_splits(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_tables = []
    test_rows = []
    best_choice = None  # (metrics, k, cw, spec)

    for spec in CONTEXT_VARIANTS:
        vec_bundle = fit_vectorizer(
            splits,
            variant=spec,
            enable_abbrev_norm=config.enable_abbrev_norm,
            negation_window=config.negation_window,
        )
        sweep_df, best_k, best_cw, selector = _chi2_sweep(
            splits, vec_bundle, variant_label=spec.name
        )
        sweep_df["variant"] = spec.name
        val_tables.append(sweep_df)

        best_val_row = sweep_df.sort_values(
            by=["recall_0", "f2_0", "precision_0", "k"],
            ascending=[False, False, False, True],
        ).iloc[0]

        # Train LR on train+val with selector fit on train only
        cw_value = _cw_value(best_cw)
        y_train = splits.train["label"].values
        y_val = splits.val["label"].values
        X_train_sel = selector.transform(vec_bundle.X_train)
        X_val_sel = selector.transform(vec_bundle.X_val)
        X_trainval_sel = vstack([X_train_sel, X_val_sel])
        y_trainval = np.concatenate([y_train, y_val])
        model = lr_model("l2", class_weight=cw_value)
        model.fit(X_trainval_sel, y_trainval)

        y_test = splits.test["label"].values
        test_probs = model.predict_proba(selector.transform(vec_bundle.X_test))[:, 1]
        test_metrics = metrics_from_probs(y_test, test_probs, threshold=0.5)

        # Uncertainty + fallback
        cleaned_test = [
            clean_text(
                t,
                enable_abbrev_norm=config.enable_abbrev_norm,
                enable_negation=spec.use_negation,
                negation_window=config.negation_window,
            )
            for t in splits.test["text"]
        ]
        decisions = apply_uncertainty_rule(
            test_probs,
            cleaned_test,
            vec_bundle.X_test,
            thresholds=config.thresholds,
            min_nnz=config.min_nnz,
        )
        sel_metrics = selective_metrics(y_test, decisions)
        fallback_rate_test = decisions["reason"].isin(["too_short", "sparse_vector"]).mean()

        fallback_rate_3star = np.nan
        if vec_bundle.X_3star is not None and not splits.three_star.empty:
            probs_3 = model.predict_proba(selector.transform(vec_bundle.X_3star))[:, 1]
            clean_3 = [
                clean_text(
                    t,
                    enable_abbrev_norm=config.enable_abbrev_norm,
                    enable_negation=spec.use_negation,
                    negation_window=config.negation_window,
                )
                for t in splits.three_star["text"]
            ]
            decisions_3 = apply_uncertainty_rule(
                probs_3,
                clean_3,
                vec_bundle.X_3star,
                thresholds=config.thresholds,
                min_nnz=config.min_nnz,
            )
            fallback_rate_3star = decisions_3["reason"].isin(
                ["too_short", "sparse_vector"]
            ).mean()

        test_rows.append(
            {
                "variant": spec.name,
                "k": best_k,
                "class_weight": best_cw,
                **test_metrics,
                **sel_metrics,
                "fallback_rate_test_strong": fallback_rate_test,
                "fallback_rate_3star": fallback_rate_3star,
            }
        )

        if negative_first_better(
            best_val_row.to_dict(), best_choice[0] if best_choice else None, best_k, best_choice[1] if best_choice else best_k
        ):
            best_choice = (best_val_row.to_dict(), best_k, best_cw, spec)

    val_table = pd.concat(val_tables, ignore_index=True)
    val_table.to_csv(out_dir / "06b_variants_val_table.csv", index=False)

    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(out_dir / "06b_variants_test_table.csv", index=False)

    best_metrics, best_k, best_cw, best_spec = best_choice
    best_txt = (
        f"variant={best_spec.name}, k={best_k}, class_weight={best_cw}, "
        f"description={best_spec.description}, negation={best_spec.use_negation}, clause_split={best_spec.use_clause}, char={best_spec.use_char}"
    )
    (out_dir / "06b_best_variant.txt").write_text(best_txt + "\n")

    # Simple figures
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.bar(test_table["variant"], test_table["recall_0"], color="steelblue")
    plt.ylabel("Test recall_0")
    plt.title("Recall_0 by variant (best config per variant)")
    plt.tight_layout()
    plt.savefig(out_dir / "06b_recall0_by_variant.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(test_table["variant"], test_table["precision_0"], color="darkorange")
    plt.ylabel("Test precision_0")
    plt.title("Precision_0 by variant (best config per variant)")
    plt.tight_layout()
    plt.savefig(out_dir / "06b_precision0_by_variant.png", dpi=200)
    plt.close()

    lines = [
        "# Step 06b - Context Feature Variants",
        "- Variants V0-V6 swept with K in {1000,2000,5000,10000} and class_weight grid.",
        f"- Best variant (val negative-first): {best_spec.name} with k={best_k}, class_weight={best_cw}.",
        "- Full validation grid saved to 06b_variants_val_table.csv.",
        "- Test metrics for best per variant saved to 06b_variants_test_table.csv.",
        "- Recall/precision bar plots saved to 06b_recall0_by_variant.png / 06b_precision0_by_variant.png.",
        f"- Best descriptor saved to 06b_best_variant.txt ({best_spec.description}).",
    ]
    (out_dir / "06b_variants_summary.md").write_text("\n".join(lines))
    return best_spec, best_k, best_cw


def step07_embedded_l1(config: DM2Config) -> None:
    _, splits, vec_bundle = _vectorize(config, variant=CONTEXT_VARIANTS[0])
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = _parse_chosen(out_dir / "06_chosen_k.txt")
    cw_label = chosen[1] if chosen else "balanced"
    cw_value = _cw_value(cw_label)

    y_train = splits.train["label"].values
    y_test = splits.test["label"].values

    model = lr_model("l1", class_weight=cw_value)
    model.fit(vec_bundle.X_train, y_train)
    test_probs = model.predict_proba(vec_bundle.X_test)[:, 1]
    metrics = metrics_from_probs(y_test, test_probs, threshold=0.5)
    non_zero = int(np.count_nonzero(model.coef_))

    pd.DataFrame([{"split": "test", "class_weight": cw_label, **metrics}]).to_csv(
        out_dir / "07_l1_metrics.csv", index=False
    )
    (out_dir / "07_l1_feature_count.txt").write_text(str(non_zero))

    lines = [
        "# Step 07 · Embedded L1 LR",
        f"- class_weight: {cw_label}",
        f"- Non-zero features: {non_zero}",
        f"- Test recall_0: {metrics['recall_0']:.3f}, precision_0: {metrics['precision_0']:.3f}, F2_0: {metrics['f2_0']:.3f}",
    ]
    (out_dir / "07_l1_summary.md").write_text("\n".join(lines))


def _ensure_selector(config: DM2Config, splits, vec_bundle):
    out_dir = Path(config.output_dir)
    chosen = _parse_chosen(out_dir / "06_chosen_k.txt")
    if not chosen:
        # Generate selection outputs if missing
        step06_feature_selection(config)
        chosen = _parse_chosen(out_dir / "06_chosen_k.txt")
    best_k, best_cw = chosen
    selector = SelectKBest(chi2, k=best_k)
    selector.fit(vec_bundle.X_train, splits.train["label"].values)
    return best_k, best_cw, selector


def _train_best_lr(config: DM2Config):
    _, splits, vec_bundle = _vectorize(config)
    best_k, best_cw, selector = _ensure_selector(config, splits, vec_bundle)
    cw_value = _cw_value(best_cw)

    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    X_trainval = vstack([vec_bundle.X_train, vec_bundle.X_val])
    y_trainval = np.concatenate([y_train, y_val])
    X_trainval_sel = selector.transform(X_trainval)
    model = lr_model("l2", class_weight=cw_value)
    model.fit(X_trainval_sel, y_trainval)
    return splits, vec_bundle, selector, model, best_k, best_cw


def _train_best_variant_lr(config: DM2Config):
    """
    Train LR for the best context-aware variant (from step06b; triggers sweep if missing).
    """
    out_dir = Path(config.output_dir)
    best = _parse_best_variant(out_dir)
    if best is None:
        best_spec, best_k, best_cw = step06b_context_feature_variants_sweep(config)
    else:
        best_spec, best_k, best_cw = best

    _, splits, vec_bundle = _vectorize(config, variant=best_spec)
    selector = SelectKBest(chi2, k=best_k)
    selector.fit(vec_bundle.X_train, splits.train["label"].values)

    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    X_trainval_sel = selector.transform(vstack([vec_bundle.X_train, vec_bundle.X_val]))
    y_trainval = np.concatenate([y_train, y_val])
    cw_value = _cw_value(best_cw)
    model = lr_model("l2", class_weight=cw_value)
    model.fit(X_trainval_sel, y_trainval)
    return splits, vec_bundle, selector, model, best_k, best_cw, best_spec


def step08_ensemble(config: DM2Config) -> None:
    _, splits, vec_bundle = _vectorize(config, variant=CONTEXT_VARIANTS[0])
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_k, best_cw, selector = _ensure_selector(config, splits, vec_bundle)
    y_train = splits.train["label"].values
    y_test = splits.test["label"].values

    X_train_sel = selector.transform(vec_bundle.X_train)
    X_test_sel = selector.transform(vec_bundle.X_test)

    rows = []
    best_rf_metrics = None
    best_rf_probs = None

    for cw_label, cw_value in CLASS_WEIGHT_GRID_TREE:
        dt = decision_tree(class_weight=cw_value)
        dt.fit(X_train_sel, y_train)
        probs = dt.predict_proba(X_test_sel)[:, 1]
        metrics = metrics_from_probs(y_test, probs, threshold=0.5)
        rows.append(
            {
                "model": "decision_tree",
                "class_weight": cw_label,
                "k": best_k,
                **metrics,
            }
        )

    for cw_label, cw_value in CLASS_WEIGHT_GRID_TREE:
        rf = random_forest(class_weight=cw_value)
        rf.fit(X_train_sel, y_train)
        probs = rf.predict_proba(X_test_sel)[:, 1]
        metrics = metrics_from_probs(y_test, probs, threshold=0.5)
        rows.append(
            {
                "model": "random_forest",
                "class_weight": cw_label,
                "k": best_k,
                **metrics,
            }
        )
        if negative_first_better(metrics, best_rf_metrics, best_k, best_k):
            best_rf_metrics = metrics
            best_rf_probs = probs

    for dummy_name, strategy in [
        ("dummy_most_frequent", "most_frequent"),
        ("dummy_stratified", "stratified"),
    ]:
        dummy = DummyClassifier(strategy=strategy, random_state=SEED)
        dummy.fit(X_train_sel, y_train)
        probs_2d = dummy.predict_proba(X_test_sel)
        classes = list(getattr(dummy, "classes_", []))
        if 1 in classes:
            pos_idx = classes.index(1)
            probs = probs_2d[:, pos_idx]
        else:
            probs = np.zeros(X_test_sel.shape[0], dtype=float)
        metrics = metrics_from_probs(y_test, probs, threshold=0.5)
        rows.append(
            {
                "model": dummy_name,
                "class_weight": "none",
                "k": best_k,
                **metrics,
            }
        )

    # Include strongest LR reference so downstream scoreboards can compare against
    # the best context-aware linear model, not only tree ensembles.
    (
        best_lr_splits,
        best_lr_vec_bundle,
        best_lr_selector,
        best_lr_model,
        best_lr_k,
        best_lr_cw,
        best_lr_spec,
    ) = _train_best_variant_lr(config)
    best_lr_probs = best_lr_model.predict_proba(
        best_lr_selector.transform(best_lr_vec_bundle.X_test)
    )[:, 1]
    best_lr_metrics = metrics_from_probs(
        best_lr_splits.test["label"].values, best_lr_probs, threshold=0.5
    )
    rows.append(
        {
            "model": "logreg_best_variant",
            "variant": best_lr_spec.name,
            "class_weight": best_lr_cw,
            "k": best_lr_k,
            **best_lr_metrics,
        }
    )

    pd.DataFrame(rows).to_csv(out_dir / "08_ensemble_metrics.csv", index=False)

    if best_rf_probs is not None:
        y_pred = (best_rf_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plot_confusion(cm, out_dir / "08_confusion_matrix_rf.png", ["Negative", "Positive"])

    # Fallback stats
    fallback_rows = []
    fallback_rows.append(
        {"dataset": "test", **_fallback_statistics(splits.test["clean_text"], vec_bundle.X_test, config.min_nnz)}
    )
    if vec_bundle.X_3star is not None:
        fallback_rows.append(
            {
                "dataset": "three_star",
                **_fallback_statistics(
                    splits.three_star["clean_text"], vec_bundle.X_3star, config.min_nnz
                ),
            }
        )
    pd.DataFrame(fallback_rows).to_csv(out_dir / "08_fallback_stats.csv", index=False)

    lines = [
        "# Step 08 · Ensemble DT/RF",
        f"- Using Chi2 K*={best_k} (class_weight* from LR: {best_cw}).",
        "- Decision Tree/Random Forest, dummy baselines, and strongest LR reference saved in 08_ensemble_metrics.csv.",
        "- RF confusion matrix saved to 08_confusion_matrix_rf.png.",
        "- Fallback stats (empty/sparse cases) saved to 08_fallback_stats.csv.",
    ]
    (out_dir / "08_ensemble_summary.md").write_text("\n".join(lines))


def step09_uncertainty_eval(config: DM2Config) -> None:
    splits, vec_bundle, selector, model, best_k, best_cw, best_spec = _train_best_variant_lr(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_test = splits.test["label"].values
    X_test_sel = selector.transform(vec_bundle.X_test)
    test_probs = model.predict_proba(X_test_sel)[:, 1]

    # Uncertainty with fallback
    cleaned_test = [
        clean_text(
            t,
            enable_abbrev_norm=config.enable_abbrev_norm,
            enable_negation=best_spec.use_negation,
            negation_window=config.negation_window,
        )
        for t in splits.test["text"]
    ]
    decisions = apply_uncertainty_rule(
        test_probs,
        cleaned_test,
        vec_bundle.X_test,
        thresholds=config.thresholds,
        min_nnz=config.min_nnz,
    )
    sel_metrics = selective_metrics(y_test, decisions)
    fallback_rate = (
        decisions["reason"].isin(["too_short", "sparse_vector"]).mean()
        if len(decisions)
        else np.nan
    )

    # Confusion on covered subset
    covered_mask = decisions["decision"] != -1
    if covered_mask.any():
        cm = confusion_matrix(
            y_test[covered_mask.values],
            decisions.loc[covered_mask, "decision"].astype(int),
            labels=[0, 1],
        )
        plot_confusion(cm, out_dir / "09_confusion_matrix_covered.png", ["Negative", "Positive"])

    pd.DataFrame(
        [
            {
                "variant": best_spec.name,
                "k": best_k,
                "class_weight": best_cw,
                "threshold_low": config.thresholds[0],
                "threshold_high": config.thresholds[1],
                "fallback_rate": fallback_rate,
                **sel_metrics,
            }
        ]
    ).to_csv(out_dir / "09_uncertainty_test.csv", index=False)

    prob_hist(test_probs, y_test, out_dir / "09_prob_hist_test_strong.png", "Test P(Positive)")

    three_star_payload = []
    if vec_bundle.X_3star is not None and vec_bundle.X_3star.shape[0] > 0:
        X3_sel = selector.transform(vec_bundle.X_3star)
        probs_3 = model.predict_proba(X3_sel)[:, 1]
        cleaned_3 = [
            clean_text(
                t,
                enable_abbrev_norm=config.enable_abbrev_norm,
                enable_negation=best_spec.use_negation,
                negation_window=config.negation_window,
            )
            for t in splits.three_star["text"]
        ]
        decisions_3 = apply_uncertainty_rule(
            probs_3,
            cleaned_3,
            vec_bundle.X_3star,
            thresholds=config.thresholds,
            min_nnz=config.min_nnz,
        )
        pos_frac = (decisions_3["decision"] == 1).mean() if len(decisions_3) else np.nan
        neg_frac = (decisions_3["decision"] == 0).mean() if len(decisions_3) else np.nan
        uncertain_frac = (decisions_3["decision"] == -1).mean() if len(decisions_3) else np.nan
        three_star_payload.append(
            {
                "k": best_k,
                "class_weight": best_cw,
                "positive_frac": pos_frac,
                "negative_frac": neg_frac,
                "uncertain_rate": uncertain_frac,
            }
        )
        simple_prob_hist(
            probs_3, out_dir / "09_prob_hist_3star.png", "3-Star P(Positive)"
        )
    else:
        three_star_payload.append(
            {
                "k": best_k,
                "class_weight": best_cw,
                "positive_frac": np.nan,
                "negative_frac": np.nan,
                "uncertain_rate": np.nan,
            }
        )
    pd.DataFrame(three_star_payload).to_csv(out_dir / "09_uncertainty_3star.csv", index=False)

    # Hard-case comparison: baseline V0 vs best variant
    base_vec_bundle = fit_vectorizer(
        splits,
        variant=CONTEXT_VARIANTS[0],
        enable_abbrev_norm=config.enable_abbrev_norm,
        negation_window=config.negation_window,
    )
    base_k, base_cw, base_selector = _ensure_selector(config, splits, base_vec_bundle)
    base_model = lr_model("l2", class_weight=_cw_value(base_cw))
    base_model.fit(
        base_selector.transform(vstack([base_vec_bundle.X_train, base_vec_bundle.X_val])),
        np.concatenate([splits.train["label"].values, splits.val["label"].values]),
    )
    hard_cases = [
        "not bad",
        "not good",
        "good but late delivery",
        "great product but support is awful",
        "idk",
        "gr8",
        "thx",
        "",
        "!!!",
    ]
    hard_rows = []
    for txt in hard_cases:
        clean_base = clean_text(txt, enable_abbrev_norm=config.enable_abbrev_norm)
        tfidf_base = base_vec_bundle.vectorizer.transform([txt])
        prob_base = base_model.predict_proba(base_selector.transform(tfidf_base))[:, 1]
        dec_base = apply_uncertainty_rule(
            prob_base,
            [clean_base],
            tfidf_base,
            thresholds=config.thresholds,
            min_nnz=config.min_nnz,
        ).iloc[0]

        clean_best = clean_text(
            txt,
            enable_abbrev_norm=config.enable_abbrev_norm,
            enable_negation=best_spec.use_negation,
            negation_window=config.negation_window,
        )
        tfidf_best = vec_bundle.vectorizer.transform([txt])
        prob_best = model.predict_proba(selector.transform(tfidf_best))[:, 1]
        dec_best = apply_uncertainty_rule(
            prob_best,
            [clean_best],
            tfidf_best,
            thresholds=config.thresholds,
            min_nnz=config.min_nnz,
        ).iloc[0]

        hard_rows.append(
            {
                "text": txt,
                "baseline_p_pos": float(prob_base[0]),
                "baseline_label": _decision_label(dec_base["decision"]),
                "best_p_pos": float(prob_best[0]),
                "best_label": _decision_label(dec_best["decision"]),
                "fallback_reason": dec_best["reason"] or dec_base["reason"],
            }
        )
    pd.DataFrame(hard_rows).to_csv(out_dir / "hard_cases_comparison.csv", index=False)

    # Persist artifacts for downstream / demo use
    models_dir = Path("models")
    persist_core_artifacts(models_dir, vec_bundle.vectorizer, selector, model)
    save_json(
        models_dir / "variant_meta.json",
        {
            "variant": best_spec.name,
            "description": best_spec.description,
            "k": best_k,
            "class_weight": best_cw,
            "enable_abbrev_norm": config.enable_abbrev_norm,
            "negation": best_spec.use_negation,
            "negation_window": config.negation_window,
            "clause_split": best_spec.use_clause,
            "char_ngrams": best_spec.use_char,
            "thresholds": config.thresholds,
            "min_nnz": config.min_nnz,
        },
    )

    lines = [
        "# Step 09 - Uncertainty Evaluation",
        f"- Best variant {best_spec.name} (Chi2 K*={best_k}, class_weight={best_cw}) trained on train+val (selector fit on train).",
        f"- Thresholds: {config.thresholds[0]:.2f}/{config.thresholds[1]:.2f}; coverage={sel_metrics['coverage']:.3f}, recall_0@covered={sel_metrics['selective_recall_0']:.3f}.",
        f"- Fallback rate (empty/sparse) on test: {fallback_rate:.3f}.",
        "- 3-star uncertainty fractions saved to 09_uncertainty_3star.csv.",
        "- Hard cases comparison (baseline vs best variant) saved to hard_cases_comparison.csv.",
        "- Artifacts saved to models/: tfidf_vectorizer.joblib, chi2_selector.joblib, best_lr_model.joblib.",
    ]
    (out_dir / "09_uncertainty_summary.md").write_text("\n".join(lines))


def step10_threshold_sweep(config: DM2Config) -> None:
    splits, vec_bundle, selector, model, best_k, best_cw, best_spec = _train_best_variant_lr(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_test = splits.test["label"].values
    test_probs = model.predict_proba(selector.transform(vec_bundle.X_test))[:, 1]

    sweep_rows = []
    for low, high in CS_THRESHOLD_PAIRS:
        decisions = apply_uncertainty_rule(
            test_probs,
            [
                clean_text(
                    t,
                    enable_abbrev_norm=config.enable_abbrev_norm,
                    enable_negation=best_spec.use_negation,
                    negation_window=config.negation_window,
                )
                for t in splits.test["text"]
            ],
            vec_bundle.X_test,
            thresholds=(low, high),
            min_nnz=config.min_nnz,
        )
        metrics = selective_metrics(y_test, decisions)
        metrics.update(
            {
                "threshold_low": low,
                "threshold_high": high,
                "fallback_rate": decisions["reason"]
                .isin(["too_short", "sparse_vector"])
                .mean(),
            }
        )
        sweep_rows.append(metrics)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "10_threshold_sweep.csv", index=False)

    # Recommendation
    feasible = sweep_df[sweep_df["selective_precision_0"] >= 0.50]
    pool = feasible if not feasible.empty else sweep_df
    best_row = pool.sort_values(
        by=["selective_recall_0", "coverage"], ascending=[False, False]
    ).iloc[0]
    recommended = (best_row["threshold_low"], best_row["threshold_high"])
    (out_dir / "10_recommended_threshold.txt").write_text(
        f"{recommended[0]:.2f},{recommended[1]:.2f}\n"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(sweep_df["coverage"], sweep_df["selective_recall_0"], marker="o")
    for _, row in sweep_df.iterrows():
        label = f"{row['threshold_low']:.2f}|{row['threshold_high']:.2f}"
        plt.text(row["coverage"], row["selective_recall_0"], label, fontsize=8, ha="right", va="bottom")
    plt.xlabel("Coverage")
    plt.ylabel("Selective recall_0")
    plt.title("Threshold trade-off (coverage vs recall_0)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "10_threshold_tradeoff.png", dpi=200)
    plt.close()

    lines = [
        "# Step 10 - Threshold Sweep",
        f"- Variant: {best_spec.name}; tested pairs: {CS_THRESHOLD_PAIRS}",
        f"- Recommended thresholds (precision_0>=0.50 constraint): {recommended[0]:.2f}/{recommended[1]:.2f}",
        "- Full sweep saved to 10_threshold_sweep.csv; trade-off plot saved to 10_threshold_tradeoff.png.",
    ]
    (out_dir / "10_threshold_summary.md").write_text("\n".join(lines))


def _load_trained_artifacts():
    models_dir = Path("models")
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    sel_path = models_dir / "chi2_selector.joblib"
    model_path = models_dir / "best_lr_model.joblib"
    meta_path = models_dir / "variant_meta.json"
    meta = {}
    if vec_path.exists() and sel_path.exists() and model_path.exists():
        vectorizer = joblib.load(vec_path)
        selector = joblib.load(sel_path)
        model = joblib.load(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        return vectorizer, selector, model, meta
    return None, None, None, meta


def step11_demo_one_review(config: DM2Config, text: str) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer, selector, model, meta = _load_trained_artifacts()
    if vectorizer is None:
        splits, vec_bundle, selector, model, best_k, best_cw, best_spec = _train_best_variant_lr(config)
        vectorizer = vec_bundle.vectorizer
        persist_core_artifacts(Path("models"), vectorizer, selector, model)
        meta = {
            "variant": best_spec.name,
            "description": best_spec.description,
            "k": best_k,
            "class_weight": best_cw,
            "enable_abbrev_norm": config.enable_abbrev_norm,
            "negation": best_spec.use_negation,
            "negation_window": config.negation_window,
            "clause_split": best_spec.use_clause,
            "char_ngrams": best_spec.use_char,
            "thresholds": config.thresholds,
            "min_nnz": config.min_nnz,
        }
        save_json(Path("models") / "variant_meta.json", meta)

    thresholds = tuple(meta.get("thresholds", config.thresholds))
    min_nnz = int(meta.get("min_nnz", config.min_nnz))
    enable_abbrev = bool(meta.get("enable_abbrev_norm", config.enable_abbrev_norm))
    enable_neg = bool(meta.get("negation", False))
    neg_window = int(meta.get("negation_window", config.negation_window))

    cleaned = clean_text(
        text,
        enable_abbrev_norm=enable_abbrev,
        enable_negation=enable_neg,
        negation_window=neg_window,
    )
    token_count = len(cleaned.split())
    nnz = 0
    prob = np.nan
    decision = "Uncertain"
    fallback_reason = None

    if cleaned.strip() == "" or token_count < 2:
        fallback_reason = "too_short"
    else:
        tfidf_vec = vectorizer.transform([text])
        nnz = tfidf_vec.nnz
        if nnz < min_nnz:
            fallback_reason = "sparse_vector"
        else:
            prob = float(model.predict_proba(selector.transform(tfidf_vec))[:, 1][0])
            if prob >= thresholds[1]:
                decision = "Positive"
            elif prob <= thresholds[0]:
                decision = "Negative"
            else:
                decision = "Uncertain"
                fallback_reason = "threshold_band"

    result_lines = [
        "=== Demo One Review ===",
        f"raw_text: {text}",
        f"cleaned_text: {cleaned}",
        f"token_count: {token_count}, nnz: {nnz}",
        f"p_positive: {prob if not np.isnan(prob) else 'n/a'}",
        f"decision: {decision}",
        f"fallback_reason: {fallback_reason}",
    ]
    print("\n".join(result_lines))

    log_path = out_dir / "11_demo_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(result_lines) + "\n")
