import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import vstack
from scipy.stats import chi2 as chi2_dist
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src.dm2_steps.common import (
    DEFAULT_THRESHOLDS,
    fit_vectorizer,
    load_data,
    make_splits,
    selective_metrics,
)
from src.issue_steps.common import (
    ISSUE_LABELS,
    clean_with_stage1,
    load_issue_bundle,
    load_stage1_cleaning_config,
)
from src.text_features import CONTEXT_VARIANTS


SEED = 42


def _metrics_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1_vals, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_0": float(precision[0]),
        "recall_0": float(recall[0]),
        "f1_0": float(f1_vals[0]),
        "f2_0": float(fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0)),
        "precision_1": float(precision[1]),
        "recall_1": float(recall[1]),
        "f1_1": float(f1_vals[1]),
    }


def _softmax_rows(mat: np.ndarray) -> np.ndarray:
    stable = mat - mat.max(axis=1, keepdims=True)
    exp = np.exp(stable)
    return exp / exp.sum(axis=1, keepdims=True)


def _subsample_train(X, y: np.ndarray, max_train_samples: int, seed: int):
    if max_train_samples <= 0 or len(y) <= max_train_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx_neg = np.where(y == 0)[0]
    idx_pos = np.where(y == 1)[0]
    n_neg = int(round(max_train_samples * (len(idx_neg) / len(y))))
    n_neg = min(len(idx_neg), max(1, n_neg))
    n_pos = max_train_samples - n_neg
    n_pos = min(len(idx_pos), max(1, n_pos))
    choose_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    choose_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    idx = np.concatenate([choose_neg, choose_pos])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _decision_from_probs(
    probs: np.ndarray, clean_texts: List[str], low: float, high: float
) -> pd.DataFrame:
    rows = []
    for prob, txt in zip(probs, clean_texts):
        token_count = len(str(txt).split())
        if token_count < 2 or str(txt).strip() == "":
            rows.append({"decision": -1, "reason": "too_short"})
            continue
        if prob >= high:
            rows.append({"decision": 1, "reason": None})
        elif prob <= low:
            rows.append({"decision": 0, "reason": None})
        else:
            rows.append({"decision": -1, "reason": "threshold_band"})
    return pd.DataFrame(rows)


def _pick_variant(name: str):
    for spec in CONTEXT_VARIANTS:
        if spec.name == name:
            return spec
    return next(v for v in CONTEXT_VARIANTS if v.name == "V6")


@dataclass
class SyllabusRunContext:
    splits: object
    X_train: object
    X_val: object
    X_test: object
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def _encode_texts_for_lstm(
    texts: List[str],
    vocab: Dict[str, int],
    max_len: int,
) -> np.ndarray:
    arr = np.zeros((len(texts), max_len), dtype=np.int64)
    unk = 1
    for i, text in enumerate(texts):
        toks = str(text).split()[:max_len]
        ids = [vocab.get(tok, unk) for tok in toks]
        if ids:
            arr[i, : len(ids)] = ids
    return arr


def _build_vocab_for_lstm(train_texts: List[str], max_vocab: int) -> Dict[str, int]:
    counter = Counter()
    for text in train_texts:
        counter.update(str(text).split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in counter.most_common(max(0, max_vocab - len(vocab))):
        vocab[tok] = len(vocab)
    return vocab


def _build_context(args) -> SyllabusRunContext:
    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation=args.enable_negation_tagging,
        negation_window=args.negation_window,
    )
    spec = _pick_variant(args.variant)
    vec_bundle = fit_vectorizer(
        splits,
        variant=spec,
        enable_abbrev_norm=args.enable_abbrev_norm,
        negation_window=args.negation_window,
    )
    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    y_test = splits.test["label"].values
    return SyllabusRunContext(
        splits=splits,
        X_train=vec_bundle.X_train,
        X_val=vec_bundle.X_val,
        X_test=vec_bundle.X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def run_classic_syllabus_bench(args) -> None:
    np.random.seed(SEED)
    ctx = _build_context(args)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _subsample_train(
        ctx.X_train, ctx.y_train, args.max_train_samples, seed=SEED
    )
    X_val = ctx.X_val
    X_test = ctx.X_test
    y_val = ctx.y_val
    y_test = ctx.y_test

    rows = []
    test_comp = []

    models = {
        "logreg_l2": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=800,
            class_weight="balanced",
            random_state=SEED,
        ),
        "multinomial_nb": MultinomialNB(alpha=0.3),
        "complement_nb": ComplementNB(alpha=0.3),
        "perceptron": Perceptron(
            max_iter=25, class_weight="balanced", random_state=SEED, tol=1e-3
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=SEED,
            dual="auto",
            max_iter=5000,
        ),
        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            max_iter=35,
            class_weight="balanced",
            random_state=SEED,
            tol=1e-3,
        ),
    }

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - t0

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        val_metrics = _metrics_from_labels(y_val, y_val_pred)
        test_metrics = _metrics_from_labels(y_test, y_test_pred)

        rows.append(
            {
                "model": name,
                "split": "val",
                "train_seconds": train_seconds,
                **val_metrics,
            }
        )
        rows.append(
            {
                "model": name,
                "split": "test",
                "train_seconds": train_seconds,
                **test_metrics,
            }
        )
        test_comp.append(
            {
                "model": name,
                "recall_0": test_metrics["recall_0"],
                "precision_0": test_metrics["precision_0"],
                "f2_0": test_metrics["f2_0"],
            }
        )

        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            val_dec = _decision_from_probs(
                val_probs,
                ctx.splits.val["clean_text"].tolist(),
                args.threshold_low,
                args.threshold_high,
            )
            test_dec = _decision_from_probs(
                test_probs,
                ctx.splits.test["clean_text"].tolist(),
                args.threshold_low,
                args.threshold_high,
            )
            val_sel = selective_metrics(y_val, val_dec)
            test_sel = selective_metrics(y_test, test_dec)
            rows.append(
                {
                    "model": f"{name}_selective",
                    "split": "val",
                    "train_seconds": train_seconds,
                    **val_sel,
                }
            )
            rows.append(
                {
                    "model": f"{name}_selective",
                    "split": "test",
                    "train_seconds": train_seconds,
                    **test_sel,
                }
            )

    # Vector semantics + FFNN path (SVD embedding)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=SEED)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    X_test_svd = svd.transform(X_test)

    sem_lr = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        random_state=SEED,
    )
    t0 = time.time()
    sem_lr.fit(X_train_svd, y_train)
    sem_lr_seconds = time.time() - t0
    sem_lr_val = _metrics_from_labels(y_val, sem_lr.predict(X_val_svd))
    sem_lr_test = _metrics_from_labels(y_test, sem_lr.predict(X_test_svd))
    rows.append(
        {"model": "svd_semantic_logreg", "split": "val", "train_seconds": sem_lr_seconds, **sem_lr_val}
    )
    rows.append(
        {"model": "svd_semantic_logreg", "split": "test", "train_seconds": sem_lr_seconds, **sem_lr_test}
    )
    test_comp.append(
        {
            "model": "svd_semantic_logreg",
            "recall_0": sem_lr_test["recall_0"],
            "precision_0": sem_lr_test["precision_0"],
            "f2_0": sem_lr_test["f2_0"],
        }
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=args.mlp_max_iter,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    t0 = time.time()
    mlp.fit(X_train_svd, y_train)
    mlp_seconds = time.time() - t0
    mlp_val = _metrics_from_labels(y_val, mlp.predict(X_val_svd))
    mlp_test = _metrics_from_labels(y_test, mlp.predict(X_test_svd))
    rows.append(
        {"model": "ffnn_mlp_svd", "split": "val", "train_seconds": mlp_seconds, **mlp_val}
    )
    rows.append(
        {"model": "ffnn_mlp_svd", "split": "test", "train_seconds": mlp_seconds, **mlp_test}
    )
    test_comp.append(
        {
            "model": "ffnn_mlp_svd",
            "recall_0": mlp_test["recall_0"],
            "precision_0": mlp_test["precision_0"],
            "f2_0": mlp_test["f2_0"],
        }
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "nlp_syllabus_bench_metrics.csv", index=False)
    pd.DataFrame(test_comp).to_csv(out_dir / "nlp_syllabus_bench_test_summary.csv", index=False)

    comp_df = pd.DataFrame(test_comp).sort_values(by=["recall_0", "f2_0"], ascending=False)

    plt.figure(figsize=(10, 4.8))
    plt.bar(comp_df["model"], comp_df["recall_0"], color="#2f6f9f")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recall for negative class")
    plt.title("Test recall_0 across NLP syllabus baselines")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_syllabus_bench_recall0.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    plt.bar(comp_df["model"], comp_df["f2_0"], color="#4c9a2a")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("F2 for negative class")
    plt.title("Test F2_0 across NLP syllabus baselines")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_syllabus_bench_f2.png", dpi=220)
    plt.close()

    best = comp_df.iloc[0]
    summary_lines = [
        "# NLP Syllabus Bench Summary",
        "",
        f"Variant for sparse features: {args.variant}",
        f"Train subsample size: {len(y_train)} (max_train_samples={args.max_train_samples})",
        f"Threshold band for selective scoring: {args.threshold_low:.2f}/{args.threshold_high:.2f}",
        "",
        "Best test model by negative-first rule:",
        f"model={best['model']}, recall_0={best['recall_0']:.3f}, precision_0={best['precision_0']:.3f}, f2_0={best['f2_0']:.3f}",
        "",
        "Detailed metrics: nlp_syllabus_bench_metrics.csv",
        "Test summary: nlp_syllabus_bench_test_summary.csv",
    ]
    (out_dir / "nlp_syllabus_bench_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"[NLP EXT] Syllabus bench saved to {out_dir}")


def run_classic_ablation(args) -> None:
    np.random.seed(SEED)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation=args.enable_negation_tagging,
        negation_window=args.negation_window,
    )

    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    y_test = splits.test["label"].values

    train_index = np.arange(len(y_train))
    if int(args.max_train_samples) > 0 and len(train_index) > int(args.max_train_samples):
        rng = np.random.default_rng(SEED)
        idx_neg = np.where(y_train == 0)[0]
        idx_pos = np.where(y_train == 1)[0]
        neg_target = int(round(int(args.max_train_samples) * (len(idx_neg) / len(y_train))))
        neg_target = min(len(idx_neg), max(1, neg_target))
        pos_target = int(args.max_train_samples) - neg_target
        pos_target = min(len(idx_pos), max(1, pos_target))
        sampled_neg = rng.choice(idx_neg, size=neg_target, replace=False)
        sampled_pos = rng.choice(idx_pos, size=pos_target, replace=False)
        train_index = np.concatenate([sampled_neg, sampled_pos])
        rng.shuffle(train_index)

    ablation_specs = [
        _pick_variant("V0"),
        _pick_variant("V2"),
        _pick_variant("V5"),
        _pick_variant("V6"),
        _pick_variant("V7"),
    ]

    rows: List[Dict[str, object]] = []
    for spec in ablation_specs:
        vec_bundle = fit_vectorizer(
            splits,
            variant=spec,
            enable_abbrev_norm=args.enable_abbrev_norm,
            negation_window=args.negation_window,
        )
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=800,
            class_weight="balanced",
            random_state=SEED,
        )
        t0 = time.time()
        model.fit(vec_bundle.X_train[train_index], y_train[train_index])
        train_seconds = time.time() - t0

        val_probs = model.predict_proba(vec_bundle.X_val)[:, 1]
        test_probs = model.predict_proba(vec_bundle.X_test)[:, 1]
        val_pred = (val_probs >= 0.5).astype(int)
        test_pred = (test_probs >= 0.5).astype(int)
        val_metrics = _metrics_from_labels(y_val, val_pred)
        test_metrics = _metrics_from_labels(y_test, test_pred)

        test_dec = _decision_from_probs(
            test_probs,
            splits.test["clean_text"].tolist(),
            args.threshold_low,
            args.threshold_high,
        )
        test_sel = selective_metrics(y_test, test_dec)

        rows.append(
            {
                "variant": spec.name,
                "description": spec.description,
                "use_negation": int(spec.use_negation),
                "use_char_ngrams": int(spec.use_char),
                "use_clause_split": int(spec.use_clause),
                "use_lexicon": int(spec.use_lexicon),
                "split": "val",
                "train_seconds": train_seconds,
                **val_metrics,
                "coverage": np.nan,
                "selective_recall_0": np.nan,
                "selective_precision_0": np.nan,
                "selective_f2_0": np.nan,
            }
        )
        rows.append(
            {
                "variant": spec.name,
                "description": spec.description,
                "use_negation": int(spec.use_negation),
                "use_char_ngrams": int(spec.use_char),
                "use_clause_split": int(spec.use_clause),
                "use_lexicon": int(spec.use_lexicon),
                "split": "test",
                "train_seconds": train_seconds,
                **test_metrics,
                "coverage": float(test_sel.get("coverage", np.nan)),
                "selective_recall_0": float(test_sel.get("selective_recall_0", np.nan)),
                "selective_precision_0": float(test_sel.get("selective_precision_0", np.nan)),
                "selective_f2_0": float(test_sel.get("selective_f2_0", np.nan)),
            }
        )

    ablation_df = pd.DataFrame(rows)
    ablation_csv = out_dir / "nlp_ablation.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    test_df = ablation_df[ablation_df["split"] == "test"].copy()
    test_df = test_df.sort_values(
        by=["recall_0", "f2_0", "precision_0"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    best = test_df.iloc[0]

    plt.figure(figsize=(8.5, 4.2))
    plt.bar(test_df["variant"], test_df["f2_0"], color="#2f6f9f")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test F2_0")
    plt.title("Classic ablation: F2_0 by variant")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_ablation_f2.png", dpi=220)
    plt.close()

    summary_lines = [
        "# Classic Ablation Summary",
        "",
        f"Threshold band for selective metrics: {args.threshold_low:.2f}/{args.threshold_high:.2f}",
        f"Train samples used: {len(train_index)} (max_train_samples={args.max_train_samples})",
        "Variants include targeted toggles for negation, character n-grams, and lexicon features.",
        "",
        "Best test variant under negative-first criterion:",
        (
            f"variant={best['variant']}, recall_0={best['recall_0']:.3f}, "
            f"precision_0={best['precision_0']:.3f}, f2_0={best['f2_0']:.3f}"
        ),
        "",
        "Files:",
        "nlp_ablation.csv",
        "nlp_ablation_f2.png",
    ]
    (out_dir / "nlp_ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] Classic ablation outputs saved to {out_dir}")


PRIMARY_METRICS = ["accuracy", "precision_0", "recall_0", "f1_0", "f2_0", "f1"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _predict_binary_with_scores(model, X) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return probs.astype(float), preds.astype(int)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        probs = _sigmoid(scores)
        preds = (scores >= 0.0).astype(int)
        return probs.astype(float), preds.astype(int)
    preds = np.asarray(model.predict(X)).astype(int).ravel()
    probs = preds.astype(float)
    return probs, preds


def _stable_seed_offset(key: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(key)) % 10000)


def _load_eval_bundle(args):
    model_meta = {}
    meta_path = Path("models/variant_meta.json")
    if meta_path.exists():
        try:
            model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            model_meta = {}

    enable_abbrev = bool(model_meta.get("enable_abbrev_norm", args.enable_abbrev_norm))
    enable_neg = bool(model_meta.get("negation", args.enable_negation_tagging))
    neg_window = int(model_meta.get("negation_window", args.negation_window))
    variant_name = str(model_meta.get("variant", args.variant))

    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=enable_abbrev,
        enable_negation=enable_neg,
        negation_window=neg_window,
    )
    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    y_test = splits.test["label"].values

    vec_path = Path("models/tfidf_vectorizer.joblib")
    sel_path = Path("models/chi2_selector.joblib")
    model_path = Path("models/best_lr_model.joblib")
    has_saved_artifacts = vec_path.exists() and sel_path.exists() and model_path.exists()

    if has_saved_artifacts:
        vectorizer = joblib.load(vec_path)
        selector = joblib.load(sel_path)
        classic_model = joblib.load(model_path)

        X_train = selector.transform(vectorizer.transform(splits.train["text"].tolist()))
        X_val = selector.transform(vectorizer.transform(splits.val["text"].tolist()))
        X_test = selector.transform(vectorizer.transform(splits.test["text"].tolist()))
        source = "saved_classic_artifacts"
    else:
        spec = _pick_variant(variant_name)
        vec_bundle = fit_vectorizer(
            splits,
            variant=spec,
            enable_abbrev_norm=enable_abbrev,
            negation_window=neg_window,
        )
        X_train = vec_bundle.X_train
        X_val = vec_bundle.X_val
        X_test = vec_bundle.X_test
        classic_model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=800,
            class_weight="balanced",
            random_state=SEED,
        )
        X_trainval = vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        classic_model.fit(X_trainval, y_trainval)
        source = f"trained_fallback_{variant_name}"

    return {
        "splits": splits,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "classic_model": classic_model,
        "source": source,
    }


def _bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str, iters: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(max(1, int(iters))):
        idx = rng.integers(0, n, size=n)
        sampled = _metrics_from_labels(y_true[idx], y_pred[idx])
        values.append(float(sampled[metric_name]))
    arr = np.array(values, dtype=float)
    point = float(_metrics_from_labels(y_true, y_pred)[metric_name])
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _bootstrap_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_name: str,
    iters: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(max(1, int(iters))):
        idx = rng.integers(0, n, size=n)
        m_a = _metrics_from_labels(y_true[idx], y_pred_a[idx])[metric_name]
        m_b = _metrics_from_labels(y_true[idx], y_pred_b[idx])[metric_name]
        diffs.append(float(m_a - m_b))
    arr = np.array(diffs, dtype=float)
    point = float(_metrics_from_labels(y_true, y_pred_a)[metric_name] - _metrics_from_labels(y_true, y_pred_b)[metric_name])
    return {
        "point_diff": point,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _mcnemar_significance(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    n01 = int(np.logical_and(a_correct, ~b_correct).sum())
    n10 = int(np.logical_and(~a_correct, b_correct).sum())
    denom = n01 + n10
    if denom == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(n01 - n10) - 1.0) ** 2 / float(denom)
        p_value = float(chi2_dist.sf(chi2_stat, df=1))
    return {
        "n01": n01,
        "n10": n10,
        "chi2_cc": float(chi2_stat),
        "p_value": p_value,
    }


def _taxonomy_category(text: str) -> str:
    raw = str(text)
    lower = f" {raw.lower()} "
    token_count = len(raw.split())
    punctuation_count = sum(ch in "!?.,;:" for ch in raw)
    alpha_count = sum(ch.isalpha() for ch in raw)

    if token_count < 3:
        return "short_text"
    if any(tok in lower for tok in [" gr8 ", " thx ", " idk ", " imo ", " w/ ", " w/o "]):
        return "slang_or_abbrev"
    if any(tok in lower for tok in [" not ", " no ", " never ", " cannot ", " can't ", " dont ", " didn't ", " didnt "]):
        return "negation_pattern"
    if any(tok in lower for tok in [" but ", " however ", " although ", " though ", " yet "]):
        return "contrast_pattern"
    if any(tok in lower for tok in [" redeem", " code", " refund", " return", " delivery", " shipping", " support", " scam", " fraud"]):
        return "domain_issue_term"
    if punctuation_count >= max(3, int(alpha_count * 0.3)):
        return "punctuation_heavy"
    return "other"


def _write_error_taxonomy(
    out_dir: Path,
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: List[Dict[str, object]] = []
    for idx, (text, true_label, pred_label, prob) in enumerate(zip(texts, y_true, y_pred, probs)):
        error_type = None
        if int(true_label) == 0 and int(pred_label) == 1:
            error_type = "false_negative_0"
        elif int(true_label) == 1 and int(pred_label) == 0:
            error_type = "false_positive_0"
        if error_type is None:
            continue
        detail_rows.append(
            {
                "row_id": idx,
                "error_type": error_type,
                "category": _taxonomy_category(text),
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "prob_positive": float(prob),
                "text": str(text),
            }
        )

    details_df = pd.DataFrame(detail_rows)
    details_csv = out_dir / "nlp_eval_error_details.csv"
    details_df.to_csv(details_csv, index=False, encoding="utf-8")

    if details_df.empty:
        summary_df = pd.DataFrame(columns=["error_type", "category", "count", "share"])
        summary_csv = out_dir / "nlp_eval_error_taxonomy.csv"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
        md_path = out_dir / "nlp_eval_error_taxonomy.md"
        md_path.write_text("# Error Taxonomy Report\n\nNo misclassified test rows found.\n", encoding="utf-8")
        return summary_df, details_df

    grouped = (
        details_df.groupby(["error_type", "category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["error_type", "count", "category"], ascending=[True, False, True], kind="mergesort")
    )
    grouped["share"] = grouped.groupby("error_type")["count"].transform(lambda x: x / x.sum())
    summary_df = grouped.rename(columns={"category": "taxonomy_category"})
    summary_csv = out_dir / "nlp_eval_error_taxonomy.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    lines = [
        "# Error Taxonomy Report",
        "",
        f"- total_misclassified_rows: {len(details_df)}",
        f"- false_negative_0: {int((details_df['error_type'] == 'false_negative_0').sum())}",
        f"- false_positive_0: {int((details_df['error_type'] == 'false_positive_0').sum())}",
        "",
        "## Taxonomy Counts",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    for error_type in ["false_negative_0", "false_positive_0"]:
        subset = details_df[details_df["error_type"] == error_type]
        if subset.empty:
            continue
        lines.append(f"## Examples: {error_type}")
        top_categories = (
            subset["category"].value_counts().head(3).index.tolist()
        )
        for category in top_categories:
            cat_rows = subset[subset["category"] == category].head(2)
            lines.append(f"- {category}:")
            for _, row in cat_rows.iterrows():
                snippet = str(row["text"]).replace("\n", " ").strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                lines.append(
                    f"  row={int(row['row_id'])}, p_pos={row['prob_positive']:.3f}, text=\"{snippet}\""
                )
        lines.append("")

    md_path = out_dir / "nlp_eval_error_taxonomy.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_df, details_df


def run_eval_rigor(args) -> None:
    np.random.seed(SEED)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_eval_bundle(args)
    splits = bundle["splits"]
    X_train = bundle["X_train"]
    X_val = bundle["X_val"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_val = bundle["y_val"]
    y_test = bundle["y_test"]
    classic_model = bundle["classic_model"]

    classic_probs, classic_pred = _predict_binary_with_scores(classic_model, X_test)
    classic_metrics = _metrics_from_labels(y_test, classic_pred)

    X_trainval = vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    X_fit, y_fit = _subsample_train(X_trainval, y_trainval, int(args.max_train_samples), seed=SEED)

    baseline_models = {
        "multinomial_nb": MultinomialNB(alpha=0.3),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=SEED,
            dual="auto",
            max_iter=5000,
        ),
    }
    predictions = {
        "classic_main": {
            "probs": classic_probs,
            "pred": classic_pred,
            "metrics": classic_metrics,
        }
    }

    for name, model in baseline_models.items():
        model.fit(X_fit, y_fit)
        probs, pred = _predict_binary_with_scores(model, X_test)
        predictions[name] = {
            "probs": probs,
            "pred": pred,
            "metrics": _metrics_from_labels(y_test, pred),
        }

    metrics_rows = []
    for model_name, payload in predictions.items():
        row = {"model": model_name}
        row.update(payload["metrics"])
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "nlp_eval_metrics.csv", index=False, encoding="utf-8")

    ci_rows = []
    for model_name, payload in predictions.items():
        for metric_name in PRIMARY_METRICS:
            key = f"{model_name}:{metric_name}"
            ci = _bootstrap_metric_ci(
                y_true=y_test,
                y_pred=payload["pred"],
                metric_name=metric_name,
                iters=int(args.bootstrap_iters),
                seed=SEED + _stable_seed_offset(key),
            )
            ci_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "point_estimate": ci["point_estimate"],
                    "ci_low_95": ci["ci_low"],
                    "ci_high_95": ci["ci_high"],
                    "bootstrap_iters": int(args.bootstrap_iters),
                }
            )
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out_dir / "nlp_eval_ci_bootstrap.csv", index=False, encoding="utf-8")

    significance_rows = []
    for baseline_name in ["multinomial_nb", "linear_svm"]:
        sig = _mcnemar_significance(
            y_true=y_test,
            y_pred_a=predictions["classic_main"]["pred"],
            y_pred_b=predictions[baseline_name]["pred"],
        )
        diff_recall = _bootstrap_diff_ci(
            y_true=y_test,
            y_pred_a=predictions["classic_main"]["pred"],
            y_pred_b=predictions[baseline_name]["pred"],
            metric_name="recall_0",
            iters=int(args.bootstrap_iters),
            seed=SEED + _stable_seed_offset(f"{baseline_name}:recall_0"),
        )
        diff_f2 = _bootstrap_diff_ci(
            y_true=y_test,
            y_pred_a=predictions["classic_main"]["pred"],
            y_pred_b=predictions[baseline_name]["pred"],
            metric_name="f2_0",
            iters=int(args.bootstrap_iters),
            seed=SEED + _stable_seed_offset(f"{baseline_name}:f2_0"),
        )
        significance_rows.append(
            {
                "model_a": "classic_main",
                "model_b": baseline_name,
                "metric_a_recall_0": predictions["classic_main"]["metrics"]["recall_0"],
                "metric_b_recall_0": predictions[baseline_name]["metrics"]["recall_0"],
                "metric_a_f2_0": predictions["classic_main"]["metrics"]["f2_0"],
                "metric_b_f2_0": predictions[baseline_name]["metrics"]["f2_0"],
                "diff_recall_0": diff_recall["point_diff"],
                "diff_recall_0_ci_low_95": diff_recall["ci_low"],
                "diff_recall_0_ci_high_95": diff_recall["ci_high"],
                "diff_f2_0": diff_f2["point_diff"],
                "diff_f2_0_ci_low_95": diff_f2["ci_low"],
                "diff_f2_0_ci_high_95": diff_f2["ci_high"],
                "n01": sig["n01"],
                "n10": sig["n10"],
                "chi2_cc": sig["chi2_cc"],
                "p_value": sig["p_value"],
                "significant_0_05": int(sig["p_value"] < 0.05),
                "bootstrap_iters": int(args.bootstrap_iters),
            }
        )
    significance_df = pd.DataFrame(significance_rows)
    significance_df.to_csv(out_dir / "nlp_eval_significance.csv", index=False, encoding="utf-8")

    taxonomy_df, details_df = _write_error_taxonomy(
        out_dir=out_dir,
        texts=splits.test["text"].tolist(),
        y_true=y_test,
        y_pred=predictions["classic_main"]["pred"],
        probs=predictions["classic_main"]["probs"],
    )

    summary_lines = [
        "# Evaluation Rigor Summary",
        "",
        f"- source_classic_model: {bundle['source']}",
        f"- bootstrap_iters: {int(args.bootstrap_iters)}",
        f"- test_size: {len(y_test)}",
        "",
        "## Main model metrics (classic_main)",
        "",
        f"- recall_0: {predictions['classic_main']['metrics']['recall_0']:.3f}",
        f"- precision_0: {predictions['classic_main']['metrics']['precision_0']:.3f}",
        f"- f2_0: {predictions['classic_main']['metrics']['f2_0']:.3f}",
        "",
        "## Files",
        "",
        "- nlp_eval_metrics.csv",
        "- nlp_eval_ci_bootstrap.csv",
        "- nlp_eval_significance.csv",
        "- nlp_eval_error_taxonomy.csv",
        "- nlp_eval_error_details.csv",
        "- nlp_eval_error_taxonomy.md",
        "",
        f"- taxonomy_rows: {len(taxonomy_df)}",
        f"- error_detail_rows: {len(details_df)}",
    ]
    (out_dir / "nlp_eval_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] Evaluation rigor outputs saved to {out_dir}")


def _prepare_issue_multilabel_dataframe(
    labels_path: Path,
    data_path: Path,
    seed: int,
    max_samples: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_path not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    if "id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df.insert(0, "id", np.arange(len(labels_df), dtype=int).astype(str))
    labels_df["id"] = labels_df["id"].astype(str)

    base_df = load_data(data_path).copy()
    if "id" not in base_df.columns:
        base_df.insert(0, "id", np.arange(len(base_df), dtype=int).astype(str))
    base_df["id"] = base_df["id"].astype(str)
    base_df = base_df[["id", "text", "rating"]]

    merged = labels_df.merge(base_df, on="id", how="left", suffixes=("", "_base"))
    if "text" not in merged.columns:
        merged["text"] = merged["text_base"]
    else:
        merged["text"] = merged["text"].fillna(merged["text_base"])
    merged["text"] = merged["text"].fillna("").astype(str)

    missing_labels = [label for label in ISSUE_LABELS if label not in merged.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns in labels CSV: {missing_labels}")

    label_df = merged[ISSUE_LABELS].apply(pd.to_numeric, errors="coerce")
    valid_mask = label_df.isin([0, 1]).all(axis=1)
    invalid_rows = int((~valid_mask).sum())
    merged = merged.loc[valid_mask].copy()
    label_df = label_df.loc[valid_mask].astype(int)

    row_sum = label_df.sum(axis=1)
    zero_rows = int((row_sum == 0).sum())
    keep_mask = row_sum > 0
    merged = merged.loc[keep_mask].copy()
    label_df = label_df.loc[keep_mask].copy()

    cleaning_cfg = load_stage1_cleaning_config(Path("."))
    merged["clean_text"] = merged["text"].apply(lambda txt: clean_with_stage1(txt, cleaning_cfg))

    sampled_rows = 0
    if max_samples > 0 and len(merged) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(merged), size=int(max_samples), replace=False)
        merged = merged.iloc[idx].reset_index(drop=True)
        label_df = label_df.iloc[idx].reset_index(drop=True)
        sampled_rows = int(max_samples)
    else:
        merged = merged.reset_index(drop=True)
        label_df = label_df.reset_index(drop=True)

    y = label_df[ISSUE_LABELS].to_numpy(dtype=np.int64)
    prep_info = {
        "rows_total_input": int(len(labels_df)),
        "rows_invalid_labels_dropped": int(invalid_rows),
        "rows_zero_labels_dropped": int(zero_rows),
        "rows_after_cleaning": int(len(merged)),
        "rows_sampled_cap": int(sampled_rows),
        "cleaning": cleaning_cfg,
    }
    return merged, y, prep_info


def _issue_labelset_codes(y: np.ndarray) -> np.ndarray:
    codes: List[str] = []
    for row in y:
        active = [ISSUE_LABELS[idx] for idx, value in enumerate(row) if int(value) == 1]
        codes.append("|".join(active) if active else "none")
    return np.array(codes, dtype=object)


def _can_issue_stratify(codes: np.ndarray) -> bool:
    if len(codes) == 0:
        return False
    counts = pd.Series(codes).value_counts()
    return bool(len(counts) > 1 and int(counts.min()) >= 2)


def _split_issue_indices(
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    idx = np.arange(len(y))
    if len(idx) < 5:
        raise ValueError("Need at least 5 labeled rows for split.")

    codes = _issue_labelset_codes(y)
    try:
        if not _can_issue_stratify(codes):
            raise ValueError("Insufficient support for labelset stratification.")
        idx_train, idx_temp = train_test_split(
            idx,
            test_size=0.30,
            random_state=seed,
            stratify=codes,
        )
        temp_codes = codes[idx_temp]
        if _can_issue_stratify(temp_codes):
            idx_val, idx_test = train_test_split(
                idx_temp,
                test_size=2.0 / 3.0,
                random_state=seed,
                stratify=temp_codes,
            )
            return idx_train, idx_val, idx_test, "stratified_labelset"
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=2.0 / 3.0,
            random_state=seed,
            shuffle=True,
        )
        return idx_train, idx_val, idx_test, "partially_stratified"
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
        return idx_train, idx_val, idx_test, "random"


def _issue_metrics_overall(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "label_cardinality_true": float(np.mean(y_true.sum(axis=1))),
        "label_cardinality_pred": float(np.mean(y_pred.sum(axis=1))),
    }


def _issue_metrics_per_label(y_true: np.ndarray, y_pred: np.ndarray, split: str, model: str) -> pd.DataFrame:
    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    pred_pos = y_pred.sum(axis=0)
    rows: List[Dict[str, object]] = []
    for idx, label in enumerate(ISSUE_LABELS):
        rows.append(
            {
                "model": model,
                "split": split,
                "label": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1_vals[idx]),
                "support_true": int(support[idx]),
                "predicted_positive": int(pred_pos[idx]),
            }
        )
    return pd.DataFrame(rows)


def _tune_issue_thresholds(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    grid = np.arange(0.10, 0.91, 0.05)
    tuned: Dict[str, float] = {}
    for idx, label in enumerate(ISSUE_LABELS):
        y_col = y_true[:, idx]
        if np.unique(y_col).size < 2:
            tuned[label] = 0.50
            continue
        best_thr = 0.50
        best_f1 = -1.0
        best_dist = 10.0
        for thr in grid:
            pred = (probs[:, idx] >= float(thr)).astype(int)
            f1_val = float(f1_score(y_col, pred, zero_division=0))
            dist = abs(float(thr) - 0.50)
            if f1_val > best_f1 + 1e-12 or (abs(f1_val - best_f1) <= 1e-12 and dist < best_dist):
                best_f1 = f1_val
                best_thr = float(thr)
                best_dist = dist
        tuned[label] = float(best_thr)
    return tuned


def _apply_issue_thresholds(probs: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    thr = np.array([float(thresholds.get(label, 0.5)) for label in ISSUE_LABELS], dtype=float)
    return (probs >= thr).astype(np.int64)


def _classic_issue_predict(texts: List[str], model_dir: Path) -> Optional[Dict[str, object]]:
    bundle = load_issue_bundle(model_dir)
    if bundle is None:
        return None
    cleaned = [clean_with_stage1(txt, bundle.cleaning) for txt in texts]
    X = bundle.vectorizer.transform(cleaned)
    if bundle.selector is not None:
        X = bundle.selector.transform(X)
    probs = bundle.model.predict_scores(X)
    thresholds = {label: float(bundle.thresholds.get(label, 0.5)) for label in ISSUE_LABELS}
    preds = _apply_issue_thresholds(probs, thresholds)
    return {
        "bundle": bundle,
        "probs": probs,
        "pred": preds,
        "thresholds": thresholds,
        "cleaned": cleaned,
    }


def run_issue_transformer_multilabel(args) -> None:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print(
            "[NLP EXT] transformers/torch not installed. "
            "Install optional deps: pip install -r requirements-optional.txt"
        )
        return

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    max_train_samples = int(args.max_train_samples)
    max_total_samples = int(args.max_total_samples)
    max_length = int(args.max_length)
    epochs = float(args.epochs)
    batch_size = int(args.batch_size)
    if args.fast_mode:
        max_train_samples = min(max_train_samples, int(args.fast_max_train_samples))
        max_total_samples = min(max_total_samples, int(args.fast_max_total_samples))
        max_length = min(max_length, int(args.fast_max_length))
        epochs = min(epochs, float(args.fast_epochs))
        print(
            "[NLP EXT] issue transformer fast_mode enabled: "
            f"max_total_samples={max_total_samples}, max_train_samples={max_train_samples}, "
            f"max_length={max_length}, epochs={epochs}"
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, y, prep_info = _prepare_issue_multilabel_dataframe(
        labels_path=args.labels_path,
        data_path=args.data_path,
        seed=int(args.seed),
        max_samples=max_total_samples,
    )
    idx_train, idx_val, idx_test, split_method = _split_issue_indices(y, seed=int(args.seed))

    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    if max_train_samples > 0 and len(train_df) > max_train_samples:
        rng = np.random.default_rng(int(args.seed))
        sub_idx = rng.choice(len(train_df), size=max_train_samples, replace=False)
        train_df = train_df.iloc[sub_idx].reset_index(drop=True)
        y_train = y_train[sub_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class MultiLabelIssueDataset(Dataset):
        def __init__(self, texts: List[str], labels: np.ndarray):
            self.enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return int(self.labels.shape[0])

        def __getitem__(self, idx: int):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.labels[idx]
            return item

    train_ds = MultiLabelIssueDataset(train_df["clean_text"].tolist(), y_train)
    val_ds = MultiLabelIssueDataset(val_df["clean_text"].tolist(), y_val)
    test_ds = MultiLabelIssueDataset(test_df["clean_text"].tolist(), y_test)

    pos_counts = y_train.sum(axis=0).astype(float)
    neg_counts = float(len(y_train)) - pos_counts
    pos_weight = torch.tensor((neg_counts + 1.0) / (pos_counts + 1.0), dtype=torch.float32)

    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(model.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ISSUE_LABELS),
        problem_type="multi_label_classification",
    )
    train_args = TrainingArguments(
        output_dir=str(out_dir / "hf_runs_issue"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=float(args.lr),
        evaluation_strategy="no",
        logging_strategy="steps",
        logging_steps=50 if args.fast_mode else 250,
        save_strategy="no",
        dataloader_num_workers=0,
        seed=int(args.seed),
        report_to=[],
    )
    trainer = MultiLabelTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    val_logits = trainer.predict(val_ds).predictions
    test_logits = trainer.predict(test_ds).predictions
    val_probs = _sigmoid(np.asarray(val_logits))
    test_probs = _sigmoid(np.asarray(test_logits))

    thresholds = {label: 0.5 for label in ISSUE_LABELS}
    if bool(args.tune_thresholds):
        thresholds = _tune_issue_thresholds(y_val, val_probs)

    val_pred = _apply_issue_thresholds(val_probs, thresholds)
    test_pred = _apply_issue_thresholds(test_probs, thresholds)

    overall_rows = []
    per_label_frames = []
    for split_name, y_true, y_pred in [("val", y_val, val_pred), ("test", y_test, test_pred)]:
        row = {"model": "transformer_multilabel", "split": split_name}
        row.update(_issue_metrics_overall(y_true, y_pred))
        overall_rows.append(row)
        per_label_frames.append(
            _issue_metrics_per_label(
                y_true=y_true,
                y_pred=y_pred,
                split=split_name,
                model="transformer_multilabel",
            )
        )
    overall_df = pd.DataFrame(overall_rows)
    per_label_df = pd.concat(per_label_frames, ignore_index=True)
    overall_df.to_csv(out_dir / "nlp_issue_transformer_metrics_overall.csv", index=False)
    per_label_df.to_csv(out_dir / "nlp_issue_transformer_metrics_per_label.csv", index=False)

    threshold_df = pd.DataFrame(
        {"label": ISSUE_LABELS, "threshold": [float(thresholds[label]) for label in ISSUE_LABELS]}
    )
    threshold_df.to_csv(out_dir / "nlp_issue_transformer_thresholds.csv", index=False)

    plt.figure(figsize=(10, 4))
    test_f1 = (
        per_label_df[per_label_df["split"] == "test"]
        .set_index("label")
        .loc[ISSUE_LABELS]["f1"]
    )
    plt.bar(ISSUE_LABELS, test_f1.values, color="#2a9d8f")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1")
    plt.title("Per-label F1 (test, transformer multi-label)")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_issue_transformer_per_label_f1.png", dpi=220)
    plt.close()

    hybrid_rows: List[Dict[str, object]] = []
    routing_df = pd.DataFrame()
    classic_payload = _classic_issue_predict(test_df["text"].tolist(), model_dir=args.model_dir)
    if classic_payload is not None:
        classic_pred = classic_payload["pred"]
        classic_probs = classic_payload["probs"]
        thr = np.array([classic_payload["thresholds"][label] for label in ISSUE_LABELS], dtype=float)
        min_margin = np.min(np.abs(classic_probs - thr), axis=1)
        uncertain_mask = min_margin < float(args.hybrid_margin)
        max_route_rate = float(np.clip(float(args.hybrid_max_route_rate), 0.0, 1.0))
        if max_route_rate <= 0.0:
            uncertain_mask = np.zeros(len(min_margin), dtype=bool)
        elif max_route_rate < 1.0 and float(np.mean(uncertain_mask)) > max_route_rate:
            cutoff = float(np.quantile(min_margin, max_route_rate))
            uncertain_mask = min_margin <= cutoff

        hybrid_pred = classic_pred.copy()
        hybrid_pred[uncertain_mask] = test_pred[uncertain_mask]

        for name, pred in [
            ("classic_issue_model", classic_pred),
            ("transformer_multilabel", test_pred),
            ("hybrid_route", hybrid_pred),
        ]:
            row = {"model": name, "split": "test"}
            row.update(_issue_metrics_overall(y_test, pred))
            hybrid_rows.append(row)

        routing_df = pd.DataFrame(
            {
                "id": test_df["id"].astype(str),
                "classic_min_margin": min_margin.astype(float),
                "route_to_transformer": uncertain_mask.astype(int),
                "classic_label_count": classic_pred.sum(axis=1).astype(int),
                "transformer_label_count": test_pred.sum(axis=1).astype(int),
                "hybrid_label_count": hybrid_pred.sum(axis=1).astype(int),
            }
        )
        routing_df.to_csv(out_dir / "nlp_issue_hybrid_routing.csv", index=False)
        pd.DataFrame(hybrid_rows).to_csv(out_dir / "nlp_issue_hybrid_metrics.csv", index=False)

    if not args.skip_model_save:
        model_save_dir = args.model_save_dir
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)
        (model_save_dir / "label_list.json").write_text(json.dumps(ISSUE_LABELS, indent=2), encoding="utf-8")
        (model_save_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(f"[NLP EXT] issue transformer model saved to {model_save_dir}")

    quantized_path = None
    fp32_state_path = None
    if bool(args.export_quantized):
        fp32_state_path = out_dir / "nlp_issue_transformer_fp32_state.pt"
        quantized_path = out_dir / "nlp_issue_transformer_quantized_state.pt"
        cpu_model = model.to("cpu")
        torch.save(cpu_model.state_dict(), fp32_state_path)
        quantized_model = torch.quantization.quantize_dynamic(
            cpu_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        torch.save(quantized_model.state_dict(), quantized_path)

    summary_lines = [
        "# Issue Transformer Multi-label Summary",
        "",
        f"model_name: {args.model_name}",
        f"split_method: {split_method}",
        f"train_size: {len(train_df)}",
        f"val_size: {len(val_df)}",
        f"test_size: {len(test_df)}",
        f"threshold_mode: {'tuned' if bool(args.tune_thresholds) else 'fixed_0.5'}",
        f"hybrid_margin: {float(args.hybrid_margin):.3f}",
        f"hybrid_max_route_rate: {float(args.hybrid_max_route_rate):.3f}",
        "",
        "Main test metrics (transformer_multilabel):",
    ]
    test_row = overall_df[overall_df["split"] == "test"].iloc[0]
    summary_lines.extend(
        [
            f"- micro_f1: {float(test_row['micro_f1']):.4f}",
            f"- macro_f1: {float(test_row['macro_f1']):.4f}",
            f"- subset_accuracy: {float(test_row['subset_accuracy']):.4f}",
            f"- hamming_loss: {float(test_row['hamming_loss']):.4f}",
            "",
            "Files:",
            "- nlp_issue_transformer_metrics_overall.csv",
            "- nlp_issue_transformer_metrics_per_label.csv",
            "- nlp_issue_transformer_thresholds.csv",
            "- nlp_issue_transformer_per_label_f1.png",
        ]
    )
    if not routing_df.empty:
        summary_lines.extend(
            [
                "- nlp_issue_hybrid_metrics.csv",
                "- nlp_issue_hybrid_routing.csv",
                f"- hybrid_routed_rows: {int(routing_df['route_to_transformer'].sum())}",
            ]
        )
    if quantized_path is not None and fp32_state_path is not None:
        fp32_size_mb = fp32_state_path.stat().st_size / (1024 * 1024)
        quant_size_mb = quantized_path.stat().st_size / (1024 * 1024)
        summary_lines.extend(
            [
                "- nlp_issue_transformer_fp32_state.pt",
                "- nlp_issue_transformer_quantized_state.pt",
                f"- fp32_state_size_mb: {fp32_size_mb:.2f}",
                f"- quantized_state_size_mb: {quant_size_mb:.2f}",
            ]
        )
    (out_dir / "nlp_issue_transformer_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    train_config = {
        "labels_path": str(args.labels_path),
        "data_path": str(args.data_path),
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "epochs": float(epochs),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "max_total_samples": int(max_total_samples),
        "max_train_samples": int(max_train_samples),
        "split_method": split_method,
        "tune_thresholds": bool(args.tune_thresholds),
        "hybrid_margin": float(args.hybrid_margin),
        "hybrid_max_route_rate": float(args.hybrid_max_route_rate),
        "classic_model_dir": str(args.model_dir),
        "skip_model_save": bool(args.skip_model_save),
        "export_quantized": bool(args.export_quantized),
        "prep_info": prep_info,
    }
    (out_dir / "nlp_issue_transformer_train_config.json").write_text(
        json.dumps(train_config, indent=2),
        encoding="utf-8",
    )
    print(f"[NLP EXT] Issue transformer outputs saved to {out_dir}")


def run_rnn_lstm_baseline(args) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[NLP EXT] torch not installed. Install optional deps to run LSTM baseline.")
        return

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ctx = _build_context(args)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_texts_full = ctx.splits.train["clean_text"].tolist()
    val_texts = ctx.splits.val["clean_text"].tolist()
    test_texts = ctx.splits.test["clean_text"].tolist()
    y_train_full = ctx.y_train

    if args.max_train_samples > 0 and len(y_train_full) > args.max_train_samples:
        rng = np.random.default_rng(SEED)
        idx_neg = np.where(y_train_full == 0)[0]
        idx_pos = np.where(y_train_full == 1)[0]
        n_neg = int(round(args.max_train_samples * (len(idx_neg) / len(y_train_full))))
        n_neg = min(len(idx_neg), max(1, n_neg))
        n_pos = args.max_train_samples - n_neg
        n_pos = min(len(idx_pos), max(1, n_pos))
        choose_neg = rng.choice(idx_neg, size=n_neg, replace=False)
        choose_pos = rng.choice(idx_pos, size=n_pos, replace=False)
        sampled_idx = np.concatenate([choose_neg, choose_pos])
        rng.shuffle(sampled_idx)
        train_texts = [train_texts_full[i] for i in sampled_idx]
        y_train = y_train_full[sampled_idx]
    else:
        train_texts = train_texts_full
        y_train = y_train_full

    vocab = _build_vocab_for_lstm(train_texts, max_vocab=args.lstm_max_vocab)
    X_train = _encode_texts_for_lstm(train_texts, vocab=vocab, max_len=args.lstm_max_len)
    X_val = _encode_texts_for_lstm(val_texts, vocab=vocab, max_len=args.lstm_max_len)
    X_test = _encode_texts_for_lstm(test_texts, vocab=vocab, max_len=args.lstm_max_len)
    y_val = ctx.y_val.astype(np.int64)
    y_test = ctx.y_test.astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.lstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.lstm_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.lstm_batch_size, shuffle=False)

    class LSTMBaseline(nn.Module):
        def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            emb = self.embedding(x)
            out, _ = self.lstm(emb)
            pooled = out[:, -1, :]
            pooled = self.dropout(pooled)
            return self.head(pooled).squeeze(1)

    model = LSTMBaseline(
        vocab_size=len(vocab),
        emb_dim=args.lstm_emb_dim,
        hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_num_layers,
        dropout=args.lstm_dropout,
    ).to(device)

    neg = max(1, int((y_train == 0).sum()))
    pos = max(1, int((y_train == 1).sum()))
    pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lstm_lr)

    def _collect_logits(loader):
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())
        return np.concatenate(all_logits), np.concatenate(all_y)

    best_state = None
    best_val_f2 = -1.0
    t0 = time.time()
    for _epoch in range(args.lstm_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_logits, val_true = _collect_logits(val_loader)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_pred = (val_probs >= 0.5).astype(int)
        val_metrics = _metrics_from_labels(val_true, val_pred)
        if val_metrics["f2_0"] > best_val_f2:
            best_val_f2 = val_metrics["f2_0"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_true = _collect_logits(val_loader)
    test_logits, test_true = _collect_logits(test_loader)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    val_pred = (val_probs >= 0.5).astype(int)
    test_pred = (test_probs >= 0.5).astype(int)
    val_metrics = _metrics_from_labels(val_true, val_pred)
    test_metrics = _metrics_from_labels(test_true, test_pred)

    val_dec = _decision_from_probs(
        val_probs,
        val_texts,
        args.threshold_low,
        args.threshold_high,
    )
    test_dec = _decision_from_probs(
        test_probs,
        test_texts,
        args.threshold_low,
        args.threshold_high,
    )
    val_sel = selective_metrics(y_val, val_dec)
    test_sel = selective_metrics(y_test, test_dec)

    rows = [
        {"model": "lstm_text", "split": "val", "train_seconds": train_seconds, **val_metrics},
        {"model": "lstm_text", "split": "test", "train_seconds": train_seconds, **test_metrics},
        {"model": "lstm_text_selective", "split": "val", "train_seconds": train_seconds, **val_sel},
        {"model": "lstm_text_selective", "split": "test", "train_seconds": train_seconds, **test_sel},
    ]
    metrics_path = out_dir / "nlp_rnn_lstm_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary_lines = [
        "# RNN/LSTM Baseline Summary",
        "",
        f"Model: single-direction LSTM ({args.lstm_num_layers} layer(s), hidden={args.lstm_hidden_dim})",
        f"Train samples: {len(y_train)}",
        f"Vocab size: {len(vocab)}",
        f"Max sequence length: {args.lstm_max_len}",
        f"Epochs: {args.lstm_epochs}",
        "",
        "Test metrics (threshold=0.5):",
        f"recall_0={test_metrics['recall_0']:.3f}, precision_0={test_metrics['precision_0']:.3f}, f2_0={test_metrics['f2_0']:.3f}",
        "",
        "Selective test metrics (threshold band):",
        f"coverage={test_sel.get('coverage', np.nan):.3f}, selective_recall_0={test_sel.get('selective_recall_0', np.nan):.3f}, selective_f2_0={test_sel.get('selective_f2_0', np.nan):.3f}",
        "",
        "File:",
        "nlp_rnn_lstm_metrics.csv",
    ]
    (out_dir / "nlp_rnn_lstm_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] RNN/LSTM baseline outputs saved to {out_dir}")


def run_mlm_probe(args) -> None:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        print("[NLP EXT] transformers/torch not installed. Install optional deps first.")
        return

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.eval()

    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise SystemExit(f"Model {args.model_name} has no mask token, cannot run mlm_probe.")

    probes = [
        {
            "probe_id": "p1_positive_quality",
            "template": f"this gift card is {mask_token}.",
            "expected_tokens": ["great", "good", "excellent", "useful"],
        },
        {
            "probe_id": "p2_negative_delivery",
            "template": f"the delivery was {mask_token} and frustrating.",
            "expected_tokens": ["slow", "late", "bad", "terrible"],
        },
        {
            "probe_id": "p3_customer_support",
            "template": f"customer support was very {mask_token}.",
            "expected_tokens": ["helpful", "responsive", "kind", "good"],
        },
        {
            "probe_id": "p4_refund_problem",
            "template": f"getting a refund was {mask_token}.",
            "expected_tokens": ["hard", "difficult", "impossible", "slow"],
        },
        {
            "probe_id": "p5_value",
            "template": f"overall value for money is {mask_token}.",
            "expected_tokens": ["good", "great", "poor", "bad"],
        },
    ]

    rows = []
    for probe in probes:
        text = probe["template"]
        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits

        mask_positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if len(mask_positions) == 0:
            continue
        mask_idx = int(mask_positions[0].item())
        token_logits = logits[0, mask_idx]
        top_ids = torch.topk(token_logits, k=args.top_k).indices.tolist()
        top_tokens_raw = tokenizer.convert_ids_to_tokens(top_ids)
        top_tokens = []
        for tok in top_tokens_raw:
            norm = tok.strip().lower().replace("##", "")
            norm = re.sub(r"^[^a-z0-9]+", "", norm)
            norm = re.sub(r"[^a-z0-9]+$", "", norm)
            top_tokens.append(norm)
        expected = [t.strip().lower() for t in probe["expected_tokens"]]
        hit = int(any(tok in expected for tok in top_tokens))

        rows.append(
            {
                "probe_id": probe["probe_id"],
                "template": text,
                "expected_tokens": "|".join(expected),
                "top_tokens": "|".join(top_tokens),
                "hit_at_k": hit,
            }
        )

    if not rows:
        raise SystemExit("mlm_probe produced no rows. Check tokenizer mask token handling.")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "nlp_mlm_probe.csv"
    df.to_csv(csv_path, index=False)

    hit_rate = float(df["hit_at_k"].mean())
    summary_lines = [
        "# MLM Probe Summary",
        "",
        f"Model: {args.model_name}",
        f"Top-k: {args.top_k}",
        f"Probe count: {len(df)}",
        f"Hit@k: {hit_rate:.3f}",
        "",
        "File:",
        "nlp_mlm_probe.csv",
    ]
    (out_dir / "nlp_mlm_probe.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] MLM probe outputs saved to {out_dir}")


def run_llm_prompt_baseline(args) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[NLP EXT] sentence-transformers not installed. Install optional deps first.")
        return

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation=False,
        negation_window=args.negation_window,
    )

    model = SentenceTransformer(args.model_name)

    prompt_negative = "Instruction: classify as NEGATIVE. The review expresses complaints, failures, delays, or poor quality."
    prompt_positive = "Instruction: classify as POSITIVE. The review expresses satisfaction, quality, smooth delivery, or value."
    prompt_emb = model.encode(
        [prompt_negative, prompt_positive],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    def _eval_split(split_df: pd.DataFrame, y_true: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        texts = split_df["clean_text"].tolist()
        text_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sims = text_emb @ prompt_emb.T  # [n,2] : neg, pos
        # Stretch similarity gap into a more separable probability signal.
        diff = sims[:, 1] - sims[:, 0]
        probs = 1.0 / (1.0 + np.exp(-args.logit_scale * diff))  # P(positive)
        y_pred = (probs >= 0.5).astype(int)

        base_metrics = _metrics_from_labels(y_true, y_pred)
        decisions = _decision_from_probs(
            probs,
            texts,
            args.threshold_low,
            args.threshold_high,
        )
        sel_metrics = selective_metrics(y_true, decisions)
        return base_metrics, sel_metrics

    y_val = splits.val["label"].values
    y_test = splits.test["label"].values

    val_base, val_sel = _eval_split(splits.val, y_val)
    test_base, test_sel = _eval_split(splits.test, y_test)

    rows = [
        {"model": "llm_prompt_semantic", "split": "val", **val_base},
        {"model": "llm_prompt_semantic", "split": "test", **test_base},
        {"model": "llm_prompt_semantic_selective", "split": "val", **val_sel},
        {"model": "llm_prompt_semantic_selective", "split": "test", **test_sel},
    ]
    metrics_path = out_dir / "nlp_llm_prompt_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary_lines = [
        "# LLM Prompt Baseline Summary",
        "",
        f"Embedding model: {args.model_name}",
        "Method: prompt-style semantic similarity classification.",
        f"Logit scale: {args.logit_scale:.2f}",
        f"Threshold band: {args.threshold_low:.2f}/{args.threshold_high:.2f}",
        "",
        "Test metrics (hard decision):",
        f"recall_0={test_base['recall_0']:.3f}, precision_0={test_base['precision_0']:.3f}, f2_0={test_base['f2_0']:.3f}",
        "",
        "Selective test metrics:",
        f"coverage={test_sel.get('coverage', np.nan):.3f}, selective_recall_0={test_sel.get('selective_recall_0', np.nan):.3f}, selective_f2_0={test_sel.get('selective_f2_0', np.nan):.3f}",
        "",
        "File:",
        "nlp_llm_prompt_metrics.csv",
    ]
    (out_dir / "nlp_llm_prompt_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] LLM prompt baseline outputs saved to {out_dir}")


def _tokenize_for_lm(text: str) -> List[str]:
    tokens = [tok for tok in str(text).split() if tok]
    return ["<s>"] + tokens + ["</s>"]


@dataclass
class NGramLM:
    order: int
    unigram: Counter
    bigram: Dict[str, Counter]
    vocab_size: int
    total_unigrams: int
    k: float

    def unigram_prob(self, w: str) -> float:
        return (self.unigram.get(w, 0) + self.k) / (
            self.total_unigrams + self.k * self.vocab_size
        )

    def bigram_prob(self, w_prev: str, w: str) -> float:
        c_prev = self.unigram.get(w_prev, 0)
        c_pair = self.bigram.get(w_prev, Counter()).get(w, 0)
        return (c_pair + self.k) / (c_prev + self.k * self.vocab_size)

    def sentence_log_prob(self, tokens: List[str]) -> float:
        if self.order == 1:
            return float(sum(math.log(self.unigram_prob(tok)) for tok in tokens[1:]))
        total = 0.0
        for i in range(1, len(tokens)):
            total += math.log(self.bigram_prob(tokens[i - 1], tokens[i]))
        return float(total)


def _fit_ngram_lm(texts: List[str], order: int = 2, k: float = 1.0) -> NGramLM:
    unigram = Counter()
    bigram = defaultdict(Counter)
    for text in texts:
        toks = _tokenize_for_lm(text)
        unigram.update(toks[1:])
        for i in range(1, len(toks)):
            bigram[toks[i - 1]][toks[i]] += 1
    vocab = set(unigram.keys())
    vocab.add("</s>")
    vocab_size = max(1, len(vocab))
    total_uni = int(sum(unigram.values()))
    return NGramLM(
        order=order,
        unigram=unigram,
        bigram=dict(bigram),
        vocab_size=vocab_size,
        total_unigrams=total_uni,
        k=k,
    )


def _perplexity(model: NGramLM, texts: List[str]) -> Tuple[float, float]:
    total_log_prob = 0.0
    total_tokens = 0
    for text in texts:
        toks = _tokenize_for_lm(text)
        total_log_prob += model.sentence_log_prob(toks)
        total_tokens += max(1, len(toks) - 1)
    avg_log_prob = total_log_prob / max(1, total_tokens)
    ppl = float(math.exp(-avg_log_prob))
    return ppl, float(avg_log_prob)


def _sample_next_from_counter(counter: Counter, rng: random.Random) -> str:
    if not counter:
        return "</s>"
    words, counts = zip(*counter.items())
    total = float(sum(counts))
    r = rng.random() * total
    cum = 0.0
    for w, c in zip(words, counts):
        cum += c
        if cum >= r:
            return w
    return words[-1]


def _generate_bigram(model: NGramLM, seed: str, max_len: int, rng: random.Random) -> str:
    prev = seed if seed else "<s>"
    if prev not in model.bigram:
        prev = "<s>"
    out = []
    for _ in range(max_len):
        nxt = _sample_next_from_counter(model.bigram.get(prev, Counter()), rng)
        if nxt == "</s>":
            break
        out.append(nxt)
        prev = nxt
    return " ".join(out)


def run_ngram_language_model(args) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation=False,
        negation_window=args.negation_window,
    )

    train_texts = splits.train["clean_text"].tolist()
    val_texts = splits.val["clean_text"].tolist()
    test_texts = splits.test["clean_text"].tolist()

    unigram_lm = _fit_ngram_lm(train_texts, order=1, k=args.add_k)
    bigram_lm = _fit_ngram_lm(train_texts, order=2, k=args.add_k)

    rows = []
    for split_name, texts in [("val", val_texts), ("test", test_texts)]:
        ppl_u, alp_u = _perplexity(unigram_lm, texts)
        ppl_b, alp_b = _perplexity(bigram_lm, texts)
        rows.append(
            {
                "model": "unigram_add_k",
                "split": split_name,
                "perplexity": ppl_u,
                "avg_log_prob": alp_u,
                "add_k": args.add_k,
            }
        )
        rows.append(
            {
                "model": "bigram_add_k",
                "split": split_name,
                "perplexity": ppl_b,
                "avg_log_prob": alp_b,
                "add_k": args.add_k,
            }
        )
    lm_df = pd.DataFrame(rows)
    lm_df.to_csv(out_dir / "nlp_ngram_lm_metrics.csv", index=False)

    rng = random.Random(SEED)
    seeds = ["<s>", "great", "not", "delivery", "refund"]
    gen_rows = []
    for seed in seeds:
        for i in range(3):
            txt = _generate_bigram(bigram_lm, seed=seed, max_len=args.gen_max_len, rng=rng)
            gen_rows.append({"seed": seed, "sample_id": i + 1, "generated_text": txt})
    pd.DataFrame(gen_rows).to_csv(out_dir / "nlp_ngram_generated_samples.csv", index=False)

    test_u = lm_df[(lm_df["model"] == "unigram_add_k") & (lm_df["split"] == "test")].iloc[0]
    test_b = lm_df[(lm_df["model"] == "bigram_add_k") & (lm_df["split"] == "test")].iloc[0]
    summary_lines = [
        "# N-gram Language Model Summary",
        "",
        f"Add-k smoothing value: {args.add_k:.2f}",
        f"Unigram test perplexity: {test_u['perplexity']:.3f}",
        f"Bigram test perplexity: {test_b['perplexity']:.3f}",
        "",
        "Lower perplexity is better. Bigram should improve over unigram on this corpus.",
        "",
        "Files:",
        "nlp_ngram_lm_metrics.csv",
        "nlp_ngram_generated_samples.csv",
    ]
    (out_dir / "nlp_ngram_lm_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] N-gram LM outputs saved to {out_dir}")


def build_course_fit_matrix(args) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = out_dir / "nlp_syllabus_bench_test_summary.csv"
    ablation_path = out_dir / "nlp_ablation.csv"
    ngram_path = out_dir / "nlp_ngram_lm_metrics.csv"
    rnn_path = out_dir / "nlp_rnn_lstm_metrics.csv"
    mlm_path = out_dir / "nlp_mlm_probe.csv"
    llm_prompt_path = out_dir / "nlp_llm_prompt_metrics.csv"
    issue_tf_path = out_dir / "nlp_issue_transformer_metrics_overall.csv"
    trans_path = Path("results/nlp_ext/nlp_metrics.csv")
    issue_path = Path("results/issue_steps_char_demo/02_metrics_overall.csv")

    has_bench = bench_path.exists()
    has_ablation = ablation_path.exists()
    has_ngram = ngram_path.exists()
    has_rnn = rnn_path.exists()
    has_mlm = mlm_path.exists()
    has_llm_prompt = llm_prompt_path.exists()
    has_issue_tf = issue_tf_path.exists()
    has_trans = trans_path.exists()
    has_issue = issue_path.exists()

    # Simple coverage points by topic, max 1.0 each.
    topics = [
        ("Probability", 1.0 if has_ngram or has_bench else 0.5),
        ("Regular Expressions", 0.5),
        ("Evaluation Measures", 1.0 if has_bench or has_ablation or has_issue or has_issue_tf else 0.7),
        ("N-Gram Language Models", 1.0 if has_ngram else 0.2),
        ("Naive Bayes", 1.0 if has_bench else 0.2),
        ("Perceptron", 1.0 if has_bench else 0.2),
        ("Logistic Regression", 1.0),
        ("Feed-forward Neural Networks", 1.0 if has_bench else 0.3),
        ("Pytorch Basic", 1.0 if has_rnn or has_trans or has_mlm or has_issue_tf else 0.2),
        ("Vector Semantics and Embeddings", 0.8 if has_bench or has_ablation else 0.4),
        ("Recurrent Neural Networks", 1.0 if has_rnn else 0.2),
        ("Transformers and LLMs", 1.0 if has_trans or has_mlm or has_llm_prompt or has_issue_tf else 0.3),
        ("Masked Language Models", 1.0 if has_mlm else 0.2),
        ("Applications of LLMs", 0.9 if has_llm_prompt else 0.3),
    ]
    df = pd.DataFrame(topics, columns=["topic", "coverage_score"])
    df["coverage_percent"] = (df["coverage_score"] * 100).round(1)
    overall = float(df["coverage_score"].mean() * 100.0)
    df.to_csv(out_dir / "nlp_course_fit_matrix.csv", index=False)

    lines = [
        "# NLP Course Fit Matrix",
        "",
        f"Estimated overall syllabus coverage: {overall:.1f}%",
        "This matrix is score-based and uses available project artifacts only.",
        "",
        "Detailed rows are in nlp_course_fit_matrix.csv",
    ]
    (out_dir / "nlp_course_fit_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[NLP EXT] Course-fit matrix saved to {out_dir}")
