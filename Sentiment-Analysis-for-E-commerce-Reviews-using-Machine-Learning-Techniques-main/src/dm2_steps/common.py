import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.text_features import (
    CONTEXT_VARIANTS,
    DEFAULT_ABBREV_MAP,
    DEFAULT_NEGATION_WINDOW,
    VariantSpec,
    build_vectorizer_from_spec,
    clean_text,
)

# Headless plotting for CLI use
matplotlib.use("Agg")

SEED = 42
DEFAULT_DATA_PATH = "data/Gift_Cards.jsonl"
DEFAULT_THRESHOLDS: Tuple[float, float] = (0.40, 0.60)
CS_THRESHOLD_PAIRS: List[Tuple[float, float]] = [
    (0.40, 0.60),
    (0.45, 0.60),
    (0.50, 0.65),
    (0.55, 0.70),
]
K_GRID = [1000, 2000, 5000, 10000]
CLASS_WEIGHT_GRID_LR: List[Tuple[str, Optional[Dict[int, int]]]] = [
    ("none", None),
    ("balanced", "balanced"),
    ("w2", {0: 2, 1: 1}),
    ("w5", {0: 5, 1: 1}),
    ("w10", {0: 10, 1: 1}),
]
CLASS_WEIGHT_GRID_TREE: List[Tuple[str, Optional[Dict[int, int]]]] = [
    ("none", None),
    ("balanced", "balanced"),
    ("w2", {0: 2, 1: 1}),
    ("w5", {0: 5, 1: 1}),
    ("w10", {0: 10, 1: 1}),
]
MIN_NNZ_DEFAULT = 2


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    three_star: pd.DataFrame


@dataclass
class VectorizerBundle:
    vectorizer: object
    X_train: csr_matrix
    X_val: csr_matrix
    X_test: csr_matrix
    X_3star: Optional[csr_matrix]


@dataclass
class DM2Config:
    data_path: Path
    output_dir: Path
    enable_abbrev_norm: bool = False
    enable_negation_tagging: bool = False
    enable_clause_split: bool = False
    enable_char_ngrams: bool = False
    negation_window: int = DEFAULT_NEGATION_WINDOW
    min_nnz: int = MIN_NNZ_DEFAULT
    thresholds: Tuple[float, float] = DEFAULT_THRESHOLDS


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    df = pd.read_json(path, lines=True)
    required_cols = {"text", "rating"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols}, found {set(df.columns)}. "
            f"Available keys: {list(df.columns)}"
        )
    return df


def make_splits(
    df: pd.DataFrame,
    enable_abbrev_norm: bool = False,
    enable_negation: bool = False,
    negation_window: int = DEFAULT_NEGATION_WINDOW,
) -> SplitData:
    df = df.copy()
    df["clean_text"] = df["text"].fillna("").apply(
        lambda t: clean_text(
            t,
            enable_abbrev_norm=enable_abbrev_norm,
            enable_negation=enable_negation,
            negation_window=negation_window,
        )
    )
    df_strong = df[df["rating"].isin([1, 2, 4, 5])].copy()
    df_strong["label"] = df_strong["rating"].apply(lambda r: 0 if r in [1, 2] else 1)
    df_three = df[df["rating"] == 3].copy()

    X_clean = df_strong["clean_text"]
    y = df_strong["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_clean, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, stratify=y_temp, random_state=SEED
    )

    df_train = pd.DataFrame(
        {
            "text": df_strong.loc[X_train.index, "text"].values,
            "clean_text": X_train.values,
            "label": y_train,
            "rating": df_strong.loc[X_train.index, "rating"].values,
        }
    )
    df_val = pd.DataFrame(
        {
            "text": df_strong.loc[X_val.index, "text"].values,
            "clean_text": X_val.values,
            "label": y_val,
            "rating": df_strong.loc[X_val.index, "rating"].values,
        }
    )
    df_test = pd.DataFrame(
        {
            "text": df_strong.loc[X_test.index, "text"].values,
            "clean_text": X_test.values,
            "label": y_test,
            "rating": df_strong.loc[X_test.index, "rating"].values,
        }
    )

    return SplitData(train=df_train, val=df_val, test=df_test, three_star=df_three)


def fit_vectorizer(
    splits: SplitData,
    variant: VariantSpec = CONTEXT_VARIANTS[0],
    enable_abbrev_norm: bool = False,
    negation_window: int = DEFAULT_NEGATION_WINDOW,
) -> VectorizerBundle:
    """
    Fit a context-aware vectorizer (FeatureUnion) on train-only texts.
    """
    vectorizer = build_vectorizer_from_spec(
        variant, enable_abbrev_norm=enable_abbrev_norm, negation_window=negation_window
    )
    X_train = vectorizer.fit_transform(splits.train["text"])
    X_val = vectorizer.transform(splits.val["text"])
    X_test = vectorizer.transform(splits.test["text"])
    X_three = (
        vectorizer.transform(splits.three_star["text"])
        if not splits.three_star.empty
        else None
    )
    return VectorizerBundle(
        vectorizer=vectorizer,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_3star=X_three,
    )


def nnz_per_row(matrix: csr_matrix) -> np.ndarray:
    return np.diff(matrix.indptr)


def metrics_from_probs(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "f2_0": fbeta_score(y_true, y_pred, beta=2, pos_label=0),
        "precision_1": precision[1],
        "recall_1": recall[1],
        "f1_1": f1[1],
    }


def apply_uncertainty_rule(
    probs: np.ndarray,
    clean_texts: List[str],
    matrix: csr_matrix,
    thresholds: Tuple[float, float],
    min_nnz: int = MIN_NNZ_DEFAULT,
) -> pd.DataFrame:
    nnz_counts = nnz_per_row(matrix)
    low, high = thresholds
    records = []
    for i, prob in enumerate(probs):
        text = clean_texts[i] if isinstance(clean_texts, list) else clean_texts.iloc[i]
        token_count = len(text.split())
        nnz = int(nnz_counts[i])
        decision = -1
        reason = None
        if token_count < 2 or text.strip() == "":
            reason = "too_short"
        elif nnz < min_nnz:
            reason = "sparse_vector"
        else:
            if prob >= high:
                decision = 1
            elif prob <= low:
                decision = 0
            else:
                reason = "threshold_band"
        records.append(
            {
                "prob": float(prob),
                "nnz": nnz,
                "token_count": token_count,
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(records)


def selective_metrics(
    y_true: np.ndarray, decisions: pd.DataFrame
) -> Dict[str, float]:
    mask = decisions["decision"] != -1
    coverage = mask.mean()
    metrics = {
        "coverage": coverage,
        "uncertain_rate": 1 - coverage,
        "selective_precision_0": np.nan,
        "selective_recall_0": np.nan,
        "selective_f2_0": np.nan,
        "fnr_covered_0": np.nan,
    }
    if coverage > 0:
        covered_true = y_true[mask.values]
        covered_pred = decisions.loc[mask, "decision"].astype(int).values
        precision, recall, _, _ = precision_recall_fscore_support(
            covered_true, covered_pred, labels=[0, 1], zero_division=0
        )
        metrics.update(
            {
                "selective_precision_0": precision[0],
                "selective_recall_0": recall[0],
                "selective_f2_0": fbeta_score(
                    covered_true, covered_pred, beta=2, pos_label=0
                ),
            }
        )
        neg_mask = covered_true == 0
        if neg_mask.sum() > 0:
            fn_0 = np.logical_and(neg_mask, covered_pred == 1).sum()
            metrics["fnr_covered_0"] = fn_0 / neg_mask.sum()
    return metrics


def negative_first_better(
    cand: Dict[str, float], best: Optional[Dict[str, float]], k: int, best_k: int
) -> bool:
    if best is None:
        return True
    cand_key = (cand.get("recall_0", 0), cand.get("f2_0", 0), cand.get("precision_0", 0), -k)
    best_key = (best.get("recall_0", 0), best.get("f2_0", 0), best.get("precision_0", 0), -best_k)
    return cand_key > best_key


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def plot_bar(values: Dict[str, int], path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    keys = list(values.keys())
    plt.bar(keys, [values[k] for k in keys], color="steelblue")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_hist(data: np.ndarray, path: Path, title: str, bins: int = 30) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color="steelblue", alpha=0.85)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, path: Path, labels: List[str]) -> None:
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def prob_hist(
    probs: np.ndarray, labels: np.ndarray, path: Path, title: str
) -> None:
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 31)
    plt.hist(probs[labels == 0], bins=bins, alpha=0.6, label="Negative (0)")
    plt.hist(probs[labels == 1], bins=bins, alpha=0.6, label="Positive (1)")
    plt.xlabel("P(Positive)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def simple_prob_hist(probs: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 31)
    plt.hist(probs, bins=bins, color="steelblue", alpha=0.8)
    plt.xlabel("P(Positive)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def lr_model(penalty: str = "l2", class_weight=None) -> LogisticRegression:
    solver = "liblinear" if penalty in {"l1", "l2"} else "lbfgs"
    n_jobs = None if solver == "liblinear" else -1
    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=800,
        random_state=SEED,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )


def decision_tree(class_weight=None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=SEED, class_weight=class_weight)


def random_forest(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        max_features="sqrt",
        class_weight=class_weight,
    )


def persist_core_artifacts(
    models_dir: Path,
    vectorizer,
    selector,
    model,
) -> None:
    ensure_dir(models_dir)
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(selector, models_dir / "chi2_selector.joblib")
    joblib.dump(model, models_dir / "best_lr_model.joblib")
