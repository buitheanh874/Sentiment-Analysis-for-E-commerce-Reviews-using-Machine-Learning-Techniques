import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.text_features import clean_text

ISSUE_LABELS: List[str] = [
    "delivery_shipping",
    "redemption_activation",
    "product_quality",
    "customer_service",
    "refund_return",
    "usability",
    "value_price",
    "fraud_scam",
    "other",
]

ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "delivery_shipping": [
        "delivery",
        "shipping",
        "ship",
        "shipped",
        "arrived",
        "arrival",
        "late",
        "delay",
        "delayed",
        "slow",
        "lost package",
        "tracking",
    ],
    "redemption_activation": [
        "redeem",
        "redemption",
        "activate",
        "activation",
        "code",
        "pin",
        "invalid code",
        "code not working",
        "cannot redeem",
        "card wont work",
        "card does not work",
    ],
    "product_quality": [
        "broken",
        "defective",
        "damaged",
        "poor quality",
        "flimsy",
        "cheap",
        "not as described",
        "bad quality",
    ],
    "customer_service": [
        "customer service",
        "support",
        "representative",
        "agent",
        "help desk",
        "unhelpful",
        "rude",
        "no response",
    ],
    "refund_return": [
        "refund",
        "returned",
        "return",
        "money back",
        "chargeback",
        "reimbursement",
        "exchange",
        "replacement",
    ],
    "usability": [
        "confusing",
        "difficult",
        "hard to use",
        "complicated",
        "error",
        "failed",
        "doesnt work",
        "does not work",
        "unable",
    ],
    "value_price": [
        "expensive",
        "overpriced",
        "pricey",
        "waste of money",
        "not worth",
        "too much",
    ],
    "fraud_scam": [
        "scam",
        "fraud",
        "fake",
        "stolen",
        "unauthorized",
        "suspicious",
        "phishing",
        "hack",
        "hacked",
    ],
    "other": [],
}

COMPLAINT_SIGNAL_LABELS = [label for label in ISSUE_LABELS if label != "other"]

SEED = 42
DEFAULT_DATA_PATH = Path("data/Gift_Cards.jsonl")
DEFAULT_RESULTS_DIR = Path("results/issue_steps")
DEFAULT_MODEL_DIR = Path("models/issue_classifier")


def sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, -20, 20)
    return 1.0 / (1.0 + np.exp(-clipped))


def normalize_text_for_keyword_scan(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return f" {text.lower()} "


def keyword_suggested_labels(text: str) -> List[str]:
    normalized = normalize_text_for_keyword_scan(text)
    hits: List[str] = []
    for label, phrases in ISSUE_KEYWORDS.items():
        if label == "other":
            continue
        for phrase in phrases:
            if f" {phrase} " in normalized:
                hits.append(label)
                break
    if not hits:
        hits = ["other"]
    return hits


def has_complaint_signal(text: str) -> bool:
    labels = keyword_suggested_labels(text)
    return any(label in COMPLAINT_SIGNAL_LABELS for label in labels)


def labels_to_pipe(labels: Sequence[str]) -> str:
    return "|".join(labels)


def load_stage1_cleaning_config(base_dir: Path = Path(".")) -> Dict[str, object]:
    defaults = {
        "enable_abbrev_norm": False,
        "enable_negation": False,
        "negation_window": 3,
    }
    meta_path = base_dir / "models" / "variant_meta.json"
    if not meta_path.exists():
        return defaults
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    return {
        "enable_abbrev_norm": bool(meta.get("enable_abbrev_norm", defaults["enable_abbrev_norm"])),
        "enable_negation": bool(meta.get("negation", defaults["enable_negation"])),
        "negation_window": int(meta.get("negation_window", defaults["negation_window"])),
    }


def clean_with_stage1(text: str, cleaning_cfg: Dict[str, object]) -> str:
    return clean_text(
        text,
        enable_abbrev_norm=bool(cleaning_cfg.get("enable_abbrev_norm", False)),
        enable_negation=bool(cleaning_cfg.get("enable_negation", False)),
        negation_window=int(cleaning_cfg.get("negation_window", 3)),
    )


class MultiLabelChi2Selector(BaseEstimator, TransformerMixin):
    """
    Multi-label Chi2 selector.
    Aggregates per-label Chi2 scores with max() and keeps top-k features.
    """

    def __init__(self, k: int = 5000):
        self.k = int(k)

    def fit(self, X, y):
        y_arr = np.asarray(y)
        if y_arr.ndim != 2:
            raise ValueError("Expected y to be a 2D binary matrix for multi-label Chi2.")
        n_features = X.shape[1]
        k = min(self.k, n_features)
        agg_scores = np.zeros(n_features, dtype=float)
        valid_any = False
        for col in range(y_arr.shape[1]):
            target = y_arr[:, col]
            if np.unique(target).size < 2:
                continue
            scores, _ = chi2(X, target)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            agg_scores = np.maximum(agg_scores, scores)
            valid_any = True
        if not valid_any:
            agg_scores = np.zeros(n_features, dtype=float)
        ranked = np.argsort(-agg_scores)[:k]
        self.selected_idx_ = np.sort(ranked)
        self.n_features_in_ = n_features
        self.scores_ = agg_scores
        return self

    def transform(self, X):
        if not hasattr(self, "selected_idx_"):
            raise RuntimeError("Selector must be fit before transform.")
        return X[:, self.selected_idx_]

    def get_support(self, indices: bool = False):
        if not hasattr(self, "selected_idx_"):
            raise RuntimeError("Selector must be fit before get_support.")
        if indices:
            return self.selected_idx_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_idx_] = True
        return mask

    @property
    def k_(self) -> int:
        if not hasattr(self, "selected_idx_"):
            return self.k
        return int(len(self.selected_idx_))


@dataclass
class PerLabelOVRModel:
    estimators: List[object]
    label_names: List[str]
    model_kind: str
    train_notes: Dict[str, str]

    def _scores_from_estimator(self, estimator, X) -> np.ndarray:
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X)
            classes = getattr(estimator, "classes_", None)
            if probs.ndim == 2 and probs.shape[1] == 1:
                if classes is not None and len(classes) == 1 and int(classes[0]) == 1:
                    return np.ones(probs.shape[0], dtype=float)
                return np.zeros(probs.shape[0], dtype=float)
            if probs.ndim == 2 and probs.shape[1] >= 2 and classes is not None:
                classes_list = list(classes)
                if 1 in classes_list:
                    pos_idx = classes_list.index(1)
                else:
                    pos_idx = int(np.argmax(classes))
                return probs[:, pos_idx].astype(float)
            return probs.ravel().astype(float)

        if hasattr(estimator, "decision_function"):
            raw = estimator.decision_function(X)
            raw = np.asarray(raw).ravel()
            return sigmoid(raw).astype(float)

        preds = estimator.predict(X)
        return np.asarray(preds, dtype=float).ravel()

    def predict_scores(self, X) -> np.ndarray:
        cols = [self._scores_from_estimator(est, X) for est in self.estimators]
        if not cols:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.vstack(cols).T

    def predict_binary(self, X, thresholds: Dict[str, float]) -> np.ndarray:
        scores = self.predict_scores(X)
        thr_vec = np.array([float(thresholds.get(label, 0.5)) for label in self.label_names], dtype=float)
        return (scores >= thr_vec).astype(int)


@dataclass
class BlendedOVRModel:
    """
    Score-level blend of two OVR-style models.
    final_score = alpha * primary + (1 - alpha) * secondary
    where alpha is tuned per label on validation split.
    """

    primary_model: Any
    secondary_model: Any
    label_names: List[str]
    blend_weights: Dict[str, float]
    train_notes: Dict[str, str]
    model_kind: str = "blended_ovr"

    def _alpha_vector(self) -> np.ndarray:
        return np.array(
            [float(self.blend_weights.get(label, 0.5)) for label in self.label_names],
            dtype=float,
        )

    def predict_scores(self, X) -> np.ndarray:
        primary_scores = np.asarray(self.primary_model.predict_scores(X), dtype=float)
        secondary_scores = np.asarray(self.secondary_model.predict_scores(X), dtype=float)
        if primary_scores.shape != secondary_scores.shape:
            raise ValueError(
                "Primary and secondary model score shapes must match for blending: "
                f"{primary_scores.shape} vs {secondary_scores.shape}"
            )
        alpha = self._alpha_vector().reshape(1, -1)
        return alpha * primary_scores + (1.0 - alpha) * secondary_scores

    def predict_binary(self, X, thresholds: Dict[str, float]) -> np.ndarray:
        scores = self.predict_scores(X)
        thr_vec = np.array(
            [float(thresholds.get(label, 0.5)) for label in self.label_names],
            dtype=float,
        )
        return (scores >= thr_vec).astype(int)


def _build_base_estimator(
    model_kind: str = "logreg",
    class_weight: Optional[object] = "balanced",
    random_state: int = SEED,
):
    if class_weight == "none":
        class_weight = None
    if model_kind == "logreg":
        return LogisticRegression(
            solver="liblinear",
            class_weight=class_weight,
            max_iter=2000,
            random_state=random_state,
        )
    if model_kind == "linearsvm":
        return LinearSVC(
            class_weight=class_weight,
            random_state=random_state,
            dual="auto",
            max_iter=5000,
        )
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def train_per_label_ovr(
    X,
    y: np.ndarray,
    label_names: Sequence[str],
    model_kind: str = "logreg",
    class_weight: Optional[object] = "balanced",
    class_weight_map: Optional[Dict[str, object]] = None,
    calibrate_probs: bool = False,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    random_state: int = SEED,
) -> PerLabelOVRModel:
    y_arr = np.asarray(y)
    if y_arr.ndim != 2:
        raise ValueError("Expected y to be 2D for multi-label training.")

    estimators: List[object] = []
    notes: Dict[str, str] = {}

    def _weight_label(value: Any) -> str:
        if value is None:
            return "none"
        if isinstance(value, str):
            return value
        return str(value)

    for idx, label in enumerate(label_names):
        target = y_arr[:, idx]
        unique = np.unique(target)
        if unique.size < 2:
            constant_value = int(unique[0]) if unique.size == 1 else 0
            dummy = DummyClassifier(strategy="constant", constant=constant_value)
            dummy.fit(X, target)
            estimators.append(dummy)
            notes[label] = f"constant_{constant_value};cw=none;calibrated=no"
            continue

        label_weight = (
            class_weight_map.get(label, class_weight)
            if class_weight_map is not None
            else class_weight
        )
        base_estimator = _build_base_estimator(
            model_kind=model_kind,
            class_weight=label_weight,
            random_state=random_state,
        )
        estimator = clone(base_estimator)
        calibrated = False
        fold_count = int(np.min(np.bincount(target.astype(int), minlength=2)))
        cv_folds = max(2, min(int(calibration_cv), fold_count))

        if calibrate_probs and model_kind in {"logreg", "linearsvm"} and fold_count >= 2:
            calibrated_estimator = CalibratedClassifierCV(
                estimator=estimator,
                method=calibration_method,
                cv=cv_folds,
            )
            calibrated_estimator.fit(X, target)
            estimators.append(calibrated_estimator)
            calibrated = True
        else:
            estimator.fit(X, target)
            estimators.append(estimator)

        notes[label] = (
            f"trained;cw={_weight_label(label_weight)};"
            f"calibrated={'yes' if calibrated else 'no'};"
            f"calibration_method={calibration_method if calibrated else 'none'};"
            f"calibration_cv={cv_folds if calibrated else 0}"
        )

    return PerLabelOVRModel(
        estimators=estimators,
        label_names=list(label_names),
        model_kind=model_kind,
        train_notes=notes,
    )


@dataclass
class IssueInferenceBundle:
    vectorizer: object
    selector: Optional[object]
    model: PerLabelOVRModel
    thresholds: Dict[str, float]
    label_list: List[str]
    cleaning: Dict[str, object]
    model_dir: Path


def has_issue_model(model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
    required = [
        model_dir / "vectorizer.joblib",
        model_dir / "ovr_model.joblib",
        model_dir / "thresholds.json",
        model_dir / "label_list.json",
    ]
    return all(path.exists() for path in required)


def load_issue_bundle(model_dir: Path = DEFAULT_MODEL_DIR) -> Optional[IssueInferenceBundle]:
    if not has_issue_model(model_dir):
        return None

    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    selector_path = model_dir / "chi2_selector.joblib"
    selector = joblib.load(selector_path) if selector_path.exists() else None
    model_obj = joblib.load(model_dir / "ovr_model.joblib")
    if not hasattr(model_obj, "predict_scores"):
        raise ValueError("Unsupported ovr_model.joblib format: missing predict_scores().")
    label_list = json.loads((model_dir / "label_list.json").read_text(encoding="utf-8"))
    thresholds_payload = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    if isinstance(thresholds_payload, dict) and "thresholds" in thresholds_payload:
        thresholds = thresholds_payload.get("thresholds", {})
        cleaning = thresholds_payload.get("cleaning", load_stage1_cleaning_config(Path(".")))
    else:
        thresholds = thresholds_payload
        cleaning = load_stage1_cleaning_config(Path("."))

    normalized_thresholds = {label: float(thresholds.get(label, 0.5)) for label in label_list}
    return IssueInferenceBundle(
        vectorizer=vectorizer,
        selector=selector,
        model=model_obj,
        thresholds=normalized_thresholds,
        label_list=list(label_list),
        cleaning=cleaning,
        model_dir=model_dir,
    )


def predict_issue_labels(text: str, bundle: IssueInferenceBundle) -> Dict[str, object]:
    cleaned = clean_with_stage1(text, bundle.cleaning)
    X = bundle.vectorizer.transform([cleaned])
    if bundle.selector is not None:
        X = bundle.selector.transform(X)

    scores = bundle.model.predict_scores(X)[0]
    confidences = {label: float(score) for label, score in zip(bundle.label_list, scores)}
    predicted = [
        {
            "label": label,
            "confidence": float(confidences[label]),
            "threshold": float(bundle.thresholds.get(label, 0.5)),
        }
        for label in bundle.label_list
        if confidences[label] >= float(bundle.thresholds.get(label, 0.5))
    ]
    predicted = sorted(predicted, key=lambda row: row["confidence"], reverse=True)
    return {
        "clean_text": cleaned,
        "predicted_labels": predicted,
        "confidences": confidences,
        "thresholds": {label: float(bundle.thresholds.get(label, 0.5)) for label in bundle.label_list},
    }
