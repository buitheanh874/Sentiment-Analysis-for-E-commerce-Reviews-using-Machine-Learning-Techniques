"""
Prediction runtime used by FastAPI web UI.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from demo import (
    load_issue_model,
    load_models,
    predict_sentiment as predict_classic_sentiment,
)


LABEL_PRIORITY = {
    "NEGATIVE": 4,
    "NEEDS_ATTENTION": 3,
    "UNCERTAIN": 2,
    "POSITIVE": 1,
}

ISSUE_TAG_TO_LABEL = {
    "shipping": "delivery_shipping",
    "quality": "product_quality",
    "packaging": "product_quality",
    "service": "customer_service",
    "usability": "usability",
    "value": "value_price",
    "general": "other",
}


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_classic_runtime():
    base_dir = _base_dir()
    try:
        vectorizer, selector, model, meta, model_info = load_models(base_dir, verbose=False)
    except SystemExit as exc:                                               
        raise RuntimeError("Classic model artifacts are missing or invalid.") from exc

    issue_bundle = load_issue_model(base_dir, verbose=False)
    return vectorizer, selector, model, meta, model_info, issue_bundle


@lru_cache(maxsize=1)
def load_transformer_runtime():
    base_dir = _base_dir()
    try:
        from demo_transformer import load_transformer_model

        tokenizer, model = load_transformer_model(base_dir, verbose=False)
    except SystemExit as exc:                                               
        raise RuntimeError("Transformer model/dependencies are unavailable.") from exc
    except Exception as exc:                
        raise RuntimeError(str(exc)) from exc
    return tokenizer, model


def to_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted


def summarize_issue_labels(result: Dict[str, Any]) -> str:
    issue_rows = result.get("issue_labels", [])
    if issue_rows:
        return ", ".join(f"{row['label']}:{row['confidence']:.2f}" for row in issue_rows)
    fallback = result.get("issue_tags", [])
    if fallback:
        return ", ".join(fallback)
    return "-"


def _resolve_issue_labels(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Resolve issue labels for UI:
    - Keep trained labels when they are specific.
    - If trained output is only `other` but rule-based tags contain a specific issue,
      prefer the specific issue labels for clearer demo behavior.
    """
    issue_rows = result.get("issue_labels", [])
    fallback_tags = [str(tag).strip().lower() for tag in result.get("issue_tags", []) if str(tag).strip()]

    trained_labels = [str(row.get("label", "")).strip() for row in issue_rows]
    has_specific_trained = any(label and label != "other" for label in trained_labels)

    mapped_rule_labels: List[str] = []
    for tag in fallback_tags:
        mapped = ISSUE_TAG_TO_LABEL.get(tag)
        if mapped and mapped not in mapped_rule_labels:
            mapped_rule_labels.append(mapped)

                                                            
    if has_specific_trained:
        return issue_rows

                                                                                    
    specific_rule_labels = [label for label in mapped_rule_labels if label != "other"]
    if not specific_rule_labels:
        return issue_rows

    other_conf = 0.75
    if issue_rows:
        best_other = max(
            [to_float(row.get("confidence")) for row in issue_rows if str(row.get("label", "")).strip() == "other"],
            default=None,
        )
        if best_other is not None:
            other_conf = max(0.55, min(0.95, float(best_other) - 0.05))

    return [
        {
            "label": label,
            "confidence": float(other_conf),
            "threshold": 0.5,
        }
        for label in specific_rule_labels
    ]


def build_risk_score(label: str, probability: Optional[float]) -> float:
    base = LABEL_PRIORITY.get(label, 0) * 100.0
    prob = 0.5 if probability is None else max(0.0, min(1.0, probability))

    if label == "NEGATIVE":
        severity = (1.0 - prob) * 20.0
    elif label == "NEEDS_ATTENTION":
        severity = (0.8 - abs(prob - 0.5)) * 15.0
    elif label == "UNCERTAIN":
        severity = 10.0
    else:
        severity = prob * 5.0

    return round(base + max(severity, 0.0), 1)


def collect_issue_summary(classic_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in classic_raw:
        issue_rows = item["classic"].get("issue_labels", [])
        for issue in issue_rows:
            rows.append(
                {
                    "label": str(issue.get("label", "")),
                    "confidence": float(issue.get("confidence", 0.0)),
                }
            )

    if not rows:
        return []

    df = pd.DataFrame(rows)
    summary_df = (
        df.groupby("label", as_index=False)
        .agg(
            count=("label", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .sort_values(["count", "avg_confidence", "label"], ascending=[False, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary_df["avg_confidence"] = summary_df["avg_confidence"].round(3)
    return summary_df.to_dict(orient="records")


def build_overview_metrics(classic_df: pd.DataFrame) -> Dict[str, int]:
    labels = classic_df["classic_label"].astype(str)
    negative = int((labels == "NEGATIVE").sum())
    needs_attention = int((labels == "NEEDS_ATTENTION").sum())
    uncertain = int((labels == "UNCERTAIN").sum())
    positive = int((labels == "POSITIVE").sum())
    return {
        "total": int(len(classic_df)),
        "flagged": negative + needs_attention,
        "negative": negative,
        "needs_attention": needs_attention,
        "uncertain": uncertain,
        "positive": positive,
    }


def build_label_distribution(classic_df: pd.DataFrame) -> List[Dict[str, Any]]:
    ordered_labels = ["NEGATIVE", "NEEDS_ATTENTION", "UNCERTAIN", "POSITIVE"]
    counts = classic_df["classic_label"].astype(str).value_counts()
    total = max(len(classic_df), 1)
    return [
        {
            "label": label,
            "count": int(counts.get(label, 0)),
            "share_percent": round((int(counts.get(label, 0)) / total) * 100, 1),
        }
        for label in ordered_labels
    ]


def build_attention_queue(classic_df: pd.DataFrame) -> List[Dict[str, Any]]:
    queue = classic_df[classic_df["classic_label"].isin(["NEGATIVE", "NEEDS_ATTENTION", "UNCERTAIN"])].copy()
    if queue.empty:
        return []
    queue = queue.sort_values(["risk_score", "classic_probability"], ascending=[False, True], na_position="last")
    queue = queue[["text", "classic_label", "classic_probability", "issue_summary", "risk_score"]].head(20)

                                                                              
    if "classic_probability" in queue.columns:
        queue["classic_probability"] = queue["classic_probability"].where(
            pd.notna(queue["classic_probability"]),
            None,
        )

    records = queue.to_dict(orient="records")
    for row in records:
        row["classic_probability"] = to_float(row.get("classic_probability"))
    return records


def _parse_texts(raw_texts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for item in raw_texts:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            cleaned.append(text)
    return cleaned


def model_status(include_transformer: bool = False) -> Dict[str, Any]:
    _, _, _, _, model_info, issue_bundle = load_classic_runtime()
    issue_mode = "trained classifier" if issue_bundle is not None else "rule-based fallback"

    result: Dict[str, Any] = {
        "classic": {
            "loaded": True,
            "message": "Trained scikit-learn runtime online",
            "model_info": model_info,
            "issue_mode": issue_mode,
        },
        "transformer": {
            "requested": include_transformer,
            "loaded": False,
            "message": "disabled",
        },
    }

    if include_transformer:
        try:
            load_transformer_runtime()
            result["transformer"]["loaded"] = True
            result["transformer"]["message"] = "ready"
        except RuntimeError as exc:
            result["transformer"]["loaded"] = False
            result["transformer"]["message"] = str(exc)

    return result


def analyze_reviews(raw_texts: List[str], include_transformer: bool = False) -> Dict[str, Any]:
    texts = _parse_texts(raw_texts)
    if not texts:
        raise ValueError("No valid input texts found.")

    vectorizer, selector, model, meta, model_info, issue_bundle = load_classic_runtime()

    transformer_tokenizer = None
    transformer_model = None
    transformer_status = {
        "requested": include_transformer,
        "loaded": False,
        "message": "disabled",
    }

    if include_transformer:
        try:
            transformer_tokenizer, transformer_model = load_transformer_runtime()
            transformer_status = {
                "requested": True,
                "loaded": True,
                "message": "ready",
            }
        except RuntimeError as exc:
            transformer_status = {
                "requested": True,
                "loaded": False,
                "message": str(exc),
            }

    classic_rows: List[Dict[str, Any]] = []
    classic_raw: List[Dict[str, Any]] = []

    for text in texts:
        classic_result = predict_classic_sentiment(
            text,
            vectorizer,
            selector,
            model,
            meta,
            issue_bundle=issue_bundle,
        )
        resolved_issue_rows = _resolve_issue_labels(classic_result)
        classic_result["issue_labels"] = resolved_issue_rows

        classic_probability = to_float(classic_result.get("probability"))
        classic_label = str(classic_result.get("label", "N/A"))
        issue_rows = resolved_issue_rows
        fallback_tags = classic_result.get("issue_tags", [])

        row = {
            "text": text,
            "classic_label": classic_label,
            "classic_probability": classic_probability,
            "classic_confidence": classic_result.get("confidence", "N/A"),
            "fallback_reason": classic_result.get("fallback_reason") or "-",
            "issue_summary": summarize_issue_labels(classic_result),
            "issue_count": len(issue_rows) if issue_rows else len(fallback_tags),
            "risk_score": build_risk_score(classic_label, classic_probability),
            "transformer_label": None,
            "transformer_probability": None,
            "transformer_confidence": None,
            "transformer_reason": None,
            "agreement": None,
        }

        if transformer_tokenizer is not None and transformer_model is not None:
            from demo_transformer import predict_sentiment as predict_transformer_sentiment

            transformer_result = predict_transformer_sentiment(text, transformer_tokenizer, transformer_model)
            t_label = str(transformer_result.get("label", "N/A"))
            row["transformer_label"] = t_label
            row["transformer_probability"] = to_float(transformer_result.get("probability"))
            row["transformer_confidence"] = transformer_result.get("confidence", "N/A")
            row["transformer_reason"] = transformer_result.get("fallback_reason") or "-"
            row["agreement"] = "match" if t_label == classic_label else "mismatch"

        classic_rows.append(row)
        classic_raw.append({"text": text, "classic": classic_result})

    classic_df = pd.DataFrame(classic_rows)

    issue_summary = collect_issue_summary(classic_raw)
    summary = build_overview_metrics(classic_df)
    label_distribution = build_label_distribution(classic_df)
    attention_queue = build_attention_queue(classic_df)

    mismatch_count = None
    if transformer_status["loaded"]:
        mismatch_count = int((classic_df["agreement"] == "mismatch").sum())

    return {
        "status": {
            "classic": {
                "loaded": True,
                "message": "Trained scikit-learn runtime online",
                "model_info": model_info,
                "issue_mode": "trained classifier" if issue_bundle is not None else "rule-based fallback",
            },
            "transformer": transformer_status,
        },
        "summary": summary,
        "label_distribution": label_distribution,
        "attention_queue": attention_queue,
        "issue_summary": issue_summary,
        "mismatch_count": mismatch_count,
        "predictions": classic_rows,
    }
