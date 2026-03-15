"""FastAPI backend for the standalone demo UI."""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ISSUE_LABELS = [
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
POSITIVE_KEYWORDS = {"good", "great", "perfect", "excellent", "love", "fast", "happy", "satisfied"}
NEGATIVE_KEYWORDS = {"bad", "poor", "terrible", "awful", "hate", "broken", "late", "slow", "scam"}
ATTENTION_KEYWORDS = {"but", "however", "although", "issue", "problem", "support", "refund"}
ISSUE_KEYWORDS = {
    "delivery_shipping": ("ship", "shipping", "delivery", "arrive", "courier", "late"),
    "redemption_activation": ("redeem", "activation", "activate", "code", "claim"),
    "product_quality": ("quality", "print", "card", "damaged", "broken", "defect"),
    "customer_service": ("support", "service", "agent", "staff", "response", "replied"),
    "refund_return": ("refund", "return", "money back", "replace", "exchange"),
    "usability": ("use", "usable", "confusing", "steps", "difficult", "instructions"),
    "value_price": ("price", "cost", "value", "expensive", "cheap"),
    "fraud_scam": ("fraud", "scam", "fake", "stolen", "hack"),
}

DEMO_MODEL_INFO = {
    "variant": "demo_heuristic_ui",
    "k_features": "N/A",
    "thresholds": "heuristic",
    "trained_at": "demo mode",
}

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
ITEMS_DIR = PROJECT_DIR / "items"

ITEM_NAME_PRESETS = [
    "Amazon Gift Card - Floral Sleeve",
    "Amazon Gift Card - Birthday Edition",
    "Amazon Gift Card - Thank You Theme",
    "Amazon Gift Card - Elegant Gold Pack",
    "Amazon Gift Card - Celebration Box",
    "Amazon Gift Card - Premium Envelope",
    "Amazon Gift Card - Holiday Special",
    "Amazon Gift Card - Minimalist Pack",
    "Amazon Gift Card - Family Gift Bundle",
]
ITEM_SUBTITLE_PRESETS = [
    "Fast delivery and ready-to-gift packaging.",
    "Top pick for birthdays and quick gifting.",
    "Professional style for business gifting.",
    "Premium print with envelope included.",
    "Popular seasonal design.",
    "Clean design and easy redemption.",
    "Great for teams and family events.",
    "Simple and modern card look.",
    "Great value bundle for repeat gifts.",
]


app = FastAPI(
    title="NLP Review API",
    version="2.0.0",
    description="Lightweight demo backend for the web UI.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ITEMS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/items", StaticFiles(directory=ITEMS_DIR), name="items")


class PredictRequest(BaseModel):
    texts: list[str]
    include_transformer: bool = False


def _issue_flags(*labels: str) -> dict[str, int]:
    active = set(labels)
    return {label: int(label in active) for label in ISSUE_LABELS}


FALLBACK_REVIEWS = [
    {
        "id": "demo_1",
        "rating": 5,
        "text": "Perfect gift card, arrived quickly and easy to redeem.",
        "issue_flags": _issue_flags("delivery_shipping", "redemption_activation"),
    },
    {
        "id": "demo_2",
        "rating": 2,
        "text": "Delivery was late and support did not answer in time.",
        "issue_flags": _issue_flags("delivery_shipping", "customer_service"),
    },
    {
        "id": "demo_3",
        "rating": 3,
        "text": "Code worked but instructions were confusing.",
        "issue_flags": _issue_flags("redemption_activation", "usability"),
    },
    {
        "id": "demo_4",
        "rating": 1,
        "text": "This looks fake, very bad experience and no refund yet.",
        "issue_flags": _issue_flags("fraud_scam", "refund_return"),
    },
    {
        "id": "demo_5",
        "rating": 4,
        "text": "Nice design and fair price for the value.",
        "issue_flags": _issue_flags("value_price"),
    },
]


def _normalize_rating(value: Any) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, parsed))


def _normalized_issue_flags(row: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for label in ISSUE_LABELS:
        raw = row.get(label, 0)
        try:
            normalized[label] = 1 if float(str(raw).strip()) >= 1 else 0
        except (TypeError, ValueError):
            normalized[label] = 0
    return normalized


def _item_meta(index: int) -> dict[str, Any]:
    title = ITEM_NAME_PRESETS[index % len(ITEM_NAME_PRESETS)]
    subtitle = ITEM_SUBTITLE_PRESETS[index % len(ITEM_SUBTITLE_PRESETS)]
    price_vnd = 149000 + index * 25000
    rating = round(min(4.9, 4.3 + (index % 5) * 0.12), 1)
    badge = "Best Seller" if index in {0, 3, 6} else "Prime"
    return {
        "display_name": title,
        "subtitle": subtitle,
        "price_vnd": int(price_vnd),
        "rating": float(rating),
        "badge": badge,
    }


def _candidate_review_paths() -> list[Path]:
    paths: list[Path] = []
    env_csv = os.getenv("NLP_REVIEW_POOL_CSV", "").strip()
    if env_csv:
        paths.append(Path(env_csv))

    paths.extend(
        [
            PROJECT_DIR / "data" / "issue_labels.csv",
            PROJECT_DIR / "data" / "issue_labels_21000_template.csv",
            PROJECT_DIR / "data" / "tung_labeled.csv",
            PROJECT_DIR / "data" / "Tung_labeled.csv",
        ]
    )

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _read_review_csv(path: Path, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding, errors="replace", newline="") as file_handle:
                reader = csv.DictReader(file_handle)
                fieldnames = [str(name).strip().lower() for name in (reader.fieldnames or [])]
                if "text" not in fieldnames or "rating" not in fieldnames:
                    continue

                rows: list[dict[str, Any]] = []
                for row_index, row in enumerate(reader, start=1):
                    normalized_row = {
                        str(key).strip().lower(): value
                        for key, value in row.items()
                        if key is not None
                    }

                    text = str(normalized_row.get("text", "")).strip()
                    if not text:
                        continue

                    review_id = str(normalized_row.get("id") or f"{path.stem}_{row_index}").strip()
                    rows.append(
                        {
                            "id": review_id,
                            "rating": _normalize_rating(normalized_row.get("rating")),
                            "text": text,
                            "issue_flags": _normalized_issue_flags(normalized_row),
                        }
                    )
                    if len(rows) >= limit:
                        break

                if rows:
                    return rows
        except OSError:
            return []

    return []


@lru_cache(maxsize=1)
def load_review_pool() -> dict[str, Any]:
    for path in _candidate_review_paths():
        if not path.exists() or not path.is_file():
            continue
        rows = _read_review_csv(path, limit=5000)
        if rows:
            return {"source": str(path), "rows": rows}

    fallback_rows = [
        {
            "id": row["id"],
            "rating": row["rating"],
            "text": row["text"],
            "issue_flags": dict(row["issue_flags"]),
        }
        for row in FALLBACK_REVIEWS
    ]
    return {"source": "demo_fallback", "rows": fallback_rows}


def _status_payload(include_transformer: bool) -> dict[str, Any]:
    transformer_message = "disabled in demo mode"
    if include_transformer:
        transformer_message = "requested but disabled in demo mode"

    return {
        "classic": {
            "loaded": True,
            "message": "demo heuristic ready",
            "issue_mode": "keyword heuristic",
            "model_info": DEMO_MODEL_INFO,
        },
        "transformer": {
            "requested": include_transformer,
            "loaded": False,
            "message": transformer_message,
        },
    }


def _catalog_items() -> list[dict[str, Any]]:
    files = [
        path
        for path in sorted(ITEMS_DIR.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    items = []
    for index, path in enumerate(files):
        items.append(
            {
                "name": path.name,
                "url": f"/items/{path.name}",
                **_item_meta(index),
            }
        )
    return items


def _infer_issue_flags(text: str) -> dict[str, int]:
    lowered = text.lower()
    flags = {label: 0 for label in ISSUE_LABELS}

    for label, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            flags[label] = 1

    if not any(flags.values()) and any(token in lowered for token in ATTENTION_KEYWORDS):
        flags["other"] = 1
    return flags


def _keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for token in keywords if token in lowered)


def _infer_sentiment(text: str) -> tuple[str, float, str]:
    positive_hits = _keyword_hits(text, POSITIVE_KEYWORDS)
    negative_hits = _keyword_hits(text, NEGATIVE_KEYWORDS)
    attention_hits = _keyword_hits(text, ATTENTION_KEYWORDS)
    score = positive_hits - negative_hits

    if negative_hits >= positive_hits + 1:
        label = "NEGATIVE"
        probability = 0.18
    elif score == -1 or attention_hits >= 2:
        label = "NEEDS_ATTENTION"
        probability = 0.42
    elif score == 0:
        label = "UNCERTAIN"
        probability = 0.50
    else:
        label = "POSITIVE"
        probability = 0.84

    intensity = max(positive_hits, negative_hits)
    if intensity >= 3:
        confidence = "High"
    elif intensity >= 1 or attention_hits >= 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    return label, probability, confidence


def _risk_score(label: str, probability: float, issue_count: int) -> float:
    base_by_label = {
        "NEGATIVE": 410.0,
        "NEEDS_ATTENTION": 305.0,
        "UNCERTAIN": 205.0,
        "POSITIVE": 110.0,
    }
    base = base_by_label.get(label, 100.0)
    label_boost = (1.0 - probability) * 30.0
    return round(base + label_boost + issue_count * 9.0, 1)


def _clean_texts(raw_texts: list[str]) -> list[str]:
    return [text.strip() for text in raw_texts if isinstance(text, str) and text.strip()]


def _build_prediction(text: str) -> dict[str, Any]:
    issue_flags = _infer_issue_flags(text)
    active_issues = [label for label, value in issue_flags.items() if value >= 1]
    label, probability, confidence = _infer_sentiment(text)

    return {
        "text": text,
        "classic_label": label,
        "classic_probability": probability,
        "classic_confidence": confidence,
        "fallback_reason": "demo_heuristic",
        "issue_summary": ", ".join(active_issues) if active_issues else "-",
        "issue_count": len(active_issues),
        "risk_score": _risk_score(label, probability, len(active_issues)),
        "transformer_label": None,
        "transformer_probability": None,
        "transformer_confidence": None,
        "transformer_reason": None,
        "agreement": None,
    }


def _build_summary(predictions: list[dict[str, Any]]) -> dict[str, int]:
    labels = [str(row.get("classic_label", "")) for row in predictions]
    negative = labels.count("NEGATIVE")
    needs_attention = labels.count("NEEDS_ATTENTION")
    uncertain = labels.count("UNCERTAIN")
    positive = labels.count("POSITIVE")
    return {
        "total": len(predictions),
        "flagged": negative + needs_attention,
        "negative": negative,
        "needs_attention": needs_attention,
        "uncertain": uncertain,
        "positive": positive,
    }


def _build_label_distribution(summary: dict[str, int]) -> list[dict[str, Any]]:
    total = max(summary.get("total", 0), 1)
    ordered = [
        ("NEGATIVE", summary.get("negative", 0)),
        ("NEEDS_ATTENTION", summary.get("needs_attention", 0)),
        ("UNCERTAIN", summary.get("uncertain", 0)),
        ("POSITIVE", summary.get("positive", 0)),
    ]
    return [
        {
            "label": label,
            "count": count,
            "share_percent": round((count / total) * 100, 1),
        }
        for label, count in ordered
    ]


def _build_issue_summary(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in predictions:
        issue_summary = str(row.get("issue_summary", "")).strip()
        if not issue_summary or issue_summary == "-":
            continue
        for label in issue_summary.split(","):
            normalized = label.strip()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "label": label,
            "count": count,
            "avg_confidence": 1.0,
        }
        for label, count in sorted_counts
    ]


def _build_attention_queue(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = [
        row
        for row in predictions
        if row.get("classic_label") in {"NEGATIVE", "NEEDS_ATTENTION", "UNCERTAIN"}
    ]
    queue.sort(
        key=lambda row: (-float(row.get("risk_score", 0.0)), float(row.get("classic_probability", 0.5)))
    )
    return [
        {
            "text": row.get("text", ""),
            "classic_label": row.get("classic_label"),
            "classic_probability": row.get("classic_probability"),
            "issue_summary": row.get("issue_summary", "-"),
            "risk_score": row.get("risk_score", 0.0),
        }
        for row in queue[:20]
    ]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status(include_transformer: bool = False) -> dict[str, Any]:
    return _status_payload(include_transformer=include_transformer)


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    return {"items": _catalog_items()}


@app.get("/api/review_pool")
def review_pool(limit: int = 1000) -> dict[str, Any]:
    requested_limit = max(1, min(3000, int(limit)))
    pool = load_review_pool()
    rows = []
    for row in pool.get("rows", [])[:requested_limit]:
        rows.append(
            {
                "id": row.get("id", ""),
                "rating": _normalize_rating(row.get("rating", 3)),
                "text": str(row.get("text", "")),
                "issue_flags": _normalized_issue_flags(dict(row.get("issue_flags", {}))),
            }
        )

    return {
        "source": str(pool.get("source", "")),
        "count": len(rows),
        "reviews": rows,
    }


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    if len(payload.texts) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 input rows per request.")

    texts = _clean_texts(payload.texts)
    if not texts:
        raise HTTPException(status_code=400, detail="No valid input texts found.")

    predictions = [_build_prediction(text) for text in texts]
    summary = _build_summary(predictions)
    return {
        "status": _status_payload(include_transformer=payload.include_transformer),
        "summary": summary,
        "label_distribution": _build_label_distribution(summary),
        "attention_queue": _build_attention_queue(predictions),
        "issue_summary": _build_issue_summary(predictions),
        "mismatch_count": None,
        "predictions": predictions,
    }
