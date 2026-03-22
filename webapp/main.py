"""FastAPI backend for the NLPSHOP web UI."""

from __future__ import annotations

import csv
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.predictor import analyze_reviews, load_classic_runtime, model_status

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


def _runtime_error_payload(include_transformer: bool, message: str) -> dict[str, Any]:
    return {
        "classic": {
            "loaded": False,
            "message": message,
            "issue_mode": "unavailable",
            "model_info": {},
        },
        "transformer": {
            "requested": include_transformer,
            "loaded": False,
            "message": "unavailable",
        },
    }


def _ensure_classic_runtime(request: Request) -> None:
    if getattr(request.app.state, "classic_runtime_ready", False):
        return

    try:
        load_classic_runtime()
    except RuntimeError as exc:
        request.app.state.classic_runtime_ready = False
        request.app.state.classic_runtime_error = str(exc)
        raise

    request.app.state.classic_runtime_ready = True
    request.app.state.classic_runtime_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classic_runtime_ready = False
    app.state.classic_runtime_error = None

    try:
        load_classic_runtime()
    except RuntimeError as exc:
        app.state.classic_runtime_ready = False
        app.state.classic_runtime_error = str(exc)
    else:
        app.state.classic_runtime_ready = True
        app.state.classic_runtime_error = None

    yield


app = FastAPI(
    title="NLP Review API",
    version="3.0.0",
    description="FastAPI backend for the NLPSHOP review understanding UI.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ITEMS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/items", StaticFiles(directory=ITEMS_DIR), name="items")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "classic_runtime_ready": bool(getattr(request.app.state, "classic_runtime_ready", False)),
    }


@app.get("/api/status")
def status(request: Request, include_transformer: bool = False) -> dict[str, Any]:
    try:
        _ensure_classic_runtime(request)
    except RuntimeError:
        message = getattr(request.app.state, "classic_runtime_error", None) or "Classic model is unavailable."
        return _runtime_error_payload(include_transformer=include_transformer, message=message)

    try:
        return model_status(include_transformer=include_transformer)
    except RuntimeError as exc:
        return _runtime_error_payload(include_transformer=include_transformer, message=str(exc))


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
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    if len(payload.texts) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 input rows per request.")

    try:
        _ensure_classic_runtime(request)
    except RuntimeError:
        message = getattr(request.app.state, "classic_runtime_error", None) or "Classic model is unavailable."
        raise HTTPException(status_code=503, detail=message)

    try:
        return analyze_reviews(payload.texts, include_transformer=payload.include_transformer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
