import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


ISSUE_KEYS = {
    "customer_service",
    "delivery_shipping",
    "fraud_scam",
    "other",
    "product_quality",
    "redemption_activation",
    "refund_return",
    "usability",
    "value_price",
}


def test_review_pool_rows_are_normalized_for_frontend_consumption():
    with TestClient(app) as client:
        response = client.get("/api/review_pool?limit=3")

    assert response.status_code == 200
    rows = response.json()["reviews"]

    assert rows
    for row in rows:
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["text"], str) and row["text"]
        assert 1 <= row["rating"] <= 5
        assert set(row["issue_flags"]) == ISSUE_KEYS
        assert set(row["issue_flags"].values()) <= {0, 1}
