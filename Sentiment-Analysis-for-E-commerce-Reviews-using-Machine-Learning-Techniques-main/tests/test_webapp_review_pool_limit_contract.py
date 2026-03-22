import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_review_pool_limit_is_clamped_to_at_least_one_row():
    with TestClient(app) as client:
        response = client.get("/api/review_pool?limit=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1


def test_review_pool_limit_is_clamped_to_upper_bound():
    with TestClient(app) as client:
        response = client.get("/api/review_pool?limit=99999")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] <= 3000
