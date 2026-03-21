import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_webapp_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["classic_runtime_ready"], bool)


def test_webapp_root_serves_html():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Persona Toggle" in response.text


def test_webapp_catalog_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert "items" in payload
    assert isinstance(payload["items"], list)


def test_webapp_review_pool_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/review_pool?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert "source" in payload
    assert "count" in payload
    assert "reviews" in payload
    assert isinstance(payload["reviews"], list)
    assert payload["count"] == len(payload["reviews"])
    assert payload["count"] <= 5
    if payload["reviews"]:
        first = payload["reviews"][0]
        assert "issue_flags" in first
        assert isinstance(first["issue_flags"], dict)


def test_webapp_predict_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={
                "texts": [
                    "Terrible support and very late delivery.",
                    "Gift card worked perfectly and arrived fast.",
                ],
                "include_transformer": False,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["classic"]["loaded"] is True
    assert len(payload["predictions"]) == 2
    first = payload["predictions"][0]
    assert "classic_label" in first
    assert "classic_probability" in first
    assert "issue_summary" in first
