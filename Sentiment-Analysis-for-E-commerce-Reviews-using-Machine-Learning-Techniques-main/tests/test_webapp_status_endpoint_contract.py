import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_status_endpoint_exposes_classic_runtime_model_info():
    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()

    assert payload["classic"]["loaded"] is True
    assert payload["classic"]["issue_mode"] == "trained classifier"
    assert payload["classic"]["message"] == "Trained scikit-learn runtime online"
    assert payload["classic"]["model_info"]["variant"] == "V6"
    assert payload["classic"]["model_info"]["thresholds"] == "0.40/0.60"
    assert payload["classic"]["model_info"]["k_features"] == 10000
    assert payload["classic"]["model_info"]["trained_at"] == "2026-03-18 21:51"
