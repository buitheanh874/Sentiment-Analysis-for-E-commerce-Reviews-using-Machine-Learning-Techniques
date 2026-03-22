import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_predict_rejects_more_than_five_hundred_inputs():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={"texts": ["x"] * 501, "include_transformer": False},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Maximum 500 input rows per request."


def test_predict_rejects_blank_or_invalid_text_payloads():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={"texts": ["", "   "], "include_transformer": False},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "No valid input texts found."


def test_predict_rejects_non_string_rows_at_request_schema_layer():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={"texts": ["ok", None], "include_transformer": False},
        )

    assert response.status_code == 422
