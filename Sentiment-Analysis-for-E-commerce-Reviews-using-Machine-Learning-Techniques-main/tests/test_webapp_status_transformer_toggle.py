import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_status_endpoint_marks_transformer_request_state():
    with TestClient(app) as client:
        response = client.get("/api/status?include_transformer=true")

    assert response.status_code == 200
    payload = response.json()

    assert payload["transformer"]["requested"] is True
    assert isinstance(payload["transformer"]["loaded"], bool)
    assert isinstance(payload["transformer"]["message"], str)
    assert payload["classic"]["loaded"] is True
