import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_root_contains_manual_review_console_inputs_and_quick_prompts():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text

    assert "Manual Review Test" in html
    assert "Late delivery" in html
    assert "Redeemed already" in html
    assert "Mixed complaint" in html
    assert 'id="review-compose-text"' in html
    assert 'id="review-compose-submit"' in html
    assert 'id="review-compose-cancel"' in html
