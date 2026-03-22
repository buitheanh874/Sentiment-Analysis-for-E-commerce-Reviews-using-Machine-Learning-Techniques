import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_root_contains_admin_navigation_and_filter_controls():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text

    assert 'data-app-view="admin"' in html
    assert 'id="date-range-select"' in html
    assert 'id="mock-product-filter"' in html
    assert 'href="#issue-mix"' in html
    assert 'href="#summary-panel"' in html
    assert 'href="#attention-queue"' in html
    assert 'href="#manual-test"' in html
