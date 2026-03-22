import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_admin_dashboard_root_contains_current_sections():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text

    assert "Customer Experience Operations Dashboard" in html
    assert "Batch Snapshot" in html
    assert "Triage Focus" in html
    assert "Selected Review Details" in html
    assert "Manual Review Test" in html


def test_admin_dashboard_root_contains_left_rail_hooks():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text

    assert 'class="catalog-rail"' in html
    assert 'id="catalog-context-card"' in html
    assert 'id="rail-snapshot-grid"' in html
    assert 'id="triage-focus-panel"' in html
