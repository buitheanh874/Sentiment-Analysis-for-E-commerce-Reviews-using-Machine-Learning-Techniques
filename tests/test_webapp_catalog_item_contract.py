import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_catalog_items_expose_expected_display_fields():
    with TestClient(app) as client:
        response = client.get("/api/catalog")

    assert response.status_code == 200
    items = response.json()["items"]

    assert items
    first = items[0]
    assert isinstance(first["display_name"], str) and first["display_name"]
    assert isinstance(first["subtitle"], str) and first["subtitle"]
    assert isinstance(first["badge"], str) and first["badge"]
    assert isinstance(first["price_vnd"], int)
    assert first["price_vnd"] > 0
    assert 0.0 <= float(first["rating"]) <= 5.0
    assert first["url"].startswith("/items/")
