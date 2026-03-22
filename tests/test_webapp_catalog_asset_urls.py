import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_catalog_items_have_resolvable_static_image_urls():
    with TestClient(app) as client:
        catalog = client.get("/api/catalog").json()

        for item in catalog["items"][:3]:
            response = client.get(item["url"])
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("image/")
