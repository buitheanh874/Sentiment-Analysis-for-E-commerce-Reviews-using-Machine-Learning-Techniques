import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_predict_response_exposes_summary_and_label_distribution():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={
                "texts": [
                    "Terrible support and very late delivery.",
                    "Gift card worked perfectly and arrived fast.",
                    "Okay product but support was slow.",
                ],
                "include_transformer": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    summary = payload["summary"]
    distribution = payload["label_distribution"]

    assert summary["total"] == len(payload["predictions"])
    assert summary["flagged"] <= summary["total"]
    assert len(distribution) == 4
    assert sum(item["count"] for item in distribution) == summary["total"]
    assert {item["label"] for item in distribution} == {
        "NEGATIVE",
        "NEEDS_ATTENTION",
        "UNCERTAIN",
        "POSITIVE",
    }
