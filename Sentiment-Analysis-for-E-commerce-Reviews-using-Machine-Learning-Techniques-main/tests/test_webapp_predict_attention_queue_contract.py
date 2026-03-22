import pytest


pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.main import app


def test_predict_attention_queue_is_sorted_by_risk_score_descending():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={
                "texts": [
                    "Terrible support and very late delivery.",
                    "Gift card worked perfectly and arrived fast.",
                    "This looks fake and I still have not received a refund.",
                ],
                "include_transformer": False,
            },
        )

    assert response.status_code == 200
    queue = response.json()["attention_queue"]

    assert queue
    scores = [row["risk_score"] for row in queue]
    assert scores == sorted(scores, reverse=True)
    for row in queue:
        assert isinstance(row["text"], str) and row["text"]
        assert isinstance(row["classic_label"], str) and row["classic_label"]
        assert isinstance(row["issue_summary"], str)
