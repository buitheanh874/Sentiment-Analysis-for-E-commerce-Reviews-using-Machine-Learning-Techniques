from types import SimpleNamespace

import pandas as pd

from src.issue_steps.steps import _template_id_sort_key, cmd_make_template


def test_template_id_sort_key_orders_numeric_before_text():
    ids = ["10", "2", "A-2", "1", "b-1"]
    assert sorted(ids, key=_template_id_sort_key) == ["1", "2", "10", "A-2", "b-1"]


def test_make_template_uses_natural_id_order(tmp_path):
    data_path = tmp_path / "sample.jsonl"
    out_path = tmp_path / "issue_labels_template.csv"

    pd.DataFrame(
        [
            {"id": 10, "text": "late delivery", "rating": 1},
            {"id": 2, "text": "bad support", "rating": 2},
            {"id": 1, "text": "good value", "rating": 5},
            {"id": "A3", "text": "code failed", "rating": 1},
        ]
    ).to_json(data_path, orient="records", lines=True)

    args = SimpleNamespace(
        data_path=data_path,
        out=out_path,
        sample_size=None,
        only_queue=False,
        seed=42,
        init_zero=True,
        queue_strategy="priority",
    )
    cmd_make_template(args)

    df = pd.read_csv(out_path)
    assert df["id"].tolist() == ["1", "2", "10", "A3"]


def test_make_template_queue_priority_sampling(monkeypatch, tmp_path):
    data_path = tmp_path / "sample_queue.jsonl"
    out_path = tmp_path / "issue_labels_template.csv"

    pd.DataFrame(
        [
            {"id": 1, "text": "ok", "rating": 1},
            {"id": 2, "text": "bad but late delivery", "rating": 3},
            {"id": 3, "text": "refund issue", "rating": 2},
            {"id": 4, "text": "support not helpful however solved late", "rating": 3},
        ]
    ).to_json(data_path, orient="records", lines=True)

    def _fake_stage1(df, base_dir):
        frame = pd.DataFrame(
            {
                "stage1_label": ["NEGATIVE", "NEEDS_ATTENTION", "NEEDS_ATTENTION", "NEEDS_ATTENTION"],
                "stage1_prob": [0.20, 0.49, 0.41, 0.45],
            },
            index=df.index,
        )
        return frame, "fake_stage1"

    monkeypatch.setattr("src.issue_steps.steps._infer_queue_stage1_labels", _fake_stage1)

    args = SimpleNamespace(
        data_path=data_path,
        out=out_path,
        sample_size=2,
        only_queue=True,
        seed=42,
        init_zero=True,
        queue_strategy="priority",
    )
    cmd_make_template(args)

    df = pd.read_csv(out_path)
    assert set(df["id"].astype(str)) == {"2", "4"}
    assert "queue_priority" in df.columns
