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
    )
    cmd_make_template(args)

    df = pd.read_csv(out_path)
    assert df["id"].tolist() == ["1", "2", "10", "A3"]
