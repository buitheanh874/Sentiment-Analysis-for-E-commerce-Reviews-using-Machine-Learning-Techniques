from types import SimpleNamespace

import pandas as pd
import pytest

from src.issue_steps.common import ISSUE_LABELS
from src.issue_steps.steps import cmd_validate


def _row(id_value, text, rating, active_labels):
    row = {"id": id_value, "text": text, "rating": rating}
    for label in ISSUE_LABELS:
        row[label] = 1 if label in active_labels else 0
    return row


def test_validate_fails_on_duplicate_label_conflicts(tmp_path):
    labels_path = tmp_path / "labels.csv"
    out_dir = tmp_path / "out"
    df = pd.DataFrame(
        [
            _row("1", "late delivery", 1, {"delivery_shipping"}),
            _row("1", "scam report", 1, {"other"}),
            _row("2", "bad support", 2, {"customer_service"}),
        ]
    )
    df.to_csv(labels_path, index=False, encoding="utf-8")

    args = SimpleNamespace(
        labels_path=labels_path,
        output_dir=out_dir,
        strict_other=False,
        fail_on_duplicate_conflicts=True,
    )
    with pytest.raises(SystemExit, match="duplicate ids with conflicting labels"):
        cmd_validate(args)

    assert (out_dir / "01_duplicate_id_conflicts.csv").exists()


def test_validate_fails_when_strict_other_enabled(tmp_path):
    labels_path = tmp_path / "labels.csv"
    out_dir = tmp_path / "out"
    df = pd.DataFrame(
        [
            _row("1", "mixed issue", 1, {"other", "delivery_shipping"}),
            _row("2", "bad support", 2, {"customer_service"}),
        ]
    )
    df.to_csv(labels_path, index=False, encoding="utf-8")

    args = SimpleNamespace(
        labels_path=labels_path,
        output_dir=out_dir,
        strict_other=True,
        fail_on_duplicate_conflicts=False,
    )
    with pytest.raises(SystemExit, match="other=1"):
        cmd_validate(args)
