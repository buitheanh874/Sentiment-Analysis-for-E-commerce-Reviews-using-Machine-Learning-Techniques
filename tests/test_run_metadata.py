import argparse
import json
from pathlib import Path

from src.run_metadata import begin_run, end_run


def test_run_metadata_lifecycle(tmp_path):
    args = argparse.Namespace(data_path=Path("data/Gift_Cards.jsonl"), sample_size=100, flag=True)
    metadata_dir = tmp_path / "_run_metadata"
    record = begin_run(
        command_name="tests.dummy_command",
        args=args,
        metadata_dir=metadata_dir,
    )
    assert record.path.exists()
    end_run(record, status="success", extra={"note": "ok"})

    payload = json.loads(record.path.read_text(encoding="utf-8"))
    assert payload["command_name"] == "tests.dummy_command"
    assert payload["status"] == "success"
    assert payload["args"]["sample_size"] == 100
    assert payload["duration_seconds"] >= 0
    assert payload["extra"]["note"] == "ok"
