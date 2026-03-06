import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "scoreboard" / "model_scoreboard.csv"


def test_scoreboard_builder_outputs_files():
    proc = subprocess.run(
        [sys.executable, "scripts/build_scoreboard.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert OUT_CSV.exists()

    df = pd.read_csv(OUT_CSV)
    assert len(df) > 0
    assert "task" in df.columns
    assert "model" in df.columns
    assert "source_file" in df.columns
    tasks = set(df["task"].astype(str))
    assert "sentiment" in tasks
    assert "issue_multilabel" in tasks
