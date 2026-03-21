import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    cmd = [sys.executable] + args
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def test_run_all_rejects_out_of_range_threshold():
    proc = _run(["-m", "src.run_all", "--threshold_low", "-0.1"])
    assert proc.returncode != 0
    err = proc.stderr.lower()
    assert "threshold" in err
    assert "between 0 and 1" in err


def test_run_all_rejects_low_above_high():
    proc = _run(
        [
            "-m",
            "src.run_all",
            "--threshold_low",
            "0.8",
            "--threshold_high",
            "0.6",
        ]
    )
    assert proc.returncode != 0
    err = proc.stderr.lower()
    assert "threshold_low" in err
    assert "<=" in err
