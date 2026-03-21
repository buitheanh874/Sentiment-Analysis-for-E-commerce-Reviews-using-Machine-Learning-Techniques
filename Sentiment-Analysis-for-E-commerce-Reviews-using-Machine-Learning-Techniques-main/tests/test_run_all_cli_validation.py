import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    cmd = [sys.executable] + args
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def test_run_all_rejects_non_numeric_until_step():
    proc = _run(["-m", "src.run_all", "--until_step", "x"])
    assert proc.returncode != 0
    err = proc.stderr.lower()
    assert "until_step" in err
    assert "integer between 1 and 10" in err


def test_run_all_rejects_out_of_range_until_step():
    proc = _run(["-m", "src.run_all", "--until_step", "11"])
    assert proc.returncode != 0
    err = proc.stderr.lower()
    assert "until_step" in err
    assert "integer between 1 and 10" in err
