import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    cmd = [sys.executable] + args
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def test_demo_cli_runs():
    proc = _run(["demo.py", "great product and fast shipping"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.lower()
    assert (
        "sentiment" in out
        or "positive" in out
        or "negative" in out
        or "uncertain" in out
    )


def test_issue_predict_cli_runs():
    proc = _run(["-m", "src.issue_steps", "predict", "--text", "good but late delivery"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.lower()
    assert "labels" in out or "prediction" in out
    metadata_dir = ROOT / "results" / "issue_steps" / "_run_metadata"
    assert metadata_dir.exists()
    assert any(metadata_dir.glob("*.json"))


def test_pipeline_help():
    proc = _run(["-m", "src.run_all", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "Run all NLP sentiment lab steps sequentially.".lower() in proc.stdout.lower()
