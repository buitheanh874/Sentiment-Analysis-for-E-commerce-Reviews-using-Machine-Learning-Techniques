import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        return value or None
    except Exception:
        return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _args_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if hasattr(args, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(args).items()}
    return {"value": _to_jsonable(args)}


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text.lower())


@dataclass
class RunRecord:
    path: Path
    payload: Dict[str, Any]
    start_monotonic: float


def begin_run(
    command_name: str,
    args: Any,
    metadata_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> RunRecord:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_safe_name(command_name)}_{uuid.uuid4().hex[:8]}"
    path = metadata_dir / f"{run_id}.json"
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "command_name": command_name,
        "argv": [_to_jsonable(arg) for arg in sys.argv],
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git_commit": _git_commit(),
        "started_at_utc": _utc_now_iso(),
        "args": _args_to_dict(args),
    }
    if extra:
        payload["extra"] = _to_jsonable(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RunRecord(path=path, payload=payload, start_monotonic=time.monotonic())


def end_run(
    record: RunRecord,
    status: str,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = dict(record.payload)
    payload["status"] = status
    payload["finished_at_utc"] = _utc_now_iso()
    payload["duration_seconds"] = round(time.monotonic() - record.start_monotonic, 3)
    if error is not None:
        payload["error"] = error
    if extra:
        merged = dict(payload.get("extra", {}))
        merged.update(_to_jsonable(extra))
        payload["extra"] = merged
    record.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

