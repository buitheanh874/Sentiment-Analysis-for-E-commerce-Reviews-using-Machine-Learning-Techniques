# Demo Runbook

## Preconditions
- Activate venv.
- Install dependencies from `requirements.txt` and `requirements-optional.txt`.

## Demo Command (Tests Only)
```bash
python -m pytest D:\Code\NLP_code\tests\test_smoke_cli.py -q
```

## Backup Command (Focused Smoke)
```bash
python -m pytest tests/test_smoke_cli.py -q
```

## Expected Direction
- The test run should collect and execute test cases from `tests/`.
- Smoke tests should pass and end with exit code `0`.
