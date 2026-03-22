from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_metadata import begin_run, end_run

RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "scoreboard"


def _exists(path: Path) -> bool:
    return path.exists()


def _score_ratio(checks: List[Tuple[str, bool]]) -> float:
    if not checks:
        return 0.0
    return float(sum(1 for _, ok in checks if ok) / len(checks))


def _weighted_score(checks: List[Tuple[str, bool, float]], cap: float = 1.0) -> float:
    score = 0.0
    for _name, ok, weight in checks:
        if ok:
            score += float(weight)
    return min(float(cap), float(score))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _shortlog_author_count() -> int:
    import subprocess

    proc = subprocess.run(
        ["git", "shortlog", "-sn", "--all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return len(lines)


def build_assessment() -> Dict[str, object]:
    scoreboard = _safe_read_csv(RESULTS_DIR / "scoreboard" / "model_scoreboard.csv")
    course_fit = _safe_read_csv(RESULTS_DIR / "nlp_ext" / "syllabus_upgrade" / "nlp_course_fit_matrix.csv")
    author_count = _shortlog_author_count()

    content_checks = [
        ("readme", _exists(ROOT / "README.md"), 0.16),
        ("run_all_pipeline", _exists(ROOT / "src" / "run_all.py"), 0.16),
        ("demo_tests", _exists(ROOT / "tests" / "test_smoke_cli.py"), 0.14),
        ("scoreboard", _exists(RESULTS_DIR / "scoreboard" / "model_scoreboard.csv"), 0.16),
        ("report_pdf", _exists(RESULTS_DIR / "reports" / "NLP_project_report_20260306.pdf"), 0.12),
        ("smoke_tests", _exists(ROOT / "tests" / "test_smoke_cli.py"), 0.12),
        ("slide_deck_file", bool(list((ROOT / "docs").glob("*.pptx"))), 0.14),
    ]
    significance_checks = [
        ("problem_framing", _exists(ROOT / "README.md"), 0.25),
        ("uncertainty_handling", _exists(RESULTS_DIR / "dm2_steps" / "09_uncertainty_summary.md"), 0.20),
        ("hard_case_analysis", _exists(RESULTS_DIR / "nlp_ext" / "hard_cases_comparison.csv"), 0.20),
        ("issue_task_metrics", _exists(RESULTS_DIR / "issue_steps" / "02_metrics_overall.csv"), 0.15),
        ("transformer_metrics", _exists(RESULTS_DIR / "nlp_ext" / "nlp_metrics.csv"), 0.10),
        ("report_pdf", _exists(RESULTS_DIR / "reports" / "NLP_project_report_20260306.pdf"), 0.10),
    ]
    process_checks = [
        ("contribution_matrix", _exists(ROOT / "docs" / "contribution_matrix.md"), 0.20),
        ("pr_template", _exists(ROOT / ".github" / "pull_request_template.md"), 0.15),
        ("tests", _exists(ROOT / "tests" / "test_smoke_cli.py"), 0.15),
        ("run_metadata", _exists(RESULTS_DIR / "dm2_steps" / "_run_metadata"), 0.10),
        ("multi_author_git", author_count >= 4, 0.15),
        ("review_log_artifact", _exists(ROOT / "docs" / "review_log.md"), 0.25),
    ]
    results_checks = [
        ("sentiment_results", _exists(RESULTS_DIR / "dm2_steps" / "08_ensemble_metrics.csv"), 0.20),
        ("issue_results", _exists(RESULTS_DIR / "issue_steps" / "02_metrics_overall.csv"), 0.20),
        ("transformer_results", _exists(RESULTS_DIR / "nlp_ext" / "nlp_metrics.csv"), 0.15),
        ("model_diversity", (not scoreboard.empty and int(scoreboard["task"].nunique()) >= 6), 0.15),
        ("eval_rigor_ci", _exists(RESULTS_DIR / "nlp_ext" / "syllabus_upgrade" / "nlp_eval_ci_bootstrap.csv"), 0.15),
        ("eval_rigor_significance", _exists(RESULTS_DIR / "nlp_ext" / "syllabus_upgrade" / "nlp_eval_significance.csv"), 0.15),
    ]
    demo_checks = [
        ("demo_test_entry", _exists(ROOT / "tests" / "test_smoke_cli.py"), 0.45),
        ("demo_inputs", _exists(ROOT / "docs" / "demo_inputs.txt"), 0.20),
        ("expected_outputs", _exists(ROOT / "docs" / "expected_outputs.md"), 0.15),
        ("demo_runbook", _exists(ROOT / "docs" / "demo_runbook.md"), 0.10),
        ("demo_recording", _exists(ROOT / "docs" / "demo_recording.mp4"), 0.10),
    ]

                                                       
    ratios = {
        "Content representation": _weighted_score(content_checks, cap=0.95),
        "Project significance": _weighted_score(significance_checks, cap=1.00),
        "Working process": _weighted_score(process_checks, cap=0.90),
        "Results quality": _weighted_score(results_checks, cap=1.00),
        "Demo": _weighted_score(demo_checks, cap=0.90),
    }
    rubric_percent = {k: round(v * 100.0, 1) for k, v in ratios.items()}
    rubric_points = {k: round((v * 4.0), 2) for k, v in ratios.items()}

    overall_percent = round(sum(rubric_percent.values()) / max(1, len(rubric_percent)), 1)
    overall_points = round(sum(rubric_points.values()), 2)

    syllabus_overall = float(course_fit["coverage_percent"].mean()) if not course_fit.empty else 0.0

    return {
        "rubric_percent": rubric_percent,
        "rubric_points_out_of_4": rubric_points,
        "rubric_total_points_out_of_20": overall_points,
        "rubric_overall_percent": overall_percent,
        "syllabus_coverage_percent": round(syllabus_overall, 1),
        "checks": {
            "content": content_checks,
            "significance": significance_checks,
            "process": process_checks,
            "results": results_checks,
            "demo": demo_checks,
        },
        "git_author_count": int(author_count),
        "scoreboard_rows": int(len(scoreboard)),
        "course_fit_rows": int(len(course_fit)),
    }


def _to_markdown(payload: Dict[str, object]) -> str:
    rubric_percent: Dict[str, float] = payload["rubric_percent"]                            
    rubric_points: Dict[str, float] = payload["rubric_points_out_of_4"]                            
    checks = payload["checks"]                            

    lines: List[str] = [
        "# Rubric & Syllabus Assessment",
        "",
        "Estimated using current repository artifacts only (code + results + docs).",
        "",
        "## Rubric Estimate",
        "",
        "| Criterion | Estimated % | Points / 4 |",
        "|---|---:|---:|",
    ]
    for key in ["Content representation", "Project significance", "Working process", "Results quality", "Demo"]:
        lines.append(f"| {key} | {rubric_percent.get(key, 0.0):.1f}% | {rubric_points.get(key, 0.0):.2f} |")

    lines.extend(
        [
            "",
            f"- Total estimated rubric score: **{payload['rubric_total_points_out_of_20']:.2f}/20**",
            f"- Overall rubric percent: **{payload['rubric_overall_percent']:.1f}%**",
            f"- Git author count observed: **{payload.get('git_author_count', 0)}**",
            "",
            "## Syllabus Fit",
            "",
            f"- Coverage from `nlp_course_fit_matrix.csv`: **{payload['syllabus_coverage_percent']:.1f}%**",
            "",
            "## Evidence Checks",
            "",
        ]
    )

    for section in ["content", "significance", "process", "results", "demo"]:
        lines.append(f"### {section.capitalize()}")
        for name, ok, weight in checks[section]:
            lines.append(f"- {'[x]' if ok else '[ ]'} {name} (weight={weight:.2f})")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = begin_run(
        command_name="scripts.build_rubric_syllabus_assessment",
        args={"output_dir": OUT_DIR},
        metadata_dir=OUT_DIR / "_run_metadata",
    )
    try:
        payload = build_assessment()
        md_path = OUT_DIR / "rubric_syllabus_assessment.md"
        json_path = OUT_DIR / "rubric_syllabus_assessment.json"
        md_path.write_text(_to_markdown(payload), encoding="utf-8")
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote: {md_path}")
        print(f"Wrote: {json_path}")
        end_run(record, status="success")
    except Exception as exc:
        end_run(record, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
