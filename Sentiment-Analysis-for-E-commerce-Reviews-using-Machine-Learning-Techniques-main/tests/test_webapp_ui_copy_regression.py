from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_index_html_omits_old_user_facing_copy():
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")

    assert "Issue Signals" not in html
    assert "Owner Insight" not in html
    assert "Manual NLP Test Tool" not in html
    assert "Open Test Console" not in html
    assert "NLP vs star rating mismatch for the active batch" not in html
    assert "Prediction details, workflow state, and routing context" not in html


def test_app_templates_hide_debug_style_terms_from_queue_and_detail_views():
    script = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")

    assert "Probability</th>" not in script
    assert "Queue tag:" not in script
    assert "Confidence:" not in script
    assert "No issue labels" not in script

    assert "Needs Review" in script
    assert "No issue categories" in script
    assert "Cannot load service status" in script
