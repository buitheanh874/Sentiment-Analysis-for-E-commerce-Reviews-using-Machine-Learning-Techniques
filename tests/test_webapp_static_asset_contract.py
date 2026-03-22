import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_static_asset_versions_match_in_index():
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")

    css_match = re.search(r'/static/styles\.css\?v=([^"]+)', html)
    js_match = re.search(r'/static/app\.js\?v=([^"]+)', html)

    assert css_match
    assert js_match
    assert css_match.group(1) == js_match.group(1)


def test_left_rail_and_focus_styles_exist():
    css = (ROOT / "webapp" / "static" / "styles.css").read_text(encoding="utf-8")
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")

    assert ".catalog-rail .catalog-context-image" in css
    assert "height: 496px;" in css
    assert ".triage-focus-content" in css
    assert "grid-template-columns: 440px minmax(0, 1fr);" in html
