"""
Streamlit UI for NLP review understanding demo.

Run:
    streamlit run demo_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from demo import (
    load_issue_model,
    load_models,
    predict_sentiment as predict_classic_sentiment,
)


APP_TITLE = "NLP Review Demo"
DEFAULT_INPUT = "\n".join(
    [
        "great product and fast shipping",
        "terrible experience, support never replied",
        "not bad overall",
        "good but late delivery",
    ]
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f5f8fc 0%, #eef3fa 100%);
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            .hero-card {
                background: linear-gradient(135deg, #113b7a 0%, #1b5fa7 100%);
                border-radius: 14px;
                padding: 18px 20px;
                color: #ffffff;
                margin-bottom: 16px;
            }
            .hero-title {
                font-size: 1.55rem;
                font-weight: 700;
                margin-bottom: 4px;
            }
            .hero-subtitle {
                opacity: 0.92;
                line-height: 1.45;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
                margin-top: 12px;
            }
            .status-card {
                background: rgba(255, 255, 255, 0.13);
                border: 1px solid rgba(255, 255, 255, 0.24);
                border-radius: 12px;
                padding: 12px 14px;
            }
            .status-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                opacity: 0.85;
            }
            .status-value {
                font-size: 1.05rem;
                font-weight: 650;
                margin-top: 3px;
                word-break: break-word;
            }
            .panel-card {
                background: #ffffff;
                border: 1px solid #d6dfec;
                border-radius: 12px;
                padding: 14px 16px;
            }
            .panel-title {
                margin: 0 0 6px 0;
                font-size: 1.05rem;
                font-weight: 650;
                color: #113b7a;
            }
            .panel-note {
                margin: 0;
                color: #4c5a70;
                font-size: 0.92rem;
                line-height: 1.4;
            }
            @media (max-width: 900px) {
                .status-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_classic_bundle(base_dir: Path):
    vectorizer, selector, model, meta, model_info = load_models(base_dir, verbose=False)
    issue_bundle = load_issue_model(base_dir, verbose=False)
    return vectorizer, selector, model, meta, model_info, issue_bundle


@st.cache_resource(show_spinner=False)
def load_transformer_bundle(base_dir: Path):
    from demo_transformer import load_transformer_model

    tokenizer, model = load_transformer_model(base_dir, verbose=False)
    return tokenizer, model


def format_prob(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    return f"{value:.3f}"


def summarize_issue_labels(result: Dict[str, Any]) -> str:
    issue_rows = result.get("issue_labels", [])
    if issue_rows:
        return ", ".join(f"{row['label']}:{row['confidence']:.2f}" for row in issue_rows)
    fallback = result.get("issue_tags", [])
    if fallback:
        return ", ".join(fallback)
    return "-"


def build_classic_row(text: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": text,
        "classic_label": result.get("label", "N/A"),
        "classic_prob_pos": format_prob(result.get("probability")),
        "classic_confidence": result.get("confidence", "N/A"),
        "classic_reason": result.get("fallback_reason") or "-",
        "issue_tags_or_labels": summarize_issue_labels(result),
    }


def build_transformer_row(text: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": text,
        "transformer_label": result.get("label", "N/A"),
        "transformer_prob_pos": format_prob(result.get("probability")),
        "transformer_confidence": result.get("confidence", "N/A"),
        "transformer_reason": result.get("fallback_reason") or "-",
    }


def parse_inputs(raw_text: str) -> List[str]:
    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]


def _status_card(label: str, value: str) -> str:
    return (
        '<div class="status-card">'
        f'<div class="status-label">{label}</div>'
        f'<div class="status-value">{value}</div>'
        "</div>"
    )


def collect_issue_summary(classic_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in classic_raw:
        issue_rows = item["classic"].get("issue_labels", [])
        for issue in issue_rows:
            rows.append(
                {
                    "label": str(issue.get("label", "")),
                    "confidence": float(issue.get("confidence", 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["label", "count", "avg_confidence"])
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("label", as_index=False)
        .agg(
            count=("label", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .sort_values(["count", "avg_confidence", "label"], ascending=[False, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary["avg_confidence"] = summary["avg_confidence"].round(3)
    return summary


def build_overview_metrics(classic_df: pd.DataFrame) -> Dict[str, int]:
    labels = classic_df["classic_label"].astype(str)
    return {
        "total": int(len(classic_df)),
        "negative": int((labels == "NEGATIVE").sum()),
        "positive": int((labels == "POSITIVE").sum()),
        "uncertain": int((labels == "UNCERTAIN").sum()),
        "needs_attention": int((labels == "NEEDS_ATTENTION").sum()),
    }


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_styles()

    base_dir = Path(__file__).resolve().parent

    with st.sidebar:
        st.header("Options")
        enable_transformer = st.checkbox("Compare with transformer", value=False)
        show_raw_json = st.checkbox("Show raw JSON output", value=False)
        st.markdown("Model files are loaded from `models/`.")

    try:
        classic_bundle = load_classic_bundle(base_dir)
    except SystemExit:
        st.error("Classic model artifacts are missing. Train or restore `models/` first.")
        return

    vectorizer, selector, model, meta, model_info, issue_bundle = classic_bundle

    tokenizer = None
    transformer_model = None
    transformer_status = "disabled"

    if enable_transformer:
        try:
            tokenizer, transformer_model = load_transformer_bundle(base_dir)
            transformer_status = "ready"
        except Exception as exc:  # noqa: BLE001
            transformer_status = f"unavailable: {exc}"

    header_html = (
        '<div class="hero-card">'
        f'<div class="hero-title">{APP_TITLE}</div>'
        '<div class="hero-subtitle">Analyze review lines with classic sentiment + issue extraction, '
        "and optionally compare against transformer predictions.</div>"
        '<div class="status-grid">'
        + _status_card("Classic K*", str(model_info.get("k_features", "N/A")))
        + _status_card("Thresholds", str(model_info.get("thresholds", "N/A")))
        + _status_card("Transformer", transformer_status)
        + "</div></div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    if "demo_input_text" not in st.session_state:
        st.session_state["demo_input_text"] = DEFAULT_INPUT

    st.markdown(
        '<div class="panel-card"><p class="panel-title">Input Reviews</p>'
        '<p class="panel-note">Enter one review per line. Use mixed and hard cases for better comparison quality.</p></div>',
        unsafe_allow_html=True,
    )
    raw_text = st.text_area(
        "One review per line",
        key="demo_input_text",
        height=180,
    )

    col_run, col_sample, col_note = st.columns([1, 1, 2])
    run_clicked = col_run.button("Analyze", type="primary", use_container_width=True)
    if col_sample.button("Load sample", use_container_width=True):
        st.session_state["demo_input_text"] = DEFAULT_INPUT
        st.rerun()
    col_note.write("Tip: use line breaks to compare multiple reviews in one run.")

    if not run_clicked:
        st.info("Enter review lines and click Analyze.")
        return

    inputs = parse_inputs(raw_text)
    if not inputs:
        st.warning("No valid input lines found.")
        return

    classic_rows: List[Dict[str, Any]] = []
    classic_raw: List[Dict[str, Any]] = []

    for text in inputs:
        result = predict_classic_sentiment(
            text,
            vectorizer,
            selector,
            model,
            meta,
            issue_bundle=issue_bundle,
        )
        classic_rows.append(build_classic_row(text, result))
        classic_raw.append({"text": text, "classic": result})

    classic_df = pd.DataFrame(classic_rows)
    metrics = build_overview_metrics(classic_df)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Inputs", metrics["total"])
    metric_cols[1].metric("Negative", metrics["negative"])
    metric_cols[2].metric("Needs attention", metrics["needs_attention"])
    metric_cols[3].metric("Uncertain", metrics["uncertain"])
    metric_cols[4].metric("Positive", metrics["positive"])

    classic_view = classic_df.rename(
        columns={
            "text": "Review text",
            "classic_label": "Classic label",
            "classic_prob_pos": "P(positive)",
            "classic_confidence": "Confidence",
            "classic_reason": "Fallback reason",
            "issue_tags_or_labels": "Issue tags/labels",
        }
    )
    tabs = st.tabs(["Classic Results", "Issue Summary", "Model Comparison"])
    with tabs[0]:
        st.dataframe(classic_view, use_container_width=True, hide_index=True)

    with tabs[1]:
        issue_summary = collect_issue_summary(classic_raw)
        if issue_summary.empty:
            st.info("No issue labels predicted for the current batch.")
        else:
            st.dataframe(issue_summary, use_container_width=True, hide_index=True)
            st.bar_chart(issue_summary.set_index("label")["count"], use_container_width=True)

    if enable_transformer and tokenizer is not None and transformer_model is not None:
        from demo_transformer import predict_sentiment as predict_transformer_sentiment

        merged_rows: List[Dict[str, Any]] = []

        for item in classic_raw:
            text = item["text"]
            t_result = predict_transformer_sentiment(text, tokenizer, transformer_model)
            merged = {
                "text": text,
                "classic_label": item["classic"].get("label", "N/A"),
                "classic_prob_pos": format_prob(item["classic"].get("probability")),
            }
            merged.update(
                {
                    "transformer_label": t_result.get("label", "N/A"),
                    "transformer_prob_pos": format_prob(t_result.get("probability")),
                    "agreement": "match"
                    if str(item["classic"].get("label", "N/A")) == str(t_result.get("label", "N/A"))
                    else "mismatch",
                    "classic_reason": item["classic"].get("fallback_reason") or "-",
                    "transformer_reason": t_result.get("fallback_reason") or "-",
                }
            )
            merged_rows.append(merged)

        with tabs[2]:
            compare_df = pd.DataFrame(merged_rows).rename(
                columns={
                    "text": "Review text",
                    "classic_label": "Classic label",
                    "classic_prob_pos": "Classic P(positive)",
                    "transformer_label": "Transformer label",
                    "transformer_prob_pos": "Transformer P(positive)",
                    "agreement": "Agreement",
                    "classic_reason": "Classic fallback",
                    "transformer_reason": "Transformer fallback",
                }
            )
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
    elif enable_transformer:
        with tabs[2]:
            st.warning("Transformer comparison requested but model/dependencies are unavailable.")
    else:
        with tabs[2]:
            st.info("Enable transformer comparison from the sidebar to view side-by-side output.")

    if show_raw_json:
        with st.expander("Raw JSON outputs", expanded=False):
            st.json(classic_raw)


if __name__ == "__main__":
    main()
