"""
Streamlit UI for NLP review understanding demo.

Run:
    streamlit run demo_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from demo import (
    load_issue_model,
    load_models,
    predict_sentiment as predict_classic_sentiment,
)
from webapp.predictor import (
    LABEL_PRIORITY,
    build_risk_score,
    summarize_issue_labels,
    to_float,
)


APP_TITLE = "Review Ops Console"
APP_SUBTITLE = (
    "Batch-analyze customer reviews with the classic sentiment stack, "
    "issue extraction, and optional transformer comparison."
)
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
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

            :root {
                --bg-a: #f4f8ef;
                --bg-b: #dfece0;
                --ink: #123249;
                --muted: #446176;
                --panel: #ffffff;
                --line: #c8d7e2;
                --brand-a: #114f7c;
                --brand-b: #18706c;
                --warn: #a2462d;
                --good: #1e6a41;
            }

            html, body, [class*="css"] {
                font-family: "Source Sans 3", sans-serif;
                color: var(--ink);
            }

            h1, h2, h3, h4 {
                font-family: "Space Grotesk", sans-serif;
                letter-spacing: 0.01em;
            }

            .stApp {
                background: linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
            }

            .stApp::before,
            .stApp::after {
                content: "";
                position: fixed;
                border-radius: 999px;
                filter: blur(4px);
                opacity: 0.38;
                pointer-events: none;
                z-index: 0;
            }

            .stApp::before {
                width: 340px;
                height: 340px;
                top: -90px;
                right: -120px;
                background: radial-gradient(circle, #b3dbcc 0%, #b3dbcc 35%, transparent 70%);
            }

            .stApp::after {
                width: 320px;
                height: 320px;
                bottom: -120px;
                left: -110px;
                background: radial-gradient(circle, #bad6ea 0%, #bad6ea 32%, transparent 68%);
            }

            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2.2rem;
                max-width: 1180px;
                position: relative;
                z-index: 1;
            }

            @keyframes riseIn {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .reveal {
                animation: riseIn 0.55s ease both;
            }

            .hero-panel {
                background: linear-gradient(126deg, var(--brand-a) 0%, var(--brand-b) 100%);
                border-radius: 18px;
                padding: 18px 20px 16px;
                color: #f7fbff;
                box-shadow: 0 14px 28px rgba(17, 58, 88, 0.18);
                animation: riseIn 0.6s ease both;
            }

            .hero-kicker {
                margin: 0;
                font-size: 0.78rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                opacity: 0.88;
            }

            .hero-title {
                margin: 4px 0 4px;
                font-size: 1.82rem;
                line-height: 1.12;
            }

            .hero-subtitle {
                margin: 0;
                max-width: 760px;
                font-size: 1rem;
                line-height: 1.45;
                opacity: 0.95;
            }

            .status-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
                margin-top: 12px;
            }

            .status-card {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.26);
                border-radius: 12px;
                padding: 10px 12px;
            }

            .status-label {
                margin: 0;
                font-size: 0.75rem;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                opacity: 0.84;
            }

            .status-value {
                margin: 3px 0 0;
                font-size: 1.02rem;
                font-weight: 650;
                line-height: 1.22;
                word-break: break-word;
            }

            .panel-card {
                margin-top: 14px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 12px 14px;
                box-shadow: 0 5px 12px rgba(16, 46, 66, 0.07);
            }

            .panel-title {
                margin: 0;
                color: var(--brand-a);
                font-size: 1rem;
                font-weight: 700;
            }

            .panel-note {
                margin: 4px 0 0;
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.35;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 10px;
                margin-top: 14px;
            }

            .kpi-card {
                border-radius: 13px;
                padding: 10px 12px;
                background: #ffffff;
                border: 1px solid #d6e0e8;
                box-shadow: 0 5px 12px rgba(12, 36, 54, 0.06);
            }

            .kpi-card.tone-alert {
                background: #fff0eb;
                border-color: #efccbf;
            }

            .kpi-card.tone-positive {
                background: #ebf7ef;
                border-color: #c4e3cc;
            }

            .kpi-card.tone-neutral {
                background: #eef5fb;
                border-color: #cfdeea;
            }

            .kpi-label {
                margin: 0;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #4d6678;
            }

            .kpi-value {
                margin: 2px 0 1px;
                font-size: 1.42rem;
                line-height: 1.05;
                font-weight: 700;
                color: var(--ink);
            }

            .kpi-note {
                margin: 0;
                font-size: 0.83rem;
                color: #5d7281;
            }

            @media (max-width: 1050px) {
                .status-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }

                .kpi-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 700px) {
                .status-grid, .kpi-grid {
                    grid-template-columns: 1fr;
                }

                .hero-title {
                    font-size: 1.45rem;
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





def build_classic_row(text: str, result: Dict[str, Any]) -> Dict[str, Any]:
    label = str(result.get("label", "N/A"))
    probability = to_float(result.get("probability"))
    issue_rows = result.get("issue_labels", [])
    fallback_tags = result.get("issue_tags", [])
    issue_count = len(issue_rows) if issue_rows else len(fallback_tags)

    return {
        "text": text,
        "classic_label": label,
        "probability": probability,
        "confidence": result.get("confidence", "N/A"),
        "fallback_reason": result.get("fallback_reason") or "-",
        "issue_summary": summarize_issue_labels(result),
        "issue_count": int(issue_count),
        "risk_score": build_risk_score(label, probability),
    }


def parse_inputs(raw_text: str) -> List[str]:
    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]


def _status_card(label: str, value: str) -> str:
    return (
        '<div class="status-card">'
        f'<p class="status-label">{label}</p>'
        f'<p class="status-value">{value}</p>'
        "</div>"
    )


def _metric_card(label: str, value: str, note: str, tone: str) -> str:
    safe_tone = tone if tone in {"alert", "positive", "neutral"} else "neutral"
    return (
        f'<div class="kpi-card tone-{safe_tone}">'
        f'<p class="kpi-label">{label}</p>'
        f'<p class="kpi-value">{value}</p>'
        f'<p class="kpi-note">{note}</p>'
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
    negative = int((labels == "NEGATIVE").sum())
    needs_attention = int((labels == "NEEDS_ATTENTION").sum())
    return {
        "total": int(len(classic_df)),
        "negative": negative,
        "needs_attention": needs_attention,
        "uncertain": int((labels == "UNCERTAIN").sum()),
        "positive": int((labels == "POSITIVE").sum()),
        "flagged": negative + needs_attention,
    }


def build_label_distribution(classic_df: pd.DataFrame) -> pd.DataFrame:
    ordered_labels = ["NEGATIVE", "NEEDS_ATTENTION", "UNCERTAIN", "POSITIVE"]
    total = max(len(classic_df), 1)
    rows: List[Dict[str, Any]] = []
    label_counts = classic_df["classic_label"].astype(str).value_counts()

    for label in ordered_labels:
        count = int(label_counts.get(label, 0))
        rows.append(
            {
                "label": label,
                "count": count,
                "share_percent": round((count / total) * 100, 1),
            }
        )

    return pd.DataFrame(rows)


def build_attention_queue(classic_df: pd.DataFrame) -> pd.DataFrame:
    queue = classic_df[classic_df["classic_label"].isin(["NEGATIVE", "NEEDS_ATTENTION", "UNCERTAIN"])].copy()
    if queue.empty:
        return pd.DataFrame(columns=["text", "classic_label", "probability", "issue_summary", "risk_score"])
    queue = queue.sort_values(["risk_score", "probability"], ascending=[False, True], na_position="last")
    return queue[["text", "classic_label", "probability", "issue_summary", "risk_score"]].reset_index(drop=True)


def render_header(model_info: Dict[str, Any], issue_mode: str, transformer_status: str) -> None:
    header_html = (
        '<section class="hero-panel">'
        '<p class="hero-kicker">NLP Monitoring Layer</p>'
        f'<h1 class="hero-title">{APP_TITLE}</h1>'
        f'<p class="hero-subtitle">{APP_SUBTITLE}</p>'
        '<div class="status-grid">'
        + _status_card("Classic K*", str(model_info.get("k_features", "N/A")))
        + _status_card("Thresholds", str(model_info.get("thresholds", "N/A")))
        + _status_card("Variant", str(model_info.get("variant", "N/A")))
        + _status_card("Issue Mode", issue_mode)
        + _status_card("Transformer", transformer_status)
        + _status_card("Model Timestamp", str(model_info.get("trained_at", "N/A")))
        + "</div></section>"
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_overview_cards(metrics: Dict[str, int]) -> None:
    cards = [
        _metric_card("Inputs", str(metrics["total"]), "Reviews in current run", "neutral"),
        _metric_card("Flagged", str(metrics["flagged"]), "Negative + needs attention", "alert"),
        _metric_card("Negative", str(metrics["negative"]), "Strongly negative signal", "alert"),
        _metric_card("Uncertain", str(metrics["uncertain"]), "Needs manual read", "neutral"),
        _metric_card("Positive", str(metrics["positive"]), "No immediate action", "positive"),
    ]
    st.markdown('<div class="kpi-grid reveal">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:", layout="wide")
    inject_styles()

    base_dir = Path(__file__).resolve().parent

    with st.sidebar:
        st.header("Run Options")
        enable_transformer = st.checkbox("Compare with transformer", value=False)
        show_attention_only = st.checkbox("Show only non-positive rows", value=False)
        max_rows = st.slider("Rows displayed in table", min_value=10, max_value=500, value=200, step=10)
        show_raw_json = st.checkbox("Show raw JSON output", value=False)
        st.caption("Model files are loaded from `models/`.")

    try:
        classic_bundle = load_classic_bundle(base_dir)
    except SystemExit:
        st.error("Classic model artifacts are missing. Train or restore `models/` first.")
        return

    vectorizer, selector, model, meta, model_info, issue_bundle = classic_bundle
    issue_mode = "trained classifier" if issue_bundle is not None else "rule-based fallback"

    tokenizer = None
    transformer_model = None
    transformer_status = "disabled"

    if enable_transformer:
        try:
            tokenizer, transformer_model = load_transformer_bundle(base_dir)
            transformer_status = "ready"
        except Exception as exc:  # noqa: BLE001
            exc_msg = str(exc).strip() or exc.__class__.__name__
            transformer_status = f"unavailable ({exc_msg[:64]})"

    render_header(model_info, issue_mode, transformer_status)

    if "demo_input_text" not in st.session_state:
        st.session_state["demo_input_text"] = DEFAULT_INPUT

    st.markdown(
        '<div class="panel-card reveal"><p class="panel-title">Input Reviews</p>'
        '<p class="panel-note">Enter one review per line, then run analysis for batch triage.</p></div>',
        unsafe_allow_html=True,
    )
    raw_text = st.text_area(
        "One review per line",
        key="demo_input_text",
        height=190,
    )

    col_run, col_sample, col_clear, col_note = st.columns([1.1, 1, 1, 2.2])
    run_clicked = col_run.button("Analyze", type="primary", use_container_width=True)
    sample_clicked = col_sample.button("Load sample", use_container_width=True)
    clear_clicked = col_clear.button("Clear", use_container_width=True)
    col_note.write("Tip: mix easy and hard cases to stress-test threshold behavior.")

    if sample_clicked:
        st.session_state["demo_input_text"] = DEFAULT_INPUT
        st.rerun()
    if clear_clicked:
        st.session_state["demo_input_text"] = ""
        st.rerun()

    if not run_clicked:
        st.info("Enter review lines and click Analyze.")
        return

    inputs = parse_inputs(raw_text)
    if not inputs:
        st.warning("No valid input lines found.")
        return

    classic_rows: List[Dict[str, Any]] = []
    classic_raw: List[Dict[str, Any]] = []

    with st.spinner("Running inference..."):
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
    if classic_df.empty:
        st.warning("No outputs generated.")
        return

    metrics = build_overview_metrics(classic_df)
    render_overview_cards(metrics)

    distribution_df = build_label_distribution(classic_df)
    attention_df = build_attention_queue(classic_df)

    col_distribution, col_attention = st.columns([1.2, 1.0])
    with col_distribution:
        st.markdown(
            '<div class="panel-card reveal"><p class="panel-title">Label Distribution</p>'
            '<p class="panel-note">Overall label spread in the current batch.</p></div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(distribution_df.set_index("label")["count"], use_container_width=True)
        st.dataframe(
            distribution_df.rename(
                columns={
                    "label": "Label",
                    "count": "Count",
                    "share_percent": "Share (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_attention:
        st.markdown(
            '<div class="panel-card reveal"><p class="panel-title">Attention Queue</p>'
            '<p class="panel-note">Highest-risk rows first for manual follow-up.</p></div>',
            unsafe_allow_html=True,
        )
        if attention_df.empty:
            st.info("No non-positive predictions in this batch.")
        else:
            queue_view = attention_df.rename(
                columns={
                    "text": "Review text",
                    "classic_label": "Label",
                    "probability": "P(positive)",
                    "issue_summary": "Issue tags/labels",
                    "risk_score": "Risk score",
                }
            )
            st.dataframe(
                queue_view.head(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "P(positive)": st.column_config.NumberColumn(format="%.3f"),
                    "Risk score": st.column_config.NumberColumn(format="%.1f"),
                },
            )

    tabs = st.tabs(["Predictions", "Issue Summary", "Model Comparison"])

    with tabs[0]:
        prediction_df = classic_df.copy()
        if show_attention_only:
            prediction_df = prediction_df[prediction_df["classic_label"] != "POSITIVE"]

        if prediction_df.empty:
            st.info("No rows available for the selected filter.")
        else:
            display_df = prediction_df.head(max_rows).rename(
                columns={
                    "text": "Review text",
                    "classic_label": "Classic label",
                    "probability": "P(positive)",
                    "confidence": "Confidence",
                    "fallback_reason": "Fallback reason",
                    "issue_summary": "Issue tags/labels",
                    "issue_count": "Issue count",
                    "risk_score": "Risk score",
                }
            )
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "P(positive)": st.column_config.NumberColumn(format="%.3f"),
                    "Issue count": st.column_config.NumberColumn(format="%d"),
                    "Risk score": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            export_df = prediction_df.rename(
                columns={
                    "text": "review_text",
                    "classic_label": "classic_label",
                    "probability": "prob_positive",
                    "confidence": "confidence",
                    "fallback_reason": "fallback_reason",
                    "issue_summary": "issue_tags_or_labels",
                    "issue_count": "issue_count",
                    "risk_score": "risk_score",
                }
            )
            st.download_button(
                label="Download current predictions (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="review_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tabs[1]:
        issue_summary = collect_issue_summary(classic_raw)
        if issue_summary.empty:
            st.info("No issue labels predicted for the current batch.")
        else:
            st.metric("Unique issue labels", int(issue_summary["label"].nunique()))
            st.metric("Total issue hits", int(issue_summary["count"].sum()))
            st.dataframe(issue_summary, use_container_width=True, hide_index=True)
            st.bar_chart(issue_summary.set_index("label")["count"], use_container_width=True)

    with tabs[2]:
        if enable_transformer and tokenizer is not None and transformer_model is not None:
            from demo_transformer import predict_sentiment as predict_transformer_sentiment

            merged_rows: List[Dict[str, Any]] = []

            for item in classic_raw:
                text = item["text"]
                t_result = predict_transformer_sentiment(text, tokenizer, transformer_model)
                classic_label = str(item["classic"].get("label", "N/A"))
                transformer_label = str(t_result.get("label", "N/A"))
                merged_rows.append(
                    {
                        "text": text,
                        "classic_label": classic_label,
                        "classic_prob_pos": to_float(item["classic"].get("probability")),
                        "transformer_label": transformer_label,
                        "transformer_prob_pos": to_float(t_result.get("probability")),
                        "agreement": "match" if classic_label == transformer_label else "mismatch",
                        "classic_reason": item["classic"].get("fallback_reason") or "-",
                        "transformer_reason": t_result.get("fallback_reason") or "-",
                    }
                )

            compare_df = pd.DataFrame(merged_rows)
            mismatch_count = int((compare_df["agreement"] == "mismatch").sum())
            st.metric("Mismatches", mismatch_count, help="Rows where classic and transformer labels differ.")
            st.dataframe(
                compare_df.rename(
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
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Classic P(positive)": st.column_config.NumberColumn(format="%.3f"),
                    "Transformer P(positive)": st.column_config.NumberColumn(format="%.3f"),
                },
            )
        elif enable_transformer:
            st.warning("Transformer comparison was requested but model/dependencies are unavailable.")
        else:
            st.info("Enable transformer comparison from the sidebar to view side-by-side output.")

    if show_raw_json:
        with st.expander("Raw JSON outputs", expanded=False):
            st.json(classic_raw)


if __name__ == "__main__":
    main()
