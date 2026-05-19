"""
Premium EUSEE AI Copilot workspace wrapper.

Purpose
-------
This module gives the existing EUSEE AI Copilot a friendlier, cleaner,
ChatGPT-style professional UX while keeping the existing AI logic intact.

What this version adds
----------------------
- Premium AI workspace header with context metrics.
- Visible Search panel for dashboard intelligence.
- Visible Advanced Tools panel:
  - Chart Builder
  - Country Compare
  - Executive Summary
  - Anomaly Scan
  - Export Center
- Professional CSS overrides for the existing legacy chatbot classes.
- Safe fallback to the original `render_ai_assistant_panel(df)` function.

Usage in app.py
---------------
from ai_chat_ui import render_ai_workspace

if has_permission("use_ai_copilot"):
    render_ai_workspace(
        df=filtered_global,
        legacy_renderer=render_ai_assistant_panel,
        current_role=get_current_role(),
        current_email=get_current_email(),
    )
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------
# Safe helpers
# ---------------------------------------------------------------------
def _nunique(df: pd.DataFrame | None, column: str) -> int:
    """Return safe unique count for a dataframe column."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return 0
    return int(df[column].dropna().nunique())


def _record_count(df: pd.DataFrame | None) -> int:
    """Return safe row count."""
    if df is None or not isinstance(df, pd.DataFrame):
        return 0
    return int(len(df))


def _safe_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return list(df.columns)


def _categorical_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            cols.append(col)
    preferred = [
        "alert-country",
        "region",
        "continent",
        "alert-impact",
        "alert-type",
        "Actor of repression",
        "year",
        "month_name",
    ]
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    return ordered


def _numeric_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _download_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered data")
    return buffer.getvalue()


def _impact_color(value: str) -> str:
    value_clean = str(value).strip().lower()
    if value_clean == "negative":
        return "#FEE4E2"
    if value_clean == "positive":
        return "#DCFAE6"
    if value_clean == "context to watch":
        return "#FEF0C7"
    return "#F9FAFB"


# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------
def inject_ai_workspace_css() -> None:
    """Global styling layer for the AI Copilot workspace and existing ai-min classes."""
    st.markdown(
        """
        <style>
        :root {
            --ai-purple: #660094;
            --ai-purple-deep: #3B005F;
            --ai-teal: #008CAA;
            --ai-bg: #F7F8FB;
            --ai-card: #FFFFFF;
            --ai-soft: #FCFCFD;
            --ai-border: #E6E8EF;
            --ai-border-soft: #EEF0F4;
            --ai-text: #232633;
            --ai-muted: #667085;
            --ai-radius-lg: 22px;
            --ai-radius-md: 16px;
            --ai-shadow: 0 14px 34px rgba(16,24,40,.07);
            --ai-shadow-soft: 0 6px 16px rgba(16,24,40,.045);
            --ai-font: "Inter", "Segoe UI", Arial, sans-serif;
        }

        .ai-workspace-shell {
            margin-top: 18px;
            margin-bottom: 18px;
            padding: 18px;
            border-radius: var(--ai-radius-lg);
            background:
                radial-gradient(circle at 100% 0%, rgba(102,0,148,.055), transparent 32%),
                linear-gradient(180deg, #FFFFFF 0%, #FAFAFC 100%);
            border: 1px solid var(--ai-border);
            box-shadow: var(--ai-shadow);
            font-family: var(--ai-font);
        }

        .ai-workspace-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--ai-border-soft);
        }

        .ai-workspace-eyebrow {
            font-size: 10px;
            font-weight: 900;
            letter-spacing: .13em;
            text-transform: uppercase;
            color: var(--ai-purple);
            margin-bottom: 5px;
        }

        .ai-workspace-title {
            font-size: clamp(20px, 2.2vw, 26px);
            font-weight: 900;
            color: #23152F;
            line-height: 1.12;
            letter-spacing: -0.02em;
        }

        .ai-workspace-subtitle {
            font-size: 12.5px;
            color: var(--ai-muted);
            line-height: 1.5;
            margin-top: 6px;
            max-width: 780px;
        }

        .ai-context-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 7px;
            max-width: 480px;
        }

        .ai-context-pill {
            padding: 6px 10px;
            border-radius: 999px;
            background: #F4EAF8;
            color: var(--ai-purple);
            border: 1px solid #E7D4F1;
            font-size: 10.5px;
            font-weight: 850;
            line-height: 1;
            white-space: nowrap;
        }

        .ai-context-pill.teal {
            background: #EFFBFE;
            color: var(--ai-teal);
            border-color: rgba(0,140,170,.16);
        }

        .ai-context-pill.neutral {
            background: #F9FAFB;
            color: #344054;
            border-color: var(--ai-border-soft);
        }

        .ai-friendly-strip {
            display: grid;
            grid-template-columns: 1.1fr .9fr;
            gap: 10px;
            margin-top: 14px;
        }

        .ai-friendly-note,
        .ai-mode-note {
            padding: 12px 14px;
            border-radius: var(--ai-radius-md);
            background: #F9FAFB;
            border: 1px solid var(--ai-border-soft);
            color: #475467;
            font-size: 12.2px;
            line-height: 1.5;
        }

        .ai-friendly-note strong,
        .ai-mode-note strong {
            color: #23152F;
            font-weight: 900;
        }

        .ai-mode-note {
            background: #FFFFFF;
        }

        .ai-tool-panel {
            margin: 14px 0 18px 0;
            padding: 14px;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid var(--ai-border);
            box-shadow: var(--ai-shadow-soft);
            font-family: var(--ai-font);
        }

        .ai-tool-panel-title {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #23152F;
            font-size: 15px;
            font-weight: 900;
            margin-bottom: 4px;
        }

        .ai-tool-panel-subtitle {
            color: var(--ai-muted);
            font-size: 11.8px;
            line-height: 1.45;
            margin-bottom: 12px;
        }

        .ai-mini-metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 9px;
            margin: 10px 0 12px 0;
        }

        .ai-mini-metric {
            border: 1px solid var(--ai-border-soft);
            background: #FCFCFD;
            border-radius: 15px;
            padding: 10px 11px;
        }

        .ai-mini-metric span {
            display: block;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--ai-purple);
            font-weight: 900;
            margin-bottom: 4px;
        }

        .ai-mini-metric strong {
            display: block;
            font-size: 17px;
            color: #23152F;
            font-weight: 900;
        }

        .ai-search-result {
            padding: 10px 12px;
            border: 1px solid var(--ai-border-soft);
            border-radius: 14px;
            background: #FFFFFF;
            margin-bottom: 8px;
            box-shadow: 0 3px 10px rgba(16,24,40,.035);
            font-size: 12px;
            line-height: 1.45;
            color: #344054;
        }

        .ai-search-result strong {
            color: #23152F;
        }

        .ai-search-result small {
            color: var(--ai-muted);
            font-size: 10.5px;
            font-weight: 700;
        }

        .ai-exec-card {
            padding: 13px 14px;
            border: 1px solid var(--ai-border-soft);
            border-left: 4px solid var(--ai-purple);
            border-radius: 16px;
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
            box-shadow: 0 4px 12px rgba(16,24,40,.04);
            margin-bottom: 10px;
            color: #344054;
            font-size: 12.5px;
            line-height: 1.5;
        }

        .ai-exec-card strong {
            color: #23152F;
            font-weight: 900;
        }

        /* Override and polish the existing minimal chatbot classes without changing logic. */
        .ai-min-shell,
        .ai-min-panel,
        .ai-min-chat,
        .ai-min-output,
        .ai-min-tools,
        .ai-min-suggestions {
            font-family: var(--ai-font) !important;
        }

        .ai-min-panel,
        .ai-min-shell {
            border-radius: 22px !important;
            border: 1px solid var(--ai-border) !important;
            background: linear-gradient(180deg, #FFFFFF 0%, #FCFCFD 100%) !important;
            box-shadow: 0 16px 38px rgba(16,24,40,.075) !important;
            overflow: hidden !important;
        }

        .ai-min-header,
        .ai-min-topbar {
            border-bottom: 1px solid var(--ai-border-soft) !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            padding: 14px 16px !important;
        }

        .ai-min-title,
        .ai-min-heading,
        .ai-min-header-title {
            font-family: var(--ai-font) !important;
            font-size: 18px !important;
            font-weight: 900 !important;
            color: #23152F !important;
            letter-spacing: -0.01em !important;
        }

        .ai-min-subtitle,
        .ai-min-note,
        .ai-min-caption,
        .ai-min-meta {
            font-family: var(--ai-font) !important;
            font-size: 11.5px !important;
            color: var(--ai-muted) !important;
            line-height: 1.45 !important;
        }

        .ai-min-body,
        .ai-min-chat {
            background: #FFFFFF !important;
        }

        .ai-min-chat {
            border: 1px solid var(--ai-border-soft) !important;
            border-radius: 18px !important;
            padding: 14px !important;
            min-height: 360px !important;
            max-height: min(64vh, 720px) !important;
            overflow-y: auto !important;
            scroll-behavior: smooth !important;
        }

        .ai-welcome {
            background:
                radial-gradient(circle at 100% 0%, rgba(102,0,148,.055), transparent 36%),
                linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            border: 1px solid var(--ai-border) !important;
            border-radius: 18px !important;
            padding: 14px 16px !important;
            color: #344054 !important;
            font-size: 13px !important;
            line-height: 1.55 !important;
            box-shadow: var(--ai-shadow-soft) !important;
        }

        .ai-mini-strip {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            gap: 10px !important;
            margin: 12px 0 !important;
        }

        .ai-mini-strip > div {
            background: #FFFFFF !important;
            border: 1px solid var(--ai-border-soft) !important;
            border-radius: 16px !important;
            padding: 12px !important;
            box-shadow: 0 4px 12px rgba(16,24,40,.04) !important;
        }

        .ai-mini-strip span {
            display: block !important;
            font-size: 10px !important;
            font-weight: 900 !important;
            letter-spacing: .08em !important;
            text-transform: uppercase !important;
            color: var(--ai-purple) !important;
            margin-bottom: 5px !important;
        }

        .ai-mini-strip strong {
            display: block !important;
            font-size: 12px !important;
            color: #344054 !important;
            line-height: 1.35 !important;
            font-weight: 750 !important;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 18px !important;
            border: 1px solid var(--ai-border-soft) !important;
            background: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(16,24,40,.04) !important;
            padding: 10px 12px !important;
            margin-bottom: 10px !important;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] div {
            font-family: var(--ai-font) !important;
            font-size: 13px !important;
            line-height: 1.55 !important;
            color: #344054;
        }

        .ai-min-suggestions {
            margin-top: 12px !important;
            padding: 12px !important;
            border: 1px solid var(--ai-border-soft) !important;
            border-radius: 18px !important;
            background: #F9FAFB !important;
        }

        .ai-min-suggestions-title {
            font-size: 10px !important;
            font-weight: 900 !important;
            letter-spacing: .12em !important;
            text-transform: uppercase !important;
            color: var(--ai-purple) !important;
            margin-bottom: 8px !important;
        }

        .ai-min-suggestions .stButton > button,
        .ai-min-tools .stButton > button,
        .ai-min-input .stButton > button,
        .ai-tool-panel .stButton > button {
            border-radius: 999px !important;
            border: 1px solid var(--ai-border) !important;
            background: #FFFFFF !important;
            color: #344054 !important;
            font-size: 12px !important;
            font-weight: 800 !important;
            min-height: 38px !important;
            box-shadow: 0 2px 8px rgba(16,24,40,.035) !important;
            transition: all .16s ease !important;
        }

        .ai-min-suggestions .stButton > button:hover,
        .ai-min-tools .stButton > button:hover,
        .ai-min-input .stButton > button:hover,
        .ai-tool-panel .stButton > button:hover {
            border-color: var(--ai-purple) !important;
            color: var(--ai-purple) !important;
            background: #FBF7FD !important;
            transform: translateY(-1px) !important;
        }

        .ai-min-tools {
            margin-top: 12px !important;
        }

        .ai-min-tools div[data-testid="stExpander"],
        .ai-tool-panel div[data-testid="stExpander"] {
            border-radius: 18px !important;
            border: 1px solid var(--ai-border-soft) !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }

        .ai-min-tools div[data-testid="stExpander"] summary,
        .ai-tool-panel div[data-testid="stExpander"] summary {
            background: #FFFFFF !important;
            font-size: 12.5px !important;
            font-weight: 900 !important;
            color: #23152F !important;
        }

        .ai-min-input {
            position: sticky !important;
            bottom: 0 !important;
            z-index: 20 !important;
            background: linear-gradient(180deg, rgba(255,255,255,.72) 0%, #FFFFFF 45%) !important;
            border-top: 1px solid var(--ai-border-soft) !important;
            padding-top: 12px !important;
            margin-top: 12px !important;
        }

        div[data-testid="stChatInput"] {
            background: #FFFFFF !important;
            border-radius: 24px !important;
            border: 1px solid #D0D5DD !important;
            box-shadow: 0 8px 22px rgba(16,24,40,.08) !important;
            overflow: hidden !important;
        }

        div[data-testid="stChatInput"] textarea {
            font-family: var(--ai-font) !important;
            font-size: 13px !important;
            color: #232633 !important;
        }

        div[data-testid="stPlotlyChart"] {
            border-radius: 16px !important;
            overflow: hidden !important;
            border: 1px solid var(--ai-border-soft) !important;
            box-shadow: 0 6px 16px rgba(16,24,40,.045) !important;
            background: #FFFFFF !important;
            margin-top: 10px !important;
        }

        .ai-min-tools [data-baseweb="tab-list"],
        .ai-tool-panel [data-baseweb="tab-list"] {
            gap: 6px !important;
        }

        .ai-min-tools [data-baseweb="tab"],
        .ai-tool-panel [data-baseweb="tab"] {
            font-size: 12px !important;
            font-weight: 850 !important;
            border-radius: 999px !important;
            padding: 6px 10px !important;
        }

        .ai-min-tools label,
        .ai-min-tools .stSelectbox label,
        .ai-min-tools .stRadio label,
        .ai-min-tools .stSlider label,
        .ai-min-tools .stTextInput label,
        .ai-tool-panel label {
            font-size: 11px !important;
            font-weight: 900 !important;
            color: #344054 !important;
        }

        .ai-min-tools [data-baseweb="select"] > div,
        .ai-min-tools [data-baseweb="input"],
        .ai-tool-panel [data-baseweb="select"] > div,
        .ai-tool-panel [data-baseweb="input"] {
            border-radius: 12px !important;
            min-height: 38px !important;
            border: 1px solid #D0D5DD !important;
        }

        @media (max-width: 900px) {
            .ai-workspace-header,
            .ai-friendly-strip {
                grid-template-columns: 1fr !important;
                flex-direction: column !important;
                align-items: flex-start !important;
            }

            .ai-context-pills {
                justify-content: flex-start !important;
                max-width: 100% !important;
            }

            .ai-mini-strip,
            .ai-mini-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
        }

        @media (max-width: 560px) {
            .ai-workspace-shell,
            .ai-tool-panel {
                padding: 13px !important;
                border-radius: 18px !important;
            }

            .ai-workspace-title {
                font-size: 20px !important;
            }

            .ai-workspace-subtitle,
            .ai-friendly-note,
            .ai-mode-note {
                font-size: 11.8px !important;
            }

            .ai-min-chat {
                min-height: 300px !important;
                max-height: 62vh !important;
                padding: 10px !important;
            }

            .ai-mini-strip,
            .ai-mini-metric-grid {
                grid-template-columns: 1fr !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Search panel
# ---------------------------------------------------------------------
def render_ai_search_panel(df: pd.DataFrame | None) -> None:
    """Render dashboard search inside the premium AI workspace."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Search is unavailable because the filtered dataset is empty.")
        return

    st.markdown(
        """
        <div class="ai-tool-panel">
            <div class="ai-tool-panel-title">🔎 Search dashboard intelligence</div>
            <div class="ai-tool-panel-subtitle">
                Search the currently filtered dataset by country, actor, restriction, alert impact, year, or any keyword.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_cols = _safe_columns(df)
    default_scope = [
        c for c in [
            "alert-country",
            "region",
            "alert-impact",
            "alert-type",
            "Actor of repression",
            "Civil society actor affected",
            "Restrictive mechanism",
            "year",
            "month_name",
        ] if c in search_cols
    ]
    if not default_scope:
        default_scope = search_cols[: min(6, len(search_cols))]

    q_col, s_col, n_col = st.columns([2.2, 1.6, 0.8])
    with q_col:
        query = st.text_input(
            "Search",
            placeholder="Example: Kenya, incarceration, civic rights, negative alerts...",
            key="ai_search_query",
        )
    with s_col:
        scope = st.multiselect(
            "Search scope",
            options=search_cols,
            default=default_scope,
            key="ai_search_scope",
        )
    with n_col:
        max_results = st.selectbox(
            "Results",
            options=[10, 25, 50, 100],
            index=1,
            key="ai_search_max_results",
        )

    if not query.strip():
        st.caption("Enter a keyword to search within the active filtered dashboard data.")
        return

    if not scope:
        st.warning("Select at least one search scope column.")
        return

    query_clean = query.strip().lower()
    search_df = df.copy()
    available_scope = [c for c in scope if c in search_df.columns]
    mask = search_df[available_scope].astype(str).apply(
        lambda row: row.str.lower().str.contains(query_clean, na=False).any(),
        axis=1,
    )
    results = search_df.loc[mask].head(int(max_results)).copy()

    st.markdown(
        f"""
        <div class="ai-mini-metric-grid">
            <div class="ai-mini-metric"><span>Matches</span><strong>{int(mask.sum()):,}</strong></div>
            <div class="ai-mini-metric"><span>Shown</span><strong>{len(results):,}</strong></div>
            <div class="ai-mini-metric"><span>Countries</span><strong>{_nunique(results, "alert-country"):,}</strong></div>
            <div class="ai-mini-metric"><span>Impact classes</span><strong>{_nunique(results, "alert-impact"):,}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if results.empty:
        st.info("No records matched the current search term.")
        return

    preview_cols = [
        c for c in [
            "alert-country",
            "region",
            "alert-impact",
            "alert-type",
            "Actor of repression",
            "Civil society actor affected",
            "Restrictive mechanism",
            "year",
            "creation_date",
        ] if c in results.columns
    ]
    if not preview_cols:
        preview_cols = available_scope[: min(6, len(available_scope))]

    st.dataframe(
        results[preview_cols],
        use_container_width=True,
        hide_index=True,
        height=min(420, max(220, 34 * len(results) + 42)),
    )

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download search results as CSV",
        data=csv,
        file_name="eusee_ai_search_results.csv",
        mime="text/csv",
        use_container_width=True,
        key="ai_search_download_csv",
    )


# ---------------------------------------------------------------------
# Advanced tools
# ---------------------------------------------------------------------
def _render_chart_builder(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="ai-exec-card">
            <strong>Chart Builder:</strong> create quick dashboard-grounded visualizations from the active filtered data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_cols = _categorical_columns(df)
    num_cols = _numeric_columns(df)

    if not cat_cols:
        st.info("No categorical columns are available for chart building.")
        return

    c1, c2, c3 = st.columns([1.3, 1.2, 1.1])
    with c1:
        group_col = st.selectbox("Group by", cat_cols, index=0, key="ai_chart_group")
    with c2:
        metric_mode = st.selectbox(
            "Metric",
            ["Record count"] + num_cols,
            index=0,
            key="ai_chart_metric",
        )
    with c3:
        chart_type = st.selectbox(
            "Chart type",
            ["Bar", "Horizontal bar", "Line", "Area", "Treemap"],
            index=0,
            key="ai_chart_type",
        )

    top_n = st.slider("Top groups", min_value=5, max_value=30, value=12, step=1, key="ai_chart_top_n")

    if metric_mode == "Record count":
        plot_df = (
            df.groupby(group_col, dropna=False)
            .size()
            .reset_index(name="value")
            .sort_values("value", ascending=False)
            .head(top_n)
        )
        y_label = "Records"
    else:
        plot_df = (
            df.groupby(group_col, dropna=False)[metric_mode]
            .sum(numeric_only=True)
            .reset_index(name="value")
            .sort_values("value", ascending=False)
            .head(top_n)
        )
        y_label = metric_mode

    plot_df[group_col] = plot_df[group_col].astype(str)

    if plot_df.empty:
        st.info("No data available for the selected chart.")
        return

    if chart_type == "Bar":
        fig = px.bar(plot_df, x=group_col, y="value", labels={"value": y_label})
    elif chart_type == "Horizontal bar":
        fig = px.bar(plot_df.sort_values("value"), x="value", y=group_col, orientation="h", labels={"value": y_label})
    elif chart_type == "Line":
        fig = px.line(plot_df, x=group_col, y="value", markers=True, labels={"value": y_label})
    elif chart_type == "Area":
        fig = px.area(plot_df, x=group_col, y="value", labels={"value": y_label})
    else:
        fig = px.treemap(plot_df, path=[group_col], values="value")

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=30),
        height=420,
        font=dict(family="Inter, Segoe UI, Arial", size=12),
        title=f"{y_label} by {group_col}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇️ Download chart data as CSV",
        data=plot_df.to_csv(index=False).encode("utf-8"),
        file_name="eusee_ai_chart_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="ai_chart_download",
    )


def _render_country_compare(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="ai-exec-card">
            <strong>Country Compare:</strong> compare selected countries across alert impacts, alert types, and key dashboard categories.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "alert-country" not in df.columns:
        st.info("Country comparison requires an 'alert-country' column.")
        return

    countries = sorted(df["alert-country"].dropna().astype(str).unique().tolist())
    default = countries[: min(3, len(countries))]
    selected = st.multiselect(
        "Countries to compare",
        options=countries,
        default=default,
        key="ai_compare_countries",
    )

    compare_by_options = [c for c in ["alert-impact", "alert-type", "region", "Actor of repression", "year"] if c in df.columns]
    if not compare_by_options:
        st.info("No comparison category columns are available.")
        return

    compare_by = st.selectbox("Compare by", compare_by_options, index=0, key="ai_compare_by")

    if not selected:
        st.warning("Select at least one country.")
        return

    subset = df[df["alert-country"].astype(str).isin(selected)].copy()
    if subset.empty:
        st.info("No records found for the selected countries.")
        return

    comp = (
        subset.groupby(["alert-country", compare_by], dropna=False)
        .size()
        .reset_index(name="records")
    )
    comp[compare_by] = comp[compare_by].astype(str)

    fig = px.bar(
        comp,
        x="alert-country",
        y="records",
        color=compare_by,
        barmode="group",
        title=f"Country comparison by {compare_by}",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=42, b=30),
        height=430,
        font=dict(family="Inter, Segoe UI, Arial", size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = (
        subset.groupby("alert-country", dropna=False)
        .size()
        .reset_index(name="total_records")
        .sort_values("total_records", ascending=False)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_executive_summary(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="ai-exec-card">
            <strong>Executive Summary:</strong> generate a compact, evidence-based summary from the active filtered data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = len(df)
    countries = _nunique(df, "alert-country")
    years = _nunique(df, "year")
    impacts = _nunique(df, "alert-impact")

    style = st.radio(
        "Summary style",
        ["Executive", "Policy", "Donor", "Technical"],
        horizontal=True,
        key="ai_summary_style",
    )

    bullets = []
    bullets.append(f"The current filtered view contains {records:,} records across {countries:,} countries and {years:,} year(s).")
    if "alert-impact" in df.columns:
        impact_counts = df["alert-impact"].astype(str).value_counts().head(3)
        impact_text = ", ".join([f"{idx}: {val:,}" for idx, val in impact_counts.items()])
        bullets.append(f"The leading alert-impact categories are {impact_text}.")
    if "alert-country" in df.columns:
        top_countries = df["alert-country"].astype(str).value_counts().head(5)
        country_text = ", ".join([f"{idx} ({val:,})" for idx, val in top_countries.items()])
        bullets.append(f"The highest-volume countries in this filtered view are {country_text}.")
    if "Actor of repression" in df.columns:
        actors = df["Actor of repression"].astype(str).value_counts().head(3)
        actor_text = ", ".join([f"{idx} ({val:,})" for idx, val in actors.items()])
        bullets.append(f"The most frequently recorded actors are {actor_text}.")
    if "year" in df.columns:
        year_counts = df["year"].dropna().astype(str).value_counts().head(3)
        year_text = ", ".join([f"{idx}: {val:,}" for idx, val in year_counts.items()])
        bullets.append(f"The strongest year-level concentrations are {year_text}.")

    st.markdown(f"#### {style} summary")
    for item in bullets:
        st.markdown(f"- {item}")

    summary_text = f"{style} summary\n\n" + "\n".join([f"- {b}" for b in bullets])
    st.download_button(
        "⬇️ Download summary as TXT",
        data=summary_text.encode("utf-8"),
        file_name="eusee_ai_executive_summary.txt",
        mime="text/plain",
        use_container_width=True,
        key="ai_summary_download",
    )


def _render_anomaly_scan(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="ai-exec-card">
            <strong>Anomaly Scan:</strong> identify unusually high concentrations in the selected category using z-score style outlier detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_cols = _categorical_columns(df)
    if not cat_cols:
        st.info("No categorical columns are available for anomaly scanning.")
        return

    c1, c2 = st.columns([1.3, 1])
    with c1:
        scan_col = st.selectbox("Scan category", cat_cols, index=0, key="ai_anomaly_scan_col")
    with c2:
        sensitivity = st.slider("Sensitivity", 1.0, 3.0, 1.5, 0.1, key="ai_anomaly_sensitivity")

    counts = (
        df[scan_col]
        .astype(str)
        .value_counts()
        .rename_axis(scan_col)
        .reset_index(name="records")
    )

    if counts.empty or counts["records"].std(ddof=0) == 0:
        st.info("No statistical anomalies detected for the selected category.")
        st.dataframe(counts.head(20), use_container_width=True, hide_index=True)
        return

    mean = counts["records"].mean()
    std = counts["records"].std(ddof=0)
    counts["z_score"] = (counts["records"] - mean) / std
    anomalies = counts[counts["z_score"] >= sensitivity].sort_values("z_score", ascending=False)

    st.markdown(
        f"""
        <div class="ai-mini-metric-grid">
            <div class="ai-mini-metric"><span>Groups scanned</span><strong>{len(counts):,}</strong></div>
            <div class="ai-mini-metric"><span>Anomalies</span><strong>{len(anomalies):,}</strong></div>
            <div class="ai-mini-metric"><span>Mean records</span><strong>{mean:.1f}</strong></div>
            <div class="ai-mini-metric"><span>Threshold</span><strong>{sensitivity:.1f}σ</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = px.bar(
        counts.head(20),
        x=scan_col,
        y="records",
        title=f"Top concentrations by {scan_col}",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=42, b=30),
        height=420,
        font=dict(family="Inter, Segoe UI, Arial", size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    if anomalies.empty:
        st.success("No high-concentration anomalies exceed the selected threshold.")
    else:
        st.dataframe(anomalies, use_container_width=True, hide_index=True)


def _render_export_center(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="ai-exec-card">
            <strong>Export Center:</strong> download the currently filtered AI workspace data and summaries.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="eusee_ai_filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
            key="ai_export_csv",
        )
    with c2:
        try:
            xlsx_bytes = _download_excel_bytes(df)
            st.download_button(
                "⬇️ Download filtered data as XLSX",
                data=xlsx_bytes,
                file_name="eusee_ai_filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="ai_export_xlsx",
            )
        except Exception:
            st.caption("XLSX export requires xlsxwriter. CSV export is available.")


def render_advanced_tools_panel(df: pd.DataFrame | None) -> None:
    """Render the advanced tools panel directly in the premium AI workspace."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Advanced tools are unavailable because the filtered dataset is empty.")
        return

    st.markdown(
        """
        <div class="ai-tool-panel">
            <div class="ai-tool-panel-title">📊 Advanced AI tools</div>
            <div class="ai-tool-panel-subtitle">
                Use these tools to explore, compare, summarize, detect patterns, and export outputs from the active filtered view.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "Chart Builder",
        "Compare",
        "Executive Summary",
        "Anomaly Scan",
        "Export Center",
    ])

    with tabs[0]:
        _render_chart_builder(df)
    with tabs[1]:
        _render_country_compare(df)
    with tabs[2]:
        _render_executive_summary(df)
    with tabs[3]:
        _render_anomaly_scan(df)
    with tabs[4]:
        _render_export_center(df)


# ---------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------
def render_ai_workspace(
    df: pd.DataFrame | None,
    legacy_renderer: Callable[[pd.DataFrame | None], Any],
    current_role: str = "guest",
    current_email: str = "",
) -> None:
    """
    Render the premium AI workspace, search, advanced tools, and then delegate to the existing chatbot.

    Parameters
    ----------
    df:
        Currently filtered dashboard dataframe.
    legacy_renderer:
        Existing function, usually render_ai_assistant_panel(df).
    current_role:
        Active dashboard role from authz.py.
    current_email:
        Active user email from authz.py.
    """
    inject_ai_workspace_css()

    records = _record_count(df)
    countries = _nunique(df, "alert-country")
    years = _nunique(df, "year")
    impacts = _nunique(df, "alert-impact")
    role_display = str(current_role or "guest").replace("_", " ").title()

    email_part = ""
    if current_email:
        email_part = f"<div class='ai-context-pill neutral'>{current_email}</div>"

    st.markdown(
        f"""
        <div class="ai-workspace-shell">
            <div class="ai-workspace-header">
                <div>
                    <div class="ai-workspace-eyebrow">AI analytical workspace</div>
                    <div class="ai-workspace-title">EUSEE AI Copilot</div>
                    <div class="ai-workspace-subtitle">
                        A cleaner, friendlier, professional AI workspace for search, advanced tools,
                        filtered dashboard interpretation, chart building, and executive summaries.
                    </div>
                </div>
                <div class="ai-context-pills">
                    <div class="ai-context-pill">{records:,} records</div>
                    <div class="ai-context-pill teal">{countries:,} countries</div>
                    <div class="ai-context-pill neutral">{years:,} years</div>
                    <div class="ai-context-pill neutral">{impacts:,} impact classes</div>
                    <div class="ai-context-pill">{role_display}</div>
                    {email_part}
                </div>
            </div>
            <div class="ai-friendly-strip">
                <div class="ai-friendly-note">
                    <strong>Search:</strong> quickly find countries, actors, restrictions, and keywords inside the active filtered data.
                </div>
                <div class="ai-mode-note">
                    <strong>Advanced tools:</strong> build charts, compare countries, generate executive summaries,
                    scan anomalies, and export filtered outputs.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔎 Search", expanded=False):
        render_ai_search_panel(df)

    with st.expander("📊 Advanced Tools", expanded=False):
        render_advanced_tools_panel(df)

    st.markdown(
        """
        <div class="ai-tool-panel">
            <div class="ai-tool-panel-title">💬 Conversation</div>
            <div class="ai-tool-panel-subtitle">
                Continue using the existing AI Copilot conversation engine below. Search and advanced tools above now remain visible and accessible.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        legacy_renderer(df)
    except Exception as exc:
        st.error(f"AI Copilot could not be rendered: {exc}")

    # Inject again after the legacy renderer so these professional overrides win
    # if the original function also injects CSS with the same class names.
    inject_ai_workspace_css()
