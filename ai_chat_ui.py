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
    """Global styling for a consolidated floating EUSEE AI Copilot panel."""
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
            --ai-shadow: 0 22px 58px rgba(16,24,40,.20);
            --ai-shadow-soft: 0 6px 16px rgba(16,24,40,.045);
            --ai-font: "Inter", "Segoe UI", Arial, sans-serif;
        }

        /* Floating launcher */
        .st-key-eusee_ai_floating_launcher {
            position: fixed !important;
            right: 22px !important;
            bottom: 22px !important;
            z-index: 2147482300 !important;
            width: min(340px, calc(100vw - 32px)) !important;
            font-family: var(--ai-font) !important;
        }

        .st-key-eusee_ai_floating_launcher .stButton > button {
            width: 100% !important;
            min-height: 52px !important;
            border-radius: 999px !important;
            border: 1px solid rgba(102,0,148,.22) !important;
            background: linear-gradient(135deg, #660094 0%, #008CAA 100%) !important;
            color: #FFFFFF !important;
            font-size: 13px !important;
            font-weight: 900 !important;
            box-shadow: 0 18px 38px rgba(16,24,40,.24) !important;
        }

        /* Floating panel. This uses Streamlit container keys, so all Streamlit widgets stay functional. */
        .st-key-eusee_ai_floating_panel {
            position: fixed !important;
            right: 22px !important;
            bottom: 22px !important;
            width: min(720px, calc(100vw - 36px)) !important;
            height: min(86vh, 860px) !important;
            max-height: calc(100vh - 36px) !important;
            z-index: 2147482400 !important;
            padding: 0 !important;
            overflow: hidden !important;
            border-radius: 24px !important;
            background: #FFFFFF !important;
            border: 1px solid rgba(230,232,239,.98) !important;
            box-shadow: var(--ai-shadow) !important;
            font-family: var(--ai-font) !important;
        }

        .st-key-eusee_ai_floating_panel > div {
            height: 100% !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding: 0 16px 16px 16px !important;
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        }

        .st-key-eusee_ai_floating_panel > div::-webkit-scrollbar {
            width: 8px !important;
        }
        .st-key-eusee_ai_floating_panel > div::-webkit-scrollbar-thumb {
            background: #D0D5DD !important;
            border-radius: 999px !important;
        }
        .st-key-eusee_ai_floating_panel > div::-webkit-scrollbar-track {
            background: #F9FAFB !important;
        }

        .ai-floating-header {
            position: sticky;
            top: 0;
            z-index: 50;
            margin: 0 -16px 14px -16px;
            padding: 14px 16px 12px 16px;
            background: rgba(255,255,255,.96);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-bottom: 1px solid var(--ai-border-soft);
            border-radius: 24px 24px 0 0;
            font-family: var(--ai-font);
        }

        .ai-floating-header-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
        }

        .ai-floating-eyebrow {
            font-size: 9.5px;
            font-weight: 950;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: var(--ai-purple);
            margin-bottom: 4px;
        }

        .ai-floating-title {
            color: #23152F;
            font-size: 18px;
            font-weight: 950;
            line-height: 1.12;
            letter-spacing: -0.02em;
        }

        .ai-floating-subtitle {
            color: var(--ai-muted);
            font-size: 11.5px;
            line-height: 1.42;
            margin-top: 4px;
            max-width: 460px;
        }

        .ai-context-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }

        .ai-context-pill {
            padding: 5px 8px;
            border-radius: 999px;
            background: #F4EAF8;
            color: var(--ai-purple);
            border: 1px solid #E7D4F1;
            font-size: 10px;
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

        .ai-tool-panel {
            margin: 10px 0 12px 0;
            padding: 13px;
            border-radius: 18px;
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
            font-size: 14px;
            font-weight: 900;
            margin-bottom: 4px;
        }

        .ai-tool-panel-subtitle {
            color: var(--ai-muted);
            font-size: 11.5px;
            line-height: 1.45;
        }

        .ai-mini-metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin: 10px 0 12px 0;
        }

        .ai-mini-metric {
            border: 1px solid var(--ai-border-soft);
            background: #FCFCFD;
            border-radius: 14px;
            padding: 9px 10px;
        }

        .ai-mini-metric span {
            display: block;
            font-size: 9.5px;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--ai-purple);
            font-weight: 900;
            margin-bottom: 4px;
        }

        .ai-mini-metric strong {
            display: block;
            font-size: 16px;
            color: #23152F;
            font-weight: 900;
        }

        .ai-exec-card {
            padding: 12px 13px;
            border: 1px solid var(--ai-border-soft);
            border-left: 4px solid var(--ai-purple);
            border-radius: 15px;
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
            box-shadow: 0 4px 12px rgba(16,24,40,.04);
            margin-bottom: 10px;
            color: #344054;
            font-size: 12px;
            line-height: 1.5;
        }

        .ai-exec-card strong {
            color: #23152F;
            font-weight: 900;
        }

        /* Tabs become the consolidated navigation for Conversation, Search, and Tools. */
        .st-key-eusee_ai_floating_panel [data-baseweb="tab-list"] {
            gap: 6px !important;
            background: #F9FAFB !important;
            border: 1px solid var(--ai-border-soft) !important;
            border-radius: 999px !important;
            padding: 5px !important;
            margin-bottom: 10px !important;
            overflow-x: auto !important;
        }

        .st-key-eusee_ai_floating_panel [data-baseweb="tab"] {
            border-radius: 999px !important;
            padding: 7px 11px !important;
            font-size: 12px !important;
            font-weight: 900 !important;
            color: #344054 !important;
        }

        .st-key-eusee_ai_floating_panel [aria-selected="true"] {
            background: #FFFFFF !important;
            color: var(--ai-purple) !important;
            box-shadow: 0 2px 8px rgba(16,24,40,.06) !important;
        }

        /* Streamlit widget polish inside floating panel */
        .st-key-eusee_ai_floating_panel label,
        .st-key-eusee_ai_floating_panel .stSelectbox label,
        .st-key-eusee_ai_floating_panel .stRadio label,
        .st-key-eusee_ai_floating_panel .stSlider label,
        .st-key-eusee_ai_floating_panel .stTextInput label,
        .st-key-eusee_ai_floating_panel .stMultiSelect label {
            font-size: 11px !important;
            font-weight: 900 !important;
            color: #344054 !important;
            font-family: var(--ai-font) !important;
        }

        .st-key-eusee_ai_floating_panel [data-baseweb="select"] > div,
        .st-key-eusee_ai_floating_panel [data-baseweb="input"] {
            border-radius: 12px !important;
            min-height: 38px !important;
            border: 1px solid #D0D5DD !important;
            background: #FFFFFF !important;
        }

        .st-key-eusee_ai_floating_panel .stButton > button,
        .st-key-eusee_ai_floating_panel .stDownloadButton > button {
            border-radius: 999px !important;
            border: 1px solid var(--ai-border) !important;
            background: #FFFFFF !important;
            color: #344054 !important;
            font-size: 12px !important;
            font-weight: 850 !important;
            min-height: 36px !important;
            box-shadow: 0 2px 8px rgba(16,24,40,.035) !important;
            transition: all .16s ease !important;
        }

        .st-key-eusee_ai_floating_panel .stButton > button:hover,
        .st-key-eusee_ai_floating_panel .stDownloadButton > button:hover {
            border-color: var(--ai-purple) !important;
            color: var(--ai-purple) !important;
            background: #FBF7FD !important;
            transform: translateY(-1px) !important;
        }

        .ai-floating-close .stButton > button {
            width: 38px !important;
            min-width: 38px !important;
            height: 38px !important;
            min-height: 38px !important;
            padding: 0 !important;
            border-radius: 999px !important;
            background: #F9FAFB !important;
            color: #344054 !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px !important;
            overflow: hidden !important;
            border: 1px solid var(--ai-border-soft) !important;
            box-shadow: 0 4px 12px rgba(16,24,40,.04) !important;
        }

        div[data-testid="stPlotlyChart"] {
            border-radius: 16px !important;
            overflow: hidden !important;
            border: 1px solid var(--ai-border-soft) !important;
            box-shadow: 0 6px 16px rgba(16,24,40,.045) !important;
            background: #FFFFFF !important;
            margin-top: 10px !important;
        }

        /* Legacy chatbot classes are now polished inside the floating Conversation tab. */
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
            border-radius: 18px !important;
            border: 1px solid var(--ai-border) !important;
            background: linear-gradient(180deg, #FFFFFF 0%, #FCFCFD 100%) !important;
            box-shadow: 0 8px 18px rgba(16,24,40,.06) !important;
            overflow: hidden !important;
        }

        .ai-min-header,
        .ai-min-topbar {
            border-bottom: 1px solid var(--ai-border-soft) !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            padding: 12px 14px !important;
        }

        .ai-min-title,
        .ai-min-heading,
        .ai-min-header-title {
            font-family: var(--ai-font) !important;
            font-size: 16px !important;
            font-weight: 900 !important;
            color: #23152F !important;
        }

        .ai-min-subtitle,
        .ai-min-note,
        .ai-min-caption,
        .ai-min-meta {
            font-family: var(--ai-font) !important;
            font-size: 11px !important;
            color: var(--ai-muted) !important;
            line-height: 1.45 !important;
        }

        .ai-min-chat {
            border: 1px solid var(--ai-border-soft) !important;
            border-radius: 16px !important;
            padding: 12px !important;
            min-height: 280px !important;
            max-height: 46vh !important;
            overflow-y: auto !important;
            scroll-behavior: smooth !important;
            background: #FFFFFF !important;
        }

        .ai-min-input {
            position: sticky !important;
            bottom: 0 !important;
            z-index: 20 !important;
            background: linear-gradient(180deg, rgba(255,255,255,.72) 0%, #FFFFFF 45%) !important;
            border-top: 1px solid var(--ai-border-soft) !important;
            padding-top: 10px !important;
            margin-top: 10px !important;
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
            font-size: 12.8px !important;
            line-height: 1.55 !important;
            color: #344054;
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

        @media (max-width: 760px) {
            .st-key-eusee_ai_floating_panel {
                right: 10px !important;
                bottom: 10px !important;
                left: 10px !important;
                width: auto !important;
                height: calc(100vh - 20px) !important;
                max-height: calc(100vh - 20px) !important;
                border-radius: 20px !important;
            }

            .st-key-eusee_ai_floating_launcher {
                right: 12px !important;
                bottom: 12px !important;
                width: calc(100vw - 24px) !important;
            }

            .ai-floating-header {
                border-radius: 20px 20px 0 0 !important;
            }

            .ai-floating-header-row {
                align-items: flex-start !important;
            }

            .ai-mini-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
        }

        @media (max-width: 520px) {
            .ai-mini-metric-grid {
                grid-template-columns: 1fr !important;
            }
            .ai-floating-title {
                font-size: 16px !important;
            }
            .ai-floating-subtitle {
                font-size: 10.8px !important;
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
# Main floating workspace
# ---------------------------------------------------------------------
def render_ai_workspace(
    df: pd.DataFrame | None,
    legacy_renderer: Callable[[pd.DataFrame | None], Any],
    current_role: str = "guest",
    current_email: str = "",
) -> None:
    """
    Render all AI Copilot features inside one consolidated floating panel.

    The floating panel contains:
    - Conversation tab: existing AI Copilot / legacy renderer.
    - Search tab: dashboard data search.
    - Advanced Tools tab: chart builder, comparison, summary, anomaly scan, exports.

    app.py integration stays the same:
        render_ai_workspace(df=filtered_global, legacy_renderer=render_ai_assistant_panel, ...)
    """
    inject_ai_workspace_css()

    st.session_state.setdefault("eusee_ai_panel_open", False)

    records = _record_count(df)
    countries = _nunique(df, "alert-country")
    years = _nunique(df, "year")
    impacts = _nunique(df, "alert-impact")
    role_display = str(current_role or "guest").replace("_", " ").title()

    # Collapsed floating launcher only.
    if not st.session_state.get("eusee_ai_panel_open", False):
        with st.container(key="eusee_ai_floating_launcher"):
            if st.button("💬 Open EUSEE AI Copilot", key="open_eusee_ai_panel", use_container_width=True):
                st.session_state.eusee_ai_panel_open = True
                st.rerun()
        return

    # Open floating panel containing every chatbot feature together.
    with st.container(key="eusee_ai_floating_panel"):
        st.markdown(
            f"""
            <div class="ai-floating-header">
                <div class="ai-floating-header-row">
                    <div>
                        <div class="ai-floating-eyebrow">AI analytical workspace</div>
                        <div class="ai-floating-title">EUSEE AI Copilot</div>
                        <div class="ai-floating-subtitle">
                            Conversation, search, chart builder, comparison, summaries, anomaly scan, and exports are consolidated here.
                        </div>
                    </div>
                </div>
                <div class="ai-context-pills">
                    <div class="ai-context-pill">{records:,} records</div>
                    <div class="ai-context-pill teal">{countries:,} countries</div>
                    <div class="ai-context-pill neutral">{years:,} years</div>
                    <div class="ai-context-pill neutral">{impacts:,} impact classes</div>
                    <div class="ai-context-pill">{role_display}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top_cols = st.columns([1, 1, 1, 5, 1])
        with top_cols[0]:
            if st.button("💬 Chat", key="ai_float_focus_chat", use_container_width=True):
                st.session_state["ai_float_default_tab"] = "chat"
        with top_cols[1]:
            if st.button("🔎 Search", key="ai_float_focus_search", use_container_width=True):
                st.session_state["ai_float_default_tab"] = "search"
        with top_cols[2]:
            if st.button("📊 Tools", key="ai_float_focus_tools", use_container_width=True):
                st.session_state["ai_float_default_tab"] = "tools"
        with top_cols[4]:
            if st.button("×", key="close_eusee_ai_panel", use_container_width=True):
                st.session_state.eusee_ai_panel_open = False
                st.rerun()

        st.markdown(
            """
            <div class="ai-tool-panel">
                <div class="ai-tool-panel-title">Unified AI workspace</div>
                <div class="ai-tool-panel-subtitle">
                    Use the tabs below to move between conversation, search, and advanced tools without leaving the floating panel.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tabs = st.tabs(["💬 Conversation", "🔎 Search", "📊 Advanced Tools"])

        with tabs[0]:
            st.markdown(
                """
                <div class="ai-tool-panel">
                    <div class="ai-tool-panel-title">💬 Conversation</div>
                    <div class="ai-tool-panel-subtitle">
                        Ask questions naturally. The existing AI Copilot engine is preserved here, including its search prompts, outputs, and internal tools.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                legacy_renderer(df)
            except Exception as exc:
                st.error(f"AI Copilot could not be rendered: {exc}")

        with tabs[1]:
            render_ai_search_panel(df)

        with tabs[2]:
            render_advanced_tools_panel(df)

    # Re-inject after legacy renderer so the floating panel overrides win.
    inject_ai_workspace_css()
