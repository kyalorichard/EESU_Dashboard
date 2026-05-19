
"""
Premium EUSEE AI Copilot floating workspace.

This version makes all chatbot functions visibly available inside one
consolidated floating panel. It avoids hiding Search and Advanced Tools behind
Streamlit tabs, so users can immediately see the available functions after
opening the AI Copilot.

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
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return 0
    return int(df[column].dropna().nunique())


def _record_count(df: pd.DataFrame | None) -> int:
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
        try:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or col in ["year"]:
                cols.append(col)
        except Exception:
            continue
    preferred = [
        "alert-country",
        "region",
        "continent",
        "alert-impact",
        "alert-type",
        "Actor of repression",
        "Civil society actor affected",
        "Restrictive mechanism",
        "year",
        "month_name",
    ]
    return [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]


def _numeric_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    cols: list[str] = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        except Exception:
            continue
    return cols


def _download_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered data")
    return buffer.getvalue()


def _short_text(value: object, max_len: int = 90) -> str:
    text = str(value) if value is not None else ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------
def inject_ai_workspace_css() -> None:
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

        .st-key-eusee_ai_floating_launcher {
            position: fixed !important;
            right: 22px !important;
            bottom: 22px !important;
            z-index: 2147482300 !important;
            width: min(360px, calc(100vw - 32px)) !important;
            font-family: var(--ai-font) !important;
        }
        .st-key-eusee_ai_floating_launcher .stButton > button {
            width: 100% !important;
            min-height: 54px !important;
            border-radius: 999px !important;
            border: 1px solid rgba(102,0,148,.22) !important;
            background: linear-gradient(135deg, #660094 0%, #008CAA 100%) !important;
            color: #FFFFFF !important;
            font-size: 13px !important;
            font-weight: 950 !important;
            box-shadow: 0 18px 38px rgba(16,24,40,.24) !important;
        }

        .st-key-eusee_ai_floating_panel {
            position: fixed !important;
            right: 22px !important;
            bottom: 22px !important;
            width: min(760px, calc(100vw - 36px)) !important;
            height: min(88vh, 900px) !important;
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
            padding: 0 16px 18px 16px !important;
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        }
        .st-key-eusee_ai_floating_panel > div::-webkit-scrollbar { width: 8px !important; }
        .st-key-eusee_ai_floating_panel > div::-webkit-scrollbar-thumb { background: #D0D5DD !important; border-radius: 999px !important; }
        .st-key-eusee_ai_floating_panel > div::-webkit-scrollbar-track { background: #F9FAFB !important; }

        .ai-floating-header {
            position: sticky;
            top: 0;
            z-index: 50;
            margin: 0 -16px 12px -16px;
            padding: 14px 16px 12px 16px;
            background: rgba(255,255,255,.97);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-bottom: 1px solid var(--ai-border-soft);
            border-radius: 24px 24px 0 0;
            font-family: var(--ai-font);
        }
        .ai-floating-header-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
        .ai-floating-eyebrow { font-size: 9.5px; font-weight: 950; letter-spacing: .14em; text-transform: uppercase; color: var(--ai-purple); margin-bottom: 4px; }
        .ai-floating-title { color: #23152F; font-size: 18px; font-weight: 950; line-height: 1.12; letter-spacing: -0.02em; }
        .ai-floating-subtitle { color: var(--ai-muted); font-size: 11.5px; line-height: 1.42; margin-top: 4px; max-width: 510px; }
        .ai-context-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
        .ai-context-pill { padding: 5px 8px; border-radius: 999px; background: #F4EAF8; color: var(--ai-purple); border: 1px solid #E7D4F1; font-size: 10px; font-weight: 850; line-height: 1; white-space: nowrap; }
        .ai-context-pill.teal { background: #EFFBFE; color: var(--ai-teal); border-color: rgba(0,140,170,.16); }
        .ai-context-pill.neutral { background: #F9FAFB; color: #344054; border-color: var(--ai-border-soft); }

        .ai-section-card, .ai-tool-panel {
            margin: 10px 0 12px 0;
            padding: 13px;
            border-radius: 18px;
            background: #FFFFFF;
            border: 1px solid var(--ai-border);
            box-shadow: var(--ai-shadow-soft);
            font-family: var(--ai-font);
        }
        .ai-section-title, .ai-tool-panel-title { display: flex; align-items: center; gap: 8px; color: #23152F; font-size: 14px; font-weight: 950; margin-bottom: 4px; }
        .ai-section-subtitle, .ai-tool-panel-subtitle { color: var(--ai-muted); font-size: 11.5px; line-height: 1.45; }

        .ai-function-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
        .ai-function-chip { background: #FCFCFD; border: 1px solid var(--ai-border-soft); border-radius: 14px; padding: 9px 10px; font-size: 11px; font-weight: 850; color: #344054; line-height: 1.25; }
        .ai-function-chip strong { color: var(--ai-purple); font-weight: 950; }

        .ai-mini-metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 10px 0 12px 0; }
        .ai-mini-metric { border: 1px solid var(--ai-border-soft); background: #FCFCFD; border-radius: 14px; padding: 9px 10px; }
        .ai-mini-metric span { display: block; font-size: 9.5px; text-transform: uppercase; letter-spacing: .08em; color: var(--ai-purple); font-weight: 900; margin-bottom: 4px; }
        .ai-mini-metric strong { display: block; font-size: 16px; color: #23152F; font-weight: 900; }

        .ai-exec-card { padding: 12px 13px; border: 1px solid var(--ai-border-soft); border-left: 4px solid var(--ai-purple); border-radius: 15px; background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%); box-shadow: 0 4px 12px rgba(16,24,40,.04); margin-bottom: 10px; color: #344054; font-size: 12px; line-height: 1.5; }
        .ai-exec-card strong { color: #23152F; font-weight: 900; }

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
        .st-key-eusee_ai_floating_panel [data-baseweb="input"] { border-radius: 12px !important; min-height: 38px !important; border: 1px solid #D0D5DD !important; background: #FFFFFF !important; }
        .st-key-eusee_ai_floating_panel .stButton > button,
        .st-key-eusee_ai_floating_panel .stDownloadButton > button { border-radius: 999px !important; border: 1px solid var(--ai-border) !important; background: #FFFFFF !important; color: #344054 !important; font-size: 12px !important; font-weight: 850 !important; min-height: 36px !important; box-shadow: 0 2px 8px rgba(16,24,40,.035) !important; transition: all .16s ease !important; }
        .st-key-eusee_ai_floating_panel .stButton > button:hover,
        .st-key-eusee_ai_floating_panel .stDownloadButton > button:hover { border-color: var(--ai-purple) !important; color: var(--ai-purple) !important; background: #FBF7FD !important; transform: translateY(-1px) !important; }
        div[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; border: 1px solid var(--ai-border-soft) !important; box-shadow: 0 4px 12px rgba(16,24,40,.04) !important; }
        div[data-testid="stPlotlyChart"] { border-radius: 16px !important; overflow: hidden !important; border: 1px solid var(--ai-border-soft) !important; box-shadow: 0 6px 16px rgba(16,24,40,.045) !important; background: #FFFFFF !important; margin-top: 10px !important; }

        .ai-min-shell, .ai-min-panel, .ai-min-chat, .ai-min-output, .ai-min-tools, .ai-min-suggestions { font-family: var(--ai-font) !important; }
        .ai-min-panel, .ai-min-shell { border-radius: 18px !important; border: 1px solid var(--ai-border) !important; background: linear-gradient(180deg, #FFFFFF 0%, #FCFCFD 100%) !important; box-shadow: 0 8px 18px rgba(16,24,40,.06) !important; overflow: hidden !important; }
        .ai-min-header, .ai-min-topbar { border-bottom: 1px solid var(--ai-border-soft) !important; background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important; padding: 12px 14px !important; }
        .ai-min-title, .ai-min-heading, .ai-min-header-title { font-family: var(--ai-font) !important; font-size: 16px !important; font-weight: 900 !important; color: #23152F !important; }
        .ai-min-subtitle, .ai-min-note, .ai-min-caption, .ai-min-meta { font-family: var(--ai-font) !important; font-size: 11px !important; color: var(--ai-muted) !important; line-height: 1.45 !important; }
        .ai-min-chat { border: 1px solid var(--ai-border-soft) !important; border-radius: 16px !important; padding: 12px !important; min-height: 280px !important; max-height: 46vh !important; overflow-y: auto !important; scroll-behavior: smooth !important; background: #FFFFFF !important; }
        .ai-min-input { position: sticky !important; bottom: 0 !important; z-index: 20 !important; background: linear-gradient(180deg, rgba(255,255,255,.72) 0%, #FFFFFF 45%) !important; border-top: 1px solid var(--ai-border-soft) !important; padding-top: 10px !important; margin-top: 10px !important; }
        div[data-testid="stChatMessage"] { border-radius: 18px !important; border: 1px solid var(--ai-border-soft) !important; background: #FFFFFF !important; box-shadow: 0 4px 12px rgba(16,24,40,.04) !important; padding: 10px 12px !important; margin-bottom: 10px !important; }
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] div { font-family: var(--ai-font) !important; font-size: 12.8px !important; line-height: 1.55 !important; color: #344054; }
        div[data-testid="stChatInput"] { background: #FFFFFF !important; border-radius: 24px !important; border: 1px solid #D0D5DD !important; box-shadow: 0 8px 22px rgba(16,24,40,.08) !important; overflow: hidden !important; }
        div[data-testid="stChatInput"] textarea { font-family: var(--ai-font) !important; font-size: 13px !important; color: #232633 !important; }

        @media (max-width: 760px) {
            .st-key-eusee_ai_floating_panel { right: 10px !important; bottom: 10px !important; left: 10px !important; width: auto !important; height: calc(100vh - 20px) !important; max-height: calc(100vh - 20px) !important; border-radius: 20px !important; }
            .st-key-eusee_ai_floating_launcher { right: 12px !important; bottom: 12px !important; width: calc(100vw - 24px) !important; }
            .ai-floating-header { border-radius: 20px 20px 0 0 !important; }
            .ai-function-grid { grid-template-columns: 1fr !important; }
            .ai-mini-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
        }
        @media (max-width: 520px) {
            .ai-mini-metric-grid { grid-template-columns: 1fr !important; }
            .ai-floating-title { font-size: 16px !important; }
            .ai-floating-subtitle { font-size: 10.8px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Visible function overview
# ---------------------------------------------------------------------
def render_visible_function_overview() -> None:
    st.markdown(
        """
        <div class="ai-section-card">
            <div class="ai-section-title">🧭 Available AI functions</div>
            <div class="ai-section-subtitle">
                All chatbot functions are visible in this floating panel. Scroll within the panel to use Chat, Search, and Advanced Tools together.
            </div>
            <div class="ai-function-grid">
                <div class="ai-function-chip"><strong>Chat</strong><br/>Ask questions and use the existing AI Copilot engine.</div>
                <div class="ai-function-chip"><strong>Search</strong><br/>Search records, countries, actors, impacts, and restriction terms.</div>
                <div class="ai-function-chip"><strong>Chart Builder</strong><br/>Create quick charts from active filtered data.</div>
                <div class="ai-function-chip"><strong>Compare</strong><br/>Compare selected countries by impact, actor, year, or alert type.</div>
                <div class="ai-function-chip"><strong>Executive Summary</strong><br/>Generate compact briefing bullets from the current view.</div>
                <div class="ai-function-chip"><strong>Anomaly Scan</strong><br/>Detect high-concentration patterns using z-score thresholds.</div>
                <div class="ai-function-chip"><strong>Export Center</strong><br/>Download filtered data and generated outputs.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Search panel
# ---------------------------------------------------------------------
def render_ai_search_panel(df: pd.DataFrame | None) -> None:
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
    ] or search_cols[: min(6, len(search_cols))]

    q_col, s_col, n_col = st.columns([2.2, 1.6, 0.8])
    with q_col:
        query = st.text_input(
            "Search",
            placeholder="Example: Kenya, incarceration, civic rights, negative alerts...",
            key="ai_search_query_visible",
        )
    with s_col:
        scope = st.multiselect(
            "Search scope",
            options=search_cols,
            default=default_scope,
            key="ai_search_scope_visible",
        )
    with n_col:
        max_results = st.selectbox(
            "Results",
            options=[10, 25, 50, 100],
            index=1,
            key="ai_search_max_results_visible",
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
        lambda row: row.str.lower().str.contains(query_clean, na=False).any(), axis=1
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
    ] or available_scope[: min(6, len(available_scope))]

    st.dataframe(results[preview_cols], use_container_width=True, hide_index=True, height=min(420, max(220, 34 * len(results) + 42)))
    st.download_button(
        "⬇️ Download search results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="eusee_ai_search_results.csv",
        mime="text/csv",
        use_container_width=True,
        key="ai_search_download_csv_visible",
    )


# ---------------------------------------------------------------------
# Advanced tools
# ---------------------------------------------------------------------
def _render_chart_builder(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-exec-card"><strong>Chart Builder:</strong> create quick dashboard-grounded visualizations from the active filtered data.</div>', unsafe_allow_html=True)
    cat_cols = _categorical_columns(df)
    num_cols = _numeric_columns(df)
    if not cat_cols:
        st.info("No categorical columns are available for chart building.")
        return

    c1, c2, c3 = st.columns([1.3, 1.2, 1.1])
    with c1:
        group_col = st.selectbox("Group by", cat_cols, index=0, key="ai_chart_group_visible")
    with c2:
        metric_mode = st.selectbox("Metric", ["Record count"] + num_cols, index=0, key="ai_chart_metric_visible")
    with c3:
        chart_type = st.selectbox("Chart type", ["Bar", "Horizontal bar", "Line", "Area", "Treemap"], index=0, key="ai_chart_type_visible")
    top_n = st.slider("Top groups", min_value=5, max_value=30, value=12, step=1, key="ai_chart_top_n_visible")

    if metric_mode == "Record count":
        plot_df = df.groupby(group_col, dropna=False).size().reset_index(name="value").sort_values("value", ascending=False).head(top_n)
        y_label = "Records"
    else:
        plot_df = df.groupby(group_col, dropna=False)[metric_mode].sum(numeric_only=True).reset_index(name="value").sort_values("value", ascending=False).head(top_n)
        y_label = metric_mode
    plot_df[group_col] = plot_df[group_col].astype(str).map(lambda x: _short_text(x, 40))

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
    fig.update_layout(margin=dict(l=20, r=20, t=42, b=30), height=420, font=dict(family="Inter, Segoe UI, Arial", size=12), title=f"{y_label} by {group_col}")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("⬇️ Download chart data as CSV", data=plot_df.to_csv(index=False).encode("utf-8"), file_name="eusee_ai_chart_data.csv", mime="text/csv", use_container_width=True, key="ai_chart_download_visible")


def _render_country_compare(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-exec-card"><strong>Country Compare:</strong> compare selected countries across alert impacts, alert types, and key dashboard categories.</div>', unsafe_allow_html=True)
    if "alert-country" not in df.columns:
        st.info("Country comparison requires an 'alert-country' column.")
        return
    countries = sorted(df["alert-country"].dropna().astype(str).unique().tolist())
    selected = st.multiselect("Countries to compare", options=countries, default=countries[: min(3, len(countries))], key="ai_compare_countries_visible")
    compare_by_options = [c for c in ["alert-impact", "alert-type", "region", "Actor of repression", "year"] if c in df.columns]
    if not compare_by_options:
        st.info("No comparison category columns are available.")
        return
    compare_by = st.selectbox("Compare by", compare_by_options, index=0, key="ai_compare_by_visible")
    if not selected:
        st.warning("Select at least one country.")
        return
    subset = df[df["alert-country"].astype(str).isin(selected)].copy()
    if subset.empty:
        st.info("No records found for the selected countries.")
        return
    comp = subset.groupby(["alert-country", compare_by], dropna=False).size().reset_index(name="records")
    comp[compare_by] = comp[compare_by].astype(str).map(lambda x: _short_text(x, 45))
    fig = px.bar(comp, x="alert-country", y="records", color=compare_by, barmode="group", title=f"Country comparison by {compare_by}")
    fig.update_layout(margin=dict(l=20, r=20, t=42, b=30), height=430, font=dict(family="Inter, Segoe UI, Arial", size=12))
    st.plotly_chart(fig, use_container_width=True)
    summary = subset.groupby("alert-country", dropna=False).size().reset_index(name="total_records").sort_values("total_records", ascending=False)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_executive_summary(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-exec-card"><strong>Executive Summary:</strong> generate a compact, evidence-based summary from the active filtered data.</div>', unsafe_allow_html=True)
    style = st.radio("Summary style", ["Executive", "Policy", "Donor", "Technical"], horizontal=True, key="ai_summary_style_visible")
    bullets = [f"The current filtered view contains {len(df):,} records across {_nunique(df, 'alert-country'):,} countries and {_nunique(df, 'year'):,} year(s)."]
    if "alert-impact" in df.columns:
        impact_text = ", ".join([f"{idx}: {val:,}" for idx, val in df["alert-impact"].astype(str).value_counts().head(3).items()])
        bullets.append(f"The leading alert-impact categories are {impact_text}.")
    if "alert-country" in df.columns:
        country_text = ", ".join([f"{idx} ({val:,})" for idx, val in df["alert-country"].astype(str).value_counts().head(5).items()])
        bullets.append(f"The highest-volume countries in this filtered view are {country_text}.")
    if "Actor of repression" in df.columns:
        actor_text = ", ".join([f"{idx} ({val:,})" for idx, val in df["Actor of repression"].astype(str).value_counts().head(3).items()])
        bullets.append(f"The most frequently recorded actors are {actor_text}.")
    if "year" in df.columns:
        year_text = ", ".join([f"{idx}: {val:,}" for idx, val in df["year"].dropna().astype(str).value_counts().head(3).items()])
        bullets.append(f"The strongest year-level concentrations are {year_text}.")
    st.markdown(f"#### {style} summary")
    for item in bullets:
        st.markdown(f"- {item}")
    summary_text = f"{style} summary\n\n" + "\n".join([f"- {b}" for b in bullets])
    st.download_button("⬇️ Download summary as TXT", data=summary_text.encode("utf-8"), file_name="eusee_ai_executive_summary.txt", mime="text/plain", use_container_width=True, key="ai_summary_download_visible")


def _render_anomaly_scan(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-exec-card"><strong>Anomaly Scan:</strong> identify unusually high concentrations in the selected category using z-score style outlier detection.</div>', unsafe_allow_html=True)
    cat_cols = _categorical_columns(df)
    if not cat_cols:
        st.info("No categorical columns are available for anomaly scanning.")
        return
    c1, c2 = st.columns([1.3, 1])
    with c1:
        scan_col = st.selectbox("Scan category", cat_cols, index=0, key="ai_anomaly_scan_col_visible")
    with c2:
        sensitivity = st.slider("Sensitivity", 1.0, 3.0, 1.5, 0.1, key="ai_anomaly_sensitivity_visible")
    counts = df[scan_col].astype(str).value_counts().rename_axis(scan_col).reset_index(name="records")
    if counts.empty or counts["records"].std(ddof=0) == 0:
        st.info("No statistical anomalies detected for the selected category.")
        st.dataframe(counts.head(20), use_container_width=True, hide_index=True)
        return
    mean = counts["records"].mean()
    std = counts["records"].std(ddof=0)
    counts["z_score"] = (counts["records"] - mean) / std
    anomalies = counts[counts["z_score"] >= sensitivity].sort_values("z_score", ascending=False)
    st.markdown(f'<div class="ai-mini-metric-grid"><div class="ai-mini-metric"><span>Groups scanned</span><strong>{len(counts):,}</strong></div><div class="ai-mini-metric"><span>Anomalies</span><strong>{len(anomalies):,}</strong></div><div class="ai-mini-metric"><span>Mean records</span><strong>{mean:.1f}</strong></div><div class="ai-mini-metric"><span>Threshold</span><strong>{sensitivity:.1f}σ</strong></div></div>', unsafe_allow_html=True)
    chart_counts = counts.head(20).copy()
    chart_counts[scan_col] = chart_counts[scan_col].map(lambda x: _short_text(x, 45))
    fig = px.bar(chart_counts, x=scan_col, y="records", title=f"Top concentrations by {scan_col}")
    fig.update_layout(margin=dict(l=20, r=20, t=42, b=30), height=420, font=dict(family="Inter, Segoe UI, Arial", size=12))
    st.plotly_chart(fig, use_container_width=True)
    if anomalies.empty:
        st.success("No high-concentration anomalies exceed the selected threshold.")
    else:
        st.dataframe(anomalies, use_container_width=True, hide_index=True)


def _render_export_center(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-exec-card"><strong>Export Center:</strong> download the currently filtered AI workspace data and summaries.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download filtered data as CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="eusee_ai_filtered_data.csv", mime="text/csv", use_container_width=True, key="ai_export_csv_visible")
    with c2:
        try:
            xlsx_bytes = _download_excel_bytes(df)
            st.download_button("⬇️ Download filtered data as XLSX", data=xlsx_bytes, file_name="eusee_ai_filtered_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="ai_export_xlsx_visible")
        except Exception:
            st.caption("XLSX export requires xlsxwriter. CSV export is available.")


def render_advanced_tools_panel(df: pd.DataFrame | None) -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Advanced tools are unavailable because the filtered dataset is empty.")
        return
    st.markdown('<div class="ai-tool-panel"><div class="ai-tool-panel-title">📊 Advanced AI tools</div><div class="ai-tool-panel-subtitle">All tools are directly visible below. Expand each tool to use it inside the floating AI panel.</div></div>', unsafe_allow_html=True)
    with st.expander("📊 Chart Builder", expanded=True):
        _render_chart_builder(df)
    with st.expander("🌍 Country Compare", expanded=True):
        _render_country_compare(df)
    with st.expander("🧾 Executive Summary", expanded=True):
        _render_executive_summary(df)
    with st.expander("⚠️ Anomaly Scan", expanded=True):
        _render_anomaly_scan(df)
    with st.expander("⬇️ Export Center", expanded=True):
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

    Unlike the prior version, this does not rely on Streamlit tabs to hide Search
    and Advanced Tools. All functions are visible in stacked sections inside the
    same floating panel after opening the launcher.
    """
    inject_ai_workspace_css()
    st.session_state.setdefault("eusee_ai_panel_open", False)

    records = _record_count(df)
    countries = _nunique(df, "alert-country")
    years = _nunique(df, "year")
    impacts = _nunique(df, "alert-impact")
    role_display = str(current_role or "guest").replace("_", " ").title()

    if not st.session_state.get("eusee_ai_panel_open", False):
        with st.container(key="eusee_ai_floating_launcher"):
            if st.button("💬 Open EUSEE AI Copilot — Chat · Search · Tools", key="open_eusee_ai_panel", use_container_width=True):
                st.session_state.eusee_ai_panel_open = True
                st.rerun()
        return

    with st.container(key="eusee_ai_floating_panel"):
        st.markdown(
            f"""
            <div class="ai-floating-header">
                <div class="ai-floating-header-row">
                    <div>
                        <div class="ai-floating-eyebrow">AI analytical workspace</div>
                        <div class="ai-floating-title">EUSEE AI Copilot</div>
                        <div class="ai-floating-subtitle">
                            Chatbot, search, chart builder, country comparison, summaries, anomaly scan, and exports are all visible together here.
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

        close_cols = st.columns([1, 1, 1, 4, 1])
        with close_cols[0]:
            st.caption("💬 Chat")
        with close_cols[1]:
            st.caption("🔎 Search")
        with close_cols[2]:
            st.caption("📊 Tools")
        with close_cols[4]:
            if st.button("×", key="close_eusee_ai_panel", use_container_width=True):
                st.session_state.eusee_ai_panel_open = False
                st.rerun()

        render_visible_function_overview()

        st.markdown('<div class="ai-section-card"><div class="ai-section-title">💬 Conversation</div><div class="ai-section-subtitle">Your existing AI Copilot engine appears below. It keeps its original logic, chat input, outputs, and internal controls.</div></div>', unsafe_allow_html=True)
        try:
            legacy_renderer(df)
        except Exception as exc:
            st.error(f"AI Copilot could not be rendered: {exc}")

        render_ai_search_panel(df)
        render_advanced_tools_panel(df)

    inject_ai_workspace_css()
