
"""
Premium EUSEE AI Copilot floating workspace.

This version keeps all chatbot functions visibly available inside one
consolidated floating panel and upgrades the look-and-feel into a cleaner,
friendlier, executive AI command-center style.

Important UX fix
----------------
The previous version rendered the new floating panel AND the original legacy
chatbot interface through legacy_renderer(df). That produced two chatbot
interfaces. This version does NOT call the full legacy renderer by default.
Instead, it provides a single native conversation area inside the floating
panel and keeps Search, Advanced Tools, Chart Builder, Compare, Summary,
Anomaly Scan, and Export Center in the same interface.

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
    """Safe docked AI workspace styles: no fixed overlay, no z-index collisions."""
    st.markdown(
        """
        <style>
        :root {
            --ai-purple: #660094;
            --ai-teal: #008CAA;
            --ai-bg: #F8FAFC;
            --ai-card: #FFFFFF;
            --ai-border: #E6E8EF;
            --ai-border-soft: #EEF0F4;
            --ai-text: #232633;
            --ai-muted: #667085;
            --ai-font: "Inter", "Segoe UI", Arial, sans-serif;
        }

        /* Important: no fixed positioning. The AI workspace now renders inside the page flow. */
        .st-key-eusee_ai_floating_launcher,
        .st-key-eusee_ai_docked_launcher {
            position: static !important;
            z-index: auto !important;
            width: 100% !important;
            margin: 10px 0 12px 0 !important;
            font-family: var(--ai-font) !important;
        }
        .st-key-eusee_ai_floating_launcher .stButton > button,
        .st-key-eusee_ai_docked_launcher .stButton > button {
            width: 100% !important;
            min-height: 44px !important;
            border-radius: 14px !important;
            border: 1px solid rgba(102,0,148,.20) !important;
            background: linear-gradient(135deg, var(--ai-purple), var(--ai-teal)) !important;
            color: #FFFFFF !important;
            font-size: 13px !important;
            font-weight: 900 !important;
            box-shadow: 0 8px 20px rgba(16,24,40,.10) !important;
        }

        .st-key-eusee_ai_floating_panel,
        .st-key-eusee_ai_docked_panel {
            position: static !important;
            z-index: auto !important;
            width: 100% !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            padding: 0 !important;
            margin: 10px 0 18px 0 !important;
            border-radius: 20px !important;
            background: #FFFFFF !important;
            border: 1px solid var(--ai-border) !important;
            box-shadow: 0 12px 30px rgba(16,24,40,.08) !important;
            font-family: var(--ai-font) !important;
        }
        .st-key-eusee_ai_floating_panel > div,
        .st-key-eusee_ai_docked_panel > div {
            height: auto !important;
            overflow: visible !important;
            padding: 0 14px 14px 14px !important;
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            border-radius: 20px !important;
        }

        .ai-floating-header {
            margin: 0 -14px 10px -14px;
            padding: 14px 16px 12px 16px;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border-bottom: 1px solid var(--ai-border-soft);
            border-radius: 20px 20px 0 0;
            font-family: var(--ai-font);
        }
        .ai-floating-header-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
        .ai-floating-eyebrow { font-size: 9px; font-weight: 950; letter-spacing: .14em; text-transform: uppercase; color: var(--ai-purple); margin-bottom: 3px; }
        .ai-floating-title { color: #23152F; font-size: 18px; font-weight: 950; line-height: 1.15; letter-spacing: -0.02em; }
        .ai-floating-subtitle { color: var(--ai-muted); font-size: 11.5px; line-height: 1.35; margin-top: 3px; max-width: 760px; }
        .ai-context-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
        .ai-context-pill { padding: 4px 8px; border-radius: 999px; background: #F4EAF8; color: var(--ai-purple); border: 1px solid #E7D4F1; font-size: 9.5px; font-weight: 850; line-height: 1; white-space: nowrap; }
        .ai-context-pill.teal { background: #EFFBFE; color: var(--ai-teal); border-color: rgba(0,140,170,.16); }
        .ai-context-pill.neutral { background: #F9FAFB; color: #344054; border-color: var(--ai-border-soft); }
        .ai-panel-status-row { display: none !important; }

        .ai-section-card, .ai-tool-panel {
            margin: 8px 0;
            padding: 11px 12px;
            border-radius: 15px;
            background: #FFFFFF;
            border: 1px solid var(--ai-border);
            box-shadow: none;
            font-family: var(--ai-font);
        }
        .ai-section-title, .ai-tool-panel-title { display: flex; align-items: center; gap: 7px; color: #23152F; font-size: 13px; font-weight: 950; margin-bottom: 2px; }
        .ai-section-subtitle, .ai-tool-panel-subtitle { color: var(--ai-muted); font-size: 10.8px; line-height: 1.35; }

        .ai-function-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin-top: 8px; }
        .ai-function-chip { background: #FCFCFD; border: 1px solid var(--ai-border-soft); border-radius: 12px; padding: 8px 9px; font-size: 10.5px; font-weight: 800; color: #344054; line-height: 1.22; min-height: 50px; }
        .ai-function-chip strong { color: var(--ai-purple); font-weight: 950; }

        .ai-mini-metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin: 8px 0; }
        .ai-mini-metric { border: 1px solid var(--ai-border-soft); background: #FCFCFD; border-radius: 12px; padding: 7px 8px; }
        .ai-mini-metric span { display: block; font-size: 8.8px; text-transform: uppercase; letter-spacing: .08em; color: var(--ai-purple); font-weight: 900; margin-bottom: 3px; }
        .ai-mini-metric strong { display: block; font-size: 14px; color: #23152F; font-weight: 900; }
        .ai-exec-card { padding: 10px 11px; border: 1px solid var(--ai-border-soft); border-left: 3px solid var(--ai-purple); border-radius: 13px; background: #FFFFFF; margin-bottom: 8px; color: #344054; font-size: 11px; line-height: 1.4; }
        .ai-exec-card strong { color: #23152F; font-weight: 900; }

        .st-key-eusee_ai_floating_panel label,
        .st-key-eusee_ai_docked_panel label,
        .st-key-eusee_ai_floating_panel .stSelectbox label,
        .st-key-eusee_ai_docked_panel .stSelectbox label,
        .st-key-eusee_ai_floating_panel .stRadio label,
        .st-key-eusee_ai_docked_panel .stRadio label,
        .st-key-eusee_ai_floating_panel .stSlider label,
        .st-key-eusee_ai_docked_panel .stSlider label,
        .st-key-eusee_ai_floating_panel .stTextInput label,
        .st-key-eusee_ai_docked_panel .stTextInput label,
        .st-key-eusee_ai_floating_panel .stTextArea label,
        .st-key-eusee_ai_docked_panel .stTextArea label,
        .st-key-eusee_ai_floating_panel .stMultiSelect label,
        .st-key-eusee_ai_docked_panel .stMultiSelect label {
            font-size: 10.5px !important;
            font-weight: 850 !important;
            color: #344054 !important;
            font-family: var(--ai-font) !important;
        }
        .st-key-eusee_ai_floating_panel [data-baseweb="select"] > div,
        .st-key-eusee_ai_docked_panel [data-baseweb="select"] > div,
        .st-key-eusee_ai_floating_panel [data-baseweb="input"],
        .st-key-eusee_ai_docked_panel [data-baseweb="input"] { border-radius: 11px !important; min-height: 34px !important; border: 1px solid #D0D5DD !important; background: #FFFFFF !important; }
        .st-key-eusee_ai_floating_panel .stButton > button,
        .st-key-eusee_ai_docked_panel .stButton > button,
        .st-key-eusee_ai_floating_panel .stDownloadButton > button,
        .st-key-eusee_ai_docked_panel .stDownloadButton > button { border-radius: 999px !important; border: 1px solid var(--ai-border) !important; background: #FFFFFF !important; color: #344054 !important; font-size: 11px !important; font-weight: 850 !important; min-height: 32px !important; box-shadow: none !important; }
        .st-key-eusee_ai_floating_panel .stButton > button:hover,
        .st-key-eusee_ai_docked_panel .stButton > button:hover,
        .st-key-eusee_ai_floating_panel .stDownloadButton > button:hover,
        .st-key-eusee_ai_docked_panel .stDownloadButton > button:hover { border-color: var(--ai-purple) !important; color: var(--ai-purple) !important; background: #FBF7FD !important; }

        div[data-testid="stChatMessage"] { border-radius: 14px !important; border: 1px solid var(--ai-border-soft) !important; background: #FFFFFF !important; box-shadow: none !important; padding: 8px 10px !important; margin-bottom: 7px !important; }
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] div { font-family: var(--ai-font) !important; font-size: 12px !important; line-height: 1.45 !important; color: #344054; }
        textarea { font-family: var(--ai-font) !important; font-size: 12px !important; }

        .st-key-eusee_ai_floating_panel div[data-testid="stExpander"],
        .st-key-eusee_ai_docked_panel div[data-testid="stExpander"] {
            border-radius: 15px !important;
            border: 1px solid var(--ai-border) !important;
            box-shadow: none !important;
            overflow: hidden !important;
            background: #FFFFFF !important;
            margin-bottom: 7px !important;
        }
        .st-key-eusee_ai_floating_panel div[data-testid="stExpander"] summary,
        .st-key-eusee_ai_docked_panel div[data-testid="stExpander"] summary {
            background: #FFFFFF !important;
            border-bottom: 1px solid var(--ai-border-soft) !important;
            padding: 10px 12px !important;
            font-size: 12px !important;
            font-weight: 900 !important;
            color: #23152F !important;
        }
        div[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--ai-border-soft) !important; box-shadow: none !important; }
        div[data-testid="stPlotlyChart"] { border-radius: 14px !important; overflow: hidden !important; border: 1px solid var(--ai-border-soft) !important; box-shadow: none !important; background: #FFFFFF !important; margin-top: 8px !important; }

        @media (max-width: 760px) {
            .ai-function-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
            .ai-mini-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
            .ai-floating-title { font-size: 16px !important; }
            .ai-floating-subtitle { font-size: 10.5px !important; }
        }
        @media (max-width: 520px) {
            .ai-function-grid { grid-template-columns: 1fr !important; }
            .ai-mini-metric-grid { grid-template-columns: 1fr !important; }
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
                Compact workspace: chat first, then search, then tools only when opened.
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
# Single-interface conversation panel
# ---------------------------------------------------------------------
def _top_values_text(df: pd.DataFrame, column: str, limit: int = 5) -> str:
    if column not in df.columns or df.empty:
        return "not available"
    vals = df[column].dropna().astype(str).str.strip()
    vals = vals[~vals.str.lower().isin(["", "nan", "none"])]
    if vals.empty:
        return "not available"
    counts = vals.value_counts().head(limit)
    return ", ".join([f"{idx} ({val:,})" for idx, val in counts.items()])


def _generate_dashboard_response(prompt: str, df: pd.DataFrame | None) -> str:
    """Generate a lightweight dashboard-grounded response without rendering the legacy UI."""
    prompt_l = (prompt or "").lower()

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "The active filtered dataset is empty, so I cannot summarize or analyze records yet. Adjust the dashboard filters and try again."

    records = len(df)
    countries = _nunique(df, "alert-country")
    years = _nunique(df, "year")
    impacts = _nunique(df, "alert-impact")

    if any(k in prompt_l for k in ["compare", "country", "countries"]):
        return (
            f"**Country comparison snapshot**\n\n"
            f"The active view contains **{records:,} records** across **{countries:,} countries**. "
            f"The highest-volume countries are: {_top_values_text(df, 'alert-country', 6)}.\n\n"
            f"Use the **Country Compare** section below for a visual comparison by alert impact, alert type, actor, region, or year."
        )

    if any(k in prompt_l for k in ["actor", "repression", "mechanism", "restriction"]):
        actor_text = _top_values_text(df, "Actor of repression", 5)
        mechanism_text = _top_values_text(df, "Restrictive mechanism", 5)
        return (
            f"**Restriction pattern snapshot**\n\n"
            f"The leading actors in the active view are: {actor_text}.\n\n"
            f"The leading restrictive mechanisms are: {mechanism_text}.\n\n"
            f"For deeper review, use **Search** for specific actors/mechanisms or **Anomaly Scan** to identify unusual concentrations."
        )

    if any(k in prompt_l for k in ["chart", "plot", "visual", "graph"]):
        return (
            "**Chart guidance**\n\n"
            "Use the **Chart Builder** section below to create a chart from the current filtered data. "
            "Recommended starting choices: group by `alert-country`, `alert-impact`, `alert-type`, `Actor of repression`, or `year`, with metric set to `Record count`."
        )

    if any(k in prompt_l for k in ["export", "download", "brief", "report"]):
        return (
            "**Export guidance**\n\n"
            "Use the **Export Center** below to download the filtered dataset as CSV/XLSX. "
            "Use **Executive Summary** first if you want a compact evidence summary before exporting."
        )

    return (
        f"**Executive summary of the active filtered view**\n\n"
        f"- Records: **{records:,}**\n"
        f"- Countries: **{countries:,}**\n"
        f"- Years represented: **{years:,}**\n"
        f"- Alert-impact classes: **{impacts:,}**\n"
        f"- Top countries: {_top_values_text(df, 'alert-country', 5)}\n"
        f"- Top alert impacts: {_top_values_text(df, 'alert-impact', 4)}\n"
        f"- Top actors: {_top_values_text(df, 'Actor of repression', 4)}\n\n"
        f"Use Search for evidence lookup, Chart Builder for visualization, Country Compare for side-by-side analysis, and Anomaly Scan for concentration detection."
    )


def render_ai_conversation_panel(df: pd.DataFrame | None) -> None:
    """Render the single native chatbot interface. This prevents the old duplicate UI."""
    st.markdown(
        '<div class="ai-section-card"><div class="ai-section-title">💬 Conversation</div><div class="ai-section-subtitle">Ask naturally. This single conversation area replaces the nested legacy chatbot UI, so only one chatbot interface is shown.</div></div>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("eusee_ai_messages", [
        {
            "role": "assistant",
            "content": "Welcome. I can summarize the active filtered view, compare countries, explain restriction patterns, guide chart building, and help prepare exports. Use the tools below for visual outputs and evidence lookup.",
        }
    ])

    with st.container():
        for msg in st.session_state.eusee_ai_messages[-5:]:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

    suggested = st.columns(2)
    suggestions = [
        "Summarize this filtered view",
        "Compare top countries",
        "Show restriction patterns",
        "What chart should I build?",
    ]
    for i, text in enumerate(suggestions):
        with suggested[i % 2]:
            if st.button(text, key=f"ai_suggest_{i}", use_container_width=True):
                st.session_state.eusee_ai_messages.append({"role": "user", "content": text})
                st.session_state.eusee_ai_messages.append({"role": "assistant", "content": _generate_dashboard_response(text, df)})
                st.rerun()

    input_col, send_col, clear_col = st.columns([5, 1.1, 1.1])
    with input_col:
        prompt = st.text_area(
            "Ask EUSEE AI Copilot",
            placeholder="Example: summarize this filtered view, compare top countries, show restriction patterns...",
            key="eusee_single_ai_text_area",
            height=76,
            label_visibility="collapsed",
        )
    with send_col:
        st.write("")
        send_clicked = st.button("Send", key="eusee_ai_send_message", use_container_width=True)
    with clear_col:
        st.write("")
        clear_clicked = st.button("Clear", key="eusee_ai_clear_chat", use_container_width=True)

    if send_clicked and prompt.strip():
        st.session_state.eusee_ai_messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.eusee_ai_messages.append({"role": "assistant", "content": _generate_dashboard_response(prompt.strip(), df)})
        st.session_state.eusee_single_ai_text_area = ""
        st.rerun()

    if clear_clicked:
        st.session_state.eusee_ai_messages = [
            {"role": "assistant", "content": "Chat cleared. Ask a new question about the active dashboard view."}
        ]
        st.rerun()

    st.caption("The chat input is embedded in the panel, not fixed to the browser bottom, so it will not overlay tools or charts.")

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

    st.dataframe(results[preview_cols], use_container_width=True, hide_index=True, height=min(300, max(180, 30 * len(results) + 38)))
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
    fig.update_layout(margin=dict(l=20, r=20, t=42, b=30), height=330, font=dict(family="Inter, Segoe UI, Arial", size=12), title=f"{y_label} by {group_col}")
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
    fig.update_layout(margin=dict(l=20, r=20, t=42, b=30), height=340, font=dict(family="Inter, Segoe UI, Arial", size=12))
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
    fig.update_layout(margin=dict(l=20, r=20, t=42, b=30), height=330, font=dict(family="Inter, Segoe UI, Arial", size=12))
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
    st.markdown('<div class="ai-tool-panel"><div class="ai-tool-panel-title">📊 Advanced AI tools</div><div class="ai-tool-panel-subtitle">Tools stay collapsed to reduce scrolling. Open only the tool you need.</div></div>', unsafe_allow_html=True)
    with st.expander("📊 Chart Builder", expanded=False):
        _render_chart_builder(df)
    with st.expander("🌍 Country Compare", expanded=False):
        _render_country_compare(df)
    with st.expander("🧾 Executive Summary", expanded=False):
        _render_executive_summary(df)
    with st.expander("⚠️ Anomaly Scan", expanded=False):
        _render_anomaly_scan(df)
    with st.expander("⬇️ Export Center", expanded=False):
        _render_export_center(df)


# ---------------------------------------------------------------------
# Main floating workspace
# ---------------------------------------------------------------------
def render_ai_workspace(
    df: pd.DataFrame | None,
    legacy_renderer: Callable[[pd.DataFrame | None], Any] | None = None,
    current_role: str = "guest",
    current_email: str = "",
) -> None:
    """
    Render AI Copilot as a docked, in-page workspace.

    This intentionally avoids fixed/floating overlays because they were causing
    controls, charts, search, and chat inputs to overlap each other. The function
    name is preserved so app.py does not need to change.
    """
    inject_ai_workspace_css()
    st.session_state.setdefault("eusee_ai_panel_open", False)

    records = _record_count(df)
    countries = _nunique(df, "alert-country")
    years = _nunique(df, "year")
    impacts = _nunique(df, "alert-impact")
    role_display = str(current_role or "guest").replace("_", " ").title()

    if not st.session_state.get("eusee_ai_panel_open", False):
        with st.container(key="eusee_ai_docked_launcher"):
            if st.button("✨ Open EUSEE AI Copilot Workspace", key="open_eusee_ai_panel", use_container_width=True):
                st.session_state.eusee_ai_panel_open = True
                st.rerun()
        return

    with st.container(key="eusee_ai_docked_panel"):
        st.markdown(
            f"""
            <div class="ai-floating-header">
                <div class="ai-floating-header-row">
                    <div>
                        <div class="ai-floating-eyebrow">Docked AI analytical workspace</div>
                        <div class="ai-floating-title">EUSEE AI Copilot</div>
                        <div class="ai-floating-subtitle">
                            Clean in-page layout. No floating overlay, no nested full-height scroll, and no hidden controls.
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

        top_cols = st.columns([1, 1, 1, 1, 3, 1])
        with top_cols[0]:
            st.caption("💬 Chat")
        with top_cols[1]:
            st.caption("🔎 Search")
        with top_cols[2]:
            st.caption("📊 Tools")
        with top_cols[3]:
            st.caption("⬇️ Export")
        with top_cols[5]:
            if st.button("Close", key="close_eusee_ai_panel", use_container_width=True):
                st.session_state.eusee_ai_panel_open = False
                st.rerun()

        render_visible_function_overview()

        chat_tab, search_tab, tools_tab, export_tab = st.tabs([
            "💬 Chat",
            "🔎 Search",
            "📊 Analyze",
            "⬇️ Export",
        ])

        with chat_tab:
            render_ai_conversation_panel(df)

        with search_tab:
            render_ai_search_panel(df)

        with tools_tab:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                st.info("Advanced tools are unavailable because the filtered dataset is empty.")
            else:
                tool = st.radio(
                    "Choose one tool",
                    ["Chart Builder", "Country Compare", "Executive Summary", "Anomaly Scan"],
                    horizontal=True,
                    key="eusee_ai_selected_tool",
                )
                if tool == "Chart Builder":
                    _render_chart_builder(df)
                elif tool == "Country Compare":
                    _render_country_compare(df)
                elif tool == "Executive Summary":
                    _render_executive_summary(df)
                elif tool == "Anomaly Scan":
                    _render_anomaly_scan(df)

        with export_tab:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                st.info("Export is unavailable because the filtered dataset is empty.")
            else:
                _render_export_center(df)

    inject_ai_workspace_css()
