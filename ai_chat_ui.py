"""
Premium EUSEE AI Copilot workspace wrapper.

Purpose
-------
This module gives the existing EUSEE AI Copilot a friendlier, cleaner,
ChatGPT-style professional UX without changing the underlying AI logic.

How it works
------------
- app.py still owns data loading, permissions, filters, charts, and the original
  render_ai_assistant_panel(df) function.
- This module wraps that legacy renderer with a standardized AI workspace shell,
  context metrics, typography, spacing, and responsive CSS overrides.
- If anything fails, the wrapper safely falls back to the legacy renderer.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


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

        /* Workspace header shell */
        .ai-workspace-shell {
            margin-top: 18px;
            margin-bottom: 22px;
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
            max-width: 420px;
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

        .ai-min-chat::-webkit-scrollbar {
            width: 8px;
        }
        .ai-min-chat::-webkit-scrollbar-thumb {
            background: #D0D5DD;
            border-radius: 999px;
        }
        .ai-min-chat::-webkit-scrollbar-track {
            background: #F9FAFB;
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

        /* Streamlit chat message polish */
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

        div[data-testid="stChatMessage"] code {
            font-size: 12px !important;
            border-radius: 6px !important;
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
        .ai-min-input .stButton > button {
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
        .ai-min-input .stButton > button:hover {
            border-color: var(--ai-purple) !important;
            color: var(--ai-purple) !important;
            background: #FBF7FD !important;
            transform: translateY(-1px) !important;
        }

        .ai-min-tools {
            margin-top: 12px !important;
        }

        .ai-min-tools div[data-testid="stExpander"] {
            border-radius: 18px !important;
            border: 1px solid var(--ai-border-soft) !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }

        .ai-min-tools div[data-testid="stExpander"] summary {
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

        /* Make tabs, radios, selects inside the Copilot compact and consistent. */
        .ai-min-tools [data-baseweb="tab-list"] {
            gap: 6px !important;
        }

        .ai-min-tools [data-baseweb="tab"] {
            font-size: 12px !important;
            font-weight: 850 !important;
            border-radius: 999px !important;
            padding: 6px 10px !important;
        }

        .ai-min-tools label,
        .ai-min-tools .stSelectbox label,
        .ai-min-tools .stRadio label,
        .ai-min-tools .stSlider label,
        .ai-min-tools .stTextInput label {
            font-size: 11px !important;
            font-weight: 900 !important;
            color: #344054 !important;
        }

        .ai-min-tools [data-baseweb="select"] > div,
        .ai-min-tools [data-baseweb="input"] {
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

            .ai-mini-strip {
                grid-template-columns: 1fr !important;
            }
        }

        @media (max-width: 560px) {
            .ai-workspace-shell {
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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ai_workspace(
    df: pd.DataFrame | None,
    legacy_renderer: Callable[[pd.DataFrame | None], Any],
    current_role: str = "guest",
    current_email: str = "",
) -> None:
    """
    Render the premium AI workspace and then delegate to the existing chatbot.

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
                        A cleaner, friendlier, professional AI workspace for summarizing the active dashboard view,
                        comparing countries, generating filtered insights, and building analytical charts.
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
                    <strong>How to use it:</strong> ask naturally — for example, “summarize this filtered view”,
                    “compare countries with the highest restrictions”, or “plot negative alerts by year”.
                </div>
                <div class="ai-mode-note">
                    <strong>Professional mode:</strong> outputs are optimized for executive summaries, compact evidence,
                    clean charts, and dashboard-grounded interpretation.
                </div>
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
