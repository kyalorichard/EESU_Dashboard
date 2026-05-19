import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
from pathlib import Path
import streamlit.components.v1 as components
import plotly.graph_objects as go
import base64
import hashlib
from datetime import datetime
from auth import auth_ui, is_privileged, is_authenticated

# Optional admin page integration. Firebase/Auth handles login;
# authz.py resolves guest/viewer/privileged/admin roles.
try:
    from authz import (
        is_admin as admin_is_admin,
        get_current_role,
        get_current_email,
        has_permission,
        apply_data_scope,
    )
    from admin_page import render_admin_page, render_admin_sidebar_navigation
except Exception:
    def admin_is_admin():
        return False
    def get_current_role():
        return "guest"
    def get_current_email():
        return ""
    def has_permission(permission):
        # Fallback used only when authz.py/admin_page.py are unavailable.
        # Keep it permissive for local debugging, while deployed authz.py remains source of truth.
        return permission in [
            "view_dashboard",
            "view_overview",
            "view_coverage_monitored_countries",
            "view_monitored_countries_value",
            "view_maps",
            "view_negative_alerts",
            "view_analytical_flow_panel",
            "view_data_table",
            "download_data",
            "use_ai_copilot",
            "view_user_manual",
            "view_chart_overview_alert_type",
            "view_chart_overview_enabling_principles",
            "view_chart_overview_regions",
            "view_chart_overview_countries",
            "view_chart_negative_restrictive_actors",
            "view_chart_negative_affected_actors",
            "view_chart_negative_restrictive_mechanisms",
            "view_chart_negative_event_types",
            "view_chart_negative_alert_types",
            "view_chart_negative_enabling_principles",
            "view_chart_heatmap_actor_mechanism",
            "view_chart_heatmap_subject_mechanism",
            "view_chart_heatmap_actor_subject",
            "view_chart_sankey_flow",
            "view_chart_geospatial_map",
            "view_chart_ai_copilot_plots",
        ]
    def apply_data_scope(df):
        return df
    def render_admin_page(data=None):
        st.error("Admin page is not available. Confirm authz.py and admin_page.py are deployed with app.py.")
    def render_admin_sidebar_navigation():
        return "Dashboard"
import math
import paramiko
import logging
import tempfile  
import os
import re

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# OPENAI PACKAGE NOTE:
#   Add openai>=1.0.0 to requirements.txt.
#   Preferred Streamlit Cloud secrets format now uses a nested section:
#       [openai]
#       OPENAI_API_KEY = "sk-proj-..."
#       OPENAI_MODEL = "gpt-4o-mini"
#   The loader also supports flat Streamlit secrets and deployment environment variables.

# Optional dependency for real Plotly map click events.
# If not installed, the app falls back to the country drill-down dropdown.
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    plotly_events = None
    HAS_PLOTLY_EVENTS = False

# --- SFTP CONFIG ---
#sftp_secrets = st.secrets.get("sftp", {})
#SFTP_HOST = sftp_secrets.get("host")
#SFTP_PORT = 22
#SFTP_USERNAME = sftp_secrets.get("username")
#SFTP_PASSWORD = sftp_secrets.get("password")
#REMOTE_DIR = sftp_secrets.get("remote_dir", "exports")

#st.write("SFTP_HOST:", sftp_secrets.get("host"))
#st.write("SFTP_USERNAME:", sftp_secrets.get("username"))
#st.write("SFTP_PASSWORD:", sftp_secrets.get("password"))
#st.write("SFTP_REMOTE_DIR:", sftp_secrets.get("remote_dir", "exports"))

st.set_page_config(page_title="EUSEE Dashboard", layout="wide")


# ---------------- GLOBAL EXECUTIVE TYPOGRAPHY + COLOR SYSTEM ----------------
def configure_global_plotly_typography():
    """Apply a consistent executive font and color system to Plotly charts.

    This does not change chart data, traces, permissions, filters, or dashboard logic.
    It only sets default typography and neutral visual defaults for charts that do
    not explicitly override their own font settings.
    """
    executive_template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="Inter, Segoe UI, Arial, sans-serif",
                size=12,
                color="#475467",
            ),
            title=dict(
                font=dict(
                    family="Inter, Segoe UI, Arial, sans-serif",
                    size=18,
                    color="#101828",
                ),
                x=0.02,
                xanchor="left",
            ),
            legend=dict(
                font=dict(
                    family="Inter, Segoe UI, Arial, sans-serif",
                    size=11,
                    color="#475467",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=24, t=58, b=34),
            hoverlabel=dict(
                font=dict(
                    family="Inter, Segoe UI, Arial, sans-serif",
                    size=11,
                    color="#101828",
                ),
                bgcolor="#FFFFFF",
                bordercolor="#E4E7EC",
            ),
            xaxis=dict(
                title_font=dict(size=12, color="#344054"),
                tickfont=dict(size=11, color="#667085"),
                gridcolor="#EEF0F4",
                zerolinecolor="#E4E7EC",
            ),
            yaxis=dict(
                title_font=dict(size=12, color="#344054"),
                tickfont=dict(size=11, color="#667085"),
                gridcolor="#EEF0F4",
                zerolinecolor="#E4E7EC",
            ),
        )
    )
    pio.templates["eusee_executive"] = executive_template
    pio.templates.default = "plotly_white+eusee_executive"
    px.defaults.template = "plotly_white+eusee_executive"
    px.defaults.labels = px.defaults.labels or {}


configure_global_plotly_typography()

# ---------------- PROFESSIONAL CLASSIC DASHBOARD UX STYLING ----------------
def inject_classic_dashboard_css():
    """Central styling layer for a clean, classic analytical dashboard look."""
    st.markdown("""
    <style>
    :root {
        --eusee-purple: #660094;
        --eusee-purple-dark: #3b005f;
        --eusee-teal: #008CAA;
        --eusee-yellow: #FFDB58;
        --eusee-bg: #F7F8FB;
        --eusee-border: #E6E8EF;
        --eusee-text: #232633;
        --eusee-muted: #667085;
    }
    .main .block-container { padding-top: 0.25rem !important; padding-bottom: 1.4rem; max-width: 1500px; }
    header[data-testid="stHeader"] {
        height: 48px !important;
        min-height: 48px !important;
        background: rgba(247,248,251,0.92) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(230,232,239,0.75) !important;
        z-index: 999999 !important;
    }
    div[data-testid="stToolbar"] { right: 0.75rem !important; }
    button[data-testid="collapsedControl"],
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 10px !important;
        left: 12px !important;
        z-index: 1000000 !important;
        width: 38px !important;
        height: 38px !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        border: 1px solid #E6E8EF !important;
        box-shadow: 0 6px 18px rgba(16,24,40,.12) !important;
    }
    div[data-testid="stDecoration"] { display: none !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #FFFFFF 0%, #F7F8FB 100%); border-right: 1px solid var(--eusee-border); }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] label {
        font-family: Arial, sans-serif !important; font-size: 11px !important; font-weight: 800 !important;
        color: #344054 !important; letter-spacing: .01em; margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] {
        border-radius: 11px !important; border: 1px solid #D0D5DD !important; background: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.04) !important; min-height: 38px !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        background: #F4EAF8 !important; color: var(--eusee-purple) !important; border-radius: 999px !important;
        border: 1px solid #E7D4F1 !important; font-size: 10px !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 11px !important; border: 1px solid #D0D5DD !important; background: #FFFFFF !important;
        color: #344054 !important; font-weight: 800 !important; font-size: 12px !important; height: 38px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.05) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { border-color: var(--eusee-purple) !important; color: var(--eusee-purple) !important; background: #FBF7FD !important; }
    .classic-filter-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F4EAF8 100%); border: 1px solid #E7D4F1; border-radius: 15px;
        padding: 12px 13px; margin: 10px 0 12px 0; box-shadow: 0 8px 20px rgba(102,0,148,.08);
    }
    .classic-filter-eyebrow { font-size: 9.5px; font-weight: 900; color: var(--eusee-purple); letter-spacing: .12em; text-transform: uppercase; margin-bottom: 4px; }
    .classic-filter-title { font-size: 14px; font-weight: 900; color: #23152F; line-height: 1.15; }
    .classic-filter-note { font-size: 10.5px; color: var(--eusee-muted); line-height: 1.35; margin-top: 5px; }
    .classic-filter-status {
        background: #FFFFFF; border: 1px solid var(--eusee-border); border-radius: 13px; padding: 10px 11px;
        margin: 10px 0 12px 0; box-shadow: 0 4px 12px rgba(16,24,40,.05);
    }
    .classic-filter-status .status-row { display:flex; justify-content:space-between; align-items:center; padding: 3px 0; font-family: Arial, sans-serif; font-size: 10.5px; color: var(--eusee-muted); }
    .classic-filter-status .status-value { color: var(--eusee-purple); font-weight: 900; }
    div[data-testid="stExpander"] { border: 1px solid var(--eusee-border) !important; border-radius: 16px !important; box-shadow: 0 8px 22px rgba(16,24,40,.06) !important; background: #FFFFFF !important; overflow: hidden !important; }
    div[data-testid="stExpander"] summary { font-family: Arial, sans-serif !important; font-size: 13px !important; font-weight: 900 !important; color: #23152F !important; background: linear-gradient(90deg, #FFFFFF 0%, #FAF7FC 100%) !important; border-bottom: 1px solid #EEF0F4 !important; padding: 10px 14px !important; }
    .data-preview-toolbar { display:flex; justify-content:space-between; align-items:center; gap:12px; background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); border: 1px solid #EEF0F4; border-radius: 14px; padding: 11px 13px; margin: 4px 0 12px 0; font-family: Arial, sans-serif; }
    .data-preview-title { font-size: 13px; font-weight: 900; color: #23152F; line-height: 1.15; }
    .data-preview-subtitle { font-size: 10.5px; color: var(--eusee-muted); margin-top: 3px; }
    .data-preview-pill-row { display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; }
    .data-preview-pill { background:#F4EAF8; color: var(--eusee-purple); border:1px solid #E7D4F1; border-radius:999px; padding:5px 9px; font-size:10px; font-weight:900; white-space:nowrap; }
    .data-preview-footnote { font-size: 10.5px; color: var(--eusee-muted); line-height:1.4; margin-top:8px; padding: 8px 10px; background:#FFFCED; border:1px solid #F8E9A1; border-radius:11px; }
    div[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; border: 1px solid #E6E8EF !important; box-shadow: 0 6px 16px rgba(16,24,40,.05) !important; }
    .executive-table-shell {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E6E8EF;
        border-radius: 18px;
        padding: 14px;
        margin: 4px 0 14px 0;
        box-shadow: 0 10px 24px rgba(16,24,40,.06);
        font-family: Arial, sans-serif;
    }
    .executive-table-header { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:12px; }
    .executive-table-eyebrow { font-size:9.5px; font-weight:900; color:var(--eusee-purple); letter-spacing:.13em; text-transform:uppercase; margin-bottom:4px; }
    .executive-table-title { font-size:15px; font-weight:900; color:#23152F; line-height:1.15; }
    .executive-table-subtitle { font-size:11px; color:var(--eusee-muted); margin-top:4px; line-height:1.35; }
    .executive-table-badge { background:#F4EAF8; color:var(--eusee-purple); border:1px solid #E7D4F1; border-radius:999px; padding:6px 10px; font-size:10px; font-weight:900; white-space:nowrap; }
    .executive-metric-grid { display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:8px; }
    .executive-mini-kpi { background:#FFFFFF; border:1px solid #EEF0F4; border-radius:13px; padding:9px 10px; box-shadow:0 2px 8px rgba(16,24,40,.04); }
    .executive-mini-kpi span { display:block; font-size:10px; color:var(--eusee-muted); font-weight:800; margin-bottom:3px; }
    .executive-mini-kpi strong { font-size:15px; color:#23152F; font-weight:900; }
    .executive-table-status { display:flex; justify-content:space-between; align-items:center; gap:10px; background:#F9FAFB; border:1px solid #EEF0F4; border-radius:13px; padding:9px 11px; margin:9px 0 10px 0; font-size:11px; color:#344054; font-family:Arial, sans-serif; }
    .executive-table-status strong { color:var(--eusee-purple); font-weight:900; }
    .executive-table-status-note { color:var(--eusee-muted); font-size:10.5px; }
    @media (max-width: 900px) { .executive-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .executive-table-header, .executive-table-status { flex-direction:column; align-items:flex-start; } }
    

    /* ---------------- DEVICE-WIDE RESPONSIVE STABILIZATION ---------------- */
    html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }
    .main .block-container { padding-top: 0.85rem !important; padding-bottom: 7rem !important; }
    [data-testid="stSidebar"] img { max-width: 100% !important; height: auto !important; }
    div[data-testid="column"] { min-width: 0 !important; }
    .stPlotlyChart, div[data-testid="stPlotlyChart"], .js-plotly-plot, .plot-container {
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }
    iframe { max-width: 100% !important; }
    .animated-title { font-size: clamp(30px, 4vw, 48px) !important; }
    .animated-subtitle { font-size: clamp(12px, 1.2vw, 14px) !important; }
    .last-updated-badge { flex-wrap: wrap !important; }

    @media (max-width: 1100px) {
        .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        section[data-testid="stSidebar"] { width: min(86vw, 360px) !important; }
    }
    @media (max-width: 900px) {
        div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        .last-updated-badge { width: 100% !important; border-radius: 16px !important; }
        .executive-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    }
    @media (max-width: 700px) {
        .main .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
        section[data-testid="stSidebar"] { width: 90vw !important; }
        .stButton > button { width: 100% !important; }
        div[data-testid="stDataFrame"] { max-height: 70vh !important; overflow: auto !important; }
    }
</style>
    """, unsafe_allow_html=True)


def render_classic_filter_header():
    st.sidebar.markdown("""
    <div class="classic-filter-header">
        <div class="classic-filter-eyebrow">Dashboard controls</div>
        <div class="classic-filter-title">🌍 Global Filters</div>
        <div class="classic-filter-note">Use the filters to narrow the data by region, country, alert type, enabling principle, and time period if needed.</div>
    </div>
    """, unsafe_allow_html=True)


def render_filter_status_card(df):
    total = len(df) if df is not None else 0
    countries = df['alert-country'].nunique() if df is not None and not df.empty and 'alert-country' in df.columns else 0
    years = df['year'].nunique() if df is not None and not df.empty and 'year' in df.columns else 0
    st.sidebar.markdown(f"""
    <div class="classic-filter-status">
        <div class="status-row"><span>Filtered records</span><span class="status-value">{total:,}</span></div>
        <div class="status-row"><span>Countries</span><span class="status-value">{countries:,}</span></div>
        <div class="status-row"><span>Years</span><span class="status-value">{years:,}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_professional_data_preview(df, title="Data Preview and Download", key="summary_data_preview"):
    """Render a clean searchable table with alert-impact conditional formatting."""
    if df is None or df.empty:
        st.info("No records are available for the current filter selection.")
        return

    display_df = df.copy()

    # Preserve content, only improve display formatting for date-like columns.
    for date_col in ["Date of submission", "creation_date"]:
        if date_col in display_df.columns:
            display_df[date_col] = pd.to_datetime(
                display_df[date_col], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    # Detect the alert-impact column before/after user-facing renaming.
    impact_col = None
    for candidate in ["alert-impact", "Impact of alert", "Alert impact"]:
        if candidate in display_df.columns:
            impact_col = candidate
            break

    with st.expander(f"📋 {title}", expanded=False):
        st.markdown("""
        <style>
        .eusee-data-preview-note {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E6E8EF;
            border-left: 4px solid #660094;
            border-radius: 14px;
            padding: 10px 12px;
            margin: 2px 0 12px 0;
            color: #667085;
            font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
            font-size: 11.5px;
            line-height: 1.42;
            font-weight: 550;
            box-shadow: 0 6px 16px rgba(16,24,40,.045);
        }
        .eusee-data-preview-note strong {
            color: #23152F;
            font-weight: 900;
        }
        /* Data Preview table: one clean internal horizontal scrollbar, aligned with tab typography. */
        div[data-testid="stDataFrame"] {
            width: 100% !important;
            max-width: 100% !important;
            border: 1px solid #E6E8EF !important;
            border-radius: 16px !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            box-shadow: 0 10px 24px rgba(16,24,40,.06) !important;
            background: #FFFFFF !important;
            font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif) !important;
        }
        div[data-testid="stDataFrame"] > div {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
        }
        div[data-testid="stDataFrame"] div[role="grid"] {
            min-width: max-content !important;
            overflow-x: auto !important;
            overflow-y: auto !important;
        }
        div[data-testid="stDataFrame"] [data-testid="stTable"] {
            overflow-x: auto !important;
            overflow-y: hidden !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="columnheader"] * {
            background: #F4EAF8 !important;
            color: #23152F !important;
            font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif) !important;
            font-size: 11.5px !important;
            font-weight: 850 !important;
            border-bottom: 1px solid #E7D4F1 !important;
            line-height: 1.25 !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="gridcell"] * {
            color: #344054 !important;
            font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif) !important;
            font-size: 11.5px !important;
            line-height: 1.35 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stDataFrame"] ::-webkit-scrollbar {
            height: 10px !important;
            width: 10px !important;
        }
        div[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
            background: #D6BBE5 !important;
            border-radius: 999px !important;
            border: 2px solid #FFFFFF !important;
        }
        div[data-testid="stDataFrame"] ::-webkit-scrollbar-track {
            background: #F8FAFC !important;
            border-radius: 999px !important;
        }
        </style>
        <div class="eusee-data-preview-note">
            <strong>Filtered data preview:</strong> Review a sample of the filtered data and download the full dataset based on the filters currently applied
        </div>
        """, unsafe_allow_html=True)

        control_col1, control_col2 = st.columns([1.7, 0.7])

        with control_col1:
            search_text = st.text_input(
                "Search table",
                value="",
                placeholder="Search by country, alert type, principle, date or keyword.",
                key=f"{key}_search",
            )

        with control_col2:
            max_rows = st.selectbox(
                "Rows shown",
                options=[25, 50, 100, 250, 500, "All"],
                index=1,
                key=f"{key}_row_limit",
            )

        table_df = display_df.copy()

        # Search across all columns; underlying data remain unchanged.
        if search_text.strip():
            query = search_text.strip().lower()
            mask = table_df.astype(str).apply(
                lambda row: row.str.lower().str.contains(query, na=False).any(),
                axis=1,
            )
            table_df = table_df.loc[mask]

        if max_rows != "All":
            table_view = table_df.head(int(max_rows)).copy()
        else:
            table_view = table_df.copy()

        # Style the alert impact column using professional status colors.
        def style_alert_impact(value):
            value_clean = str(value).strip().lower()

            if value_clean == "negative":
                return (
                    "background-color:#FEE4E2;"
                    "color:#B42318;"
                    "font-weight:800;"
                )

            if value_clean == "positive":
                return (
                    "background-color:#DCFAE6;"
                    "color:#067647;"
                    "font-weight:800;"
                )

            if value_clean == "context to watch":
                return (
                    "background-color:#FEF0C7;"
                    "color:#B54708;"
                    "font-weight:800;"
                )

            return ""

        table_to_render = table_view
        if impact_col and impact_col in table_view.columns:
            try:
                table_to_render = table_view.style.map(
                    style_alert_impact,
                    subset=[impact_col],
                )
            except AttributeError:
                # Compatibility fallback for older pandas versions.
                table_to_render = table_view.style.applymap(
                    style_alert_impact,
                    subset=[impact_col],
                )

        # Span the full expander/panel width. Wide tables keep a single
        # internal horizontal scrollbar instead of shrinking the panel layout.
        st.dataframe(
            table_to_render,
            use_container_width=True,
            hide_index=True,
            height=min(560, max(320, 34 * min(len(table_view), 12) + 92)),
            key=key,
        )

        csv = table_df.to_csv(index=False).encode("utf-8")
        if has_permission("download_data"):

if has_permission("use_ai_copilot"):
    render_ai_assistant_panel(filtered_global)
# When unavailable, the AI Copilot status is shown in Settings / Profile instead of a sidebar alert.



# ---------------- FOOTER ----------------
# Feedback is rendered as a single collapsed responsive floating overlay near the dashboard header.
# Feedback is rendered as a single collapsed responsive floating overlay near the dashboard header.
# Footer image

# OpenAI test UI is now integrated inside the AI Copilot drawer.

# --- Load image and convert to base64 ---
footer_image_path = "assets/footer_logo.png"
with open(footer_image_path, "rb") as f:
    data = f.read()
b64 = base64.b64encode(data).decode()

# --- Render fixed footer using components.html ---
components.html(f"""
<div style="
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px 0;
    background-color:'white';
    z-index: 9999;    
">    
    <img src="data:image/png;base64,{b64}" width="900">
</div>
""", height=200)
st.markdown("<div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)


# Suggested prompts for better UX
SUGGESTED_PROMPTS = [
    "Summarize the current filtered alerts",
    "Which countries need the most attention?",
    "What are the top restrictive mechanisms?",
    "Show trends across regions",
    "Create an advanced chart from filtered data"
]


# Chatbot should render only after explicit user expansion/opening
# to reduce dashboard initial render load.
