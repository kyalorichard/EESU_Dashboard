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
#from auth import auth_ui, is_privileged, is_authenticated, init_session, restore_session, logout
import math
import paramiko
import logging
import tempfile  
import os
import re
import requests
import textwrap
import uuid
import warnings
from streamlit.elements.lib.policies import CachedWidgetWarning
import streamlit.components.v1 as components

warnings.filterwarnings(
    "ignore",
    category=CachedWidgetWarning
)

from auth import auth_ui, is_privileged, is_authenticated, init_session, restore_session, logout


st.set_page_config(page_title="EUSEE Dashboard", layout="wide", initial_sidebar_state="collapsed")

init_session()
restore_session()

# ------------------------------------------------------------------
# HIDE GITHUB / SOURCE CODE ACCESS
# KEEP SIDEBAR TOGGLE VISIBLE
# ------------------------------------------------------------------
st.markdown("""
<style>

/* Streamlit menu */
#MainMenu {
    visibility: hidden !important;
}

/* Footer */
footer {
    visibility: hidden !important;
}

/* Purple top decoration line */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Deploy button */
[data-testid="stDeployButton"] {
    display: none !important;
}

/* Source code / GitHub / toolbar actions */
[data-testid="stHeaderActionElements"] {
    display: none !important;
}

/* Extra toolbar buttons */
[data-testid="stToolbarActions"] {
    display: none !important;
}

/* Keep header visible */
header[data-testid="stHeader"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    background: rgba(247,248,251,0.95) !important;
}

/* Keep sidebar toggle visible */
button[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;

    width: auto !important;
    min-width: 92px !important;
    height: 38px !important;

    border-radius: 999px !important;
    padding: 0 12px !important;

    background: #FFFFFF !important;
    border: 1px solid #E7D4F1 !important;

    box-shadow: 0 6px 18px rgba(16,24,40,.10) !important;
}

/* Filters text beside sidebar icon */
button[data-testid="collapsedControl"]::after {
    content: " Filters";
    font-size: 12px;
    font-weight: 800;
    color: #660094;
    margin-left: 6px;
}

/* Hide any GitHub links */
a[href*="github.com"] {
    display: none !important;
}

/* Hide source-code links */
a[href*="source"] {
    display: none !important;
}



.block-container {
    padding-top: 1rem !important;
}

</style>
""", unsafe_allow_html=True)


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

    try:
        # Preferred name used by the optimized admin script.
        from admin import render_admin_page, render_admin_sidebar_navigation
    except Exception:
        # Backward-compatible fallback if your file is still named admin_page.py.
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

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    plotly_events = None
    HAS_PLOTLY_EVENTS = False



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

/* REMOVE SPACE BELOW SUBTITLE */
.animated-subtitle{
    margin-top: 0rem !important;
    margin-bottom: 0.85rem !important;
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    line-height: 1.25 !important;
}

/* REMOVE GAP BEFORE TABS */
div[data-testid="stTabs"]{
    margin-top: 0.95rem !important;
    padding-top: 0rem !important;
    margin-bottom: 0rem !important;
    padding-bottom: 0rem !important;
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



def _build_fast_table_search_mask(table_df: pd.DataFrame, search_text: str) -> pd.Series:
    """
    Boolean table search across all visible Data Preview columns.

    Supported:
    - AND: Kenya AND negative
    - OR: Kenya OR Uganda
    - NOT: Kenya NOT positive
    - quoted phrases: "civil society" AND Kenya

    Default behavior:
    - Multiple words without operators are treated as AND.
      Example: Kenya negative = Kenya AND negative
    """
    if table_df is None or table_df.empty:
        return pd.Series(dtype=bool)

    query = str(search_text or "").strip()
    if not query:
        return pd.Series(True, index=table_df.index)

    searchable_cols = [
        col for col in table_df.columns
        if (
            pd.api.types.is_object_dtype(table_df[col])
            or pd.api.types.is_string_dtype(table_df[col])
            or pd.api.types.is_categorical_dtype(table_df[col])
            or pd.api.types.is_numeric_dtype(table_df[col])
            or pd.api.types.is_datetime64_any_dtype(table_df[col])
        )
    ]

    if not searchable_cols:
        return pd.Series(False, index=table_df.index)

    # Build one searchable text string per row
    row_text = pd.Series("", index=table_df.index)

    for col in searchable_cols:
        row_text = row_text + " " + (
            table_df[col]
            .fillna("")
            .astype(str)
            .str.lower()
        )

    # Tokenize quoted phrases and Boolean operators
    tokens = re.findall(r'"[^"]+"|\bAND\b|\bOR\b|\bNOT\b|[^\s]+', query, flags=re.IGNORECASE)

    if not tokens:
        return pd.Series(True, index=table_df.index)

    def term_mask(term: str) -> pd.Series:
        term = term.strip().strip('"').lower()
        if not term:
            return pd.Series(True, index=table_df.index)
        return row_text.str.contains(term, regex=False, na=False)

    # If no Boolean operators are used, default to AND search
    has_boolean = any(t.upper() in {"AND", "OR", "NOT"} for t in tokens)

    if not has_boolean:
        mask = pd.Series(True, index=table_df.index)
        for term in tokens:
            mask &= term_mask(term)
        return mask

    # Boolean parser: left-to-right evaluation
    mask = None
    current_op = "AND"
    negate_next = False

    for token in tokens:
        upper = token.upper()

        if upper in {"AND", "OR"}:
            current_op = upper
            continue

        if upper == "NOT":
            negate_next = True
            continue

        this_mask = term_mask(token)

        if negate_next:
            this_mask = ~this_mask
            negate_next = False

        if mask is None:
            mask = this_mask
        elif current_op == "AND":
            mask &= this_mask
        elif current_op == "OR":
            mask |= this_mask

    if mask is None:
        return pd.Series(True, index=table_df.index)

    return mask

def render_professional_data_preview(df, title="Data Preview and Download", key="summary_data_preview", remove_vertical_scroll=False):
    """Render a clean, fast, searchable table with controlled scrolling and row limits.


    Notes:
    - The dataframe passed into this component is already filtered by the
      dashboard/sidebar/tab filters.
    - The table search is applied on top of those filters.
    - The Rows shown selector controls only the visible preview rows.
    - CSV download keeps the full searched/filtered table, not just the preview.
    - Overview and Negative Alerts Analysis use the same table dimensions,
      bounded height, and scroll behavior for a consistent UX.
    - remove_vertical_scroll is retained only for backward compatibility; it no
      longer changes sizing because all Data Preview tables must match.
    """
    if df is None or df.empty:
        st.info("No records are available for the current filter selection.")
        return

    
    DATA_PREVIEW_STANDARD_HEIGHT = 460

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
        /* Data Preview table: controlled vertical + horizontal scrolls for easy access. */
        div[data-testid="stDataFrame"] {
            width: 100% !important;
            max-width: 100% !important;
            border: 1px solid #E6E8EF !important;
            border-radius: 16px !important;
            overflow: auto !important;
            box-shadow: 0 10px 24px rgba(16,24,40,.06) !important;
            background: #FFFFFF !important;
            font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif) !important;
        }
        div[data-testid="stDataFrame"] > div {
            width: 100% !important;
            max-width: 100% !important;
            overflow: auto !important;
        }
        div[data-testid="stDataFrame"] div[role="grid"] {
            min-width: max-content !important;
            overflow: auto !important;
        }
        div[data-testid="stDataFrame"] [data-testid="stTable"] {
            overflow: auto !important;
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
                placeholder='Use Boolean search, e.g. Kenya AND negative, Uganda OR Kenya, "civil society" NOT positive.',                
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
        active_filter_rows = len(table_df)

        # Fast search across all display columns. The search is applied after
        # dashboard/sidebar filters, so the user sees and downloads only records
        # from the active filtered subset. Multiple words are treated as AND terms.
        search_text_clean = " ".join(str(search_text or "").split()).strip()
        if search_text_clean:
            table_df = table_df.loc[
                _build_fast_table_search_mask(table_df, search_text_clean)
            ].copy()

        if max_rows != "All":
            table_view = table_df.head(int(max_rows)).copy()
        else:
            table_view = table_df.copy()

        selected_row_limit_label = "All" if max_rows == "All" else f"{int(max_rows):,}"
        st.caption(
            f"Rows shown: {selected_row_limit_label}. Displaying {len(table_view):,} of "
            f"{len(table_df):,} matching records from {active_filter_rows:,} active-filter records."
        )

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

        # Standard compact Data Preview height.
        # All rows selected by the Rows shown control remain available inside
        # the dataframe through its internal vertical scrollbar, while wide
        # tables keep the horizontal scrollbar. This prevents the Overview
        # Data Preview panel from becoming too long.
        st.dataframe(
            table_to_render,
            use_container_width=True,
            hide_index=True,
            height=DATA_PREVIEW_STANDARD_HEIGHT,
            key=key,
        )

        csv = table_df.to_csv(index=False).encode("utf-8")
        if has_permission("download_data"):
            st.download_button(
                "⬇️ Download filtered table as CSV",
                data=csv,
                file_name=f"{key}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{key}_download",
            )
        else:
            st.caption("CSV download is disabled for your access level.")

        st.markdown("""
        <div class="data-preview-footnote">
            Interpretation note: this table reflects the active filters. Negative alerts are highlighted in red,
            positive alerts in green, and context-to-watch records in amber.
        </div>
        """, unsafe_allow_html=True)

inject_classic_dashboard_css()


# ---------------- MONITORED COUNTRIES ACCESS HELPER ----------------
def can_view_monitored_countries_value() -> bool:
    """Return True only when the active role can see the Monitored Countries numeric value.

    This must use the dedicated admin-controlled permission
    `view_monitored_countries_value`. The broader
    `view_coverage_monitored_countries` permission controls whether summary
    cards are visible; it must not expose the numeric countries_value.
    """
    try:
        return bool(has_permission("view_monitored_countries_value"))
    except Exception:
        return False


def monitored_countries_display_value(value) -> str:
    """Format monitored-country values only for permitted users."""
    if not can_view_monitored_countries_value():
        return "+80"
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


## ---------------- BASE DIRECTORIES ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ---------------- EXPORT DIRECTORY ----------------
# Use /exports if it exists (Docker volume mapping)
EXPORT_DIR = Path("/exports") if Path("/exports").exists() else BASE_DIR / "exports"

# Ensure folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EXEC_BRIEF_PATH = BASE_DIR / "docs" / "EU_SEE_Dashboard_Quick_Start_Executive.pdf"
USER_MANUAL_PATH = BASE_DIR / "docs" / "EU SEE Dashboard user manual.pdf"


# ---------------- USER MANUAL PDF HELPERS ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def _read_pdf_bytes_cached(pdf_path_str: str):
    """Read manual/brief PDF bytes once per deployment session.

    This prevents the User Manual tab from re-reading large PDF files on every
    Streamlit rerun. It also returns a clear error string instead of crashing
    the tab when the PDF is missing in deployment.
    """
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return None, f"PDF not found: {pdf_path.name}. Confirm it exists under the docs/ folder in the deployed repo."
    if not pdf_path.is_file():
        return None, f"PDF path is not a file: {pdf_path.name}."
    try:
        return pdf_path.read_bytes(), ""
    except Exception as exc:
        return None, f"Could not read {pdf_path.name}: {exc}"

def _safe_pdf_download_button(title: str, pdf_path: Path, key_prefix: str):
    """Render a PDF download button safely without blocking the tab."""
    pdf_bytes, pdf_error = _read_pdf_bytes_cached(str(pdf_path))
    if pdf_error:
        st.warning(pdf_error)
        return

    st.download_button(
        label=f"⬇ Download {title}",
        data=pdf_bytes,
        file_name=pdf_path.name,
        mime="application/pdf",
        use_container_width=True,
        key=f"{key_prefix}_{pdf_path.stem}",
    )


# ---------------- SIDEBAR-ONLY AUTH ROUTING ----------------
# Authentication routing is enabled only from the sidebar User Privilege Center.
# Restricted chart/map/tab cards remain passive locked-state messages and do not
# trigger login navigation.
st.session_state.setdefault("auth_view", False)
st.session_state.setdefault("auth_mode", "Login")
st.session_state.setdefault("auth_reset_open", False)

if is_authenticated():
    st.session_state.auth_view = False

if st.session_state.get("auth_view", False) and not is_authenticated():
    st.markdown("""
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"], .main, .main .block-container {
        filter: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        pointer-events: auto !important;
        opacity: 1 !important;
    }
    .eusee-login-route-shell {
        max-width: 760px;
        margin: 24px auto 18px auto;
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F7ECFB 100%);
        border: 1px solid rgba(102,0,148,.14);
        box-shadow: 0 14px 34px rgba(16,24,40,.08);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }
    .eusee-login-route-eyebrow {
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .13em;
        text-transform: uppercase;
        color: #660094;
        margin-bottom: 5px;
    }
    .eusee-login-route-title {
        font-size: 24px;
        font-weight: 950;
        color: #23152F;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .eusee-login-route-note {
        font-size: 12.5px;
        color: #667085;
        line-height: 1.45;
    }
    </style>

    <div class="eusee-login-route-shell">
        <div class="eusee-login-route-eyebrow">Privileged access</div>
        <div class="eusee-login-route-title">EUSEE Dashboard Sign in / Register</div>
        <div class="eusee-login-route-note">
            Sign in to access advanced features and analyses available to EUSEE partners.
        </div>
    </div>
    """, unsafe_allow_html=True)

    auth_ui()

    if st.button("← Back to dashboard", use_container_width=True, key="back_to_dashboard_from_sidebar_auth"):
        st.session_state.auth_view = False
        st.rerun()

    st.stop()


# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    parquet_file = EXPORT_DIR / "output_final.parquet"
    meta_file = EXPORT_DIR / "countries_metadata.json"

    # --- Step 1: Load Parquet file safely ---
    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        st.error(f"Failed to read Parquet file: {e}")
        return pd.DataFrame()

    if df.empty:
        st.warning("Loaded Parquet file is empty.")
        return df

    # --- Step 2: Ensure required columns exist ---
    for col in ["alert-country", "alert-impact", "alert-type", "Actor of repression"]:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in dataset.")
            df[col] = ""

    # --- Step 3: Load country metadata BEFORE ISO mapping ---
    country_meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, encoding="utf-8") as f:
                country_meta = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load countries metadata: {e}")
            country_meta = {}
    else:
        st.warning(f"Countries metadata JSON not found: {meta_file}")
        country_meta = {}

    # --- Step 4: Basic country and alert cleaning ---
    df["alert-country"] = df["alert-country"].astype(str).str.strip()
    df = df[df["alert-country"].str.lower() != "jose"]

    df["alert-impact"] = df["alert-impact"].astype(str).str.strip()
    df = df[df["alert-impact"].notna() & (df["alert-impact"] != "") & (df["alert-impact"].str.lower() != "nan")]

    # Normalize country names before ISO mapping.
    COUNTRY_FIXES = {
        "Guinea-Bissau": "Guinea Bissau",
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "Democratic Republic of Congo 2": "Democratic Republic of the Congo",
        "Congo (Brazzaville)": "Republic of Congo",
        "Congo Brazzaville": "Republic of Congo",
        "Congo-Brazzaville": "Republic of Congo",
        "Congo": "Republic of Congo",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "CÃ´te d'Ivoire": "Côte d'Ivoire",
        "Ivory Coast": "Côte d'Ivoire",
        "Tanzania, United Republic of": "Tanzania",
        "United Republic of Tanzania": "Tanzania",
        "Lao People's Democratic Republic": "Laos",
        "Lao PDR": "Laos",
        "Timor-Leste": "Timor Leste",
        "Gambia": "The Gambia",
        "Hong Kong SAR": "Hong Kong",
        "Lebanon NAR": "Lebanon",
    }

    df["alert-country"] = (
        df["alert-country"]
        .astype(str)
        .str.strip()
        .replace(COUNTRY_FIXES)
    )

    # --- Step 5: Clean alert type and remove non-alert event rows ---
    df["alert-type"] = df["alert-type"].astype(str).str.strip()
    df = df[
        (df["alert-type"].str.lower() != "event") &
        (df["alert-type"] != "") &
        (df["alert-type"].str.lower() != "nan")
    ]

    # Clean Actor of repression.
    df["Actor of repression"] = df["Actor of repression"].astype(str).str.strip()
    df["Actor of repression"] = df["Actor of repression"].replace({"VNSAs": "Violent non-state actors"})

    # --- Step 6: Map ISO codes and continent using cleaned country names ---
    df["iso_alpha3"] = df["alert-country"].apply(
        lambda x: country_meta.get(x, {}).get("iso_alpha3", None)
    )

    df["continent"] = df["alert-country"].apply(
        lambda x: country_meta.get(x, {}).get("continent", "Unknown")
    )

    # --- Step 7: Map continent to dashboard region ---
    def continent_to_region(continent):
        if continent == "Africa":
            return "Africa"
        elif continent in ["Asia", "Oceania"]:
            return "Asia and the Pacific"
        elif continent in ["Europe", "Middle East"]:
            return "The Middle East"
        elif continent in ["Americas", "North America", "South America", "Caribbean"]:
            return "Americas and the Caribbean"
        else:
            return "Unknown"

    df["region"] = df["continent"].apply(continent_to_region)

    # --- Step 8: Warn about missing ISO codes after normalization ---
    missing_countries = (
        df.loc[df["iso_alpha3"].isna(), "alert-country"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: (s.str.lower() != "none") & (s.str.lower() != "nan") & (s != "")]
        .unique()
    )

    if len(missing_countries) > 0:
        st.warning(
            "Countries missing ISO codes after metadata normalization: "
            + ", ".join(sorted(missing_countries))
        )

    # --- Step 9: Process dates and expose latest dataset date for the UX badge ---
    if "creation_date" in df.columns:
        df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
        df["year"] = df["creation_date"].dt.year
        df["month_name"] = df["creation_date"].dt.strftime("%B")

        latest_dataset_date = df["creation_date"].dropna().max()

        if pd.notna(latest_dataset_date):
            latest_dataset_date_display = latest_dataset_date.strftime("%d %B %Y")
            latest_dataset_date_iso = latest_dataset_date.strftime("%Y-%m-%d")
        else:
            latest_dataset_date_display = "Not available"
            latest_dataset_date_iso = ""

        # Keep the badge metadata tied to the exact dataset ingestion step.
        st.session_state["latest_dataset_date"] = latest_dataset_date_display
        st.session_state["latest_dataset_date_iso"] = latest_dataset_date_iso
        st.session_state["latest_dataset_date_source"] = "Based on latest loaded dataset"
        df.attrs["latest_dataset_date"] = latest_dataset_date_display
        df.attrs["latest_dataset_date_iso"] = latest_dataset_date_iso
    else:
        st.session_state["latest_dataset_date"] = "Not available"
        st.session_state["latest_dataset_date_iso"] = ""
        st.session_state["latest_dataset_date_source"] = "creation_date column not found in the loaded dataset"
        st.warning("No 'creation_date' column found in dataset.")

    # --- Step 10: Update alert-impact based on alert-type ---
    if "alert-type" in df.columns and "alert-impact" in df.columns:
        mask = df["alert-type"].astype(str).str.strip().str.lower() == "context to watch"
        df.loc[mask, "alert-impact"] = "Context to watch"

    return df

# --- Load data safely ---
data = apply_data_scope(load_data())

#### --------prepare enabling principles to be ordered-------------------------------------------
ENABLING_PRINCIPLE_ORDER = [
    "1. Respect and protection of fundamental freedoms",
    "2. Supportive legal and regulatory framework",
    "3. Accessible and sustainable resources",
    "4. Open and responsive State",
    "5. Supportive public culture and discourses on civil society",
    "6. Access to a secure digital environment"
]

ENABLING_PRINCIPLE_LABEL_MAP = {
    "Respect and protection of fundamental freedoms":"1. Respect and protection of fundamental freedoms",
    "Supportive legal and regulatory framework":"2. Supportive legal and regulatory framework",
    "Accessible and sustainable resources":"3. Accessible and sustainable resources",
    "State openness and responsiveness to civil society":"4. Open and responsive State",
    "Civic Culture and Public Discourses on Civil Society":"5. Supportive public culture and discourses on civil society",
    "Digital Environment Integrity and Security":"6. Access to a secure digital environment"
}


# ---------------- MULTISELECT WITH SELECT ALL ----------------
def safe_multiselect(label, options, session_key, sidebar=True, container=None):
    """
    Professional multiselect helper with Select all behavior.

    Fixes dependent-filter behavior by pruning stale selections whenever the
    available option list changes. This is important for the Overview Data
    Preview table because Country depends on Region, and Month depends on Year.
    """
    target = container if container is not None else (st.sidebar if sidebar else st)

    # Clean options, remove blanks, and de-duplicate by string representation.
    clean_options = []
    seen = set()
    for x in list(options):
        if pd.isna(x):
            continue
        val = x.item() if hasattr(x, "item") else x
        if isinstance(val, str):
            val = val.strip()
            if val == "" or val.lower() in ["nan", "none"]:
                continue
        sig = str(val)
        if sig not in seen:
            clean_options.append(val)
            seen.add(sig)

    options = sorted(clean_options, key=lambda v: str(v).lower())
    options_with_all = ["Select all"] + options
    widget_key = f"{session_key}_widget"
    options_signature_key = f"{session_key}_options_signature"
    options_signature = "||".join(map(str, options))
    valid_values = set(map(str, options))

    # Prune stale internal selections when upstream filters change the options.
    current_internal = st.session_state.get(session_key, options.copy())
    current_internal = [x for x in current_internal if str(x) in valid_values]

    # Empty means all currently available values are active.
    if not current_internal:
        current_internal = options.copy()

    st.session_state[session_key] = current_internal

    # Keep widget state synchronized with the current option universe. Without
    # this, a country/month chosen under a previous Region/Year can remain in
    # session_state and make the Overview table appear incorrectly filtered.
    options_changed = st.session_state.get(options_signature_key) != options_signature
    if options_changed or widget_key not in st.session_state:
        if set(map(str, current_internal)) == valid_values:
            st.session_state[widget_key] = []
        else:
            st.session_state[widget_key] = [x for x in current_internal if str(x) in valid_values]
        st.session_state[options_signature_key] = options_signature
    else:
        st.session_state[widget_key] = [x for x in st.session_state.get(widget_key, []) if x == "Select all" or str(x) in valid_values]

    selected = target.multiselect(
        label,
        options_with_all,
        key=widget_key,
        placeholder="",
        help="Leave empty or choose Select all to include all available options.",
    )

    if "Select all" in selected or len(selected) == 0:
        st.session_state[session_key] = options.copy()
        return options

    cleaned = [x for x in selected if x != "Select all" and str(x) in valid_values]
    if not cleaned:
        cleaned = options.copy()
    st.session_state[session_key] = cleaned
    return cleaned

def inject_professional_sidebar_filter_css():
    """Additional styling for the upgraded grouped sidebar filter experience."""
    st.markdown("""
    <style>

    .sidebar-profile-card {
        background: #FFFFFF;
        border: 1px solid #E6E8EF;
        border-radius: 14px;
        padding: 9px 10px;
        box-shadow: 0 6px 16px rgba(16,24,40,.045);
        font-family: Arial, sans-serif;
    }


    .sidebar-access-shell {
        margin: 12px 0 10px 0;
        padding: 12px 12px 11px 12px;
        border-radius: 16px;
        background: linear-gradient(135deg, #FFFFFF 0%, #FCF7FF 100%);
        border: 1px solid rgba(102,0,148,.16);
        box-shadow: 0 10px 24px rgba(16,24,40,.065);
        font-family: Arial, sans-serif;
        position: relative;
        overflow: hidden;
    }

    .sidebar-access-shell::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #660094 0%, #008CAA 58%, #FFDB58 100%);
    }

    .sidebar-access-top {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 3px;
    }

    .sidebar-access-icon {
        width: 36px;
        height: 36px;
        min-width: 36px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #660094;
        background: linear-gradient(135deg, rgba(102,0,148,.12), rgba(0,140,170,.10));
        border: 1px solid rgba(102,0,148,.10);
        font-size: 16px;
        font-weight: 900;
    }

    .sidebar-access-copy {
        min-width: 0;
        flex: 1;
    }

    .sidebar-access-eyebrow {
        font-size: 9px;
        font-weight: 950;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #660094;
        line-height: 1.1;
    }

    .sidebar-access-title {
        margin-top: 3px;
        color: #23152F;
        font-size: 13px;
        font-weight: 950;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .sidebar-access-note {
        margin-top: 4px;
        color: #667085;
        font-size: 10.5px;
        font-weight: 700;
        line-height: 1.35;
    }

    .sidebar-access-pill-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 10px;
    }

    .sidebar-access-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        padding: 5px 8px;
        border-radius: 999px;
        background: #EFFBFE;
        color: #008CAA;
        border: 1px solid rgba(0,140,170,.14);
        font-size: 9.5px;
        font-weight: 950;
        line-height: 1;
    }

    .sidebar-access-pill.secondary {
        background: #F4EAF8;
        color: #660094;
        border-color: #E7D4F1;
    }

    .sidebar-access-help {
        margin-top: 9px;
        padding: 8px 9px;
        border-radius: 12px;
        background: #F9FAFB;
        border: 1px solid #EEF0F4;
        color: #667085;
        font-size: 10.2px;
        line-height: 1.35;
        font-weight: 650;
    }

    .sidebar-access-center {
        margin-bottom: 8px;
    }

    .sidebar-profile-card-merged {
        margin-top: 10px;
        padding: 8px 9px;
        background: rgba(255,255,255,.92);
        border-color: #EEF0F4;
        box-shadow: none;
    }

    section[data-testid="stSidebar"] div[data-testid="column"] .stButton > button {
        height: 34px !important;
        font-size: 11px !important;
        border-radius: 10px !important;
    }

    .sidebar-profile-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
        border-bottom: 1px solid #F2F4F7;
        font-size: 10.5px;
        color: #667085;
    }

    .sidebar-profile-row:last-child {
        border-bottom: 0;
    }

    .sidebar-profile-row strong {
        color: #2D0055;
        font-size: 10.5px;
        font-weight: 900;
        text-align: right;
        max-width: 155px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* ---------------- GLOBAL SELECT / MULTISELECT COLOR SYSTEM ---------------- */
    [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #D0D5DD !important;
        border-radius: 12px !important;
        min-height: 38px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.05) !important;
        transition: all .16s ease !important;
    }

    [data-baseweb="select"] > div:hover {
        border-color: #B692C8 !important;
        box-shadow: 0 0 0 3px rgba(102,0,148,.075) !important;
    }

    [data-baseweb="select"] > div:focus-within {
        border-color: #660094 !important;
        box-shadow: 0 0 0 3px rgba(102,0,148,.14) !important;
    }

    [data-baseweb="tag"] {
        background: #F4EAF8 !important;
        color: #660094 !important;
        border: 1px solid #E7D4F1 !important;
        border-radius: 999px !important;
        font-size: 10px !important;
        font-weight: 850 !important;
    }

    [data-baseweb="tag"] svg {
        color: #660094 !important;
    }

    /* ---------------- SELECT / MULTISELECT DROPDOWN MENU ---------------- */
    /*
       Streamlit/BaseWeb renders dropdown menus in a portal outside the sidebar.
       Avoid broad listbox rules that let the menu expand across the page.
       The :has() selector limits this styling to select/multiselect dropdown popovers only.
    */
    

    .stMultiSelect label, .stSelectbox label {
        font-size: 10.8px !important;
        font-weight: 900 !important;
        color: #344054 !important;
        letter-spacing: .01em !important;
        margin-bottom: 4px !important;
    }

    .negative-filter-shell {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFFCFB 100%);
        border: 1px solid rgba(180,35,24,.12);
        border-radius: 16px;
        padding: 11px 13px;
        margin: 2px 0 13px 0;
        box-shadow: 0 8px 22px rgba(16,24,40,.055);
        font-family: Arial, sans-serif;
    }

    .negative-filter-eyebrow {
        font-size: 9.5px;
        font-weight: 900;
        color: #B42318;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .negative-filter-title {
        font-size: 14px;
        font-weight: 950;
        color: #23152F;
        line-height: 1.15;
    }

    .negative-filter-note {
        font-size: 10.7px;
        color: #667085;
        line-height: 1.35;
        margin-top: 5px;
    }

    .negative-filter-chip-row {
        display: flex;
        gap: 7px;
        flex-wrap: wrap;
        margin-top: 9px;
    }

    .negative-filter-chip {
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 9.8px;
        font-weight: 900;
        background: #FFF4ED;
        color: #B42318;
        border: 1px solid rgba(180,35,24,.14);
    }
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 15% 0%, rgba(102,0,148,.055), transparent 30%),
            linear-gradient(180deg, #FFFFFF 0%, #F7F8FB 100%) !important;
    }

    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] > div {
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }

    .sidebar-filter-section {
        font-family: Arial, sans-serif;
        font-size: 10.5px;
        color: #667085;
        line-height: 1.35;
        margin: -2px 0 9px 0;
    }

    .sidebar-filter-footer {
        background: #FFFFFF;
        border: 1px solid #E6E8EF;
        border-radius: 14px;
        padding: 9px 10px;
        margin: 9px 0 12px 0;
        box-shadow: 0 6px 16px rgba(16,24,40,.045);
        font-family: Arial, sans-serif;
    }

    .sidebar-filter-footer-title {
        font-size: 11px;
        font-weight: 900;
        color: #23152F;
        margin-bottom: 3px;
    }

    .sidebar-filter-footer-note {
        font-size: 10px;
        color: #667085;
        line-height: 1.35;
    }

    div[data-testid="stExpander"] {
        margin-bottom: 10px !important;
        border-radius: 16px !important;
        border: 1px solid #E6E8EF !important;
        background: #FFFFFF !important;
        box-shadow: 0 8px 22px rgba(16,24,40,.055) !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] summary {
        min-height: 42px !important;
        padding: 10px 13px !important;
        background: linear-gradient(90deg, #FFFFFF 0%, #FAF7FC 100%) !important;
        border-bottom: 1px solid #EEF0F4 !important;
        color: #23152F !important;
        font-family: Arial, sans-serif !important;
        font-size: 12.5px !important;
        font-weight: 900 !important;
        letter-spacing: -0.01em !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background: linear-gradient(90deg, #FFFFFF 0%, #F4EAF8 100%) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        min-height: 38px !important;
        border-radius: 12px !important;
        border: 1px solid #D0D5DD !important;
        background: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.045) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
        border-color: #B692C8 !important;
        box-shadow: 0 0 0 3px rgba(102,0,148,.07) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        background: #F4EAF8 !important;
        color: #660094 !important;
        border: 1px solid #E7D4F1 !important;
        border-radius: 999px !important;
        font-size: 10px !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] label {
        font-size: 10.8px !important;
        font-weight: 900 !important;
        color: #344054 !important;
        letter-spacing: .01em !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 12px !important;
        height: 38px !important;
        font-size: 11.5px !important;
        font-weight: 900 !important;
    }

    section[data-testid="stSidebar"] button[disabled] {
        opacity: 1 !important;
        color: #008CAA !important;
        background: #EFFBFE !important;
        border-color: rgba(0,140,170,.18) !important;
    }
    </style>
    """, unsafe_allow_html=True)



# ---------------- GLOBAL FILTERS: PROFESSIONAL COLLAPSIBLE SIDEBAR ----------------
st.sidebar.image("assets/eu-see-logo.png", width=230)

# ---------------- SIDEBAR PRIVILEGE ACCESS CENTER ----------------
def render_sidebar_access_settings_profile():
    """Render one clean, native Streamlit sidebar panel for access, account, navigation, and feature status.

    This version intentionally avoids rendering the panel body with raw HTML so users never see
    HTML tags/scripts if Streamlit sanitization or markdown rendering changes.
    """
    signed_in = is_authenticated()
    is_admin_user = bool(signed_in and admin_is_admin())

    role = get_current_role() if callable(get_current_role) else "guest"
    role_label = (role or "guest").replace("_", " ").title()

    email = get_current_email() if callable(get_current_email) else ""
    display_email = email or st.session_state.get("email", "Public user")
    display_name = st.session_state.get("name", "User") if signed_in else "Guest access"

    copilot_status = "Available" if has_permission("use_ai_copilot") else "Limited"
    export_status = "Enabled" if has_permission("download_data") else "Restricted"
    admin_status = "Enabled" if is_admin_user else "Not available"
    access_status = "Signed in" if signed_in else "Public mode"

    # Show the monitored-country value as a first-class item in the
    # privilege/access list. Use the already scoped dataframe so the value
    # respects the active role's data scope.
    try:
        raw_monitored_countries_value = (
            int(data["alert-country"].nunique())
            if "data" in globals()
            and isinstance(data, pd.DataFrame)
            and not data.empty
            and "alert-country" in data.columns
            else 0
        )
        monitored_countries_value = monitored_countries_display_value(raw_monitored_countries_value)
    except Exception:
        monitored_countries_value = 0

    st.session_state.setdefault("eusee_sidebar_workspace", "Dashboard")
    if not is_admin_user:
        st.session_state["eusee_sidebar_workspace"] = "Dashboard"

    # Small CSS only for Streamlit widgets in the privilege center; no visible HTML content is rendered.
    st.sidebar.markdown("""
    <style>
    /* ---------- User Privilege Center ---------- */

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.eusee-privilege-marker){
        gap:0.25rem;
    }

    .eusee-privilege-marker{
        display:none;
    }

    /* Panel title */
    section[data-testid="stSidebar"] h3{
        font-size:12px !important;
        font-weight:900 !important;
        color:#23152F !important;
        margin-bottom:0.4rem !important;
    }

    /* Guest access / Logged-in user name */
    section[data-testid="stSidebar"] p{
        font-size:10px !important;
        font-weight:700 !important;
        color:#23152F !important;
        line-height:1.45 !important;
    }

    /* Description */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{
        font-size:10px !important;
        font-family: Arial, sans-serif;
        line-height:1.5 !important;
        color:#667085 !important;
    }

    /* Metric cards */
    section[data-testid="stSidebar"] [data-testid="stMetric"]{
        background:#FFFFFF;
        border:1px solid #EEF0F4;
        border-radius:12px;
        padding:7px 8px;
        box-shadow:0 2px 8px rgba(16,24,40,.035);
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{
        font-size:10px !important;
        font-family: Arial, sans-serif;
        font-weight:800 !important;
        color:#667085 !important;
        text-transform:uppercase;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricValue"]{
        font-size:12px !important;
        font-weight:900 !important;
        color:#23152F !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # Use a bordered native container where available; fallback gracefully for older Streamlit versions.
    try:
        panel = st.sidebar.container(border=True)
    except TypeError:
        panel = st.sidebar.container()

    with panel:
        st.markdown('<span class="eusee-privilege-marker"></span>', unsafe_allow_html=True)
        st.markdown("### 🔐 User Privilege Center")
        st.caption("Central access, role, navigation, and feature availability.")

        st.markdown(f"**{display_name}**")
        st.caption(
            "Your dashboard permissions are controlled by your approved EUSEE role."
            if signed_in
            else "Sign in to access advanced features and analyses available to EUSEE partners."
        )

           
        if is_admin_user:
            workspace = st.radio(
                "Workspace",
                options=["Dashboard", "Admin"],
                horizontal=True,
                key="eusee_sidebar_workspace_radio",
                index=0 if st.session_state.get("eusee_sidebar_workspace") == "Dashboard" else 1,
                label_visibility="collapsed",
            )
            if workspace != st.session_state.get("eusee_sidebar_workspace"):
                st.session_state["eusee_sidebar_workspace"] = workspace
                st.rerun()

        if signed_in:
            if st.button("Logout", use_container_width=True, key="privilege_center_logout_btn"):
                logout()
        else:
            if st.button("🔐 Sign in / Register", use_container_width=True, key="privilege_center_signin_btn"):
                st.session_state.auth_view = True
                st.rerun()

render_sidebar_access_settings_profile()

render_classic_filter_header()
inject_professional_sidebar_filter_css()

# Sidebar compact/responsive override removed to restore the previous sidebar layout.

regions_labels = [
    "Africa",
    "The Middle East",
    "Asia and the Pacific",
    "Americas and the Caribbean",
]

with st.sidebar.expander("🌍 Dashboard filters", expanded=True) as sidebar_filter_box:
  
    selected_regions = safe_multiselect(
        "Region",
        regions_labels,
        "selected_regions",
        container=sidebar_filter_box,
    )

    filtered_countries = (
        data[data["region"].isin(selected_regions)]
        if (not data.empty and "region" in data.columns and selected_regions)
        else data
    )

    selected_countries = safe_multiselect(
        "Country",
        filtered_countries["alert-country"].dropna().unique()
        if not filtered_countries.empty and "alert-country" in filtered_countries.columns
        else [],
        "selected_countries",
        container=sidebar_filter_box,
    )

    selected_alert_impacts = safe_multiselect(
        "Nature of event / alert",
        data["alert-impact"].dropna().unique()
        if not data.empty and "alert-impact" in data.columns
        else [],
        "selected_alert_impacts",
        container=sidebar_filter_box,
    )

    selected_alert_types = safe_multiselect(
        "Impact of alert",
        data["alert-type"].dropna().unique()
        if not data.empty and "alert-type" in data.columns
        else [],
        "selected_alert_types",
        container=sidebar_filter_box,
    )

    principle_options = (
        data["enabling-principle"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .str.capitalize()
        .replace("", np.nan)
        .dropna()
        .unique()
        if not data.empty and "enabling-principle" in data.columns
        else []
    )

    reverse_principle_map = {
    v: k for k, v in ENABLING_PRINCIPLE_LABEL_MAP.items()
    }

    principle_display_options = [
        label
        for label in ENABLING_PRINCIPLE_ORDER
        if label in reverse_principle_map
    ]

    selected_principle_display = safe_multiselect(
        "Enabling principle",
        principle_display_options,
        "selected_enabling_principle",
        container=sidebar_filter_box,
    )

    selected_enabling_principle = [
        reverse_principle_map[p]
        for p in selected_principle_display
    ]

    selected_years = safe_multiselect(
        "Year",
        sorted(data["year"].dropna().unique())
        if not data.empty and "year" in data.columns
        else [],
        "selected_years",
        container=sidebar_filter_box,
    )

    if (
        not data.empty
        and "year" in data.columns
        and "month_name" in data.columns
        and selected_years
    ):
        available_months_source = data[data["year"].isin(selected_years)]["month_name"].dropna().unique()
    elif not data.empty and "month_name" in data.columns:
        available_months_source = data["month_name"].dropna().unique()
    else:
        available_months_source = []

    available_months = sorted(
        available_months_source,
        key=lambda m: pd.to_datetime(m, format="%B", errors="coerce").month
        if pd.notna(pd.to_datetime(m, format="%B", errors="coerce"))
        else 13,
    )

    selected_months = safe_multiselect(
        "Month",
        available_months,
        "selected_months",
        container=sidebar_filter_box,
    )

    reset_col1, reset_col2 = st.columns([1, 1])

    with reset_col1:
        reset_filters = st.button(
            "🔄 Reset",
            use_container_width=True,
            key="reset_sidebar_filters",
        )

    with reset_col2:
        st.button(
            "✅ Applied",
            use_container_width=True,
            disabled=True,
            key="filters_applied_note",
        )

if reset_filters:
    for key in [
        "selected_regions",
        "selected_countries",
        "selected_alert_types",
        "selected_enabling_principle",
        "selected_alert_impacts",
        "selected_months",
        "selected_years",
        "selected_actor_types",
        "selected_subject_types",
        "selected_mechanism_types",
        "selected_event_types",
    ]:
        st.session_state.pop(key, None)
        st.session_state.pop(f"{key}_widget", None)
        st.session_state.pop(f"{key}_options_signature", None)
    # Clear table searches too, so reset returns the Overview Data Preview to
    # the full active dataset immediately.
    for table_key in ["overview_summary_data_preview", "negative_summary_data_preview"]:
        st.session_state.pop(f"{table_key}_search", None)
        st.session_state.pop(f"{table_key}_row_limit", None)
    st.rerun()

st.sidebar.markdown(
    """
    <div class="sidebar-filter-footer">
        <div class="sidebar-filter-footer-title">Filter behavior</div>
        <div class="sidebar-filter-footer-note">Filters update the dashboard automatically. Empty selections mean all available values are included.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep the dataset update status as the final sidebar panel.


# ---------------- FILTER DATA ----------------
def contains_any(cell_value, selected_values):
    """Case-insensitive matcher for comma-separated enabling principles."""
    if selected_values is None or len(selected_values) == 0:
        return True
    if pd.isna(cell_value):
        return False

    selected_norm = {str(v).strip().lower() for v in selected_values if str(v).strip()}
    cell_terms = {
        part.strip().lower()
        for part in str(cell_value).split(",")
        if part.strip()
    }

    # Exact token match first; fallback substring keeps compatibility with older
    # records that may not use comma-separated principle values consistently.
    return bool(cell_terms & selected_norm) or any(
        sel in str(cell_value).strip().lower() for sel in selected_norm
    )

filtered_global = data[
    (data['region'].isin(selected_regions)) &
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
].copy()


st.session_state["eusee_active_filtered_df"] = filtered_global.copy()

st.session_state["eusee_active_filter_summary"] = {
    "filtered_records": int(len(filtered_global)),
    "latest_dataset_date": st.session_state.get("latest_dataset_date", "Not available"),
    "basis": "Current active sidebar/dashboard filters"
}


# ---------------- ADMIN ROUTING FROM SIDEBAR PRIVILEGE CENTER ----------------
# Admin users can switch between Dashboard and Admin inside the single User Privilege Center panel.
# IMPORTANT: this block must run BEFORE the dashboard title and st.tabs() are created.
# Otherwise Streamlit will keep showing the dashboard tab bar above the Admin page.
if is_authenticated() and admin_is_admin() and st.session_state.get("eusee_sidebar_workspace") == "Admin":
    render_admin_page(data=data)
    st.stop()

if not has_permission("view_dashboard"):
    render_access_locked("Dashboard", "view_dashboard permission")
    st.stop()


# ---------------- DASHBOARD TITLE WITH ANIMATED DIVIDER AND TITLE ----------------
st.markdown(f"""
<div class="dashboard-title-shell">

<h1 class="animated-title">
    EU SEE Dashboard
</h1>

<div class="animated-divider"></div>

<div class="animated-subtitle">
    This interactive dashboard allows exploration and analysis of data produced by the EU SEE project.
    It aggregates information reported by Network Members across 86 countries to document trends 
    in the enabling environment for civil society.
</div>

</div>

<style>
.dashboard-title-shell {{
    overflow: hidden;
    margin-top: -0.5rem !important;
    padding-top: 0rem !important;
    margin-bottom: 0.4rem !important;
}}

.animated-title {{
    margin: 0 0 0px 0 !important;
    padding: 0 !important;
    line-height: 1.02;
    color: #660094;
    font-size: 48px;
    font-family: Arial, sans-serif;
    font-weight: 700;
    opacity: 0;
    transform: translateY(-20px);
    animation: titleFadeSlide 0.8s ease-out forwards;
    animation-delay: 0.2s;
}}

@keyframes titleFadeSlide {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.animated-divider {{
    width: 15%;
    max-width: 120px;
    height: 4px;
    background: linear-gradient(to right, #FFDB58, #660094);
    border-radius: 2px;
    margin-top: 0rem !important;
    margin-bottom: 2px !important;
    opacity: 0;
    transform: translateX(-120%);
    animation: dividerSlide 1s ease-out forwards;
    animation-delay: 0.6s;
}}

@keyframes dividerSlide {{
    from {{ transform: translateX(-120%); opacity: 0; }}
    to   {{ transform: translateX(0); opacity: 1; }}
}}

.animated-subtitle {{
    font-size: 14px;
    font-family: Arial, sans-serif;
    color: #333333;
    margin-top: 0rem !important;
    margin-bottom: 3px !important;
    padding-bottom: 0px !important;
    max-width: 980px;
    line-height: 1.25;
    opacity: 0;
    animation: subtitleFade 0.8s ease-out forwards;
    animation-delay: 1.0s;
}}

@keyframes subtitleFade {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------- MAIN TABS - PLACED IMMEDIATELY AFTER SUBTITLE ----------------
# This removes the visible blank space between the dashboard subtitle and the tabs.
#tab_map disabled
# Build dashboard tabs dynamically from Admin → Visibility permissions.
# When a permission is unchecked, the entire tab is removed from the tab bar.
tab_overview = tab_negative = tab_map = tab_manual = None

_dashboard_tab_specs = []

if has_permission("view_overview"):
    _dashboard_tab_specs.append(("overview", "📊 Overview"))

if has_permission("view_negative_alerts"):
    _dashboard_tab_specs.append(("negative", "⚠️ Negative Alerts Analysis"))

if has_permission("view_maps"):
    _dashboard_tab_specs.append(("map", "🗺️ Visualization Map"))

if has_permission("view_user_manual"):
    _dashboard_tab_specs.append(("manual", "📘 User Manual"))

if _dashboard_tab_specs:
    _dashboard_tabs = st.tabs([label for _, label in _dashboard_tab_specs])
    _dashboard_tab_lookup = {tab_id: tab for (tab_id, _), tab in zip(_dashboard_tab_specs, _dashboard_tabs)}

    tab_overview = _dashboard_tab_lookup.get("overview")
    tab_negative = _dashboard_tab_lookup.get("negative")
    tab_map = _dashboard_tab_lookup.get("map")
    tab_manual = _dashboard_tab_lookup.get("manual")
else:
    st.error("No dashboard tabs are enabled for your role. Please contact the dashboard administrator.")
    st.stop()


# ---------------- COLLAPSED RESPONSIVE FLOATING FEEDBACK OVERLAY ----------------
def render_top_feedback_bar():
    """
    Inject a reliable single-control floating feedback widget.

    Fix applied:
    - Uses one click listener per button only.
    - Avoids duplicate onclick + addEventListener bindings, which caused the widget
      to open and immediately close again.
    - Uses CSS class toggling only; no competing inline display overrides.
    """
    feedback_url = "https://forms.office.com/pages/responsepage.aspx?id=aFcOUAlSoUeqnjS7rLiI3i2QH6350xBGsugTt9B-i59URUk5UEFTV0VKSDRaU0lXTEc1S1g1M0hYTi4u&route=shorturl"

    components.html(f"""
    <script>
    (function() {{
        const doc = window.parent.document;
        const rootId = "eusee-feedback-floating-root";
        const styleId = "eusee-feedback-floating-style";

        // Remove previous/cached feedback widgets and styles.
        [
            "eusee-feedback-floating-root",
            "eusee-feedback-callout",
            "eusee-feedback-tab"
        ].forEach(function(id) {{
            const el = doc.getElementById(id);
            if (el) el.remove();
        }});

        [
            "eusee-feedback-floating-style",
            "eusee-feedback-style"
        ].forEach(function(id) {{
            const el = doc.getElementById(id);
            if (el) el.remove();
        }});

        try {{
            window.localStorage.removeItem("eusee_feedback_widget_closed");
            window.localStorage.removeItem("eusee_feedback_widget_dismissed");
            window.localStorage.removeItem("eusee_feedback_widget_hidden");
        }} catch (e) {{}}

        const style = doc.createElement("style");
        style.id = styleId;
        style.innerHTML = `
            #eusee-feedback-floating-root {{
                position: fixed !important;
                top: clamp(58px, 7vh, 78px) !important;
                left: 75% !important;
                transform: translateX(-50%) !important;
                z-index: 2147482500 !important;
                font-family: Arial, sans-serif !important;
                pointer-events: auto !important;
                width: auto !important;
                max-width: calc(100vw - 24px) !important;
            }}

            #eusee-feedback-floating-root * {{
                box-sizing: border-box !important;
            }}

            .eusee-feedback-toggle {{
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 8px !important;
                min-height: 38px !important;
                padding: 8px 14px !important;
                border-radius: 999px !important;
                border: 1px solid rgba(102,0,148,.16) !important;
                background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(252,247,255,.98)) !important;
                color: #2D0055 !important;
                box-shadow: 0 12px 28px rgba(17,24,39,.13), 0 2px 8px rgba(102,0,148,.08) !important;
                font-size: 12px !important;
                font-weight: 950 !important;
                cursor: pointer !important;
                user-select: none !important;
                white-space: nowrap !important;
                backdrop-filter: blur(14px) !important;
                -webkit-backdrop-filter: blur(14px) !important;
                transition: transform .16s ease, box-shadow .16s ease, background .16s ease !important;
                appearance: none !important;
                -webkit-appearance: none !important;
            }}

            .eusee-feedback-toggle:hover {{
                transform: translateY(-1px) !important;
                background: linear-gradient(135deg, #FFFFFF, #F4EAF8) !important;
                box-shadow: 0 16px 34px rgba(17,24,39,.16), 0 3px 10px rgba(102,0,148,.10) !important;
            }}

            .eusee-feedback-toggle-icon {{
                width: 24px !important;
                height: 24px !important;
                min-width: 24px !important;
                border-radius: 9px !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                background: linear-gradient(135deg, rgba(102,0,148,.13), rgba(0,140,170,.10)) !important;
                border: 1px solid rgba(102,0,148,.10) !important;
                color: #660094 !important;
                font-size: 13px !important;
                font-weight: 900 !important;
            }}

            .eusee-feedback-toggle-caret {{
                color: #667085 !important;
                font-size: 13px !important;
                font-weight: 950 !important;
                line-height: 1 !important;
            }}

            .eusee-feedback-panel {{
                display: none !important;
                width: min(760px, calc(100vw - 28px)) !important;
                align-items: center !important;
                justify-content: space-between !important;
                gap: 12px !important;
                padding: 12px 12px 12px 14px !important;
                border-radius: 18px !important;
                background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(252,247,255,.98)) !important;
                border: 1px solid rgba(102,0,148,.14) !important;
                box-shadow: 0 16px 38px rgba(17,24,39,.14), 0 2px 8px rgba(102,0,148,.08) !important;
                backdrop-filter: blur(14px) !important;
                -webkit-backdrop-filter: blur(14px) !important;
            }}

            #eusee-feedback-floating-root.is-open > #eusee-feedback-toggle {{
                display: none !important;
            }}

            #eusee-feedback-floating-root.is-open .eusee-feedback-panel {{
                display: flex !important;
            }}

            .eusee-feedback-panel-left {{
                display: flex !important;
                align-items: center !important;
                gap: 10px !important;
                min-width: 0 !important;
                flex: 1 !important;
            }}

            .eusee-feedback-panel-icon {{
                width: 34px !important;
                height: 34px !important;
                min-width: 34px !important;
                border-radius: 12px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                background: linear-gradient(135deg, rgba(102,0,148,.13), rgba(0,140,170,.10)) !important;
                border: 1px solid rgba(102,0,148,.10) !important;
                color: #660094 !important;
                font-size: 15px !important;
                font-weight: 900 !important;
            }}

            .eusee-feedback-panel-copy {{
                color: #344054 !important;
                font-size: 12px !important;
                line-height: 1.32 !important;
                font-weight: 750 !important;
                white-space: normal !important;
                min-width: 0 !important;
            }}

            .eusee-feedback-panel-copy strong {{
                color: #2D0055 !important;
                font-weight: 950 !important;
            }}

            .eusee-feedback-panel-actions {{
                display: inline-flex !important;
                align-items: center !important;
                justify-content: flex-end !important;
                gap: 8px !important;
                flex-shrink: 0 !important;
                align-self: center !important;
            }}

            .eusee-feedback-panel-button {{
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                min-height: 34px !important;
                padding: 8px 14px !important;
                border-radius: 999px !important;
                background: linear-gradient(90deg, #660094 0%, #008CAA 100%) !important;
                color: #FFFFFF !important;
                text-decoration: none !important;
                font-size: 11px !important;
                font-weight: 900 !important;
                white-space: nowrap !important;
                box-shadow: 0 8px 18px rgba(102,0,148,.18) !important;
                transition: all .16s ease !important;
            }}

            .eusee-feedback-panel-button:hover {{
                transform: translateY(-1px) !important;
                filter: brightness(1.04) !important;
            }}

            .eusee-feedback-panel-toggle {{
                min-height: 34px !important;
                padding: 8px 12px !important;
                gap: 6px !important;
                box-shadow: none !important;
                background: rgba(255,255,255,.92) !important;
            }}

            @media (max-width: 900px) {{
                #eusee-feedback-floating-root {{
                    top: 62px !important;
                    max-width: calc(100vw - 20px) !important;
                }}

                .eusee-feedback-panel {{
                    width: min(640px, calc(100vw - 20px)) !important;
                    padding: 11px 12px !important;
                }}
            }}

            @media (max-width: 700px) {{
                #eusee-feedback-floating-root {{
                    top: 58px !important;
                    left: 50% !important;
                    width: calc(100vw - 18px) !important;
                    max-width: calc(100vw - 18px) !important;
                }}

                .eusee-feedback-toggle {{
                    width: fit-content !important;
                    max-width: 92vw !important;
                    margin: 0 auto !important;
                    min-height: 36px !important;
                    padding: 7px 12px !important;
                    font-size: 11.5px !important;
                }}

                .eusee-feedback-panel {{
                    width: 100% !important;
                    max-height: calc(100vh - 88px) !important;
                    overflow-y: auto !important;
                    flex-direction: column !important;
                    align-items: stretch !important;
                    gap: 10px !important;
                    padding: 11px !important;
                    border-radius: 16px !important;
                }}

                .eusee-feedback-panel-left {{
                    align-items: flex-start !important;
                }}

                .eusee-feedback-panel-copy {{
                    font-size: 11.5px !important;
                    line-height: 1.35 !important;
                }}

                .eusee-feedback-panel-actions {{
                    width: 100% !important;
                    gap: 8px !important;
                    align-items: center !important;
                    justify-content: space-between !important;
                }}

                .eusee-feedback-panel-button {{
                    flex: 1 !important;
                    width: auto !important;
                }}
            }}
        `;
        doc.head.appendChild(style);

        const root = doc.createElement("div");
        root.id = rootId;
        root.className = "is-collapsed";
        root.innerHTML = `
            <button class="eusee-feedback-toggle" id="eusee-feedback-toggle" type="button" aria-label="Open feedback panel" aria-expanded="false" title="Open feedback panel">
                <span class="eusee-feedback-toggle-icon">💬</span>
                <span id="eusee-feedback-toggle-label">Feedback</span>
                <span class="eusee-feedback-toggle-caret" id="eusee-feedback-toggle-caret">+</span>
            </button>

            <div class="eusee-feedback-panel" id="eusee-feedback-panel" role="dialog" aria-label="Feedback panel">
                <div class="eusee-feedback-panel-left">
                    <div class="eusee-feedback-panel-icon">💬</div>
                    <div class="eusee-feedback-panel-copy">
                        <strong>Share your feedback</strong> on usability, insights, and dashboard improvements.
                    </div>
                </div>

                <div class="eusee-feedback-panel-actions">
                    <a class="eusee-feedback-panel-button" href="{feedback_url}" target="_blank" rel="noopener noreferrer">
                        Fill in the form
                    </a>
                    <button class="eusee-feedback-toggle eusee-feedback-panel-toggle" id="eusee-feedback-panel-toggle" type="button" aria-label="Collapse feedback panel" title="Collapse feedback panel">
                        <span>Collapse</span>
                        <span class="eusee-feedback-toggle-caret">−</span>
                    </button>
                </div>
            </div>
        `;
        doc.body.appendChild(root);

        const openButton = doc.getElementById("eusee-feedback-toggle");
        const closeButton = doc.getElementById("eusee-feedback-panel-toggle");

        function setFeedbackOpen(isOpen) {{
            root.classList.toggle("is-open", isOpen);
            root.classList.toggle("is-collapsed", !isOpen);
            openButton.setAttribute("aria-expanded", isOpen ? "true" : "false");
        }}

        openButton.addEventListener("click", function(event) {{
            event.preventDefault();
            event.stopPropagation();
            setFeedbackOpen(true);
        }});

        closeButton.addEventListener("click", function(event) {{
            event.preventDefault();
            event.stopPropagation();
            setFeedbackOpen(false);
        }});

        doc.addEventListener("keydown", function(event) {{
            if (event.key === "Escape") setFeedbackOpen(false);
        }});

        setFeedbackOpen(false);
    }})();
    </script>
    """, height=0, width=0)

render_top_feedback_bar()  # Single-button floating dashboard feedback overlay.



# ---------------- TAB 2: Negative Events ----------------
# Filter negative alerts
reactive_df = filtered_global[filtered_global['alert-impact'] == "Negative"].copy()

# Ensure all required columns exist
required_columns = [
    'Actor of repression',
    'Subject of repression',
    'Mechanism of repression',
    'Type of event',
    'alert-type',
    'enabling-principle'
]

for col in required_columns:
    if col not in reactive_df.columns:
        reactive_df[col] = np.nan
        st.warning(f"Column '{col}' was missing and has been added as empty.")

# ---------------- LABEL WRAPPING ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = str(label).split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)

def info_tooltip(message: str) -> str:
    """
    Returns a question mark HTML with tooltip.
    Use with st.markdown(..., unsafe_allow_html=True)
    """
    return f'<span style="font-weight:bold; cursor: help; color: #660094;" title="{message}">❓</span>'

# ---------------- RESPONSIVE SUMMARY CARDS ----------------
def render_summary_cards(df, base_bar_height=25, show_breakdown=True, card_key="summary"):
    """
    Render compact, equal-height professional KPI cards:
    1. Monitored Countries
    2. Total Alerts
    3. Alerts Breakdown as a contained donut plot
    """
    total_countries = df['alert-country'].nunique() if not df.empty else 0
    total_alerts = len(df) if not df.empty else 0
    negative = int((df['alert-impact'] == "Negative").sum()) if not df.empty else 0
    positive = int((df['alert-impact'] == "Positive").sum()) if not df.empty else 0
    context = int((df['alert-impact'] == "Context to watch").sum()) if not df.empty else 0
    total_np = negative + positive + context

    neg_pct = round((negative / total_np) * 100, 1) if total_np else 0
    pos_pct = round((positive / total_np) * 100, 1) if total_np else 0
    context_pct = round((context / total_np) * 100, 1) if total_np else 0

    neg_stop = neg_pct
    pos_stop = neg_pct + pos_pct

    if total_np:
        donut_gradient = (
            f"conic-gradient(#FFDB58 0% {neg_stop}%, "
            f"#660094 {neg_stop}% {pos_stop}%, "
            f"#008CAA {pos_stop}% 100%)"
        )
    else:
        donut_gradient = "conic-gradient(#E5E7EB 0% 100%)"

    st.markdown("""
    <style>
    /* ---------------- CLEAN PROFESSIONAL KPI SUMMARY CARDS ---------------- */
    .eusee-kpi-card {
        height: auto;
        min-height: 172px;
        background:
            radial-gradient(circle at 100% 0%, rgba(102, 0, 148, 0.055), transparent 34%),
            linear-gradient(180deg, #FFFFFF 0%, #FCFAFF 100%);
        border: 1px solid rgba(102, 0, 148, 0.115);
        border-radius: 18px;
        box-shadow: 0 12px 26px rgba(17, 24, 39, 0.070), inset 0 1px 0 rgba(255,255,255,0.95);
        padding: 14px 15px 13px 15px;
        margin: 2px 0 8px 0;
        box-sizing: border-box;
        overflow: visible;
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    /* Remove only the old top color strip; keep the card background shading. */
    .eusee-kpi-card::before {
        display: none !important;
        content: none !important;
        background: transparent !important;
        height: 0 !important;
    }

    .eusee-kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 32px rgba(17, 24, 39, 0.090), inset 0 1px 0 rgba(255,255,255,0.95);
        border-color: rgba(102, 0, 148, 0.180);
    }

    .eusee-kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-top: 0;
    }

    .eusee-kpi-eyebrow {
        color: #667085;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: .11em;
        text-transform: uppercase;
        line-height: 1;
        margin-bottom: 4px;
    }

    .eusee-kpi-title {
        color: #23152F;
        font-size: 12.5px;
        font-weight: 900;
        line-height: 1.08;
        letter-spacing: -.01em;
    }

    .eusee-kpi-icon {
        width: 30px;
        height: 30px;
        min-width: 30px;
        border-radius: 12px;
        background: #F8FAFC;
        color: #344054;
        border: 1px solid #EEF2F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 900;
        box-shadow: none;
    }

    .eusee-kpi-value {
        font-size: 36px;
        line-height: .92;
        font-weight: 950;
        margin-top: 9px;
        letter-spacing: -0.045em;
        font-family: Arial Black, Arial, sans-serif;
    }

    .eusee-kpi-note {
        color: #667085;
        font-size: 10.5px;
        font-weight: 700;
        line-height: 1.24;
        margin-top: 6px;
        white-space: normal;
    }

    .eusee-microline {
        height: 3px;
        width: 46px;
        border-radius: 999px;
        background: #E6E8EF;
        opacity: 1;
        margin-top: 9px;
    }

    .eusee-donut-layout {
        display: grid;
        grid-template-columns: 76px 1fr;
        align-items: center;
        gap: 9px;
        margin-top: 4px;
    }

    .eusee-donut {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        position: relative;
        background: var(--donut-gradient);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.95), 0 6px 14px rgba(17,24,39,.10);
    }

    .eusee-donut::before {
        content: "";
        position: absolute;
        inset: -3px;
        border-radius: 50%;
        background: #F8FAFC;
        z-index: -1;
    }

    .eusee-donut::after {
        content: "";
        position: absolute;
        inset: 17px;
        border-radius: 50%;
        background: #FFFFFF;
        box-shadow: inset 0 0 0 1px #E6E8EF;
    }

    .eusee-donut-center {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 1;
        color: #23152F;
        font-weight: 950;
        line-height: 1;
        pointer-events: none;
        font-family: Arial Black, Arial, sans-serif;
    }

    .eusee-donut-center .num {
        font-size: 14px;
        letter-spacing: -.03em;
    }

    .eusee-donut-center .lab {
        font-size: 7.8px;
        color: #667085;
        margin-top: 2px;
        font-family: Arial, sans-serif;
        font-weight: 800;
    }

    .eusee-breakdown-list {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .eusee-breakdown-row {
        display: grid;
        grid-template-columns: 10px minmax(48px, 1fr) 42px 42px;
        align-items: center;
        gap: 6px;
        padding: 4px 6px;
        border-radius: 10px;
        background: #FFFFFF;
        border: 1px solid #EEF2F6;
        box-shadow: none;
        line-height: 1;
    }

    .eusee-breakdown-row:hover {
        background: #F9FAFB;
        border-color: #E6E8EF;
    }

    .eusee-breakdown-label {
        color: #344054;
        font-size: 9.8px;
        font-weight: 950;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .eusee-breakdown-pct {
        color: #101828;
        font-size: 10.4px;
        font-weight: 950;
        text-align: right;
        font-family: Arial Black, Arial, sans-serif;
        letter-spacing: -.035em;
    }

    .eusee-breakdown-count {
        color: #667085;
        font-size: 9.5px;
        font-weight: 850;
        text-align: right;
        white-space: nowrap;
    }

    .eusee-dot {
        width: 8px;
        height: 8px;
        min-width: 8px;
        border-radius: 999px;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(255,255,255,.85), 0 1px 3px rgba(17,24,39,.14);
    }

    .eusee-breakdown-bar {
        grid-column: 2 / 5;
        height: 3px;
        background: #F2F4F7;
        border-radius: 999px;
        overflow: hidden;
        margin-top: -1px;
    }

    .eusee-breakdown-fill {
        height: 100%;
        border-radius: 999px;
        width: var(--bar-width);
        background: var(--bar-color);
        opacity: .82;
    }

    .eusee-tooltip {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 17px;
        height: 17px;
        margin-left: 5px;
        border-radius: 999px;
        background: linear-gradient(135deg, #F4EAF8 0%, #EFFBFE 100%);
        border: 1px solid rgba(102,0,148,.20);
        color: #660094;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        font-size: 10px;
        font-weight: 950;
        line-height: 1;
        cursor: help;
        box-shadow: 0 2px 7px rgba(16,24,40,.08);
        vertical-align: middle;
    }

    .eusee-tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 50%;
        top: calc(100% + 10px);
        bottom: auto;
        transform: translateX(-50%) translateY(-4px);
        width: min(320px, 72vw);
        padding: 10px 12px;
        border-radius: 12px;
        background: #23152F;
        border: 1px solid rgba(255,255,255,.14);
        color: #FFFFFF;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        font-size: 11px;
        font-weight: 650;
        line-height: 1.42;
        letter-spacing: -0.005em;
        text-align: left;
        white-space: normal;
        box-shadow: 0 16px 34px rgba(16,24,40,.22);
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
        z-index: 999999;
        transition: opacity .16s ease, transform .16s ease, visibility .16s ease;
    }

    .eusee-tooltip::before {
        content: "";
        position: absolute;
        left: 50%;
        top: calc(100% + 4px);
        bottom: auto;
        transform: translateX(-50%);
        border-width: 0 6px 6px 6px;
        border-style: solid;
        border-color: transparent transparent #23152F transparent;
        opacity: 0;
        visibility: hidden;
        z-index: 999999;
        transition: opacity .16s ease, visibility .16s ease;
    }

    .eusee-tooltip:hover::after,
    .eusee-tooltip:focus::after,
    .eusee-tooltip:hover::before,
    .eusee-tooltip:focus::before {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(0);
    }

    @media (max-width: 700px) {
        .eusee-tooltip::after {
            left: auto;
            right: -12px;
            transform: translateY(4px);
            width: min(280px, 82vw);
        }
        .eusee-tooltip:hover::after,
        .eusee-tooltip:focus::after {
            transform: translateY(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    countries_value = monitored_countries_display_value(total_countries)
    countries_size = "38px" if can_view_monitored_countries_value() else "18px" 
    with col1:
        st.markdown(f"""
        <div class="eusee-kpi-card">
            <div>
                <div class="eusee-kpi-top">
                    <div><div class="eusee-kpi-title">Monitored Countries</div></div>
                    <div class="eusee-kpi-icon">🌍</div>
                </div>
                <div class="eusee-kpi-value" style="color:#008CAA;font-size:36px;">{countries_value}</div><div class="eusee-microline" style="color:#008CAA;"></div>
            </div>
            
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="eusee-kpi-card">
            <div>
                <div class="eusee-kpi-top">
                    <div><div class="eusee-kpi-title">Total Alerts <span class="eusee-tooltip" tabindex="0" aria-label="Total alerts interpretation note" data-tooltip="Higher numbers of alerts do not always indicate a worse situation; they may reflect better reporting or different thresholds across countries.">?</span></div></div>
                    <div class="eusee-kpi-icon">⚠️</div>
                </div>
                <div class="eusee-kpi-value" style="color:#FF6F61;">{total_alerts:,}</div><div class="eusee-microline" style="color:#FF6F61;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="eusee-kpi-card">
            <div class="eusee-kpi-top">
                <div><div class="eusee-kpi-title">Alerts Breakdown</div></div>
                <div class="eusee-kpi-icon">◔</div>
            </div>
            <div class="eusee-donut-layout">
                <div class="eusee-donut" style="--donut-gradient:{donut_gradient};" title="Negative: {negative:,} ({neg_pct}%) | Positive: {positive:,} ({pos_pct}%) | Context to watch: {context:,} ({context_pct}%)">
                    <div class="eusee-donut-center">
                        <div class="num">{total_np:,}</div>
                        <div class="lab">alerts</div>
                    </div>
                </div>
                <div class="eusee-breakdown-list">
                    <div class="eusee-breakdown-row" title="Negative alerts: {negative:,} records, {neg_pct}% of filtered alerts">
                        <span class="eusee-dot" style="background:#FFDB58;"></span>
                        <span class="eusee-breakdown-label">Negative</span>
                        <span class="eusee-breakdown-pct">{neg_pct}%</span>
                        <span class="eusee-breakdown-count">{negative:,}</span>
                        <div class="eusee-breakdown-bar"><div class="eusee-breakdown-fill" style="--bar-width:{neg_pct}%; --bar-color:#FFDB58;"></div></div>
                    </div>
                    <div class="eusee-breakdown-row" title="Positive alerts: {positive:,} records, {pos_pct}% of filtered alerts">
                        <span class="eusee-dot" style="background:#660094;"></span>
                        <span class="eusee-breakdown-label">Positive</span>
                        <span class="eusee-breakdown-pct">{pos_pct}%</span>
                        <span class="eusee-breakdown-count">{positive:,}</span>
                        <div class="eusee-breakdown-bar"><div class="eusee-breakdown-fill" style="--bar-width:{pos_pct}%; --bar-color:#660094;"></div></div>
                    </div>
                    <div class="eusee-breakdown-row" title="Context to watch alerts: {context:,} records, {context_pct}% of filtered alerts">
                        <span class="eusee-dot" style="background:#008CAA;"></span>
                        <span class="eusee-breakdown-label">Context to watch</span>
                        <span class="eusee-breakdown-pct">{context_pct}%</span>
                        <span class="eusee-breakdown-count">{context:,}</span>
                        <div class="eusee-breakdown-bar"><div class="eusee-breakdown-fill" style="--bar-width:{context_pct}%; --bar-color:#008CAA;"></div></div>
                    </div>
                </div>
            </div>
        </div>

        """, unsafe_allow_html=True)



def _top_split_item_for_negative_card(df, col, protected_label="Journalists, media and influencers"):
    """Return the most frequent comma-separated item for a negative-alert intelligence card."""
    if df is None or df.empty or col not in df.columns:
        return "Not available", 0
    placeholder = "Journalists__MEDIA__and__influencers"
    s = df[col].dropna().astype(str).str.strip()
    s = s.str.replace(r"\bVNSAs\b", "Violent non-state actors", regex=True)
    s = s.str.replace(protected_label, placeholder, regex=False)
    exploded = (
        s.str.split(",")
        .explode()
        .astype(str)
        .str.strip()
        .str.replace(placeholder, protected_label, regex=False)
    )
    exploded = exploded[(exploded != "") & (exploded.str.lower() != "nan") & (exploded.str.lower() != "none")]
    if exploded.empty:
        return "Not available", 0
    counts = exploded.value_counts()
    return str(counts.index[0]), int(counts.iloc[0])


def _compact_text_for_card(value, max_len=42):
    value = str(value or "Not available").strip()
    return value if len(value) <= max_len else value[: max_len - 1].rstrip() + "…"


def render_negative_alerts_intelligence_cards(negative_df, all_filtered_df=None, card_key="negative_intelligence"):
    """Render a Negative Alerts-specific KPI/intelligence row.

    This replaces the generic Negative/Positive/Context donut in the Negative Alerts tab,
    where all records are already negative and a composition donut adds limited value.
    """
    negative_total = len(negative_df) if negative_df is not None else 0
    all_total = len(all_filtered_df) if all_filtered_df is not None and not all_filtered_df.empty else negative_total
    negative_share = round((negative_total / all_total) * 100, 1) if all_total else 0

    top_actor, top_actor_count = _top_split_item_for_negative_card(negative_df, "Actor of repression")
    top_mechanism, top_mechanism_count = _top_split_item_for_negative_card(negative_df, "Mechanism of repression")
    top_subject, top_subject_count = _top_split_item_for_negative_card(negative_df, "Subject of repression")

    top_country = "Not available"
    top_country_count = 0
    monitored_countries = 0
    if negative_df is not None and not negative_df.empty and "alert-country" in negative_df.columns:
        country_series = negative_df["alert-country"].dropna().astype(str).str.strip().replace("", np.nan).dropna()
        monitored_countries = int(country_series.nunique()) if not country_series.empty else 0
        country_counts = country_series.value_counts()
        if not country_counts.empty:
            top_country = str(country_counts.index[0])
            top_country_count = int(country_counts.iloc[0])

    actor_pct = round((top_actor_count / negative_total) * 100, 1) if negative_total else 0
    mech_pct = round((top_mechanism_count / negative_total) * 100, 1) if negative_total else 0
    subject_pct = round((top_subject_count / negative_total) * 100, 1) if negative_total else 0

    st.markdown("""
    <style>
    .negintel-card {
        height: auto;
        min-height: 172px;
        /* Clean negative-analysis summary card background: keep the soft card shade, remove the colored top glow. */
        background: linear-gradient(180deg, #FFFFFF 0%, #FFFCFB 100%);
        border: 1px solid rgba(180, 35, 24, 0.12);
        border-radius: 17px;
        box-shadow: 0 12px 26px rgba(17, 24, 39, 0.070), inset 0 1px 0 rgba(255,255,255,0.95);
        padding: 11px 14px 10px 14px;
        margin: 2px 0 8px 0;
        box-sizing: border-box;
        overflow: visible;
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
    }
    /* Remove the colored strip/shadow from the top of Negative Alert Analysis summary cards. */
    .negintel-card::before {
        display: none !important;
        content: none !important;
        background: transparent !important;
        height: 0 !important;
    }
    .negintel-top { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-top:2px; }
    .negintel-eyebrow { color:#9A6B66; font-size:9px; font-weight:900; letter-spacing:.10em; text-transform:uppercase; line-height:1; margin-bottom:4px; }
    .negintel-title { color:#2D0055; font-size:12.5px; font-weight:900; line-height:1.05; letter-spacing:-.01em; }
    .negintel-icon { width:30px; height:30px; min-width:30px; border-radius:12px; background:linear-gradient(135deg, rgba(180,35,24,.12), rgba(255,219,88,.14)); color:#B42318; border:1px solid rgba(180,35,24,.10); display:flex; align-items:center; justify-content:center; font-size:16px; font-weight:900; }
    .negintel-value { font-size:34px; line-height:.92; font-weight:950; margin-top:8px; letter-spacing:-0.045em; font-family:Arial Black, Arial, sans-serif; color:#B42318; }
    .negintel-note { color:#667085; font-size:10px; font-weight:700; line-height:1.18; margin-top:4px; white-space:normal; }
    .negintel-pill { display:inline-flex; align-items:center; gap:5px; width:fit-content; border-radius:999px; padding:5px 9px; font-size:10px; font-weight:900; background:#FFF4ED; color:#B42318; border:1px solid rgba(180,35,24,.14); margin-top:7px; }
    .negintel-row-list { display:flex; flex-direction:column; gap:6px; margin-top:7px; }
    .negintel-row {
        display:grid;
        grid-template-columns: minmax(0, 1fr) 52px 42px;
        align-items:start;
        gap:8px;
        padding:7px 8px;
        border-radius:11px;
        background:rgba(255,255,255,.78);
        border:1px solid rgba(102,0,148,.065);
        line-height:1.18;
        min-height:34px;
    }
    .negintel-row-label {
        color:#344054;
        font-size:9.2px;
        font-weight:750;
        overflow:visible;
        text-overflow:unset;
        white-space:normal;
        overflow-wrap:anywhere;
        word-break:normal;
        line-height:1.22;
    }
    .negintel-row-label strong { color:#2D0055; font-weight:950; }
    .negintel-row-pct { color:#101828; font-size:10.4px; font-weight:950; text-align:right; font-family:Arial Black, Arial, sans-serif; letter-spacing:-.035em; white-space:nowrap; padding-top:1px; }
    .negintel-row-count { color:#667085; font-size:9.7px; font-weight:850; text-align:right; white-space:nowrap; padding-top:2px; }
    @media (max-width: 900px) {
        .negintel-row { grid-template-columns: minmax(0, 1fr) 50px 42px; }
        .negintel-row-label { font-size:10.4px; }
    }
    @media (max-width: 520px) {
        .negintel-row { grid-template-columns: minmax(0, 1fr); gap:3px; }
        .negintel-row-pct, .negintel-row-count { text-align:left; padding-top:0; }
    }
    .negintel-compact-line { font-size:10.2px; color:#344054; line-height:1.22; font-weight:850; margin-top:5px; }
    .negintel-compact-line strong { color:#2D0055; font-weight:950; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        countries_value = monitored_countries_display_value(monitored_countries)
        countries_size = "34px" if can_view_monitored_countries_value() else "18px" 
        st.markdown(f"""
        <div class="negintel-card">
            <div>
                <div class="negintel-top">
                    <div><div class="negintel-title">Monitored Countries</div></div>
                    <div class="negintel-icon">🌍</div>
                </div>
                <div class="negintel-value" style="color:#008CAA;font-size:36px;">{countries_value}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="negintel-card">
            <div>
                <div class="negintel-top">
                    <div><div class="negintel-title">Total Negative alerts</div></div>
                    <div class="negintel-icon">⚠️</div>
                </div>
                <div class="negintel-value">{negative_total:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="negintel-card">           
                <div class="negintel-top">
                    <div><div class="negintel-eyebrow">Frequent Restriction Pattern</div><div class="negintel-title"></div></div>
                    <div class="negintel-icon">⛓️</div>
                </div>
                <div class="negintel-row-list">
                    <div class="negintel-row" title="Top restrictive actor: {top_actor}"><span class="negintel-row-label"><strong> Restrictive Actor:</strong> {top_actor}</span><span class="negintel-row-pct">{actor_pct}%</span><span class="negintel-row-count">{top_actor_count:,}</span></div>
                    <div class="negintel-row" title="Top restrictive mechanism: {top_mechanism}"><span class="negintel-row-label"><strong>Restrictive Mechanism:</strong> {top_mechanism}</span><span class="negintel-row-pct">{mech_pct}%</span><span class="negintel-row-count">{top_mechanism_count:,}</span></div>
                    <div class="negintel-row" title="Top affected civil society actor: {top_subject}"><span class="negintel-row-label"><strong>Civil society actor affected:</strong> {top_subject}</span><span class="negintel-row-pct">{subject_pct}%</span><span class="negintel-row-count">{top_subject_count:,}</span></div>
                </div>
            
        </div>
        """, unsafe_allow_html=True)


def normalize_label(label: str) -> str:
    """
    Capitalize first character only, lowercase remaining characters.
    Safe for None/NaN.
    """
    if pd.isna(label):
        return ""
    label = str(label).strip()
    if len(label) == 0:
        return ""
    return label[0].upper() + label[1:].lower()

def wrap_label_by_words(label, words_per_line=3):
    """Wrap long labels for better display"""
    words = label.split()
    lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return '<br>'.join(lines)

def safe_wrap_label(label, axis="y", words_per_line=4):
    """
    Wrap labels ONLY for y-axis.
    X-axis wrapping breaks Plotly font rendering.
    """
    if axis == "x":
        return normalize_label(label)
    return wrap_label_by_words(normalize_label(label), words_per_line)


# ---------------- PROFESSIONAL CHART UX THEME ----------------
CHART_COLORS = {
    "Positive": "#660094",
    "Postive": "#660094",
    "Negative": "#FFDB58",
    "Context to watch": "#008CAA",
    "Default": "#FFDB58",
}

CHART_FONT = "Inter, Arial, sans-serif"
CHART_TITLE_COLOR = "#2D0055"
CHART_TEXT_COLOR = "#263238"
CHART_GRID_COLOR = "#EEF1F6"
CHART_AXIS_COLOR = "#D8DEE9"


def apply_classic_chart_theme(fig, title=None, height=None, horizontal=False, showlegend=True):
    """Apply one professional, classic dashboard style without changing chart data."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, size=11, color=CHART_TEXT_COLOR),
        title=dict(
            text=title if title is not None else fig.layout.title.text,
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(family=CHART_FONT, size=14, color=CHART_TITLE_COLOR),
        ),
        margin=dict(l=135 if horizontal else 46, r=28, t=58, b=58),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            font=dict(family=CHART_FONT, size=12, color=CHART_TEXT_COLOR),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(230,232,239,0.65)",
            borderwidth=1,
            font=dict(family=CHART_FONT, size=9, color="#344054"),
            title=None,
            itemsizing="trace",
            itemwidth=30,
            tracegroupgap=0,
        ),
        showlegend=showlegend,
    )
    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridwidth=1,
        gridcolor=CHART_GRID_COLOR,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=CHART_AXIS_COLOR,
        ticks="",
        tickfont=dict(family=CHART_FONT, size=10, color="#52616B"),
    )
    fig.update_yaxes(
        title=None,
        showgrid=False if horizontal else True,
        gridwidth=1,
        gridcolor=CHART_GRID_COLOR,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=CHART_AXIS_COLOR,
        ticks="",
        tickfont=dict(family=CHART_FONT, size=10, color="#52616B"),
    )
    return fig


def render_chart_shell():
    """Global chart container polish: subtle cards, spacing and consistent dashboard feel."""
    st.markdown(
        """
        <style>
        div[data-testid="stPlotlyChart"] {
            background: #FFFFFF;
            border: 1px solid #E9E2F2;
            border-radius: 18px;
            padding: 8px 10px 4px 10px;
            box-shadow: 0 10px 28px rgba(45, 0, 85, 0.055);
            margin-bottom: 18px;
            transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
        }
        div[data-testid="stPlotlyChart"]:hover {
            transform: translateY(-1px);
            border-color: #D8C7E6;
            box-shadow: 0 14px 34px rgba(45, 0, 85, 0.09);
        }
        div[data-testid="stPlotlyChart"] svg.main-svg {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

render_chart_shell()

# ---------------- PERCENT AXIS / STANDARD HEIGHT HELPERS ----------------
CHART_HEIGHT_VERTICAL = 350
CHART_HEIGHT_HORIZONTAL = 350


def _nice_percent_axis_max(max_pct):
    """Return a readable percentage axis maximum based on observed max percentage."""
    try:
        max_pct = float(max_pct)
    except Exception:
        return 10
    if max_pct <= 0:
        return 5
    if max_pct <= 2:
        return 2.5
    if max_pct <= 5:
        return 6
    if max_pct <= 10:
        return 12
    if max_pct <= 15:
        return 17
    if max_pct <= 20:
        return 25
    if max_pct <= 30:
        return 35
    if max_pct <= 40:
        return 45
    if max_pct <= 50:
        return 55
    if max_pct <= 60:
        return 70
    if max_pct <= 80:
        return 90
    return 100


def _standard_chart_height(horizontal=False):
    """Keep all bar/stacked charts visually consistent."""
    return CHART_HEIGHT_HORIZONTAL if horizontal else CHART_HEIGHT_VERTICAL


# ---------------- DYNAMIC BAR CHART ----------------
def create_bar_chart(df, x, y, title=None, horizontal=False, color_col=None, normalize_labels=True):
    """Create a percentage bar chart with standard height and dynamic percent axis."""
    df = df.copy()

    if df is None or df.empty or x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return apply_classic_chart_theme(
            fig,
            title=title,
            height=_standard_chart_height(horizontal),
            horizontal=horizontal,
            showlegend=False,
        )

    df[y] = pd.to_numeric(df[y], errors="coerce").fillna(0)
    df["raw_count"] = df[y]

    total_count = float(df["raw_count"].sum())
    df["percent_value"] = np.where(
        total_count > 0,
        (df["raw_count"] / total_count) * 100,
        0,
    ).round(1)

    df["percent_label"] = df["percent_value"].map(
        lambda v: f"{v:.1f}%" if v > 0 else ""
    )

    max_pct = float(df["percent_value"].max()) if not df.empty else 0
    axis_max = _nice_percent_axis_max(max_pct)
    height = _standard_chart_height(horizontal)

    if normalize_labels:
        df[x] = df[x].apply(
            lambda l: wrap_label_by_words(
                normalize_label(l) if x not in ["alert-country", "region"] else str(l),
                words_per_line=3,
            )
        )
    else:
        df[x] = df[x].astype(str).apply(
            lambda l: wrap_label_by_words(l, words_per_line=3)
        )

    if "Other" in df[x].values:
        df_other = df[df[x] == "Other"]
        df_main = df[df[x] != "Other"]
        df = pd.concat([df_main, df_other], ignore_index=True)
        if horizontal:
            df = df[::-1].reset_index(drop=True)

    fig = px.bar(
        df,
        x="percent_value" if horizontal else x,
        y=x if horizontal else "percent_value",
        orientation="h" if horizontal else "v",
        color=color_col,
        color_discrete_sequence=[CHART_COLORS["Default"]],
        text="percent_label",
        custom_data=["raw_count", "percent_value"],
    )

    fig.update_traces(
        textposition=[
            "inside" if val >= (axis_max * 0.12) else "outside"
            for val in df["percent_value"]
        ],
        insidetextanchor="end",
        texttemplate="%{text}",
        textfont=dict(size=11, color="#1F2937", family=CHART_FONT),
        marker_line=dict(color="rgba(255,255,255,0.75)", width=0.8),
        hovertemplate=(
            "<b>%{y}</b><br>" if horizontal else "<b>%{x}</b><br>"
        ) + "Share: %{customdata[1]:.1f}%<br>Count: %{customdata[0]:,.0f}<extra></extra>",
    )

    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_xaxes(
            title="Percent of total",
            ticksuffix="%",
            range=[0, axis_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_yaxes(
            title="Percent of total",
            ticksuffix="%",
            range=[0, axis_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

    fig = apply_classic_chart_theme(
        fig,
        title=title,
        height=height,
        horizontal=horizontal,
        showlegend=bool(color_col),
    )

    if horizontal:
        fig.update_xaxes(title="Percent of total", ticksuffix="%", range=[0, axis_max])
    else:
        fig.update_yaxes(title="Percent of total", ticksuffix="%", range=[0, axis_max])

    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="black"),
        opacity=0.035,
        xanchor="center",
        yanchor="middle",
    )

    return fig

# ---------------- STACKED BAR LABEL CONTRAST HELPER ----------------
def readable_stacked_bar_label_color(hex_color):
    """Return readable value-label color for stacked-bar segments."""
    try:
        value = str(hex_color or "").strip().lower().replace(" ", "")

        purple_tokens = {
            "#660094",
            "660094",
            "purple",
            "rgb(102,0,148)",
            "rgba(102,0,148,1)",
            "rgba(102,0,148,1.0)",
        }

        if value in purple_tokens:
            return "#FFFFFF"

        if value.startswith("rgba") or value.startswith("rgb"):
            nums = re.findall(r"[0-9.]+", value)
            if len(nums) >= 3:
                r, g, b = [int(float(n)) for n in nums[:3]]
                if (r, g, b) == (102, 0, 148):
                    return "#FFFFFF"

        if value.startswith("#"):
            hex_value = value.replace("#", "")
            if len(hex_value) == 3:
                hex_value = "".join(ch * 2 for ch in hex_value)
            if hex_value == "660094":
                return "#FFFFFF"

        return "#111827"
    except Exception:
        return "#111827"

# ---------------- HORIZONTAL STACKED BAR ----------------
def create_h_stacked_bar(df, y, x="count", color_col="alert-impact", title=None, horizontal=False, normalize_labels=True):
    """Create a stacked bar chart using percent of grand total with standard height."""
    df = df.copy()

    if df is None or df.empty or y not in df.columns or x not in df.columns or color_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return apply_classic_chart_theme(
            fig,
            title=title,
            height=_standard_chart_height(horizontal),
            horizontal=horizontal,
            showlegend=False,
        )

    df[x] = pd.to_numeric(df[x], errors="coerce").fillna(0)
    df["raw_count"] = df[x]

    grand_total = float(df["raw_count"].sum())
    df["percent_value"] = np.where(
        grand_total > 0,
        (df["raw_count"] / grand_total) * 100,
        0,
    ).round(1)

    df["percent_label"] = df["percent_value"].map(
        lambda v: f"{v:.1f}%" if v > 0 else ""
    )

    if "Other" in df[y].astype(str).values:
        df_other = df[df[y].astype(str) == "Other"]
        df_main = df[df[y].astype(str) != "Other"]
        df = pd.concat([df_main, df_other], ignore_index=True)

    if normalize_labels:
        df[y] = df[y].apply(
            lambda l: wrap_label_by_words(normalize_label(l), words_per_line=4)
        )
    else:
        df[y] = df[y].apply(lambda l: wrap_label_by_words(l, words_per_line=4))

    ordered_y = list(dict.fromkeys(df[y].tolist()))[::-1] if horizontal else list(dict.fromkeys(df[y].tolist()))

    max_pct = float(df.groupby(y)["percent_value"].sum().max()) if not df.empty else 0
    axis_max = _nice_percent_axis_max(max_pct)
    height = _standard_chart_height(horizontal)

    categories = sorted(df[color_col].dropna().unique())
    category_colors = CHART_COLORS

    fig = go.Figure()

    for cat in categories:
        df_cat = df[df[color_col] == cat].copy()
        df_cat[y] = pd.Categorical(df_cat[y], categories=ordered_y, ordered=True)
        df_cat = df_cat.sort_values(y)

        bar_color = category_colors.get(cat, "#660094")
        label_color = readable_stacked_bar_label_color(bar_color)

        fig.add_trace(go.Bar(
            x=df_cat["percent_value"] if horizontal else df_cat[y],
            y=df_cat[y] if horizontal else df_cat["percent_value"],
            name=cat,
            orientation="h" if horizontal else "v",
            marker_color=bar_color,
            text=df_cat["percent_label"],
            customdata=np.stack([df_cat["raw_count"], df_cat["percent_value"]], axis=-1),
            textposition="inside",
            insidetextanchor="middle",
            texttemplate="%{text}",
            textfont=dict(color=label_color, size=11, family=CHART_FONT),
            marker_line=dict(color="rgba(255,255,255,0.72)", width=0.8),
            hovertemplate=(
                f"<b>%{{y}}</b><br>{cat}: %{{customdata[1]:.1f}}% of total<br>Count: %{{customdata[0]:,.0f}}<extra></extra>"
                if horizontal else
                f"<b>%{{x}}</b><br>{cat}: %{{customdata[1]:.1f}}% of total<br>Count: %{{customdata[0]:,.0f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        height=height,
        margin=dict(l=120 if horizontal else 20, r=20, t=20, b=20),
    )

    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_xaxes(
            title="Percent of total",
            ticksuffix="%",
            range=[0, axis_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_yaxes(
            title="Percent of total",
            ticksuffix="%",
            range=[0, axis_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

    fig = apply_classic_chart_theme(
        fig,
        title=title,
        height=height,
        horizontal=horizontal,
        showlegend=True,
    )

    fig.update_layout(barmode="stack")

    if horizontal:
        fig.update_xaxes(title="Percent of total", ticksuffix="%", range=[0, axis_max])
    else:
        fig.update_yaxes(title="Percent of total", ticksuffix="%", range=[0, axis_max])

    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="black"),
        opacity=0.035,
        xanchor="center",
        yanchor="middle",
    )

    return fig


# ---------------- HELPER FUNCTIONS ----------------
def filter_top_n(df, row_col, col_col, top_n=None):
    """
    Creates a pivot table for heatmaps, keeping only top-N rows if specified.
    """
    pivot_df = (
        df.groupby([row_col, col_col])
        .size()
        .reset_index(name='count')
    )

    if top_n is not None:
        top_rows = (
            pivot_df.groupby(row_col)['count']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        pivot_df = pivot_df[pivot_df[row_col].isin(top_rows)]

    heatmap_df = pivot_df.pivot(index=row_col, columns=col_col, values='count').fillna(0)
    return heatmap_df

# ---------------- PROFESSIONAL RELATIONSHIP ANALYTICS HELPERS ----------------
def _safe_chart_label(label, words_per_line=3, max_chars=42):
    """Readable compact label for dense heatmaps/Sankey nodes."""
    text = normalize_label(label) if 'normalize_label' in globals() else str(label)
    text = str(text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return wrap_label_by_words(text, words_per_line=words_per_line)


def render_analytics_module_header(title, subtitle, badges=None):
    """Consistent executive-style header for complex analytical modules."""
    badges = badges or []
    badge_html = "".join([f'<span class="analytics-badge">{b}</span>' for b in badges])
    st.markdown(f"""
    <style>
    .analytics-panel {{
        background: linear-gradient(180deg, #FFFFFF 0%, #FBFAFD 100%);
        border: 1px solid #E8E1F0;
        border-radius: 18px;
        padding: 14px 16px 12px 16px;
        margin: 12px 0 14px 0;
        box-shadow: 0 8px 26px rgba(45, 0, 85, 0.055);
    }}
    .analytics-panel-title {{
        font-family: Inter, Arial, sans-serif;
        font-size: 15px;
        font-weight: 900;
        color: #2D0055;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }}
    .analytics-panel-subtitle {{
        font-family: Inter, Arial, sans-serif;
        font-size: 11.8px;
        color: #52616B;
        line-height: 1.45;
        max-width: 980px;
    }}
    .analytics-badge {{
        display: inline-block;
        background: #F5EFFA;
        color: #660094;
        border: 1px solid #E6D7F0;
        border-radius: 999px;
        padding: 4px 9px;
        margin: 7px 6px 0 0;
        font-family: Inter, Arial, sans-serif;
        font-size: 10.5px;
        font-weight: 800;
    }}
    .chart-card-caption {{
        font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
        font-size: 9.5px;
        font-weight: 550;
        color: var(--eusee-text-muted, #667085);
        line-height: 1.42;
        letter-spacing: -0.005em;
        margin-top: -4px;
        margin-bottom: 10px;
        max-width: 980px;
    }}
    @media (max-width: 900px) {{
        .chart-card-caption {{
            font-size: 9.2px;
            line-height: 1.4;
        }}
    }}
    </style>
    <div class="analytics-panel">
        <div class="analytics-panel-title">{title}</div>
        <div class="analytics-panel-subtitle">{subtitle}</div>
        <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def create_heatmap(pivot_df, title="Heatmap", x_label="", y_label=""):
    """Professional Plotly heatmap for relationship matrices."""
    if pivot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No matching relationship data",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=13, color="#64748B", family=CHART_FONT)
        )
        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=48, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=title, x=0.02, xanchor="left", font=dict(size=13, family=CHART_FONT, color=CHART_TITLE_COLOR))
        )
        return fig

    plot_df = pivot_df.copy()
    plot_df.index = [_safe_chart_label(i, words_per_line=2, max_chars=36) for i in plot_df.index]
    plot_df.columns = [_safe_chart_label(i, words_per_line=2, max_chars=34) for i in plot_df.columns]

    z_values = plot_df.values.astype(float)
    zmax = float(np.nanmax(z_values)) if z_values.size else 1

    colorscale = [
        [0.00, "#F8FAFC"],
        [0.18, "#EEF7FA"],
        [0.42, "#CFEAF0"],
        [0.68, "#9D77BF"],
        [1.00, "#660094"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=plot_df.values,
        x=plot_df.columns,
        y=plot_df.index,
        colorscale=colorscale,
        zmin=0,
        zmax=zmax if zmax > 0 else 1,
        xgap=2,
        ygap=2,
        hovertemplate="<b>%{y}</b><br>↳ <b>%{x}</b><br><span style='color:#64748B'>Alerts:</span> <b>%{z}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Alerts", font=dict(size=11, family=CHART_FONT, color="#52616B")),
            tickfont=dict(size=10, family=CHART_FONT, color="#52616B"),
            thickness=9,
            len=0.70,
            outlinewidth=0,
            bgcolor="rgba(255,255,255,0)",
        ),
    ))

    if plot_df.shape[0] <= 8 and plot_df.shape[1] <= 8:
        annotations = []
        for iy, yv in enumerate(plot_df.index):
            for ix, xv in enumerate(plot_df.columns):
                val = plot_df.values[iy][ix]
                if val > 0:
                    annotations.append(dict(
                        x=xv, y=yv, text=str(int(val)), showarrow=False,
                        font=dict(size=10, family=CHART_FONT, color="#FFFFFF" if val >= zmax * 0.55 else "#2D0055")
                    ))
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=13.5, family=CHART_FONT, color=CHART_TITLE_COLOR)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=max(340, min(560, 275 + plot_df.shape[0] * 34)),
        margin=dict(l=105, r=30, t=58, b=105),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=CHART_FONT, size=10.5, color=CHART_TEXT_COLOR),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#D9E2EC", font=dict(color=CHART_TEXT_COLOR, family=CHART_FONT, size=11)),
    )
    fig.update_xaxes(
        tickangle=-30, showgrid=False, zeroline=False, showline=False, ticks="",
        tickfont=dict(size=9.8, family=CHART_FONT, color="#52616B"),
        title_font=dict(size=10.5, family=CHART_FONT, color="#64748B"),
    )
    fig.update_yaxes(
        autorange="reversed", showgrid=False, zeroline=False, showline=False, ticks="",
        tickfont=dict(size=9.8, family=CHART_FONT, color="#52616B"),
        title_font=dict(size=10.5, family=CHART_FONT, color="#64748B"),
    )
    return fig


# ---------------- HELPER: Get Top-N Items ----------------
def get_top_n_items(df, col, top_n):
    counts = df[col].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return counts.index.tolist()


# ---------------- PROFESSIONAL HEATMAP RENDER FUNCTION ----------------
def render_heatmaps(df, top_n=5):
    if df.empty:
        st.warning("No data available for heatmaps.")
        return

    protected_label = "Journalists, media and influencers"
    placeholder = "Journalists__MEDIA__and__influencers"

    def safe_split(x):
        if pd.isna(x):
            return []
        x = str(x).strip()
        if not x:
            return []
        x = x.replace(protected_label, placeholder)
        parts = [i.strip() for i in x.split(",") if str(i).strip()]
        return [p.replace(placeholder, protected_label) for p in parts]

    df_exploded = df.copy()
    explode_cols = ["Actor of repression", "Subject of repression", "Mechanism of repression"]
    for col in explode_cols:
        df_exploded[col] = df_exploded[col].apply(safe_split)
        df_exploded = df_exploded.explode(col)
        df_exploded[col] = df_exploded[col].astype(str).str.strip()

    df_exploded = df_exploded[
        (df_exploded["Actor of repression"] != "") &
        (df_exploded["Subject of repression"] != "") &
        (df_exploded["Mechanism of repression"] != "")
    ].copy()

    if df_exploded.empty:
        st.info("No actor–mechanism–subject relationship data are available under the current filters.")
        return

    top_actors = get_top_n_items(df_exploded, "Actor of repression", top_n)
    top_subjects = get_top_n_items(df_exploded, "Subject of repression", top_n)
    top_mechanisms = get_top_n_items(df_exploded, "Mechanism of repression", top_n)

    df_top = df_exploded[
        df_exploded["Actor of repression"].isin(top_actors) &
        df_exploded["Subject of repression"].isin(top_subjects) &
        df_exploded["Mechanism of repression"].isin(top_mechanisms)
    ].copy()

    if df_top.empty:
        st.warning("No heatmap data available after applying the Top-N selection.")
        return

    actor_mechanism_pivot = filter_top_n(df_top, "Actor of repression", "Mechanism of repression", top_n)
    subject_mechanism_pivot = filter_top_n(df_top, "Subject of repression", "Mechanism of repression", top_n)
    actor_subject_pivot = filter_top_n(df_top, "Actor of repression", "Subject of repression", top_n)

    all_values = pd.concat([actor_mechanism_pivot.stack(), subject_mechanism_pivot.stack(), actor_subject_pivot.stack()])
    zmax = float(all_values.max()) if not all_values.empty else 1

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        fig1 = create_heatmap(actor_mechanism_pivot, title="What are the mechanisms used<br>by restrictive actors?", x_label="Restrictive Mechanism", y_label="Restrictive Actor")
        fig1.update_traces(zmin=0, zmax=zmax)
        render_dashboard_plotly_chart(fig1, plot_df=actor_mechanism_pivot.stack().reset_index(name="count"), visual_type="heatmap", x_col="Actor of repression", group_col="Mechanism of repression", dashboard_df=df_top, config={"displayModeBar": False}, key="heatmap_actor_mechanism_pro", permission_key="view_chart_heatmap_actor_mechanism", permission_label="Actor × mechanism heatmap")
        
    with c2:
        fig2 = create_heatmap(subject_mechanism_pivot, title="What are the restrictive mechanisms<br>affecting civil society actors?", x_label="Restrictive Mechanism", y_label="Affected civil society group")
        fig2.update_traces(zmin=0, zmax=zmax)
        render_dashboard_plotly_chart(fig2, plot_df=subject_mechanism_pivot.stack().reset_index(name="count"), visual_type="heatmap", x_col="Subject of repression", group_col="Mechanism of repression", dashboard_df=df_top, config={"displayModeBar": False}, key="heatmap_subject_mechanism_pro", permission_key="view_chart_heatmap_subject_mechanism", permission_label="Affected actor × mechanism heatmap")
        
    with c3:
        fig3 = create_heatmap(actor_subject_pivot, title="Who are the actors restricting<br>civil society?", x_label="Affected civil society group", y_label="Restrictive actor")
        fig3.update_traces(zmin=0, zmax=zmax)
        render_dashboard_plotly_chart(fig3, plot_df=actor_subject_pivot.stack().reset_index(name="count"), visual_type="heatmap", x_col="Actor of repression", group_col="Subject of repression", dashboard_df=df_top, config={"displayModeBar": False}, key="heatmap_actor_subject_pro", permission_key="view_chart_heatmap_actor_subject", permission_label="Actor × affected actor heatmap")
        
# ---------------- PROFESSIONAL SANKEY FUNCTION ----------------
def render_sankey(df, top_n=None, width=900, wrap_width=22):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for flow analysis", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    protected_label = "Journalists, media and influencers"
    placeholder = "Journalists__MEDIA__and__influencers"

    def split_values(x):
        if pd.isna(x):
            return []
        x = str(x).strip().replace(protected_label, placeholder)
        parts = [i.strip() for i in x.split(",") if i.strip()]
        return [p.replace(placeholder, protected_label) for p in parts]

    flow_df = df.copy()
    for col in ["Actor of repression", "Mechanism of repression", "Subject of repression"]:
        flow_df[col] = flow_df[col].apply(split_values)
        flow_df = flow_df.explode(col)
        flow_df[col] = flow_df[col].astype(str).str.strip()

    flow_df = flow_df[
        (flow_df["Actor of repression"] != "") &
        (flow_df["Mechanism of repression"] != "") &
        (flow_df["Subject of repression"] != "")
    ].copy()

    if flow_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No complete actor–mechanism–subject flows available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    def top_nodes(col):
        values = flow_df[col].value_counts()
        return values.head(top_n).index.tolist() if top_n is not None else values.index.tolist()

    top_actors = top_nodes("Actor of repression")
    top_mechanisms = top_nodes("Mechanism of repression")
    top_subjects = top_nodes("Subject of repression")

    flow_df = flow_df[
        flow_df["Actor of repression"].isin(top_actors) &
        flow_df["Mechanism of repression"].isin(top_mechanisms) &
        flow_df["Subject of repression"].isin(top_subjects)
    ].copy()

    if flow_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No flows remain after Top-N filtering", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    def wrap_node(prefix, label):
        label = str(label).strip()
        words = label.split()
        lines, line = [], ""
        for word in words:
            if len((line + " " + word).strip()) <= wrap_width:
                line = (line + " " + word).strip()
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        wrapped = "<br>".join(lines[:3])
        if len(lines) > 3:
            wrapped += "…"
        return f"<b>{prefix}</b><br>{wrapped}"

    actor_nodes = [wrap_node("Actor", a) for a in top_actors]
    mechanism_nodes = [wrap_node("Mechanism", m) for m in top_mechanisms]
    subject_nodes = [wrap_node("Affected group", s) for s in top_subjects]
    nodes = actor_nodes + mechanism_nodes + subject_nodes
    node_index = {name: i for i, name in enumerate(nodes)}

    # Sankey labels use one global font color in Plotly. To keep labels readable,
    # use light, high-contrast node fills and a dark global label font.
    actor_color = "#F7D95C"       # readable yellow with dark labels
    mechanism_color = "#D9B8F2"   # light purple, keeps EUSEE identity without hiding text
    subject_color = "#BFEAF2"     # light teal-blue, readable with dark labels
    node_colors = ([actor_color] * len(actor_nodes) + [mechanism_color] * len(mechanism_nodes) + [subject_color] * len(subject_nodes))

    links = []
    am = flow_df.groupby(["Actor of repression", "Mechanism of repression"]).size().reset_index(name="value")
    for _, r in am.iterrows():
        links.append(dict(
            source=node_index[wrap_node("Actor", r["Actor of repression"])],
            target=node_index[wrap_node("Mechanism", r["Mechanism of repression"])],
            value=int(r["value"]),
            color="rgba(102,0,148,0.16)"
        ))
    ms = flow_df.groupby(["Mechanism of repression", "Subject of repression"]).size().reset_index(name="value")
    for _, r in ms.iterrows():
        links.append(dict(
            source=node_index[wrap_node("Mechanism", r["Mechanism of repression"])],
            target=node_index[wrap_node("Affected group", r["Subject of repression"])],
            value=int(r["value"]),
            color="rgba(0,140,170,0.16)"
        ))

    fig_height = max(500, min(780, 350 + len(nodes) * 23))
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=26,
            thickness=20,
            line=dict(color="rgba(45,0,85,0.30)", width=0.8),
            label=nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><extra></extra>",
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color=[l["color"] for l in links],
            hovertemplate="<b>%{value}</b> linked alerts<extra></extra>",
        ),
         textfont=dict(
            family="Arial",
            size=10,
            color="#000000"
        )
    ))

    for name, color in [("Restrictive actors", actor_color), ("Restrictive mechanisms", mechanism_color), ("Affected civil society groups", subject_color)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=11, color=color, line=dict(color="rgba(45,0,85,0.30)", width=0.8)),
            name=name
        ))

    fig.update_layout(
        title=dict(
            text="Pathway: Restrictive actors → Restrictive mechanism → Affected civil society group",
            x=0.02,
            xanchor="left",
            font=dict(size=15, family=CHART_FONT, color=CHART_TITLE_COLOR)
        ),
        # Critical readability setting: dark labels on deliberately light node fills.
        font=dict(size=11.2, family=CHART_FONT, color="#17212B"),
        height=fig_height,
        width=width,
        margin=dict(l=22, r=22, t=62, b=34),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D9E2EC",
            font=dict(color="#17212B", family=CHART_FONT, size=12)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.09,
            xanchor="left",
            x=0,
            font=dict(size=10.8, family=CHART_FONT, color="#52616B")
        ),
        # Hide Cartesian axes created by the invisible Scatter traces used only for the Sankey legend.
        # This removes x/y axis values without affecting Sankey nodes, links, labels, or hover tooltips.
        xaxis=dict(
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=""
        ),
        yaxis=dict(
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=""
        ),
    )
    return fig

# ---------------- HIGH-END ANALYTICAL FLOW PANEL ----------------
def render_analytical_flow_panel(df):
    """Unified panel combining relationship heatmaps and Sankey flow."""
    if "top_n_option" not in st.session_state:
        st.session_state.top_n_option = "Top 5"

    total_records = len(df) if df is not None else 0
    top_n_map = {"Top 2": 2, "Top 3": 3, "Top 4": 4, "Top 5": 5}
    if st.session_state.get("top_n_option") not in top_n_map:
        st.session_state.top_n_option = "Top 5"

    st.markdown("""
    <style>
    .flow-panel-shell {
        background: linear-gradient(180deg, #FFFFFF 0%, #FBF9FE 100%);
        border: 1px solid #E7DDF2;
        border-radius: 22px;
        padding: 18px 18px 14px 18px;
        margin: 18px 0 18px 0;
        box-shadow: 0 12px 34px rgba(45, 0, 85, 0.075);
    }
    .flow-panel-eyebrow {font-family: Inter, Arial, sans-serif; font-size: 10.5px; font-weight: 900; letter-spacing: .10em; text-transform: uppercase; color: #008CAA; margin-bottom: 4px;}
    .flow-panel-title {font-family: Inter, Arial, sans-serif; font-size: 19px; font-weight: 950; color: #2D0055; margin-bottom: 4px;}
    .flow-panel-subtitle {font-family: Inter, Arial, sans-serif; font-size: 12px; color: #64748B; line-height: 1.45; max-width: 980px; margin-bottom: 12px;}
    .flow-panel-badges {display: flex; flex-wrap: wrap; gap: 7px; margin: 8px 0 4px 0;}
    .flow-panel-badge {background: #F3ECF8; border: 1px solid #E1D2EC; border-radius: 999px; padding: 5px 9px; font-family: Inter, Arial, sans-serif; font-size: 10.8px; font-weight: 800; color: #4B006E;}
    .flow-guide-card {background: #FFFFFF; border: 1px solid #E8EEF3; border-radius: 16px; padding: 11px 13px; min-height: 76px; box-shadow: 0 5px 16px rgba(15, 23, 42, 0.045);}
    .flow-guide-title {font-family: Inter, Arial, sans-serif; font-size: 11.8px; font-weight: 900; color: #2D0055; margin-bottom: 3px;}
    .flow-guide-text {font-family: Inter, Arial, sans-serif; font-size: 10.8px; color: #64748B; line-height: 1.35;}
    .flow-section-label {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        font-size: 15px;
        font-weight: 850;
        color: #101828;
        line-height: 1.25;
        letter-spacing: -0.015em;
        margin: 16px 0 6px 0;
    }
    .flow-section-note {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        font-size: 12px;
        font-weight: 500;
        color: #667085;
        line-height: 1.5;
        margin-bottom: 10px;
    }
    .flow-info-panel {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E4E7EC;
        border-left: 4px solid #660094;
        border-radius: 16px;
        padding: 13px 15px;
        margin: 14px 0 12px 0;
        box-shadow: 0 8px 20px rgba(16,24,40,.045);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }
    .flow-info-panel .flow-section-label {
        margin: 0 0 6px 0;
        color: #101828;
        font-size: 15px;
        font-weight: 850;
    }
    .flow-info-panel .flow-section-note {
        margin: 0;
        color: #667085;
        font-size: 12px;
        font-weight: 500;
        line-height: 1.55;
    }
    .flow-divider {height: 1px; background: linear-gradient(90deg, rgba(102,0,148,.22), rgba(0,140,170,.16), rgba(255,219,88,.10)); margin: 14px 0 10px 0;}
    </style>
    <div class="flow-panel-shell">
        <div class="flow-panel-eyebrow">Negative Events Relationship Analysis</div>
        <div class="flow-panel-title">Relationship Explorer</div>
        <div class="flow-panel-subtitle">
            This section helps you explore how restrictive actors, restrictive mechanisms, and affected civil society groups are connected. 
            Use the heatmaps to identify the strongest links between them, and the flow diagram to follow the pathway from actor to mechanism to affected group.
        </div>
       
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        st.markdown('<div class="flow-guide-card"><div class="flow-guide-title">1. Identify key links</div><div class="flow-guide-text">Use the heatmaps to see which restrictive actors, mechanisms, and affected civil society groups appear most frequently together. Darker cells indicate stronger links.</div></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="flow-guide-card"><div class="flow-guide-title">2. Follow the pathway</div><div class="flow-guide-text">Use the flow diagram to see how restrictive actors are connected to specific mechanisms, and how these mechanisms affect different civil society groups.</div></div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="flow-guide-card"><div class="flow-guide-title">3. Adjust the level of detail</div><div class="flow-guide-text">Use the Top-N selector to choose how many restrictive actors, mechanisms, and affected civil society groups are shown. Lower values simplify the view; higher values provide a more detailed analysis.</div></div>', unsafe_allow_html=True)

    ctrl_left, ctrl_right = st.columns([1.15, 2.85], gap="large")
    with ctrl_left:
        selected = st.selectbox(
            "Top-N selector",
            options=list(top_n_map.keys()),
            index=list(top_n_map.keys()).index(st.session_state.get("top_n_option", "Top 5")),
            help="Select how many top restrictive actors, mechanisms, and affected civil society groups are shown in the heatmaps and flow diagram.",
            key="flow_panel_top_n_select",
        )
        st.session_state.top_n_option = selected
        top_n = top_n_map[selected]
        st.session_state.top_n = top_n
    with ctrl_right:
        st.markdown(f"""
        <div class="flow-panel-badges" style="margin-top: 27px;">
            <span class="flow-panel-badge">View: {'All categories' if top_n is None else 'Top ' + str(top_n)}</span>
            <span class="flow-panel-badge">Tip: Hover over the heatmap squares and flow lines to see the number of alerts</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="flow-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-section-label">Heatmaps</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-section-note">Use these matrices to see which restrictive actors, mechanisms, and affected civil society groups appear most frequently together. Darker cells indicate stronger links.</div>', unsafe_allow_html=True)
    render_heatmaps(df, top_n=top_n)

    st.markdown('<div class="flow-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-info-panel">
            <div class="flow-section-label">Flow diagram</div>
            <div class="flow-section-note">
                Use the flow diagram to see how restrictive actors are connected to specific mechanisms,
                and how these mechanisms affect different civil society groups.<br>
                Wider lines show where more alerts connect restrictive actors, restrictive mechanisms,
                and affected civil society groups under the selected filters.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_dashboard_plotly_chart(render_sankey(df, top_n=top_n), plot_df=df, visual_type="sankey flow diagram", x_col="Actor of repression", group_col="Mechanism of repression", dashboard_df=df, config={"displayModeBar": False}, key="negative_events_analytical_flow_panel_sankey", permission_key="view_chart_sankey_flow", permission_label="Analytical Sankey flow")

# ---------------- TOP-N BAR HELPER ----------------
def top_n_bar(df, col, top_n=None):
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=[col, "count"])
    
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    
    if top_n is not None:
        counts = counts.head(top_n)    
    return counts
# ---------------- EXPLODE MULTI-VALUED COLUMNS ----------------
def explode_multi_valued_columns(df, cols):
    """
    Explodes comma-separated values in specified columns.
    Each comma-separated value becomes a separate row.
    """
    df_exploded = df.copy()
    for col in cols:
        if col in df_exploded.columns:
            df_exploded[col] = df_exploded[col].fillna("").astype(str).str.split(",")
            df_exploded = df_exploded.explode(col)
            df_exploded[col] = df_exploded[col].str.strip()
    return df_exploded





# ---------------- EXECUTIVE ACCESS STATE CARDS ----------------
def inject_access_state_card_css():
    """Central polished access-state styling for restricted tabs, charts, maps and tools."""
    st.markdown("""
    <style>
    .eusee-access-card {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid rgba(102,0,148,.14);
        background:
            radial-gradient(circle at top right, rgba(102,0,148,.07), transparent 30%),
            linear-gradient(135deg, #FFFFFF 0%, #FCFAFF 100%);
        padding: 22px;
        margin: 10px 0 18px 0;
        box-shadow: 0 16px 40px rgba(16,24,40,.08), inset 0 1px 0 rgba(255,255,255,.95);
        font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
    }
    .eusee-access-card.compact {
        border-radius: 18px;
        padding: 17px 18px;
        margin: 6px 0 14px 0;
        box-shadow: 0 10px 24px rgba(16,24,40,.06), inset 0 1px 0 rgba(255,255,255,.95);
    }
    .eusee-access-topbar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #660094 0%, #8E24AA 50%, #008CAA 100%);
    }
    .eusee-access-header {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 18px;
    }
    .eusee-access-card.compact .eusee-access-header {
        gap: 12px;
        margin-bottom: 13px;
    }
    .eusee-access-icon {
        width: 62px;
        height: 62px;
        min-width: 62px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        background: linear-gradient(135deg, rgba(102,0,148,.12), rgba(0,140,170,.10));
        border: 1px solid rgba(102,0,148,.10);
    }
    .eusee-access-card.compact .eusee-access-icon {
        width: 46px;
        height: 46px;
        min-width: 46px;
        border-radius: 15px;
        font-size: 21px;
    }
    .eusee-access-eyebrow {
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #660094;
        margin-bottom: 5px;
    }
    .eusee-access-title {
        font-size: 21px;
        font-weight: 950;
        line-height: 1.12;
        color: #23152F;
        margin-bottom: 7px;
    }
    .eusee-access-card.compact .eusee-access-title {
        font-size: 16px;
        margin-bottom: 5px;
    }
    .eusee-access-copy {
        font-size: 12.5px;
        line-height: 1.5;
        color: #667085;
        max-width: 820px;
        font-weight: 550;
    }
    .eusee-access-card.compact .eusee-access-copy {
        font-size: 11.5px;
        line-height: 1.42;
    }
    .eusee-access-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }
    .eusee-access-card.compact .eusee-access-meta-grid {
        gap: 8px;
        margin-bottom: 13px;
    }
    .eusee-access-meta-card {
        border-radius: 16px;
        border: 1px solid #E7D4F1;
        background: rgba(255,255,255,.84);
        padding: 14px;
    }
    .eusee-access-card.compact .eusee-access-meta-card {
        border-radius: 13px;
        padding: 10px 11px;
    }
    .eusee-access-meta-label {
        display: block;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #6941C6;
        margin-bottom: 6px;
    }
    .eusee-access-meta-value {
        font-size: 13px;
        font-weight: 850;
        color: #23152F;
        word-break: break-word;
    }
    .eusee-access-card.compact .eusee-access-meta-label {
        font-size: 9px;
        margin-bottom: 4px;
    }
    .eusee-access-card.compact .eusee-access-meta-value {
        font-size: 11.5px;
    }
    .eusee-access-actions {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        flex-wrap: wrap;
        padding-top: 14px;
        border-top: 1px solid #EEF0F4;
    }
    .eusee-access-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: #F4EAF8;
        border: 1px solid #E7D4F1;
        color: #660094;
        font-size: 11px;
        font-weight: 850;
        white-space: nowrap;
    }
    .eusee-access-badge.success {
        background: #ECFDF3;
        border-color: #ABEFC6;
        color: #067647;
    }
    .eusee-access-action-copy {
        font-size: 11.5px;
        color: #667085;
        font-weight: 600;
        line-height: 1.4;
    }
    @media (max-width: 900px) {
        .eusee-access-header { flex-direction: column; }
        .eusee-access-meta-grid { grid-template-columns: 1fr; }
        .eusee-access-title { font-size: 18px; }
        .eusee-access-card { padding: 18px; }
    }
    </style>
    """, unsafe_allow_html=True)


inject_access_state_card_css()


def _safe_current_role() -> str:
    try:
        role = str(get_current_role() or "guest").replace("_", " ").strip()
        return role.title() if role else "Guest"
    except Exception:
        return "Guest"


def _access_icon_for_permission(permission_key: str) -> str:
    key = str(permission_key or "").lower()
    if "ai" in key or "copilot" in key:
        return "🤖"
    if "map" in key or "geo" in key:
        return "🗺️"
    if "sankey" in key or "flow" in key:
        return "🔄"
    if "heatmap" in key:
        return "🔥"
    if "download" in key or "export" in key:
        return "⬇️"
    if "country" in key or "countries" in key:
        return "🌍"
    if "admin" in key:
        return "🛡️"
    if "chart" in key or "plot" in key:
        return "📊"
    return "🔒"


def _required_role_for_permission(permission_key: str, fallback: str = "Privileged User") -> str:
    key = str(permission_key or "").lower()
    if "admin" in key:
        return "Administrator"
    if "download" in key:
        return "Approved Export User"
    if "ai" in key or "copilot" in key:
        return "Privileged AI User"
    if "map" in key or "geo" in key or "chart" in key or "plot" in key or "heatmap" in key or "sankey" in key:
        return "Privileged Analyst"
    return fallback


def _html_escape(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def render_access_locked(section_title: str, required_level: str = "logged-in user"):
    """Render a premium restricted-section card for full tabs or major panels."""
    render_permission_locked_card(
        section_title=section_title,
        permission_key=str(required_level),
        required_role=str(required_level).title(),
        feature_icon="🔐",
        feature_description="This section is not available for your current access level. Use the sidebar access controls or contact an administrator if this feature should be enabled for your account.",
        compact=False,
    )


def can_render_feature(permission_key: str) -> bool:
    """Safe wrapper around admin-configured permissions. Admins are allowed by authz.has_permission."""
    try:
        return bool(has_permission(permission_key))
    except Exception:
        return False

# Central chart/map privilege catalogue. Every visual listed here is rendered
# through render_dashboard_plotly_chart() or render_permission_locked_card() so
# restricted users see the same polished locked-state card instead of a plain message.
RESTRICTED_VISUAL_PERMISSION_LABELS = {
    "view_chart_overview_alert_type": "Overview alert type distribution",
    "view_chart_overview_enabling_principles": "Overview enabling-principle distribution",
    "view_chart_overview_regions": "Overview regional distribution",
    "view_chart_overview_countries": "Overview country distribution",
    "view_chart_negative_restrictive_actors": "Restrictive actors chart",
    "view_chart_negative_affected_actors": "Civil society actors affected chart",
    "view_chart_negative_restrictive_mechanisms": "Restrictive mechanisms chart",
    "view_chart_negative_event_types": "Negative event types chart",
    "view_chart_negative_alert_types": "Negative alert types chart",
    "view_chart_negative_enabling_principles": "Negative enabling-principles chart",
    "view_chart_heatmap_actor_mechanism": "Actor × mechanism heatmap",
    "view_chart_heatmap_subject_mechanism": "Affected actor × mechanism heatmap",
    "view_chart_heatmap_actor_subject": "Actor × affected actor heatmap",
    "view_chart_sankey_flow": "Analytical Sankey flow",
    "view_chart_geospatial_map": "Geospatial intelligence map",
    "view_chart_ai_copilot_plots": "AI Copilot plots",
}

def render_permission_locked_card(
    section_title: str,
    permission_key: str,
    container=None,
    *,
    required_role: str | None = None,
    feature_icon: str | None = None,
    feature_description: str | None = None,
    compact: bool = True,
):
    """Polished access-state card for restricted tabs, charts, maps, tools and downloads."""
    target = container if container is not None else st
    feature_icon = feature_icon or _access_icon_for_permission(permission_key)
    session_label = "Restricted access"
    badge_class = "eusee-access-badge"
    action_copy = (
        "Use the User Privilege Center or contact an administrator if this feature should be enabled for your account."
    )
    description = feature_description or "This dashboard component is restricted by the active permission settings."
    card_class = "eusee-access-card compact" if compact else "eusee-access-card"

    safe_title = _html_escape(section_title)
    safe_description = _html_escape(description)
    safe_session = _html_escape(session_label)
    safe_action = _html_escape(action_copy)

    target.markdown(f"""
    <div class="{card_class}">
        <div class="eusee-access-topbar"></div>
        <div class="eusee-access-header">
            <div class="eusee-access-icon">{feature_icon}</div>
            <div class="eusee-access-heading-block">
                <div class="eusee-access-eyebrow">Restricted feature</div>
                <div class="eusee-access-title">{safe_title}</div>
                <div class="eusee-access-copy">{safe_description}</div>
            </div>
        </div>
        <div class="eusee-access-actions">
            <div class="{badge_class}">{safe_session}</div>
            <div class="eusee-access-action-copy">{safe_action}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_if_permitted(permission_key: str, section_title: str, render_fn, container=None):
    """Render any chart/widget only when its admin-configured permission is enabled."""
    if can_render_feature(permission_key):
        return render_fn()
    render_permission_locked_card(section_title, permission_key, container=container)
    return None

# ---------------- TABS ALREADY RENDERED DIRECTLY BELOW SUBTITLE ----------------
SOURCE_TEXT = "Source: EU SEE Dashboard. Data compiled by EU SEE Network."
def add_source_line(fig, y_offset=-0.15, font_size=12, font_color="gray"):
    """
    Adds a source line below the chart.
    - y_offset: vertical position (negative values go below the plot)
    """
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=y_offset,
        showarrow=False,
        text=SOURCE_TEXT,
        font=dict(size=font_size, color=font_color),
        xanchor="center",
        yanchor="top"
    )
    return fig

# ---------------- PROFESSIONAL TITLE-AWARE IN-CHART INFO BADGE ----------------
def _strip_plotly_html(text):
    """Return plain title text for width estimation only."""
    if text is None:
        return ""
    return re.sub(r"<[^>]+>", "", str(text)).replace("&nbsp;", " ").strip()

def _estimate_badge_x_from_title(
    title_text,
    title_x=0.5,
    title_xanchor="center",
    title_font_size=14,
    chart_width_px=620,
    right_padding=0.018,
    max_x=0.985,
):
    """Estimate a Plotly paper-coordinate x position immediately after the title.

    Plotly does not expose rendered title pixel width to Python/Streamlit before
    rendering, so this uses a conservative text-width estimate. It keeps the badge
    visually attached to the title while staying inside the Plotly chart area.
    """
    clean_title = _strip_plotly_html(title_text)
    if not clean_title:
        return min(max(title_x + 0.055, 0.04), max_x)

    # Approximate average glyph width for Arial-like dashboard font.
    estimated_title_px = len(clean_title) * title_font_size * 0.50
    title_width_paper = estimated_title_px / max(float(chart_width_px), 1.0)

    if title_xanchor == "left":
        badge_x = title_x + title_width_paper + right_padding
    elif title_xanchor == "right":
        badge_x = title_x + right_padding
    else:
        # Centered title: right edge is center + half the title width.
        badge_x = title_x + (title_width_paper / 2.0) + right_padding

    return min(max(badge_x, 0.04), max_x)

def _wrap_chart_tooltip_text(message, line_length=82):
    """Format long tooltip text so Plotly hover labels remain readable."""
    raw = _strip_plotly_html(message)
    if not raw:
        return ""
    words = raw.split()
    lines = []
    current = []
    current_len = 0

    for word in words:
        extra = 1 if current else 0
        if current and current_len + len(word) + extra > line_length:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + extra

    if current:
        lines.append(" ".join(current))

    return "<br>".join(lines)

def _figure_has_chart_info_badge(fig):
    """Avoid duplicate info badges when a chart already received one manually."""
    try:
        for ann in list(fig.layout.annotations or []):
            ann_text = str(getattr(ann, "text", "") or "").lower()
            ann_hover = str(getattr(ann, "hovertext", "") or "")
            if (
                "eusee-chart-info-badge" in ann_text
                or (ann_text in ["<b>i</b>", "i", "<b>ⓘ</b>", "ⓘ"] and ann_hover.strip())
            ):
                return True
    except Exception:
        return False
    return False

def _escape_plotly_title_attr(value):
    """Escape text used inside the Plotly title HTML tooltip attribute."""
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&#39;")
    )

def _build_plotly_title_with_info(title_text, tooltip_text):
    """Build a Plotly-safe title with the info icon locked beside the title.

    This keeps the badge inside the chart title area instead of using Plotly
    annotations or a separate Streamlit markdown header. It is intentionally
    compact so it does not disturb chart layout or column spacing.
    """
    clean_title = _escape_plotly_title_attr(_strip_plotly_html(title_text))
    clean_tooltip = _escape_plotly_title_attr(_strip_plotly_html(tooltip_text))

    if not clean_title or not clean_tooltip:
        return title_text

    return f"""
<span style="display:inline-flex;align-items:center;gap:7px;white-space:nowrap;">
    <span>{clean_title}</span>
    <span title="{clean_tooltip}" style="
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:17px;
        height:17px;
        border-radius:50%;
        background:#F4EAF8;
        border:1px solid #E7D4F1;
        color:#660094;
        font-size:10px;
        font-weight:900;
        line-height:17px;
        vertical-align:middle;
        cursor:help;
    ">i</span>
</span>
""".strip()

def add_chart_info_badge(
    fig,
    message,
    x=None,
    y=1.065,
    badge_text="<b>ⓘ Tip</b>",
    chart_width_px=620,
    title_x=None,
    title_xanchor=None,
):
    """Add a reliable in-chart tooltip aid for the two enabling-principle charts.

    This version avoids fragile Streamlit DOM JavaScript and unreliable Plotly
    title HTML hover. It keeps the chart title layout stable, adds a small
    visible Tip pill inside the Plotly title band, and adds a near-invisible
    hover zone across the chart rows so users see the same note when moving
    the mouse inside the chart area.
    """
    if fig is None or not message:
        return fig

    title = fig.layout.title
    raw_title_text = getattr(title, "text", "") or ""
    plain_title = _strip_plotly_html(raw_title_text)

    if not plain_title:
        return fig

    # Avoid duplicate info aids on reruns or repeated layout calls.
    if _figure_has_chart_info_badge(fig):
        return fig

    title_font = getattr(title, "font", None)
    title_font_size = getattr(title_font, "size", None) or 15
    title_font_family = getattr(title_font, "family", None) or CHART_FONT
    title_font_color = getattr(title_font, "color", None) or "#23152F"

    inferred_xanchor = title_xanchor
    if inferred_xanchor is None:
        inferred_xanchor = getattr(title, "xanchor", None) or "left"

    current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}

    # Keep the title clean and stable. The visible pill sits inside the chart
    # title band, not in a separate Streamlit block.
    fig.update_layout(
        title=dict(
            text=plain_title,
            x=float(title_x) if title_x is not None else (getattr(title, "x", None) or 0.01),
            xanchor=str(inferred_xanchor),
            y=getattr(title, "y", None) or 0.97,
            yanchor=getattr(title, "yanchor", None) or "top",
            font=dict(
                family=title_font_family,
                size=title_font_size,
                color=title_font_color,
            ),
        ),
        margin=dict(
            l=current_margin.get("l", 135),
            r=current_margin.get("r", 28),
            t=max(int(current_margin.get("t", 58) or 58), 76),
            b=current_margin.get("b", 58),
        ),
        hovermode="closest",
    )

    wrapped_message = _wrap_chart_tooltip_text(message, line_length=80)

    # 1) Always-visible, compact in-chart Tip pill. This guarantees users can
    # see that contextual help exists even if browser/Plotly hover behavior
    # changes across deployments.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=float(0.005 if x is None else x),
        y=float(y),
        xanchor="left",
        yanchor="top",
        text=badge_text,
        hovertext=wrapped_message,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E6E8EF",
            font=dict(size=11, color="#344054", family=CHART_FONT),
        ),
        showarrow=False,
        align="center",
        bgcolor="rgba(244,234,248,0.98)",
        bordercolor="#E7D4F1",
        borderwidth=1,
        borderpad=5,
        font=dict(size=10, color="#660094", family=CHART_FONT),
        opacity=1,
        captureevents=True,
    )

    # 2) Plotly-native hover zone inside the chart body. This is more reliable
    # than JavaScript because Plotly itself handles the hoverlabel.
    try:
        y_values = []
        numeric_x_values = []

        for trace in list(fig.data or []):
            orientation = str(getattr(trace, "orientation", "") or "").lower()
            trace_y = list(getattr(trace, "y", []) or [])
            trace_x = list(getattr(trace, "x", []) or [])

            # The two target charts are horizontal bar charts with categories on y.
            if orientation == "h" and trace_y:
                for val in trace_y:
                    if val is not None and str(val).strip() and str(val).lower() != "nan":
                        if val not in y_values:
                            y_values.append(val)
                for val in trace_x:
                    try:
                        numeric_x_values.append(float(val))
                    except Exception:
                        pass

        if y_values:
            max_x = max(numeric_x_values) if numeric_x_values else 1.0
            hover_x = max(max_x * 0.72, 1.0)
            fig.add_trace(
                go.Scatter(
                    x=[hover_x] * len(y_values),
                    y=y_values,
                    mode="markers",
                    marker=dict(
                        size=44,
                        color="rgba(102,0,148,0.001)",
                        line=dict(width=0, color="rgba(102,0,148,0)"),
                    ),
                    text=[wrapped_message] * len(y_values),
                    hovertemplate="<b>Tip</b><br>%{text}<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="#FFFFFF",
                        bordercolor="#E6E8EF",
                        font=dict(size=11, color="#344054", family=CHART_FONT),
                    ),
                    showlegend=False,
                    name="EUSEE chart information",
                    cliponaxis=False,
                )
            )
    except Exception:
        # The visible Tip pill above remains available even if the invisible
        # hover zone cannot be created for a specific Plotly version.
        pass

    return fig

def build_default_chart_tooltip(fig, visual_type="chart", x_col=None, group_col=None):
    """Create a concise fallback tooltip for charts that do not have a custom note."""
    try:
        title_text = _strip_plotly_html(getattr(fig.layout.title, "text", "") or "")
    except Exception:
        title_text = ""

    chart_label = str(visual_type or "chart").strip().lower()
    parts = []

    if title_text:
        parts.append(f"{title_text}.")
    else:
        parts.append("This chart summarizes the filtered dashboard records.")

    if x_col and group_col:
        parts.append(f"It compares {x_col} and groups the results by {group_col}.")
    elif x_col:
        parts.append(f"It summarizes results by {x_col}.")
    elif group_col:
        parts.append(f"It groups the filtered records by {group_col}.")

    parts.append("Values update automatically when the dashboard filters change.")
    parts.append("Use Plotly hover for exact counts and legend controls to isolate categories.")

    return " ".join(parts)


def apply_title_adjacent_tooltip(
    fig,
    *,
    message=None,
    visual_type="chart",
    x_col=None,
    group_col=None,
    chart_width_px=620,
):
    """Apply the standardized title-adjacent tooltip to any dashboard Plotly figure."""
    if fig is None:
        return fig

    try:
        title_text = _strip_plotly_html(getattr(fig.layout.title, "text", "") or "")
    except Exception:
        title_text = ""

    # Only add title-adjacent badges to charts with visible titles.
    if not title_text:
        return fig

    tooltip_message = message or build_default_chart_tooltip(
        fig,
        visual_type=visual_type,
        x_col=x_col,
        group_col=group_col,
    )

    return add_chart_info_badge(
        fig,
        tooltip_message,
        chart_width_px=chart_width_px,
    )

# ---------------- STANDARD IN-CHART INFO BADGES ----------------
def render_chart_floating_tip(*args, **kwargs):
    """Deprecated compatibility wrapper.

    Chart interpretation notes are now integrated directly into Plotly charts
    through add_chart_info_badge(...), so no floating Streamlit overlay is rendered.
    """
    return None

# ---------------- SMALL-SCREEN RESPONSIVENESS + NON-INTRUSIVE LEGEND PATCH ----------------
def inject_full_tab_responsive_css():
    """Responsive shell that stacks tab content only on small screens and preserves chart legend placement."""
    st.markdown("""
    <style>
    /* ---------- Streamlit tab shell: keep desktop/tablet layouts intact ---------- */
    div[data-testid="stTabs"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow: visible !important;
    }

    div[data-testid="stTabs"] div[role="tablist"] {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)) !important;
        gap: 6px !important;
        border-bottom: 1px solid #E6E8EF !important;
        overflow-x: hidden !important;
        overflow-y: visible !important;
        padding: 4px 0 7px 0 !important;
        scrollbar-width: none !important;
        align-items: stretch !important;
    }

    div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {
        display: none !important;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 32px !important;
        height: 32px !important;
        max-height: 32px !important;
        border-radius: 999px !important;
        padding: 5px 8px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-width: 0 !important;
        width: 100% !important;
        flex: 1 1 auto !important;
        background: #FFFFFF !important;
        border: 1px solid #E6E8EF !important;
        color: #344054 !important;
        font-size: clamp(10px, 0.8vw, 12px) !important;
        line-height: 1.05 !important;
        font-weight: 850 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.035) !important;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #660094 !important;
        background: #F4EAF8 !important;
        border-color: #E7D4F1 !important;
    }

    div[data-testid="stTabs"] div[role="tabpanel"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        padding-top: 12px !important;
    }

    /* ---------- Universal containment without forcing desktop columns to stack ---------- */
    .main .block-container,
    .element-container,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"] {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="column"] {
        min-width: 0 !important;
        overflow: visible !important;
    }

    /* ---------- Plotly: compact, professional legends without changing legend location ---------- */
    .js-plotly-plot .legend {
        pointer-events: auto !important;
    }

    .js-plotly-plot .legend rect.bg {
        fill: rgba(255,255,255,0.82) !important;
        stroke: rgba(230,232,239,0.65) !important;
        stroke-width: 1px !important;
        rx: 8px !important;
        ry: 8px !important;
    }

    .js-plotly-plot .legend .traces {
        opacity: 0.98 !important;
    }

    .js-plotly-plot .legend text,
    .js-plotly-plot .legendtext {
        font-family: Arial, sans-serif !important;
        font-size: clamp(8.5px, 0.75vw, 10px) !important;
        font-weight: 750 !important;
        letter-spacing: -0.01em !important;
    }

    /* ---------- Plotly: responsive legends without changing legend location ---------- */
    div[data-testid="stPlotlyChart"],
    .stPlotlyChart,
    .js-plotly-plot,
    .plot-container,
    .svg-container {
        width: 100% !important;
        max-width: 100% !important;
        overflow: visible !important;
        box-sizing: border-box !important;
    }

    .js-plotly-plot .legend text {
        font-family: Arial, sans-serif !important;
        font-size: clamp(9px, 1.1vw, 11px) !important;
    }

    .js-plotly-plot .legendtoggle {
        cursor: pointer !important;
    }

    /* ---------- KPI cards: equal height and visible descriptions ---------- */
    .eusee-kpi-card {
        height: 190px !important;
        min-height: 190px !important;
        max-height: none !important;
        overflow: visible !important;
        gap: 8px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    .eusee-kpi-note {
        display: block !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.32 !important;
        min-height: 28px !important;
    }

    .eusee-donut-layout {
        grid-template-columns: minmax(70px, 78px) minmax(0, 1fr) !important;
        align-items: center !important;
        min-width: 0 !important;
    }

    .eusee-breakdown-row {
        grid-template-columns: 10px minmax(0, 1fr) minmax(34px, 42px) minmax(36px, 46px) !important;
    }

    .eusee-breakdown-label {
        min-width: 0 !important;
    }

    /* ---------- Tables and dataframes ---------- */
    div[data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow: auto !important;
    }

    iframe,
    canvas,
    svg {
        max-width: 100% !important;
    }

    /* ---------- Tablet: preserve side-by-side layout where Streamlit columns fit ---------- */
    @media (max-width: 1100px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        div[data-testid="stTabs"] div[role="tablist"] {
            gap: 5px !important;
            grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)) !important;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            font-size: 10.5px !important;
            padding-left: 6px !important;
            padding-right: 6px !important;
        }

        .eusee-kpi-card {
            height: 200px !important;
            min-height: 200px !important;
        }

        .eusee-donut-layout {
            grid-template-columns: 68px minmax(0, 1fr) !important;
            gap: 7px !important;
        }

        .eusee-donut {
            width: 66px !important;
            height: 66px !important;
        }
    }

    /* ---------- Small screens only: stack Streamlit columns ---------- */
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.8rem !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }

        div[data-testid="stTabs"] div[role="tablist"] {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 6px !important;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            height: 32px !important;
            min-height: 32px !important;
            max-height: 32px !important;
            font-size: 10.5px !important;
            padding: 5px 7px !important;
        }

        .eusee-kpi-card {
            height: 190px !important;
            min-height: 190px !important;
            padding: 13px 14px 12px 14px !important;
        }

        .eusee-donut-layout {
            grid-template-columns: 82px minmax(0, 1fr) !important;
        }

        .eusee-donut {
            width: 76px !important;
            height: 76px !important;
        }
    }

    /* ---------- Very small phones ---------- */
    @media (max-width: 430px) {
        .main .block-container {
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
        }

        .eusee-donut-layout {
            grid-template-columns: 1fr !important;
            justify-items: center !important;
            gap: 10px !important;
        }

        .eusee-breakdown-list {
            width: 100% !important;
        }

        .eusee-kpi-value {
            font-size: 30px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def apply_responsive_plotly_layout(fig, *, legend_bottom=False):
    """Make Plotly charts responsive while preserving each chart's original legend location."""
    if fig is None:
        return fig

    try:
        current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    except Exception:
        current_margin = {}

    try:
        fig.update_layout(
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=max(int(current_margin.get("l", 40) or 40), 36),
                r=max(int(current_margin.get("r", 24) or 24), 24),
                t=max(int(current_margin.get("t", 50) or 50), 52),
                b=max(int(current_margin.get("b", 44) or 44), 48),
            ),
            uniformtext_minsize=9,
            uniformtext_mode="hide",
        )
    except Exception:
        pass

    # Preserve existing x/y/orientation. Only make legend text and boxes adaptive.
    try:
        existing_legend = fig.layout.legend.to_plotly_json() if fig.layout.legend else {}
        existing_legend.update({
            "bgcolor": existing_legend.get("bgcolor", "rgba(255,255,255,0.86)"),
            "bordercolor": existing_legend.get("bordercolor", "rgba(230,232,239,0.60)"),
            "borderwidth": existing_legend.get("borderwidth", 1),
            "font": dict(size=9, family="Arial", color="#344054"),
            # Keep Plotly's native colored markers visible. Do not force symbol scaling via CSS.
            "itemsizing": existing_legend.get("itemsizing", "trace"),
            # 30 is Plotly's practical compact minimum; larger values create excessive label gaps.
            "itemwidth": min(int(existing_legend.get("itemwidth", 30) or 30), 30),
            "tracegroupgap": 0,
        })
        fig.update_layout(legend=existing_legend)
    except Exception:
        pass

    # Do not set legend_entrywidth: fixed entry widths create large gaps between legend labels.
    # Long labels keep their native Plotly spacing and are handled by smaller font + compact itemwidth.

    try:
        fig.update_xaxes(automargin=True, tickfont=dict(size=10), title_standoff=8)
        fig.update_yaxes(automargin=True, tickfont=dict(size=10), title_standoff=8)
    except Exception:
        pass

    return fig

inject_full_tab_responsive_css()

# ---------------- COMPACT TAB + PROFESSIONAL LEGEND FINAL OVERRIDE ----------------
def inject_compact_tabs_and_legend_ux():
    """Final UI polish: all tabs visible without horizontal scrolling; compact legends preserve original placement."""
    st.markdown("""
    <style>
    div[data-testid="stTabs"] div[role="tablist"] {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)) !important;
        gap: 5px !important;
        overflow-x: hidden !important;
        padding: 3px 0 6px 0 !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        width: 100% !important;
        min-width: 0 !important;
        height: 31px !important;
        min-height: 31px !important;
        max-height: 31px !important;
        padding: 4px 7px !important;
        border-radius: 999px !important;
        font-size: clamp(9.5px, .72vw, 11.2px) !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    @media (max-width: 720px) {
        div[data-testid="stTabs"] div[role="tablist"] {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 6px !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            font-size: 10.2px !important;
            height: 32px !important;
            min-height: 32px !important;
        }
    }
    @media (max-width: 360px) {
        div[data-testid="stTabs"] button[role="tab"] {
            font-size: 9.4px !important;
            padding-left: 5px !important;
            padding-right: 5px !important;
        }
    }
    .js-plotly-plot .legend rect.bg {
        fill: rgba(255,255,255,.86) !important;
        stroke: rgba(230,232,239,.65) !important;
        stroke-width: 1px !important;
    }
    .js-plotly-plot .legend text,
    .js-plotly-plot .legendtext {
        font-size: clamp(8px, .68vw, 9.6px) !important;
        font-weight: 750 !important;
    }    .eusee-kpi-card {
        height: 190px !important;
        min-height: 190px !important;
        max-height: none !important;
    }
    @media (max-width: 1100px) {
        .eusee-kpi-card { height: 200px !important; min-height: 200px !important; }
    }
    @media (max-width: 640px) {
        .eusee-kpi-card { height: 190px !important; min-height: 190px !important; }
    }
    @media (max-width: 430px) {
        .eusee-kpi-card { height: 210px !important; min-height: 210px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

inject_compact_tabs_and_legend_ux()


# ---------------- FINAL RESPONSIVE TAB TEXT UX OVERRIDE ----------------
def inject_final_responsive_tab_text_ux():
    """Final override for responsive Streamlit tabs.

    Purpose:
    - Keep all tab labels readable on desktop, tablet, and mobile.
    - Allow long labels such as "Negative Alerts Analysis" to wrap cleanly.
    - Avoid horizontal scrolling, clipped text, and ellipsis-only labels.
    - Preserve a compact professional tab style for nested tabs as well.
    """
    st.markdown("""
    <style>
    /* Apply to all Streamlit tabs, including nested dashboard/AI tabs. */
    div[data-testid="stTabs"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow: visible !important;
    }

    div[data-testid="stTabs"] div[role="tablist"],
    div[data-testid="stTabs"] [role="tablist"] {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)) !important;
        gap: 8px !important;
        width: 100% !important;
        max-width: 100% !important;
        padding: 5px 0 9px 0 !important;
        margin: 0 0 4px 0 !important;
        overflow-x: hidden !important;
        overflow-y: visible !important;
        align-items: stretch !important;
        border-bottom: 1px solid #E8E2EF !important;
        scrollbar-width: none !important;
    }

    div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar,
    div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
        display: none !important;
    }

    div[data-testid="stTabs"] button[role="tab"],
    div[data-testid="stTabs"] [role="tab"] {
        width: 100% !important;
        min-width: 0 !important;
        height: auto !important;
        min-height: 42px !important;
        max-height: none !important;
        padding: 8px 10px !important;
        margin: 0 !important;
        border-radius: 13px !important;
        background: #FFFFFF !important;
        border: 1px solid #E6E8EF !important;
        color: #344054 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.035) !important;
        font-family: Arial, sans-serif !important;
        font-size: clamp(10.5px, 0.9vw, 12.5px) !important;
        font-weight: 850 !important;
        line-height: 1.16 !important;
        text-align: center !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
        hyphens: auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background .18s ease, color .18s ease, border-color .18s ease, box-shadow .18s ease !important;
    }

    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] [role="tab"] p,
    div[data-testid="stTabs"] button[role="tab"] span,
    div[data-testid="stTabs"] [role="tab"] span {
        margin: 0 !important;
        padding: 0 !important;
        max-width: 100% !important;
        line-height: 1.16 !important;
        text-align: center !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
    }

    div[data-testid="stTabs"] button[role="tab"]:hover,
    div[data-testid="stTabs"] [role="tab"]:hover {
        background: #F4EAF8 !important;
        color: #660094 !important;
        border-color: #E7D4F1 !important;
        box-shadow: inset 0 -3px 0 #660094, 0 2px 6px rgba(16,24,40,.045) !important;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #660094 0%, #7A1FA2 100%) !important;
        color: #FFFFFF !important;
        border-color: #660094 !important;
        box-shadow: inset 0 -3px 0 #FFDB58, 0 3px 9px rgba(102,0,148,.12) !important;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    div[data-testid="stTabs"] div[role="tabpanel"],
    div[data-testid="stTabs"] [role="tabpanel"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        padding-top: 10px !important;
    }

    /* Tablet: two-column tab grid with readable wrapped labels. */
    @media (max-width: 900px) {
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [role="tablist"] {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 7px !important;
            padding-bottom: 8px !important;
        }

        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] [role="tab"] {
            min-height: 44px !important;
            padding: 8px 9px !important;
            font-size: 11.5px !important;
            line-height: 1.18 !important;
        }
    }

    /* Mobile: keep labels readable and prevent compressed/clipped tab text. */
    @media (max-width: 520px) {
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [role="tablist"] {
            grid-template-columns: 1fr !important;
            gap: 6px !important;
            position: relative !important;
            top: auto !important;
        }

        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] [role="tab"] {
            min-height: 40px !important;
            padding: 8px 10px !important;
            font-size: 11.2px !important;
            border-radius: 12px !important;
        }
    }

    /* Very narrow devices: still no ellipsis, but slightly tighter spacing. */
    @media (max-width: 360px) {
        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] [role="tab"] {
            min-height: 38px !important;
            padding: 7px 8px !important;
            font-size: 10.6px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
inject_final_responsive_tab_text_ux()

# ---------------- FINAL TOP TAB SPACING OVERRIDE ----------------
def inject_final_top_tab_spacing_override():
    """Final override to remove dead space above the main dashboard tabs.

    This is intentionally loaded after all tab styling functions because earlier
    responsive tab CSS reintroduces padding/margins around the Streamlit tab bar.
    It only changes vertical spacing around the tabs, not tab behavior, filters,
    charts, maps, permissions, or chatbot logic.
    """
    st.markdown("""
    <style>
    /* Keep the title/subtitle block compact before the tabs. */
    .animated-title {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        line-height: 1.02 !important;
    }

    .animated-divider {
        margin-top: 0rem !important;
        margin-bottom: 0.08rem !important;
    }

    .animated-subtitle {
        margin-top: 0rem !important;
        margin-bottom: 2.5rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        line-height: 1.25 !important;
    }

    /* Pull Streamlit tabs directly upward under the subtitle. */
    div[data-testid="stTabs"] {
        margin-top: -1.35rem !important;
        padding-top: 0rem !important;
    }

    div[data-testid="stTabs"] > div {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Put tab buttons at the top edge of the tabs container. */
    div[data-testid="stTabs"] div[role="tablist"],
    div[data-testid="stTabs"] [role="tablist"] {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
        margin-bottom: 0.12rem !important;
        padding-bottom: 0.12rem !important;
    }

    /* Remove the default gap between tab buttons and tab panel content. */
    div[data-testid="stTabs"] div[role="tabpanel"],
    div[data-testid="stTabs"] [role="tabpanel"] {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    /* Remove empty iframe/component wrappers that can create invisible gaps. */
    .main .block-container iframe[width="0"],
    .main .block-container iframe[height="0"],
    .main .block-container div[data-testid="stElementContainer"]:has(iframe[height="0"]),
    .main .block-container div[data-testid="stElementContainer"]:has(iframe[width="0"]) {
        height: 0px !important;
        min-height: 0px !important;
        max-height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)
inject_final_top_tab_spacing_override()

# ---------------- FINAL LEGEND COLOR + SPACING FIX ----------------
def inject_plotly_legend_color_spacing_fix():
    """Preserve Plotly legend color swatches and tighten label spacing without relocating legends."""
    st.markdown("""
    <style>
    .js-plotly-plot .legend rect.bg {
        fill: rgba(255,255,255,.88) !important;
        stroke: rgba(230,232,239,.62) !important;
        stroke-width: 1px !important;
    }
    .js-plotly-plot .legend .traces,
    .js-plotly-plot .legendpoints,
    .js-plotly-plot .legendsymbols {
        opacity: 1 !important;
    }
    .js-plotly-plot .legendpoints path,
    .js-plotly-plot .legendpoints circle,
    .js-plotly-plot .legendpoints rect,
    .js-plotly-plot .legendsymbols path,
    .js-plotly-plot .legendsymbols circle,
    .js-plotly-plot .legendsymbols rect {
        opacity: 1 !important;
        visibility: visible !important;
    }
    .js-plotly-plot .legend text,
    .js-plotly-plot .legendtext {
        font-family: Arial, sans-serif !important;
        font-size: clamp(8px, .66vw, 9.2px) !important;
        font-weight: 760 !important;
        letter-spacing: -0.025em !important;
    }
    </style>
    """, unsafe_allow_html=True)
inject_plotly_legend_color_spacing_fix()

# ---------------- CHATBOT-ONLY CHART / MAP EXPLANATION SUPPORT ----------------
import html
import re

def _strip_plotly_html(value):
    value = "" if value is None else str(value)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(value.split()).strip()


def _escape_chart_header_html(value):
    return html.escape("" if value is None else str(value), quote=True)


def render_dashboard_plotly_chart(
    fig,
    *,
    plot_df=None,
    visual_type="chart",
    x_col=None,
    group_col=None,
    dashboard_df=None,
    title=None,
    key=None,
    container=None,
    use_container_width=True,
    config=None,
    expanded=False,
    chart_info=None,
    show_title_tooltip=False,
    chart_width_px=620,
    permission_key=None,
    permission_label=None,
):
    target = container if container is not None else st

    if permission_key and not can_render_feature(permission_key):
        render_permission_locked_card(
            permission_label or title or visual_type.title(),
            permission_key,
            container=target
        )
        return None

    # Keep title and info badge inside the Plotly chart area
    if show_title_tooltip and chart_info:
        try:
            fig = add_chart_info_badge(
                fig,
                chart_info,
                x=0.10,
                y=1.265,
                chart_width_px=chart_width_px,
            )
        except Exception:
            pass

    # Keep chart title centered inside Plotly canvas
    current_title = title or fig.layout.title.text or ""

    if current_title:
        fig.update_layout(
            title=dict(
                text=current_title,
                x=0.5,
                xanchor="center",
                y=0.965,
                yanchor="top"
            ),
            margin=dict(
                t=85
            )
        )

    fig = apply_responsive_plotly_layout(fig)

    # Re-apply title and top margin after responsive layout,
    # in case apply_responsive_plotly_layout overrides layout values.
    if current_title:
        fig.update_layout(
            title=dict(
                text=current_title,
                x=0.5,
                xanchor="center",
                y=0.965,
                yanchor="top"
            ),
            margin=dict(
                t=85
            )
        )

    target.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config=config,
        key=key
    )
# ---------------- TAB 1 ------------------------
if tab_overview is not None:
    with tab_overview:

        if has_permission("view_overview"):
            #st.subheader("Overview Metrics")
            if has_permission("view_coverage_monitored_countries"):
                render_summary_cards(filtered_global, card_key="overview_summary")
            a1 = filtered_global.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
            df_clean = filtered_global.assign(**{"enabling-principle": filtered_global["enabling-principle"].str.split(",")}).explode("enabling-principle")
            df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().map(ENABLING_PRINCIPLE_LABEL_MAP)
            df_clean["enabling-principle"] = pd.Categorical(df_clean["enabling-principle"],categories=ENABLING_PRINCIPLE_ORDER,ordered=True)
            a2 = df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count').sort_values("enabling-principle",ascending=False)
            a3 = filtered_global.groupby(["region","alert-impact"]).size().reset_index(name='count')
            #a4 = filtered_global.groupby(["alert-country","alert-impact"]).size().reset_index(name='count').sort_values(by='count', ascending=False)
            # Top 10 countries by total alert count
            top10_countries = (
                filtered_global
                .groupby("alert-country")
                .size()
                .nlargest(15)
                .index
            )

            # Keep only those countries
            a4 = (
                filtered_global[
                    filtered_global["alert-country"].isin(top10_countries)
                ]
                .groupby(["alert-country", "alert-impact"])
                .size()
                .reset_index(name="count")
            )

            # Percentage of total alerts within the Top 10 countries
            a4["percentage"] = (
                a4["count"] / a4["count"].sum() * 100
            )

            # Sort countries by their total counts
            country_order = (
                filtered_global
                .groupby("alert-country")
                .size()
                .loc[top10_countries]
                .sort_values(ascending=False)
                .index
            )

            a4["alert-country"] = pd.Categorical(
                a4["alert-country"],
                categories=country_order,
                ordered=True
            )

            a4 = a4.sort_values(
                ["alert-country", "count"],
                ascending=[True, False]
            )

            r1c1,r1c2 = st.columns(2)
            r2c1,r2c2 = st.columns(2)


            render_dashboard_plotly_chart(create_h_stacked_bar(a1,y="alert-type",x="count",color_col="alert-impact",title="Alert type distribution", horizontal=True, normalize_labels=True), plot_df=a1, visual_type="stacked bar chart", x_col="alert-type", group_col="alert-impact", dashboard_df=filtered_global, key="tab1_chart1", container=r1c1, permission_key="view_chart_overview_alert_type", permission_label="Overview alert type distribution")

            fig12 = create_h_stacked_bar(
                a2,
                y="enabling-principle",
                x="count",
                color_col="alert-impact",
                title="Alert distribution across enabling principles", 
                horizontal=True,
                normalize_labels=False
            )

            enabling_principle_note = (
                "Alerts may be classified under more than one enabling principle "
                "and can therefore be counted in multiple principles."
            )

            # Add source line if needed
            #fig12 = add_source_line(fig12)

            # Render chart in Streamlit with the info tooltip directly beside the title.
            render_dashboard_plotly_chart(
                fig12,
                plot_df=a2,
                visual_type="stacked bar chart",
                x_col="enabling-principle",
                group_col="alert-impact",
                dashboard_df=filtered_global,
                key="tab1_chart2",
                container=r1c2,
                chart_info=enabling_principle_note,
                show_title_tooltip=True,
                permission_key="view_chart_overview_enabling_principles",
                permission_label="Overview enabling-principle distribution",
            )
  
            #r1c2.plotly_chart(create_h_stacked_bar(a2,y="enabling-principle",x="count",color_col="alert-impact",title="Alert distribution across enabling principles", horizontal=True),use_container_width=True,  key="tab1_chart2")

            #if is_privileged():
            render_dashboard_plotly_chart(create_h_stacked_bar(a3,y="region",x="count",color_col="alert-impact",title="Alert distribution across regions", horizontal=False, normalize_labels=False), plot_df=a3, visual_type="stacked bar chart", x_col="region", group_col="alert-impact", dashboard_df=filtered_global, key="tab1_chart3", container=r2c1, permission_key="view_chart_overview_regions", permission_label="Overview regional distribution")
            render_dashboard_plotly_chart(create_h_stacked_bar(a4,y="alert-country",x="count",color_col="alert-impact",title="Alert distribution across countries", horizontal=False, normalize_labels=False), plot_df=a4, visual_type="stacked bar chart", x_col="alert-country", group_col="alert-impact", dashboard_df=filtered_global, key="tab1_chart4", container=r2c2, permission_key="view_chart_overview_countries", permission_label="Overview country distribution")

    
            cols_rename_map  = {
                "post_title": "Title of post",
                "summary": "Event summary",
                "creation_date": "Date of submission",
                "alert-country": "Country",
                "enabling-principle": "Enabling principles",
                "alert-impact": "Impact of alert",
                "alert-type": "Type of alert"
            }
                # keep only existing columns, then rename
            filtered_global_prev = (
                filtered_global
                .loc[:, filtered_global.columns.intersection(cols_rename_map.keys())]
                .rename(columns=cols_rename_map)
            )
   
                # ---------------- Tab two data preview ------------------

            if has_permission("view_data_table"):
                render_professional_data_preview(filtered_global_prev, title="Data Preview and Download", key="overview_summary_data_preview", remove_vertical_scroll=False)  
            #else:
                #st.info("Sign in with an authorized account to unlock additional detailed and disaggregated data.")   
        
        # ---------------- Negative Events ----------------
        else:
            render_access_locked("Overview", "public-summary or viewer")

if tab_negative is not None:
    with tab_negative:

        if has_permission("view_negative_alerts"):
            #st.subheader("Negative Alerts")
            # Filter negative events
            reactive_df = filtered_global[filtered_global['alert-impact'] == "Negative"].copy()


            if reactive_df.empty:
                st.warning("No negative events available for the selected filters.")
        
            else:
                # Initialize Top-N selection in session state
                if "neg_top_n" not in st.session_state:
                    st.session_state["neg_top_n"] = 5  # default Top 5
            
                # ---------------- SPELL OUT "VNSAs" ----------------
   
                reactive_df['Actor of repression'] = (reactive_df['Actor of repression'].astype(str).str.replace(r'\bVNSAs\b', 'Violent non-state actors', regex=True))
        
                # ---------------- SUMMARY CARDS ----------------
                # Show totals BEFORE exploding multi-valued columns

                protected_label = "Journalists, media and influencers"
                placeholder = "Journalists__MEDIA__and__influencers"
    
                def safe_split(x):
                    if pd.isna(x):
                        return []

                    x = x.strip()

                    # Temporarily replace protected label
                    x = x.replace(protected_label, placeholder)

                    # Split normally
                    parts = [i.strip() for i in x.split(",")]

                    # Restore protected label
                    parts = [p.replace(placeholder, protected_label) for p in parts]

                    return parts

        
                # ---------------- EXPLODE MULTI-VALUED COLUMNS ----------------
                cols_to_explode = [
                    "Actor of repression",
                    "Subject of repression",
                    "Mechanism of repression",
                    "Type of event"
                ]

                df_exploded = reactive_df.copy()

                df_exploded = df_exploded[(df_exploded['Type of event'] != "Error")]

                for col in cols_to_explode:
                    df_exploded[col] = df_exploded[col].apply(safe_split)
                    df_exploded = df_exploded.explode(col)
                    df_exploded[col] = df_exploded[col].astype(str).str.strip()

                def cap_first(s):
                    if pd.isna(s):
                        return None
                    s = str(s).strip()
                    if not s:
                        return None
                    return s[:1].upper() + s[1:]

                def formatted_options(series):
                    s = series.dropna().astype(str).str.strip()
                    s = s[s.ne("")]
                    return sorted(s.map(cap_first).dropna().unique())
    

                # ---------------- NEGATIVE ALERTS FILTERS: PROFESSIONAL GROUPED PANEL ----------------
                with st.expander("⚠️ Negative alerts filters", expanded=True):
                    st.markdown(
                        """
                        <div class="negative-filter-shell">
                            <div class="negative-filter-title">Negative Alerts Filter Panel</div>
                            <div class="negative-filter-note">
                                Explore negative alerts in more detail, including affected civil society actors, 
                                restrictive actors and mechanisms, negative event types, and alert distribution across types and enabling principles. 
                                Use the filters to focus on specific restrictive actors, affected civil society actors, mechanisms, and negative event types.
                            </div>
                      
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    neg_f1, neg_f2 = st.columns(2)

                    with neg_f1:
                        selected_actor_types = safe_multiselect(
                            "Types of restrictive actors",
                            formatted_options(df_exploded["Actor of repression"]),
                            "selected_actor_types",
                            sidebar=False,
                        )

                        selected_subject_types = safe_multiselect(
                            "Types of civil society actors affected",
                            formatted_options(df_exploded["Subject of repression"]),
                            "selected_subject_types",
                            sidebar=False,
                        )

                    with neg_f2:
                        selected_mechanism_types = safe_multiselect(
                            "Types of restrictive mechanisms",
                            formatted_options(df_exploded["Mechanism of repression"]),
                            "selected_mechanism_types",
                            sidebar=False,
                        )

                        selected_event_types = safe_multiselect(
                            "Types of negative events",
                            formatted_options(df_exploded["Type of event"]),
                            "selected_event_types",
                            sidebar=False,
                        )
                ##### -------- Tab 2 Summary card totals--------------------------
                reactive_df_updated= reactive_df[(reactive_df['Actor of repression'].apply(lambda x: contains_any(x, selected_actor_types))) &
                    (reactive_df['Subject of repression'].apply(lambda x: contains_any(x, selected_subject_types))) &
                    (reactive_df['Mechanism of repression'].apply(lambda x: contains_any(x, selected_mechanism_types))) &
                    (reactive_df['Type of event'].apply(lambda x: contains_any(x, selected_event_types)))
                ]
                render_negative_alerts_intelligence_cards(
                    reactive_df_updated,
                    all_filtered_df=filtered_global,
                    card_key="negative_events_summary"
                )

                #df_exploded['Subject of repression'] = df_exploded['Subject of repression'].apply(safe_split)

                filtered_df= df_exploded[(df_exploded['Actor of repression'].apply(lambda x: contains_any(x, selected_actor_types))) &
                    (df_exploded['Subject of repression'].apply(lambda x: contains_any(x, selected_subject_types))) &
                    (df_exploded['Mechanism of repression'].apply(lambda x: contains_any(x, selected_mechanism_types))) &
                    (df_exploded['Type of event'].apply(lambda x: contains_any(x, selected_event_types)))
                ]
    
                filtered_df1 = df_exploded.copy()
                #filtered_df = reactive_df_updated.copy()
    
                tab2_actor = reactive_df_updated.assign(**{"Actor of repression": reactive_df_updated["Actor of repression"].str.split(",")}).explode("Actor of repression")
    
                tab2_actor["Actor of repression"] = tab2_actor["Actor of repression"].str.strip()
                m1 = tab2_actor.groupby(["Actor of repression","alert-impact"]).size().reset_index(name='count')

                #tab2_subj = reactive_df_updated.assign(**{"Subject of repression": reactive_df_updated["Subject of repression"].str.split(",")}).explode("Subject of repression")
    
                tab2_subj = (
                    reactive_df_updated
                    .assign(**{
                        "Subject of repression": reactive_df_updated["Subject of repression"].apply(safe_split)
                    })
                    .explode("Subject of repression")
                )
       
                tab2_subj["Subject of repression"] = tab2_subj["Subject of repression"].str.strip()
                m2 = tab2_subj.groupby(["Subject of repression","alert-impact"]).size().reset_index(name='count')

                tab2_mech = reactive_df_updated.assign(**{"Mechanism of repression": reactive_df_updated["Mechanism of repression"].str.split(",")}).explode("Mechanism of repression")
                tab2_mech["Mechanism of repression"] = tab2_mech["Mechanism of repression"].str.strip()
                m3 = tab2_mech.groupby(["Mechanism of repression","alert-impact"]).size().reset_index(name='count')

                tab2_type = reactive_df_updated.assign(**{"Type of event": reactive_df_updated["Type of event"].str.split(",")}).explode("Type of event")
                tab2_type["Type of event"] = tab2_type["Type of event"].str.strip()
                m4 = tab2_type.groupby(["Type of event","alert-impact"]).size().reset_index(name='count')

                tab2_alert = reactive_df_updated.assign(**{"alert-type": reactive_df_updated["alert-type"].str.split(",")}).explode("alert-type")
                tab2_alert["alert-type"] = tab2_alert["alert-type"].str.strip()
                m5 = tab2_alert.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
    
                tab2_enabling_principle = reactive_df_updated.assign(**{"enabling-principle": reactive_df_updated["enabling-principle"].str.split(",")}).explode("enabling-principle")
                tab2_enabling_principle["enabling-principle"] = tab2_enabling_principle["enabling-principle"].str.strip().map(ENABLING_PRINCIPLE_LABEL_MAP)
                tab2_enabling_principle["enabling_principle"] = pd.Categorical(tab2_enabling_principle["enabling-principle"],categories=ENABLING_PRINCIPLE_ORDER,ordered=True)
                m6 = tab2_enabling_principle.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count').sort_values("enabling-principle",ascending=False)
    
                # ---------------- BAR CHARTS ----------------
                r1c1, r1c2, r1c3 = st.columns(3)
                r2c1, r2c2, r2c3 = st.columns(3)

    
                render_dashboard_plotly_chart(create_bar_chart(m1, "Actor of repression", "count",title="Types of restrictive actors", normalize_labels=True), plot_df=m1, visual_type="bar chart", x_col="Actor of repression", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart1", container=r1c1, permission_key="view_chart_negative_restrictive_actors", permission_label="Restrictive actors chart")
                render_dashboard_plotly_chart(create_bar_chart(m2, "Subject of repression", "count",title="Types of civil society actors affected", normalize_labels=True), plot_df=m2, visual_type="bar chart", x_col="Subject of repression", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart2", container=r1c2, permission_key="view_chart_negative_affected_actors", permission_label="Civil society actors affected chart")
                render_dashboard_plotly_chart(create_bar_chart(m3, "Mechanism of repression", "count",title="Types of restrictive mechanisms", normalize_labels=True), plot_df=m3, visual_type="bar chart", x_col="Mechanism of repression", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart3", container=r1c3, permission_key="view_chart_negative_restrictive_mechanisms", permission_label="Restrictive mechanisms chart")
                render_dashboard_plotly_chart(create_bar_chart(m4, "Type of event", "count",title="Types of negative events", horizontal=True, normalize_labels=True), plot_df=m4, visual_type="bar chart", x_col="Type of event", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart4", container=r2c1, permission_key="view_chart_negative_event_types", permission_label="Negative event types chart")
                render_dashboard_plotly_chart(create_bar_chart(m5, "alert-type", "count",title="Distribution of negative alert types", horizontal=True, normalize_labels=True), plot_df=m5, visual_type="bar chart", x_col="alert-type", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart5", container=r2c2, permission_key="view_chart_negative_alert_types", permission_label="Negative alert types chart")
          
                fig23= (create_bar_chart(m6, "enabling-principle", "count", title="Negative alert distribution across enabling principles", horizontal=True, normalize_labels=False))

          
                negative_enabling_principle_note = (
                    "Negative alerts may be classified under more than one enabling principle "
                    "and can therefore be counted in multiple principles."
                )

                # Render the chart in Streamlit with the info tooltip directly beside the title.
                render_dashboard_plotly_chart(
                    fig23,
                    plot_df=m6,
                    visual_type="bar chart",
                    x_col="enabling-principle",
                    group_col="alert-impact",
                    dashboard_df=reactive_df_updated,
                    key="tab2_chart6",
                    container=r2c3,
                    chart_info=negative_enabling_principle_note,
                    show_title_tooltip=True,
                    permission_key="view_chart_negative_enabling_principles",
                    permission_label="Negative enabling-principle distribution",
                )

                #r2c3.plotly_chart(create_bar_chart(m6, "enabling-principle", "count",title="Negative alert distribution across enabling principles", horizontal=True), use_container_width=True, key="tab2_chart6")

                # ---------------- ANALYTICAL FLOW PANEL ----------------
                if has_permission("view_analytical_flow_panel"):
                    render_analytical_flow_panel(filtered_df)
                #else:
                    #st.info("Analytical Flow Panel is disabled for your current access level.")

                cols_to_keep = {
                    "post_title": "Title of post",
                    "summary": "Event summary",
                    "creation_date": "Date of submission",
                    "alert-country": "Country",
                    "enabling-principle": "Enabling principles",
                    "alert-impact": "Impact of alert",
                    "alert-type": "Type of alert",
                    "Actor of repression": "Types of restrictive actors",
                    "Subject of repression": "Types of civil society actors affected",
                    "Mechanism of repression": "Types of restrictive mechanisms",
                    "Type of event": "Types of negative events"           
                }
                # keep only existing columns, then rename
                reactive_df_updated_prev = (
                    reactive_df_updated
                    .loc[:, reactive_df_updated.columns.intersection(cols_to_keep.keys())]
                    .rename(columns=cols_to_keep)
                )
        
                # ---------------- Tab two data preview ----------------
              
                if has_permission("view_data_table"):
                    render_professional_data_preview(reactive_df_updated_prev, title="Data Preview and Download", key="negative_summary_data_preview")
           
                # ---------------- TAB 3 (MAP) ----------------
        else:
            render_access_locked("Negative Alerts", "privileged")

if tab_map is not None:
    with tab_map:

        if has_permission("view_maps"):

            if has_permission("view_maps"):
                # ---------------- PREMIUM GEOSPATIAL INTELLIGENCE TAB ----------------
                if has_permission("view_coverage_monitored_countries"):
                    render_summary_cards(filtered_global, card_key="map_summary")

                MAP_FONT = "Inter, Segoe UI, Arial, sans-serif"

                st.markdown("""
                <style>
                .map-page-shell {
                    background: transparent;
                    border: 0;
                    border-radius: 0;
                    padding: 0;
                    margin: 2px 0 8px 0;
                    box-shadow: none;
                    font-family: Arial, sans-serif;
                }
                .map-intel-hero {
                    background:
                        radial-gradient(circle at 96% 10%, rgba(0,140,170,.08), transparent 28%),
                        linear-gradient(135deg, #FFFFFF 0%, #FBF7FF 100%);
                    border: 1px solid rgba(102,0,148,0.12);
                    border-radius: 18px;
                    padding: 12px 15px;
                    box-shadow: 0 8px 20px rgba(17,24,39,0.045);
                    margin: 4px 0 8px 0;
                }
                .map-hero-top {
                    display:flex;
                    justify-content:space-between;
                    align-items:flex-start;
                    gap:16px;
                    flex-wrap:wrap;
                }
                .map-intel-eyebrow {
                    font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
                    font-size: 9.5px;
                    font-weight: 850;
                    letter-spacing: .105em;
                    text-transform: uppercase;
                    color: #660094;
                    margin-bottom: 5px;
                    line-height: 1.15;
                }
                .map-intel-title {
                    font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
                    font-size: 14px;
                    font-weight: 850;
                    color: #101828;
                    margin-bottom: 5px;
                    letter-spacing: -0.025em;
                    line-height: 1.15;
                }
                .map-intel-subtitle {
                    font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
                    font-size: 11.5px;
                    font-weight: 550;
                    color: #667085;
                    line-height: 1.45;
                    max-width: 1100px;
                }
                .map-legend-chip {
                    background:#FFFFFF;
                    border:1px solid #E9E2F2;
                    border-radius:999px;
                    padding:7px 11px;
                    color:#344054;
                    font-size:11px;
                    font-weight:850;
                    box-shadow:0 4px 10px rgba(17,24,39,.045);
                    white-space:nowrap;
                }
                .map-chip-row {display:flex; flex-wrap:wrap; gap:8px; margin-top:11px;}
                .map-chip {
                    display:inline-flex;
                    align-items:center;
                    gap:6px;
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    color:#334155;
                    border-radius:999px;
                    padding:6px 10px;
                    font-size:11px;
                    font-weight:850;
                    box-shadow:0 3px 9px rgba(17,24,39,0.045);
                }
                .map-intel-card {
                    height: 128px;
                    background: #FFFFFF;
                    border: 1px solid #E8EAF0;
                    border-radius: 17px;
                    padding: 13px 14px;
                    box-shadow: 0 10px 24px rgba(17,24,39,0.055);
                    font-family: Arial, sans-serif;
                    overflow:hidden;
                    position:relative;
                }
                .map-intel-card::before {
                    content:"";
                    position:absolute;
                    left:0; right:0; top:0;
                    height:4px;
                    background:linear-gradient(90deg, #660094 0%, #008CAA 55%, #FFDB58 100%);
                }
                .map-intel-card-label {
                    font-size: 10px;
                    font-weight: 950;
                    color: #64748B;
                    text-transform: uppercase;
                    letter-spacing: .08em;
                    margin-bottom: 6px;
                }
                .map-intel-card-value {
                    font-size: 27px;
                    font-weight: 950;
                    color: #2D0055;
                    line-height:1.05;
                    letter-spacing:-.035em;
                }
                .map-intel-card-note {
                    font-size: 10.8px;
                    color: #667085;
                    line-height:1.32;
                    margin-top: 7px;
                }
                .map-insight-grid {
                    display:grid;
                    grid-template-columns: 1.25fr 1fr 1fr;
                    gap:12px;
                    margin:13px 0 14px 0;
                }
                .map-insight-card {
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    border-radius:17px;
                    padding:13px 14px;
                    box-shadow:0 10px 24px rgba(17,24,39,.055);
                    min-height:100px;
                }
                .map-insight-title {
                    font-size:10px;
                    color:#660094;
                    font-weight:950;
                    text-transform:uppercase;
                    letter-spacing:.11em;
                    margin-bottom:6px;
                }
                .map-insight-text {
                    font-size:12.4px;
                    color:#334155;
                    line-height:1.5;
                    font-weight:650;
                }
                .map-insight-text b {color:#2D0055; font-weight:950;}
                .map-method-note {
                    background:#FFFBEB;
                    border:1px solid #FDE68A;
                    border-left:4px solid #FFDB58;
                    border-radius:15px;
                    padding:11px 13px;
                    color:#4A3B00;
                    font-size:11.8px;
                    line-height:1.48;
                    margin:12px 0;
                    font-family:Arial, sans-serif;
                }
                .map-panel-card {
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    border-radius:18px;
                    padding:12px 13px;
                    box-shadow:0 10px 22px rgba(17,24,39,.052);
                    margin: 6px 0 10px 0;
                    font-family:Arial, sans-serif;
                }
                .map-layout-tight {
                    margin-top: 0;
                    margin-bottom: 0;
                }
                .map-visual-card {
                    position: relative;
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    border-radius:18px;
                    padding:8px 8px 4px 8px;
                    box-shadow:0 10px 24px rgba(17,24,39,.055);
                    margin: 4px 0 8px 0;
                    overflow:hidden;
                }
                .map-reading-strip {
                    display:flex;
                    flex-wrap:wrap;
                    gap:8px;
                    align-items:center;
                    justify-content:space-between;
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    border-radius:15px;
                    padding:9px 11px;
                    margin: 8px 0 0 0;
                    box-shadow:0 6px 14px rgba(17,24,39,.04);
                    font-family:Arial, sans-serif;
                }
                .map-reading-strip span {
                    color:#334155;
                    font-size:11.2px;
                    font-weight:750;
                    line-height:1.35;
                }
                .map-reading-strip b {color:#2D0055; font-weight:950;}
                .map-support-grid {
                    margin-top: -4px;
                    margin-bottom: 4px;
                }
                .map-panel-title {
                    color:#2D0055;
                    font-size:15px;
                    font-weight:950;
                    margin-bottom:4px;
                    letter-spacing:-.15px;
                }
                .map-panel-help {
                    color:#64748B;
                    font-size:11.5px;
                    line-height:1.45;
                    margin-bottom:10px;
                }
                .country-insight-box {
                    background:linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
                    border:1px solid #E8EAF0;
                    border-left:4px solid #660094;
                    border-radius:15px;
                    padding:13px 14px;
                    color:#334155;
                    font-size:12px;
                    line-height:1.52;
                    margin-top:10px;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,.9);
                }
                .country-insight-box b {color:#2D0055;}
                .country-mini-grid {
                    display:grid;
                    grid-template-columns: repeat(2, minmax(0,1fr));
                    gap:8px;
                    margin:10px 0;
                }
                .country-mini-kpi {
                    background:#F8FAFC;
                    border:1px solid #EEF2F6;
                    border-radius:12px;
                    padding:8px 9px;
                }
                .country-mini-kpi span {
                    display:block;
                    color:#64748B;
                    font-size:9.5px;
                    font-weight:900;
                    text-transform:uppercase;
                    letter-spacing:.06em;
                    margin-bottom:3px;
                }
                .country-mini-kpi strong {
                    color:#2D0055;
                    font-size:15px;
                    font-weight:950;
                }
                .map-action-list {
                    margin: 8px 0 0 0;
                    padding-left: 18px;
                    color:#334155;
                    font-size:11.8px;
                    line-height:1.5;
                    font-weight:650;
                }
                .map-overview-guide {
                    display:grid;
                    grid-template-columns: minmax(220px, .72fr) minmax(0, 1fr);
                    gap:12px;
                    align-items:stretch;
                    margin-top:13px;
                }
                .map-guide-card {
                    background:linear-gradient(180deg,#FFFFFF 0%,#FAF7FC 100%);
                    border:1px solid rgba(102,0,148,.14);
                    border-radius:16px;
                    padding:12px 14px;
                    box-shadow:0 8px 18px rgba(45,0,85,.065);
                    margin:0;
                    font-family:Arial, sans-serif;
                }
                .map-guide-title {
                    color:#2D0055;
                    font-size:13.5px;
                    font-weight:950;
                    letter-spacing:-.12px;
                    margin-bottom:4px;
                }
                .map-guide-sub {
                    color:#64748B;
                    font-size:10.8px;
                    line-height:1.42;
                    margin-bottom:10px;
                }
                .map-guide-step {
                    display:grid;
                    grid-template-columns:23px 1fr;
                    gap:8px;
                    align-items:flex-start;
                    padding:7px 0;
                    border-top:1px solid #EEF0F4;
                }
                .map-guide-num {
                    width:21px;
                    height:21px;
                    border-radius:8px;
                    background:linear-gradient(135deg,#660094 0%,#008CAA 100%);
                    color:#FFFFFF;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:9.5px;
                    font-weight:950;
                    box-shadow:0 4px 9px rgba(102,0,148,.18);
                }
                .map-guide-text {
                    font-size:10.8px;
                    color:#344054;
                    line-height:1.38;
                    font-weight:650;
                }
                .map-guide-text b {color:#23152F; font-weight:950;}
                .map-overview-stat-grid {
                    display:grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap:8px;
                }
                .map-overview-stat {
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    border-radius:14px;
                    padding:11px 12px;
                    box-shadow:0 6px 14px rgba(17,24,39,.04);
                }
                .map-overview-stat span {
                    display:block;
                    color:#64748B;
                    font-size:9.5px;
                    font-weight:900;
                    text-transform:uppercase;
                    letter-spacing:.06em;
                    margin-bottom:4px;
                }
                .map-overview-stat strong {
                    color:#2D0055;
                    font-size:18px;
                    font-weight:950;
                    line-height:1.05;
                }
                .map-overview-stat small {
                    display:block;
                    color:#667085;
                    font-size:10.3px;
                    line-height:1.32;
                    margin-top:5px;
                    font-weight:650;
                }
                @media (max-width: 1000px) {
                    .map-overview-guide { grid-template-columns:1fr; }
                    .map-overview-stat-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
                }
                @media (max-width: 620px) {
                    .map-overview-stat-grid { grid-template-columns:1fr; }
                }
                .priority-country-panel {
                    background:linear-gradient(180deg,#FFFFFF 0%,#FCFAFF 100%);
                    border:1px solid #E7D4F1;
                    border-radius:18px;
                    padding:12px 14px;
                    box-shadow:0 8px 18px rgba(45,0,85,.06);
                    margin:6px 0 10px 0;
                    font-family:Arial, sans-serif;
                }
                .priority-title {
                    color:#2D0055;
                    font-size:15.5px;
                    font-weight:950;
                    letter-spacing:-.15px;
                    margin-bottom:5px;
                }
                .priority-sub {
                    color:#64748B;
                    font-size:11.5px;
                    line-height:1.45;
                    margin-bottom:12px;
                }
                .priority-row {
                    display:grid;
                    grid-template-columns:30px minmax(0,1fr) auto;
                    align-items:center;
                    gap:9px;
                    padding:9px 10px;
                    margin-bottom:8px;
                    border-radius:14px;
                    background:#FFFFFF;
                    border:1px solid #EEF0F4;
                    box-shadow:0 4px 10px rgba(16,24,40,.045);
                }
                .priority-rank {
                    width:25px;
                    height:25px;
                    border-radius:10px;
                    background:linear-gradient(135deg,#660094 0%,#008CAA 100%);
                    color:#FFFFFF;
                    font-size:10px;
                    font-weight:950;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                }
                .priority-country {
                    font-size:12.2px;
                    font-weight:950;
                    color:#23152F;
                    line-height:1.15;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    white-space:nowrap;
                }
                .priority-meta {
                    font-size:10.5px;
                    color:#667085;
                    margin-top:3px;
                    line-height:1.25;
                }
                .priority-meta b {color:#2D0055; font-weight:950;}
                .priority-score {
                    text-align:right;
                    color:#660094;
                    font-size:11px;
                    font-weight:950;
                    line-height:1.12;
                    white-space:nowrap;
                }
                .priority-score span {
                    display:block;
                    color:#667085;
                    font-size:9.5px;
                    font-weight:850;
                    margin-bottom:2px;
                }
                .priority-badge {
                    display:inline-block;
                    margin-top:5px;
                    padding:3px 8px;
                    border-radius:999px;
                    background:#FFF4ED;
                    color:#B42318;
                    border:1px solid rgba(180,35,24,.16);
                    font-size:9.5px;
                    font-weight:950;
                }
                .priority-badge.priority-watch {background:#F8FAFC;color:#475467;border-color:#E8EAF0;}
                .priority-badge.priority-moderate {background:#EFFBFE;color:#008CAA;border-color:rgba(0,140,170,.18);}
                .priority-badge.priority-high {background:#FFFBEB;color:#7A3E00;border-color:#FDE68A;}
                .priority-badge.priority-very-high {background:#FFF4ED;color:#B42318;border-color:rgba(180,35,24,.16);}
                .priority-footnote {
                    margin-top:8px;
                    padding-top:9px;
                    border-top:1px solid #EEF0F4;
                    color:#667085;
                    font-size:10.5px;
                    line-height:1.35;
                    font-weight:650;
                }
                .map-quality-strip {
                    display:flex;
                    gap:8px;
                    flex-wrap:wrap;
                    margin:8px 0 0 0;
                }
                .map-quality-pill {
                    background:#F8FAFC;
                    border:1px solid #E8EAF0;
                    color:#475467;
                    border-radius:999px;
                    padding:5px 9px;
                    font-size:10.5px;
                    font-weight:850;
                }
                @media (max-width: 980px) {
                    .map-insight-grid {grid-template-columns:1fr;}
                    .map-intel-card {height:auto; min-height:118px;}
                    .country-mini-grid {grid-template-columns:1fr;}
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown('<div class="map-page-shell">', unsafe_allow_html=True)

                geo_file_candidates = [
                    Path("/exports") / "countries.geojson",
                    Path.cwd() / "exports" / "countries.geojson",
                    Path.cwd() / "exports" / "countriess.geojson",  # legacy typo fallback
                ]
                geo_file = next((p for p in geo_file_candidates if p.exists()), None)

                if geo_file is not None and geo_file.exists():
                    with open(geo_file, encoding="utf-8") as f:
                        countries_gj = json.load(f)

                    # ---------------- Base map data and intelligence metrics ----------------
                    stats = (
                        filtered_global
                        .groupby("alert-country", dropna=False)
                        .agg(
                            iso_alpha3=("iso_alpha3", lambda x: next((v for v in x.dropna().astype(str) if v.strip()), None)),
                            total_alerts=("alert-impact", "size"),
                            negative_alerts=("alert-impact", lambda x: int((x == "Negative").sum())),
                            positive_alerts=("alert-impact", lambda x: int((x == "Positive").sum())),
                            context_to_watch_alerts=("alert-impact", lambda x: int((x == "Context to watch").sum())),
                            regions=("region", lambda x: ", ".join(sorted(set(x.dropna().astype(str)))[:2])),
                        )
                        .reset_index()
                    )

                    geo_iso3 = {
                        str(f.get("properties", {}).get("ISO3166-1-Alpha-3", "")).strip()
                        for f in countries_gj.get("features", [])
                    }
                    geo_iso3 = {x for x in geo_iso3 if x and x.lower() != "none"}

                    df_map = stats[
                        stats["iso_alpha3"].notna()
                        & stats["iso_alpha3"].astype(str).isin(geo_iso3)
                    ].copy()

                    for c in ["total_alerts", "negative_alerts", "positive_alerts", "context_to_watch_alerts"]:
                        df_map[c] = pd.to_numeric(df_map[c], errors="coerce").fillna(0).astype(int)

                    df_map["perc_negative"] = np.where(
                        df_map["total_alerts"] > 0,
                        (df_map["negative_alerts"] / df_map["total_alerts"] * 100).round(1),
                        0
                    )
                    df_map["alert_balance"] = (df_map["positive_alerts"] - df_map["negative_alerts"]).astype(int)
                    df_map["priority_score"] = (df_map["negative_alerts"] * 0.65 + df_map["perc_negative"] * 0.35).round(1)
                    df_map["priority_level"] = pd.cut(
                        df_map["priority_score"],
                        bins=[-1, 20, 45, 70, float("inf")],
                        labels=["Watch", "Moderate", "High", "Very high"]
                    ).astype(str)

                    total_filtered_records = int(len(filtered_global)) if filtered_global is not None else 0
                    total_mapped = int(df_map["total_alerts"].sum()) if not df_map.empty else 0
                    unmapped_alerts = max(total_filtered_records - total_mapped, 0)
                    mapping_coverage = round((total_mapped / total_filtered_records) * 100, 1) if total_filtered_records else 0
                    mapped_countries = int(df_map["alert-country"].nunique()) if not df_map.empty else 0
                    top_country = df_map.sort_values("total_alerts", ascending=False).iloc[0]["alert-country"] if not df_map.empty else "N/A"
                    top_priority_country = df_map.sort_values("priority_score", ascending=False).iloc[0]["alert-country"] if not df_map.empty else "N/A"
                    avg_negative_share = round(df_map["perc_negative"].mean(), 1) if not df_map.empty else 0
                    very_high_count = int((df_map["priority_level"] == "Very high").sum()) if not df_map.empty else 0
                    high_count = int((df_map["priority_level"] == "High").sum()) if not df_map.empty else 0
                    mapped_negative = int(df_map["negative_alerts"].sum()) if not df_map.empty else 0
                    mapped_positive = int(df_map["positive_alerts"].sum()) if not df_map.empty else 0
                    mapped_context = int(df_map["context_to_watch_alerts"].sum()) if not df_map.empty else 0

                    if mapped_negative >= max(mapped_positive, mapped_context):
                        dominant_signal = "Negative alerts are the dominant mapped signal"
                        dominant_next_step = "prioritize restrictive-event pathways and review affected actors."
                    elif mapped_positive >= mapped_context:
                        dominant_signal = "Positive alerts are the dominant mapped signal"
                        dominant_next_step = "identify enabling-pattern examples and potential comparative lessons."
                    else:
                        dominant_signal = "Context-to-watch alerts are the dominant mapped signal"
                        dominant_next_step = "monitor emerging situations before they shift into restrictive or enabling events."

                    priority_share = round(((very_high_count + high_count) / mapped_countries) * 100, 1) if mapped_countries else 0

                    unmapped_meta = sorted(
                        set(stats.loc[stats["iso_alpha3"].isna(), "alert-country"].dropna().astype(str))
                    )
                    unmapped_geo = sorted(
                        set(stats.loc[stats["iso_alpha3"].notna(), "alert-country"].astype(str))
                        - set(df_map["alert-country"].astype(str))
                    )

                    # Render the Geographic Overview panel through an HTML component.
                    # This prevents Streamlit from displaying the HTML markup as raw text.
                    components.html(
                        f"""
                        <style>
                        html, body {{
                            margin: 0;
                            padding: 0;
                            background: transparent;
                            font-family: Arial, sans-serif;
                            overflow-x: hidden;
                        }}

                        .map-intel-hero {{
                            background:
                                radial-gradient(circle at top right, rgba(102,0,148,.07), transparent 35%),
                                linear-gradient(180deg,#FFFFFF 0%,#FCFAFF 100%);
                            border: 1px solid rgba(102,0,148,.10);
                            border-radius: 22px;
                            padding: 18px 20px;
                            box-shadow: 0 14px 34px rgba(16,24,40,.06);
                            box-sizing: border-box;
                        }}

                        .map-hero-top {{
                            display: flex;
                            justify-content: space-between;
                            align-items: flex-start;
                            gap: 16px;
                            margin-bottom: 16px;
                        }}

                        .map-intel-eyebrow {{
                            color: #660094;
                            font-size: 10px;
                            font-weight: 900;
                            letter-spacing: .10em;
                            text-transform: uppercase;
                            margin-bottom: 5px;
                        }}

                        .map-intel-title {{
                            color: #101828;
                            font-family: "Inter", "Segoe UI", Arial, sans-serif;
                            font-size: 14px;
                            font-weight: 850;
                            letter-spacing: -0.02em;
                            line-height: 1.18;
                            margin-bottom: 7px;
                        }}

                        .map-intel-subtitle {{
                            color: #667085;
                            font-size: 12px;
                            line-height: 1.55;
                            max-width: 850px;
                        }}

                        .map-legend-chip {{
                            padding: 7px 12px;
                            border-radius: 999px;
                            background: #F4EAF8;
                            color: #660094;
                            border: 1px solid #E7D4F1;
                            font-size: 9.5px;
                            font-weight: 900;
                            white-space: nowrap;
                        }}

                        .map-overview-guide {{
                            display: grid;
                            grid-template-columns: minmax(300px, 1.05fr) minmax(360px, 1fr);
                            gap: 14px;
                            align-items: stretch;
                        }}

                        .map-guide-card {{
                            background: #FFFFFF;
                            border: 1px solid rgba(102,0,148,.12);
                            border-radius: 18px;
                            padding: 15px 16px;
                            box-shadow: 0 8px 18px rgba(45,0,85,.055);
                        }}

                        .map-guide-title {{
                            color: #23152F;
                            font-size: 12px;
                            font-weight: 950;
                            margin-bottom: 5px;
                        }}

                        .map-guide-sub {{
                            color: #667085;
                            font-size: 9.5px;
                            line-height: 1.45;
                            margin-bottom: 7px;
                        }}

                        .map-guide-step {{
                            display: grid;
                            grid-template-columns: 26px 1fr;
                            gap: 10px;
                            align-items: flex-start;
                            padding: 9px 0;
                            border-top: 1px solid #EEF0F4;
                        }}

                        .map-guide-step:first-of-type {{
                            border-top: none;
                            padding-top: 0;
                        }}

                        .map-guide-num {{
                            width: 24px;
                            height: 24px;
                            border-radius: 999px;
                            background: linear-gradient(135deg,#660094 0%,#008CAA 100%);
                            color: #FFFFFF;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 9.5px;
                            font-weight: 950;
                            box-shadow: 0 4px 9px rgba(102,0,148,.18);
                        }}

                        .map-guide-text {{
                            font-size: 9.5px;
                            color: #344054;
                            line-height: 1.45;
                            font-weight: 650;
                        }}

                        .map-guide-text b {{
                            color: #23152F;
                            font-weight: 950;
                        }}

                        .map-overview-stat-grid {{
                            display: grid;
                            grid-template-columns: repeat(3, minmax(0, 1fr));
                            gap: 9px;
                        }}

                        .map-overview-stat {{
                            background: #FFFFFF;
                            border: 1px solid #E8EAF0;
                            border-radius: 16px;
                            padding: 13px 14px;
                            box-shadow: 0 6px 14px rgba(17,24,39,.04);
                        }}

                        .map-overview-stat span {{
                            display: block;
                            color: #667085;
                            font-size: 9.5px;
                            font-weight: 900;
                            text-transform: uppercase;
                            letter-spacing: .06em;
                            margin-bottom: 5px;
                        }}

                        .map-overview-stat strong {{
                            display: block;
                            color: #23152F;
                            font-size: 24px;
                            font-weight: 950;
                            line-height: 1.05;
                            margin-bottom: 5px;
                        }}

                        .map-overview-stat small {{
                            display: block;
                            color: #667085;
                            font-size: 10.3px;
                            line-height: 1.35;
                            font-weight: 650;
                        }}

                        @media (max-width: 980px) {{
                            .map-hero-top {{
                                flex-direction: column;
                            }}
                            .map-overview-guide {{
                                grid-template-columns: 1fr;
                            }}
                        }}

                        @media (max-width: 620px) {{
                            .map-intel-hero {{
                                padding: 15px;
                                border-radius: 18px;
                            }}
                            .map-overview-stat-grid {{
                                grid-template-columns: 1fr;
                            }}
                            .map-intel-title {{
                                font-size: 14px;
                            }}
                        }}
                        /* Force Visualization Map hover tooltip text to stay white. */
                        .js-plotly-plot .hoverlayer .hovertext text,
                        .js-plotly-plot .hoverlayer .hovertext tspan {{
                            fill: #FFFFFF !important;
                            color: #FFFFFF !important;
                            font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif) !important;
                        }}

                        </style>

                        <div class="map-intel-hero">
                            <div class="map-hero-top">
                                <div>
                                    <div class="map-intel-eyebrow">Geographic Overview</div>
                                    <div class="map-intel-title">Visualization Map: Alerts by Country</div>
                                    <div class="map-intel-subtitle">
                                        This map shows where alerts are concentrated across countries based on the filters selected.
                                        Use it to identify countries that may require closer review. Darker countries indicate a higher
                                        filtered alert volume.
                                    </div>
                                </div>
                            </div>

                            <div class="map-overview-guide">
                                <div class="map-guide-card">
                                    <div class="map-guide-title">🧭 How to read this map</div>
                                    <div class="map-guide-sub">
                                        Use this map to see where filtered alerts are concentrated and where follow-up review may be needed.
                                    </div>
                                    <div class="map-guide-step">
                                        <div class="map-guide-num">1</div>
                                        <div class="map-guide-text">
                                            <b>Look at color intensity:</b> darker countries indicate a higher number of filtered alerts.
                                        </div>
                                    </div>
                                    <div class="map-guide-step">
                                        <div class="map-guide-num">2</div>
                                        <div class="map-guide-text">
                                            <b>Hover for details:</b> hover over a country to see the alert breakdown and priority level.
                                        </div>
                                    </div>
                                </div>

                            </div>
                        </div>
                        """,
                        height=315,
                        scrolling=False,
                    )

                
                    if unmapped_meta or unmapped_geo:
                        issue_bits = []
                        if unmapped_meta:
                            issue_bits.append("Missing metadata: " + ", ".join(unmapped_meta[:12]) + (" ..." if len(unmapped_meta) > 12 else ""))
                        if unmapped_geo:
                            issue_bits.append("No GeoJSON geometry match: " + ", ".join(unmapped_geo[:12]) + (" ..." if len(unmapped_geo) > 12 else ""))
                        st.markdown(
                            f"""<div class="map-quality-strip"><span class="map-quality-pill">Data quality check</span><span class="map-quality-pill">{' | '.join(issue_bits)}</span></div>""",
                            unsafe_allow_html=True
                        )

                    # ---------------- Dynamic center and zoom ----------------
                    if not df_map.empty:
                        coords = []
                        country_iso_set = set(df_map["iso_alpha3"].dropna().astype(str))
                        for feature in countries_gj.get("features", []):
                            if str(feature.get("properties", {}).get("ISO3166-1-Alpha-3", "")).strip() in country_iso_set:
                                geometry = feature.get("geometry", {})
                                if geometry.get("type") == "Polygon":
                                    coords.extend(geometry.get("coordinates", [[]])[0])
                                elif geometry.get("type") == "MultiPolygon":
                                    for poly in geometry.get("coordinates", []):
                                        if poly:
                                            coords.extend(poly[0])
                        if coords:
                            lons, lats = zip(*coords)
                            center = {"lat": float(np.mean(lats)), "lon": float(np.mean(lons))}
                            lon_span = max(lons) - min(lons)
                            lat_span = max(lats) - min(lats)
                            span = max(lon_span, lat_span, 1)
                            zoom = max(1, min(4.2, 3.7 - np.log10(span + 1)))
                        else:
                            center, zoom = {"lat": 10, "lon": 0}, 1.6
                    else:
                        center, zoom = {"lat": 10, "lon": 0}, 1.6

                    # ---------------- Enlarged full-width map workspace ----------------
                    st.markdown('<div class="map-layout-tight">', unsafe_allow_html=True)

                    if df_map.empty:
                        st.info("No mapped country records are available under the current filters.")
                    else:
                        fig = px.choropleth_mapbox(
                            df_map,
                            geojson=countries_gj,
                            locations="iso_alpha3",
                            featureidkey="properties.ISO3166-1-Alpha-3",
                            color="total_alerts",
                            hover_name="alert-country",
                            color_continuous_scale=[[0, "#FFF7D6"], [0.45, "#FFDB58"], [1, "#7A3E00"]],
                            mapbox_style="carto-positron",
                            zoom=zoom,
                            center=center,
                            opacity=0.92,
                        )

                        fig.update_traces(
                            customdata=df_map[[
                                "alert-country", "total_alerts", "negative_alerts", "positive_alerts",
                                "context_to_watch_alerts", "perc_negative", "priority_level",
                                "regions", "priority_score"
                            ]].values,
                            hovertemplate=(
                                "<span style='color:#FFFFFF'><b>%{customdata[0]}</b></span><br>"
                                "<span style='color:#FFFFFF'>Region: %{customdata[7]}</span><br>"
                                "<span style='color:#FFFFFF'>● Total alerts: %{customdata[1]}</span><br>"
                                "<span style='color:#FFFFFF'>● Negative: %{customdata[2]}</span><br>"
                                "<span style='color:#FFFFFF'>● Positive: %{customdata[3]}</span><br>"
                                "<span style='color:#FFFFFF'>● Context: %{customdata[4]}</span><br>"
                            ),
                            hoverlabel=dict(
                                bgcolor="#2D0055",
                                font=dict(size=12, family=MAP_FONT, color="#FFFFFF"),
                                font_size=12,
                                font_family=MAP_FONT,
                                font_color="#FFFFFF",
                                bordercolor="#FFFFFF"
                            ),
                            marker_line_width=0.55,
                            marker_line_color="rgba(45,0,85,0.50)",
                        )

                        fig.update_layout(
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            height=720,
                            hoverlabel=dict(
                                bgcolor="#2D0055",
                                bordercolor="#FFFFFF",
                                font=dict(size=12, family=MAP_FONT, color="#FFFFFF"),
                            ),
                            coloraxis_colorbar=dict(
                                title=dict(text="Alerts", font=dict(size=11, family=MAP_FONT, color="#FFFFFF")),
                                tickfont=dict(size=10, family=MAP_FONT, color="#FFFFFF"),
                                thickness=12,
                                len=0.68,
                                x=0.985,
                                xanchor="left",
                                outlinewidth=0,
                            ),
                            mapbox=dict(
                                bearing=0,
                                pitch=0,
                            ),
                            font=dict(family=MAP_FONT, color="#FFFFFF"),
                        )

                        st.markdown('<div class="map-visual-card">', unsafe_allow_html=True)
                        render_dashboard_plotly_chart(
                            fig,
                            plot_df=df_map,
                            visual_type="map",
                            x_col="alert-country",
                            group_col="priority_level",
                            dashboard_df=filtered_global,
                            config={"displayModeBar": False, "responsive": True},
                            key="professional_geo_intelligence_map",
                            permission_key="view_chart_geospatial_map",
                            permission_label="Geospatial intelligence map",
                        )
                        st.markdown('</div>', unsafe_allow_html=True)


                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        else:
            render_access_locked("Visualization Map", "viewer or privileged")

if tab_manual is not None:
    with tab_manual:

        if has_permission("view_user_manual"):

            if has_permission("view_user_manual"):
                def _pdf_download_card(title, subtitle, audience, pdf_path: Path, icon="📄"):
                    """Professional document card for dashboard manuals/briefs."""
                    st.markdown(
                        f"""
                        <div class="manual-doc-card">
                            <div class="manual-doc-icon">{icon}</div>
                            <div class="manual-doc-body">
                                <div class="manual-doc-title">{title}</div>
                                <div class="manual-doc-subtitle">{subtitle}</div>
                                <div class="manual-doc-audience">{audience}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    _safe_pdf_download_button(
                        title=title,
                        pdf_path=pdf_path,
                        key_prefix="manual_pdf_download",
                    )

                st.markdown(
                    """
                    <style>
                    .manual-hero {
                        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                        border: 1px solid #E6E8EF;
                        border-left: 5px solid #660094;
                        border-radius: 18px;
                        padding: 18px 20px;
                        box-shadow: 0 10px 24px rgba(16, 24, 40, 0.06);
                        margin: 0 0 18px 0;
                        font-family: Arial, sans-serif;
                    }
                    .manual-eyebrow {
                        display: block;
                        color: #660094;
                        background: transparent;
                        border: 0;
                        border-radius: 0;
                        padding: 0;
                        font-size: 10px;
                        font-weight: 950;
                        letter-spacing: .14em;
                        text-transform: uppercase;
                        margin: 0 0 6px 0;
                        line-height: 1.2;
                    }
                    .manual-title {
                        color: #23152F;
                        font-size: 18px;
                        font-weight: 950;
                        margin: 0 0 10px 0;
                        line-height: 1.12;
                    }
                    .manual-title-divider {
                        width: 74px;
                        height: 4px;
                        border-radius: 999px;
                        background: linear-gradient(90deg, #660094 0%, #008CAA 100%);
                        margin: 0 0 14px 0;
                    }
                    .manual-lead {
                        color: #475467;
                        font-size: 12px;
                        line-height: 1.5;
                        max-width: 1150px;
                        margin: 0;
                        font-weight: 300;
                    }
                    .manual-access-pill {
                        display: inline-flex;
                        align-items: center;
                        padding: 5px 11px;
                        border-radius: 999px;
                        background: #F4EAF8;
                        border: 1px solid #E7D4F1;
                        color: #660094;
                        font-size: 11px;
                        font-weight: 900;
                        margin: 12px 8px 0 0;
                        line-height: 1.1;
                    }
                    .manual-access-note {
                        color: #667085;
                        font-size: 12px;
                        line-height: 1.5;
                    }
                    .manual-kpi-grid {
                        display: grid;
                        grid-template-columns: repeat(4, minmax(0, 1fr));
                        gap: 7px;
                        margin: 8px 0 11px 0;
                    }
                    .manual-mini-card {
                        display: grid;
                        grid-template-columns: 24px minmax(0, 1fr);
                        column-gap: 7px;
                        align-items: start;
                        background: #FFFFFF;
                        border: 1px solid #ECE5F3;
                        border-radius: 12px;
                        padding: 7px 8px;
                        box-shadow: 0 4px 12px rgba(54, 26, 83, 0.045);
                        min-height: 64px;
                        font-family: Arial, sans-serif;
                    }
                    .manual-mini-icon {
                        width: 22px;
                        height: 22px;
                        border-radius: 8px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: #F8F3FB;
                        color: #660094;
                        font-size: 12px;
                        margin-bottom: 0;
                        grid-row: span 2;
                    }
                    .manual-mini-title {
                        color: #2D0055;
                        font-size: 9.5px;
                        font-weight: 900;
                        margin-bottom: 2px;
                    }
                    .manual-mini-text {
                        color: #64748B;
                        font-size: 9.8px;
                        line-height: 1.25;
                    }
                    @media (max-width: 1050px) {
                        .manual-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                    }
                    @media (max-width: 560px) {
                        .manual-kpi-grid { grid-template-columns: 1fr; }
                        .manual-mini-card { min-height: auto; }
                    }
                    .manual-section-card {
                        background: #FFFFFF;
                        border: 1px solid #ECE5F3;
                        border-radius: 15px;
                        padding: 12px 13px;
                        box-shadow: 0 7px 18px rgba(54, 26, 83, 0.06);
                        margin-bottom: 11px;
                        font-family: Arial, sans-serif;
                    }
                    .manual-section-title {
                        color: #2D0055;
                        font-size: 13.5px;
                        font-weight: 900;
                        margin-bottom: 2px;
                    }
                    .manual-section-note {
                        color: #64748B;
                        font-size: 10.8px;
                        line-height: 1.28;
                        margin-bottom: 7px;
                    }
                    .manual-step {
                        display: grid;
                        grid-template-columns: 24px 1fr;
                        gap: 7px;
                        align-items: start;
                        padding: 6px 0;
                        border-bottom: 1px solid #F1EEF5;
                    }
                    .manual-step:last-child { border-bottom: none; }
                    .manual-step-num {
                        background: #660094;
                        color: white;
                        width: 21px;
                        height: 21px;
                        border-radius: 999px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 9.5px;
                        font-weight: 900;
                    }
                    .manual-step-title {
                        color: #334155;
                        font-size: 10.8px;
                        font-weight: 900;
                        margin-bottom: 1px;
                    }
                    .manual-step-text {
                        color: #64748B;
                        font-size: 10.3px;
                        line-height: 1.25;
                    }
                    .manual-doc-card {
                        display: grid;
                        grid-template-columns: 34px 1fr;
                        gap: 9px;
                        align-items: center;
                        background: #FFFFFF;
                        border: 1px solid #ECE5F3;
                        border-left: 5px solid #660094;
                        border-radius: 13px;
                        padding: 10px;
                        box-shadow: 0 8px 22px rgba(54, 26, 83, 0.07);
                        margin-bottom: 7px;
                        font-family: Arial, sans-serif;
                    }
                    .manual-doc-icon {
                        width: 32px;
                        height: 32px;
                        border-radius: 11px;
                        background: linear-gradient(135deg, #660094, #8A2DB2);
                        color: #FFFFFF;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 15px;
                    }
                    .manual-doc-title {
                        color: #2D0055;
                        font-size: 11.8px;
                        font-weight: 900;
                        margin-bottom: 2px;
                    }
                    .manual-doc-subtitle {
                        color: #475569;
                        font-size: 10.2px;
                        line-height: 1.24;
                        margin-bottom: 4px;
                    }
                    .manual-doc-audience {
                        color: #660094;
                        font-size: 9.4px;
                        font-weight: 800;
                        background: #F8F3FB;
                        border: 1px solid #E8DFF0;
                        display: inline-block;
                        padding: 2px 7px;
                        border-radius: 999px;
                    }
                    .manual-compact-note {
                        margin-top: 4px;
                        color: #64748B;
                        font-size: 10px;
                        line-height: 1.25;
                    }
                    @media (max-width: 760px) {
                        .manual-hero { padding: 13px 14px; }
                        .manual-title { font-size: 18px; }
                        .manual-lead { font-size: 10px; }
                        .manual-access-note { display: block; margin-top: 7px; }
                        .manual-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                        .manual-mini-card { min-height: auto; }
                        .manual-section-card { padding: 11px; }
                    }
                    @media (max-width: 480px) {
                        .manual-kpi-grid { grid-template-columns: 1fr; }
                    }
                    .manual-tip {
                        background: #FFF9DC;
                        border: 1px solid #F2E7A8;
                        border-radius: 12px;
                        padding: 9px 10px;
                        color: #55420A;
                        font-size: 11.5px;
                        line-height: 1.45;
                        font-family: Arial, sans-serif;
                    }

                    /* Remove internal scrolling from the User Manual tab while preserving normal page scroll. */
                    .user-manual-shell,
                    .user-manual-shell * {
                        scrollbar-width: none !important;
                    }

                    .user-manual-shell::-webkit-scrollbar,
                    .user-manual-shell *::-webkit-scrollbar {
                        width: 0 !important;
                        height: 0 !important;
                        display: none !important;
                    }

                    .user-manual-shell,
                    .user-manual-shell div,
                    .user-manual-shell section,
                    .user-manual-shell article,
                    .user-manual-shell [data-testid="stVerticalBlock"],
                    .user-manual-shell [data-testid="stHorizontalBlock"],
                    .user-manual-shell [data-testid="stExpander"],
                    .user-manual-shell [data-testid="stMarkdownContainer"] {
                        overflow: visible !important;
                        max-height: none !important;
                        height: auto !important;
                    }

                    .manual-hero,
                    .manual-kpi-grid,
                    .manual-mini-card,
                    .manual-section-card,
                    .manual-doc-card,
                    .manual-tip {
                        overflow: visible !important;
                        max-height: none !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="user-manual-shell">', unsafe_allow_html=True)

                st.markdown(
                    """
                    <div class="manual-hero">
                        <div class="manual-title">Dashboard User Guide</div>
                        <div class="manual-title-divider"></div>
                        <span class="manual-lead">
                            A quick guide to help you navigate the dashboard, apply filters, interpret charts and maps,
                            explore alert analysis, <br> search the data preview, export filtered results, and use the AI assistant
                            for additional analytical exploration.
                        </span>
                        <div>
                            <span class="manual-access-pill">Privileged access only</span>
                            <span class="manual-access-note">
                                Some advanced features, including the AI assistant and the data summary preview,
                                are available only to authorized EUSEE stakeholders.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="manual-kpi-grid">
                        <div class="manual-mini-card"><div class="manual-mini-icon">🎯</div><div class="manual-mini-title">Purpose</div><div class="manual-mini-text">Understand what the dashboard shows and how each section can support EU SEE monitoring.</div></div>
                        <div class="manual-mini-card"><div class="manual-mini-icon">🧭</div><div class="manual-mini-title">Navigation</div><div class="manual-mini-text">Find your way across the Overview, Negative Alerts Analysis, Visualization Map, Data Preview, and AI Assistant. 
                        Please note that privileged users can access the AI assistant and the data summary preview.</div></div>
                        <div class="manual-mini-card"><div class="manual-mini-icon">🔎</div><div class="manual-mini-title">Analysis</div><div class="manual-mini-text">Use filters, charts, maps, and tables to explore alert trends and country-level patterns.</div></div>
                        <div class="manual-mini-card"><div class="manual-mini-icon">⬇</div><div class="manual-mini-title">Manual</div><div class="manual-mini-text">Download the full user manual for detailed, step-by-step guidance.</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                guide_col, docs_col = st.columns([1.35, 1], gap="large")

                with guide_col:
                    st.markdown(
                        """
                        <div class="manual-section-card">
                            <div class="manual-section-title">Quick-start workflow</div>
                            <div class="manual-section-note">Recommended path for first-time users.</div>
                            <div class="manual-step"><div class="manual-step-num">1</div><div><div class="manual-step-title">The scope</div><div class="manual-step-text">Use the global filters to select the region, country, alert impact, nature of alert, enabling principle, year, and month.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">2</div><div><div class="manual-step-title">Start with the overview</div><div class="manual-step-text">Review the main figures and charts to understand the filtered data.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">3</div><div><div class="manual-step-title">Explore alert patterns</div><div class="manual-step-text">Use the Overview and Negative Alerts Analysis sections to examine distributions, trends, affected civil society actors, restrictive actors, and mechanisms.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">4</div><div><div class="manual-step-title">Use the map for country-level patterns</div><div class="manual-step-text">See where alerts are concentrated and hover over countries for more detail.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">5</div><div><div class="manual-step-title">Review the data, if available</div><div class="manual-step-text">Privileged users can use the data summary preview to search, review, and export filtered records.</div></div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        """
                        <div class="manual-section-card">
                            <div class="manual-section-title">How to interpret dashboard findings</div>
                            <div class="manual-section-note">Keep these principles in mind when using or presenting findings from the dashboard.</div>
                            <div class="manual-step"><div class="manual-step-num">✓</div><div><div class="manual-step-title">Counts are monitoring signals</div><div class="manual-step-text">Higher counts may reflect more incidents, stronger reporting, better monitoring coverage, or a combination of these factors.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">✓</div><div><div class="manual-step-title">Use filters transparently</div><div class="manual-step-text">When sharing charts or tables, mention the selected region, period, alert impact, alert type, and other relevant filters.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">✓</div><div><div class="manual-step-title">Compare different views</div><div class="manual-step-text">Use figures, charts, maps, and available data records together before drawing conclusions.</div></div></div>
                            <div class="manual-step"><div class="manual-step-num">✓</div><div><div class="manual-step-title">Cite the dashboard</div><div class="manual-step-text">When using data, charts, or findings from this dashboard, always cite the EU SEE Dashboard as follows: EU SEE Dashboard. Name of the graph/data visualization (as provided on the Dashboard website). Date of last update/consultation.</div></div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with docs_col:
                    st.markdown(
                        """
                        <div class="manual-section-card">
                            <div class="manual-section-title">Full user manual</div>
                            <div class="manual-section-note">Download the full user manual to understand the dashboard’s goal, indicators, navigation, and how to interpret the data responsibly.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    _pdf_download_card(
                        "Executive Brief",
                        "One-page dashboard overview for senior leadership, donors, and policy reporting.",
                        "Best for: executives and external briefings",
                        EXEC_BRIEF_PATH,
                        icon="📌",
                    )

                    _pdf_download_card(
                        "Full User Manual",
                        "Detailed guide covering navigation, filters, charts, map interpretation, data preview, and exports.",
                        "Best for: analysts and advanced users",
                        USER_MANUAL_PATH,
                        icon="📘",
                    )

                    st.markdown(
                        """
                        <div class="manual-tip">
                            <strong>Recommended reporting note:</strong><br>
                            Dashboard findings should be interpreted as reported monitoring evidence, not as direct prevalence estimates.
                            Always pair quantitative outputs with contextual review and partner validation.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                render_access_locked("User Manual", "guest or higher")


        else:
            render_access_locked("User Manual", "guest or higher")


# ============================================================
# EUSEE LANGFLOW CHATBOT
# LangFlow-only brain: answers + plots + memory + filtered data
# ============================================================

import json
import uuid
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

LANGFLOW_API_URL = st.secrets.get("langflow", {}).get("LANGFLOW_API_URL", "").strip()
LANGFLOW_API_KEY = st.secrets.get("langflow", {}).get("LANGFLOW_API_KEY", "").strip()

# IMPORTANT: update this if your LangFlow Prompt Template component ID is different.
LANGFLOW_PROMPT_COMPONENT_ID = st.secrets.get("langflow", {}).get(
    "LANGFLOW_PROMPT_COMPONENT_ID",
    "Prompt Template-hiUxU"
).strip()

# ---------------- CHAT HISTORY PERSISTENCE ----------------
# Stores each authenticated user's Copilot history on disk so it survives
# Streamlit reruns, browser refreshes, logout/login, and app restarts when the
# project folder or Docker volume is persistent.
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_LIMIT = 100


def _current_chat_user_key() -> str:
    """Return a stable privacy-safe key for the active user chat history."""
    email = str(st.session_state.get("email") or "").lower().strip()

    if email:
        identity = f"user::{email}"
    else:
        # Public/guest users only get a browser-session-level identity.
        # Authenticated users get persistent per-email history.
        st.session_state.setdefault("eusee_guest_chat_key", str(uuid.uuid4()))
        identity = f"guest::{st.session_state.eusee_guest_chat_key}"

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _chat_history_path(user_key: str | None = None) -> Path:
    user_key = user_key or _current_chat_user_key()
    return CHAT_HISTORY_DIR / f"{user_key}.json"


def _normalise_chat_messages(messages) -> list[dict]:
    clean_messages = []

    if not isinstance(messages, list):
        return clean_messages

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "assistant")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "assistant"

        content = str(msg.get("content", "")).strip()
        if not content:
            continue

        clean_messages.append({
            "id": str(msg.get("id") or uuid.uuid4().hex),
            "role": role,
            "content": content,
            "created_at": str(msg.get("created_at") or datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        })

    return clean_messages[-CHAT_HISTORY_LIMIT:]


def load_user_chat_history(force: bool = False) -> list[dict]:
    """Load the current user's saved Copilot history into session state."""
    user_key = _current_chat_user_key()

    if (
        not force
        and st.session_state.get("eusee_chat_history_loaded")
        and st.session_state.get("eusee_chat_user_key") == user_key
    ):
        return st.session_state.get("eusee_chat_messages", [])

    history_file = _chat_history_path(user_key)
    messages = []

    if history_file.exists():
        try:
            payload = json.loads(history_file.read_text(encoding="utf-8"))
            messages = _normalise_chat_messages(payload.get("messages", []))
        except Exception:
            messages = []

    st.session_state.eusee_chat_user_key = user_key
    st.session_state.eusee_chat_messages = messages
    st.session_state.eusee_chat_history_loaded = True
    st.session_state.eusee_chat_session_id = user_key[:32]

    return messages


def save_user_chat_history() -> None:
    """Persist the active user's Copilot history to disk."""
    user_key = st.session_state.get("eusee_chat_user_key") or _current_chat_user_key()
    messages = _normalise_chat_messages(st.session_state.get("eusee_chat_messages", []))
    st.session_state.eusee_chat_messages = messages

    payload = {
        "user_key": user_key,
        "email": str(st.session_state.get("email") or "").lower().strip(),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "message_count": len(messages),
        "messages": messages,
    }

    try:
        history_file = _chat_history_path(user_key)
        tmp_file = history_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_file.replace(history_file)
    except Exception as exc:
        if st.secrets.get("debug", {}).get("show_chat_history_errors", False):
            st.warning(f"Chat history could not be saved: {exc}")


def append_user_chat_message(role: str, content: str) -> None:
    """Append one Copilot message and immediately save the user's history."""
    load_user_chat_history()

    st.session_state.eusee_chat_messages.append({
        "id": uuid.uuid4().hex,
        "role": str(role or "assistant").lower().strip(),
        "content": str(content or "").strip(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })

    st.session_state.eusee_chat_messages = _normalise_chat_messages(
        st.session_state.eusee_chat_messages
    )
    save_user_chat_history()


def clear_user_chat_history() -> None:
    """Clear the current user's saved Copilot history."""
    user_key = st.session_state.get("eusee_chat_user_key") or _current_chat_user_key()
    st.session_state.eusee_chat_messages = []
    st.session_state.eusee_chat_history_loaded = True

    try:
        history_file = _chat_history_path(user_key)
        if history_file.exists():
            history_file.unlink()
    except Exception:
        pass


load_user_chat_history(force=True)


def _clean_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    for col in work.columns:
        if pd.api.types.is_object_dtype(work[col]) or pd.api.types.is_string_dtype(work[col]):
            work[col] = work[col].fillna("").astype(str).str.strip()

    return work


def _find_col(df, possible_names):
    lookup = {str(c).lower().strip(): c for c in df.columns}
    for name in possible_names:
        key = str(name).lower().strip()
        if key in lookup:
            return lookup[key]
    return None


def _counts(series, top_n=10):
    if series is None:
        return {}

    counts = (
        series.astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(top_n)
    )

    return {str(k): int(v) for k, v in counts.items()}


def _direct_top(ranking, label):
    if not ranking:
        return {}
    first_key = next(iter(ranking))
    return {
        label: first_key,
        "count": int(ranking[first_key])
    }


def build_filter_summary(df):
    work = _clean_df(df)

    return json.dumps({
        "filtered_records": int(len(work)),
        "latest_dataset_date": st.session_state.get("latest_dataset_date", "Not available"),
        "basis": "Current active dashboard/sidebar filters"
    }, indent=2, default=str)


def build_lookup_context(df, top_n=15):
    work = _clean_df(df)

    if work.empty:
        return json.dumps({
            "available": False,
            "message": "No records available under current filters."
        })

    country_col = _find_col(work, ["alert-country", "country", "Country"])
    region_col = _find_col(work, ["region", "Region"])
    impact_col = _find_col(work, ["alert-impact", "impact", "Alert impact", "Impact"])
    type_col = _find_col(work, ["alert-type", "alert type", "type", "Type of alert"])
    principle_col = _find_col(work, ["enabling-principle", "Enabling principle", "enabling principle"])
    actor_col = _find_col(work, ["Actor of repression", "actor of repression"])
    mechanism_col = _find_col(work, ["Mechanism of repression", "mechanism of repression"])
    affected_col = _find_col(work, ["Affected actor", "affected actor"])
    subject_col = _find_col(work, ["Subject of repression", "subject of repression"])
    year_col = _find_col(work, ["year", "Year"])
    month_col = _find_col(work, ["month_name", "month", "Month"])

    dimensions = {
        "country": country_col,
        "region": region_col,
        "alert_type": type_col,
        "enabling_principle": principle_col,
        "actor_of_repression": actor_col,
        "mechanism_of_repression": mechanism_col,
        "affected_actor": affected_col,
        "subject_of_repression": subject_col,
        "year": year_col,
        "month": month_col,
    }

    lookup = {
        "available": True,
        "filtered_records": int(len(work)),
        "columns_available": list(work.columns),
        "direct_answers": {},
        "rankings": {},
        "summaries": {},
        "natural_language_index": {}
    }

    # Overall rankings
    for dim_name, col in dimensions.items():
        if col and col in work.columns:
            ranking = _counts(work[col], top_n=top_n)
            lookup["rankings"][f"top_{dim_name}s"] = ranking
            lookup["natural_language_index"][f"top {dim_name}s"] = ranking
            lookup["natural_language_index"][f"distribution of {dim_name}s"] = ranking

            if ranking:
                lookup["direct_answers"][f"highest_{dim_name}"] = _direct_top(ranking, dim_name)

    # Impact-specific summaries
    if impact_col and impact_col in work.columns:
        impact_values = (
            work[impact_col]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )

        lookup["impact_values"] = [str(x) for x in impact_values]
        lookup["rankings"]["alert_impacts"] = _counts(work[impact_col], top_n=top_n)

        for impact_value in impact_values:
            impact_clean = str(impact_value).lower().strip()
            impact_key = impact_clean.replace(" ", "_")

            impact_df = work[
                work[impact_col].astype(str).str.lower().str.strip().eq(impact_clean)
            ]

            lookup["summaries"][f"{impact_key}_alerts"] = {
                "total": int(len(impact_df))
            }
            lookup["rankings"][f"total_{impact_key}_alerts"] = int(len(impact_df))

            for dim_name, col in dimensions.items():
                if not col or col not in impact_df.columns:
                    continue

                ranking = _counts(impact_df[col], top_n=top_n)
                if not ranking:
                    continue

                ranking_key = f"top_{dim_name}s_by_{impact_key}_alerts"
                lookup["rankings"][ranking_key] = ranking
                lookup["summaries"][f"{impact_key}_alerts"][f"top_{dim_name}s"] = ranking

                lookup["direct_answers"][f"{dim_name}_with_highest_{impact_key}_alerts"] = _direct_top(
                    ranking,
                    dim_name
                )

                lookup["natural_language_index"][f"{dim_name} with highest {impact_clean} alerts"] = ranking
                lookup["natural_language_index"][f"highest {impact_clean} alerts by {dim_name}"] = ranking
                lookup["natural_language_index"][f"top {dim_name}s with highest {impact_clean} alerts"] = ranking
                lookup["natural_language_index"][f"top {dim_name}s by {impact_clean} alerts"] = ranking
                lookup["natural_language_index"][f"{impact_clean} alerts by {dim_name}"] = ranking

                if dim_name == "country":
                    lookup["direct_answers"][f"country_with_highest_{impact_key}_alerts"] = _direct_top(ranking, "country")
                    lookup["natural_language_index"][f"country with highest {impact_clean} alerts"] = ranking
                    lookup["natural_language_index"][f"countries with highest {impact_clean} alerts"] = ranking
                    lookup["natural_language_index"][f"top countries with highest {impact_clean} alerts"] = ranking

                if dim_name == "region":
                    lookup["direct_answers"][f"region_with_highest_{impact_key}_alerts"] = _direct_top(ranking, "region")
                    lookup["natural_language_index"][f"region with highest {impact_clean} alerts"] = ranking
                    lookup["natural_language_index"][f"regions with highest {impact_clean} alerts"] = ranking
                    lookup["natural_language_index"][f"top regions with highest {impact_clean} alerts"] = ranking

            # Region-specific impact summaries, e.g. negative alerts in Africa
            if region_col and region_col in impact_df.columns:
                for region_value in impact_df[region_col].dropna().astype(str).str.strip().unique():
                    region_key = region_value.lower().replace(" ", "_").replace("-", "_")
                    region_df = impact_df[
                        impact_df[region_col].astype(str).str.lower().str.strip()
                        == region_value.lower().strip()
                    ]

                    summary_key = f"{impact_key}_alerts_in_{region_key}"
                    lookup["summaries"][summary_key] = {
                        "region": region_value,
                        "impact": impact_value,
                        "total": int(len(region_df)),
                    }

                    if country_col and country_col in region_df.columns:
                        lookup["summaries"][summary_key]["top_countries"] = _counts(region_df[country_col], top_n=top_n)

                    if type_col and type_col in region_df.columns:
                        lookup["summaries"][summary_key]["top_alert_types"] = _counts(region_df[type_col], top_n=top_n)

                    if actor_col and actor_col in region_df.columns:
                        lookup["summaries"][summary_key]["top_actors_of_repression"] = _counts(region_df[actor_col], top_n=top_n)

                    lookup["natural_language_index"][f"summarise {impact_clean} alerts in {region_value.lower()}"] = lookup["summaries"][summary_key]
                    lookup["natural_language_index"][f"summary of {impact_clean} alerts in {region_value.lower()}"] = lookup["summaries"][summary_key]
                    lookup["natural_language_index"][f"{impact_clean} alerts in {region_value.lower()}"] = lookup["summaries"][summary_key]

    return json.dumps(lookup, indent=2, default=str)


def build_dashboard_context(df, top_n=10):
    work = _clean_df(df)

    if work.empty:
        return json.dumps({
            "available": False,
            "message": "No dashboard records available under the current filters."
        })

    context = {
        "available": True,
        "filtered_records": int(len(work)),
        "latest_dataset_date": st.session_state.get("latest_dataset_date", "Not available"),
        "columns_available": list(work.columns),
        "column_summaries": {}
    }

    for col in work.columns:
        series = work[col].dropna()

        if series.empty:
            context["column_summaries"][col] = {
                "type": "empty",
                "non_empty_records": 0
            }
            continue

        if pd.api.types.is_numeric_dtype(series):
            context["column_summaries"][col] = {
                "type": "numeric",
                "non_empty_records": int(series.count()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median())
            }
        else:
            clean_series = (
                series.astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
            )

            context["column_summaries"][col] = {
                "type": "categorical_text",
                "non_empty_records": int(clean_series.count()),
                "unique_values": int(clean_series.nunique()),
                "top_values": _counts(clean_series, top_n=top_n)
            }

    return json.dumps(context, indent=2, default=str)


def extract_langflow_text(response_json):
    try:
        return response_json["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except Exception:
        return json.dumps(response_json, indent=2, default=str)


def ask_langflow(user_question, lookup_context, dashboard_context, filter_summary):
    if not LANGFLOW_API_URL:
        return json.dumps({
            "answer": "LangFlow API URL is not configured.",
            "available_in_context": False,
            "used_current_filters": False,
            "analysis_type": "configuration_error",
            "interpretation_note": "",
            "chart": {},
            "follow_up_suggestions": []
        })

    if not LANGFLOW_API_KEY:
        return json.dumps({
            "answer": "LangFlow API key is not configured.",
            "available_in_context": False,
            "used_current_filters": False,
            "analysis_type": "configuration_error",
            "interpretation_note": "",
            "chart": {},
            "follow_up_suggestions": []
        })

    chat_memory = json.dumps(
        st.session_state.eusee_chat_messages[-12:],
        indent=2,
        default=str
    )

    payload = {
        "input_value": user_question,
        "output_type": "chat",
        "input_type": "chat",
        "tweaks": {
            LANGFLOW_PROMPT_COMPONENT_ID: {
                "lookup_context": lookup_context,
                "dashboard_context": dashboard_context,
                "filter_summary": filter_summary,
                "chat_memory": chat_memory,
                "question": user_question,
            },
            "ChatInput-0gnCu": {
                "session_id": st.session_state.eusee_chat_session_id,
                "context_id": "eusee-dashboard",
                "should_store_message": True,
            },
            "ChatOutput-wP9WA": {
                "session_id": st.session_state.eusee_chat_session_id,
                "context_id": "eusee-dashboard",
                "should_store_message": True,
            },
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": LANGFLOW_API_KEY,
    }

    try:
        response = requests.post(
            LANGFLOW_API_URL,
            json=payload,
            headers=headers,
            timeout=90,
        )

        if response.status_code == 200:
            return extract_langflow_text(response.json())

        return json.dumps({
            "answer": f"Could not reach the EUSEE Copilot service. LangFlow response: {response.status_code} {response.reason}: {response.text[:700]}",
            "available_in_context": False,
            "used_current_filters": False,
            "analysis_type": "langflow_error",
            "interpretation_note": "",
            "chart": {},
            "follow_up_suggestions": []
        })

    except Exception as e:
        return json.dumps({
            "answer": f"Could not reach the EUSEE Copilot service. LangFlow error: {e}",
            "available_in_context": False,
            "used_current_filters": False,
            "analysis_type": "langflow_error",
            "interpretation_note": "",
            "chart": {},
            "follow_up_suggestions": []
        })


EUSEE_WEBSITE_REDIRECT_TEXT = (
    "\n\n---\n"
    "🌐 For a broader overview and additional qualitative insights, "
    "please visit the EUSEE website at https://eusee.org"
)


def _append_eusee_website_redirect(answer: str, result: dict) -> str:
    """Append the EUSEE website redirect only for dashboard-derived chatbot answers.

    The redirect is shown when the LangFlow JSON confirms that the answer is
    available in the supplied dashboard context and uses the active filters.
    This avoids adding the link to greetings, configuration errors, or unrelated
    responses.
    """
    answer = str(answer or "").strip()

    dashboard_related = (
        bool(result.get("available_in_context", False))
        and bool(result.get("used_current_filters", False))
    )

    if not dashboard_related:
        return answer

    if "https://eusee.org" in answer:
        return answer

    return answer + EUSEE_WEBSITE_REDIRECT_TEXT

def render_langflow_output(raw_answer, chart_instance_key=None):
    try:
        result = json.loads(raw_answer)
    except Exception:
        st.markdown(str(raw_answer))
        return

    answer = _append_eusee_website_redirect(result.get("answer", ""), result)
    if answer:
        st.markdown(answer)

    chart = result.get("chart", {})
    if not isinstance(chart, dict) or not chart:
        return

    chart_type = str(chart.get("type", "")).lower().strip()
    title = chart.get("title", "") or "Dashboard chart"
    x_label = chart.get("x_label", "Category") or "Category"
    y_label = chart.get("y_label", "Count") or "Count"
    sort_order = str(chart.get("sort_order", "")).lower().strip()

    legend_value = chart.get("legend", "Series")
    legend_label = "Series" if isinstance(legend_value, bool) else str(legend_value)

    if not chart_type:
        st.warning("Chart could not be rendered: missing chart type.")
        st.json(chart)
        return

    data = chart.get("data", [])
    if not isinstance(data, list) or not data:
        st.warning("Chart could not be rendered: missing chart.data.")
        st.json(chart)
        return

    chart_df = pd.DataFrame(data)

    x_col = chart.get("x")
    y_col = chart.get("y")
    series_col = chart.get("series")

    if not x_col or not y_col:
        st.warning("Chart could not be rendered: missing x/y field names.")
        st.json(chart)
        return

    if x_col not in chart_df.columns or y_col not in chart_df.columns:
        st.warning(
            f"Chart could not be rendered: x/y fields not found in chart.data. "
            f"x={x_col}, y={y_col}, columns={list(chart_df.columns)}"
        )
        st.json(chart)
        return

    chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[y_col])

    if chart_df.empty:
        st.warning("Chart could not be rendered: numeric values are empty after conversion.")
        st.json(chart)
        return

    if sort_order == "descending":
        chart_df = chart_df.sort_values(y_col, ascending=False)
    elif sort_order == "ascending":
        chart_df = chart_df.sort_values(y_col, ascending=True)

    if chart_instance_key is None:
        chart_instance_key = uuid.uuid4().hex

    base_key = f"eusee_ai_chart_{chart_instance_key}_{abs(hash(raw_answer))}"

    try:
        if chart_type == "bar":
            fig = px.bar(
                chart_df,
                x=x_col,
                y=y_col,
                title=title,
                text=y_col,
                labels={x_col: x_label, y_col: y_label}
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_bar")

        elif chart_type in ["horizontal_bar", "hbar"]:
            fig_df = chart_df.sort_values(y_col, ascending=True)
            fig = px.bar(
                fig_df,
                x=y_col,
                y=x_col,
                orientation="h",
                title=title,
                text=y_col,
                labels={x_col: x_label, y_col: y_label}
            )
            fig.update_layout(height=max(430, 42 * len(fig_df)))
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_hbar")

        elif chart_type == "grouped_bar":
            if not series_col or series_col not in chart_df.columns:
                st.warning("Chart could not be rendered: grouped_bar requires a valid series field.")
                st.json(chart)
                return

            fig = px.bar(
                chart_df,
                x=x_col,
                y=y_col,
                color=series_col,
                barmode="group",
                title=title,
                text=y_col,
                labels={
                    x_col: x_label,
                    y_col: y_label,
                    series_col: legend_label
                }
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_grouped_bar")

        elif chart_type in ["stacked_bar", "stacked_100_percent_bar"]:
            if not series_col or series_col not in chart_df.columns:
                st.warning("Chart could not be rendered: stacked charts require a valid series field.")
                st.json(chart)
                return

            if chart_type == "stacked_100_percent_bar":
                total_df = chart_df.groupby(x_col)[y_col].transform("sum")
                chart_df["_percent"] = chart_df[y_col] / total_df.replace(0, pd.NA) * 100
                chart_df = chart_df.dropna(subset=["_percent"])
                plot_y = "_percent"
                y_axis_title = "Percentage"
            else:
                plot_y = y_col
                y_axis_title = y_label

            fig = px.bar(
                chart_df,
                x=x_col,
                y=plot_y,
                color=series_col,
                title=title,
                text=plot_y,
                labels={
                    x_col: x_label,
                    plot_y: y_axis_title,
                    series_col: legend_label
                }
            )
            fig.update_layout(barmode="stack", height=430)
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_{chart_type}")

        elif chart_type == "pie":
            fig = px.pie(
                chart_df,
                names=x_col,
                values=y_col,
                title=title
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_pie")

        elif chart_type == "donut":
            fig = px.pie(
                chart_df,
                names=x_col,
                values=y_col,
                title=title,
                hole=0.45
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_donut")

        elif chart_type == "line":
            fig = px.line(
                chart_df,
                x=x_col,
                y=y_col,
                title=title,
                markers=True,
                labels={x_col: x_label, y_col: y_label}
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_line")

        elif chart_type == "area":
            fig = px.area(
                chart_df,
                x=x_col,
                y=y_col,
                title=title,
                labels={x_col: x_label, y_col: y_label}
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_area")

        elif chart_type == "scatter":
            fig = px.scatter(
                chart_df,
                x=x_col,
                y=y_col,
                title=title,
                labels={x_col: x_label, y_col: y_label}
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_scatter")

        elif chart_type == "table":
            st.dataframe(chart_df, use_container_width=True)

        else:
            st.warning(f"Unsupported chart type returned by LangFlow: {chart_type}")
            st.json(chart)

    except Exception as e:
        st.warning(f"Chart could not be rendered: {e}")
        st.json(chart)

# ============================================================
# SAFE EUSEE AI COPILOT POPOVER
# Opens/closes without rerunning and does not interfere with dashboard tabs/charts.
# Only submitting a Copilot question triggers the normal Streamlit rerun.
# ============================================================


def inject_eusee_ai_popover_css():
    """Scoped styling for the native Streamlit Copilot popover.

    Important fix:
    Streamlit/BaseWeb uses the same `div[data-baseweb="popover"]` portal for
    both `st.popover()` and select/multiselect dropdown menus. Therefore, broad
    rules such as `div[data-baseweb="popover"] > div { width: 430px; ... }`
    also resize sidebar multiselect dropdowns and make them appear outside the
    sidebar/window.

    This version scopes the drawer styling to popovers that contain Streamlit
    content blocks and separately keeps select/multiselect dropdown menus small.
    """
    st.markdown(
        """
        <style>
        /* Keep footer space so the Copilot control never covers the fixed footer. */
        .main .block-container {
            padding-bottom: 7rem !important;
        }

        /* Right-side Copilot launcher only. */
        div[data-testid="stPopover"] {
            position: fixed !important;
            right: 22px !important;
            bottom: 82px !important;
            z-index: 999998 !important;
            width: auto !important;
            max-width: calc(100vw - 44px) !important;
        }

        div[data-testid="stPopover"] > button {
            border-radius: 999px !important;
            min-height: 52px !important;
            padding: 0 20px !important;
            background: linear-gradient(135deg,#660094 0%,#008CAA 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255,255,255,.30) !important;
            box-shadow: 0 16px 36px rgba(102,0,148,.28) !important;
            font-weight: 950 !important;
        }

        div[data-testid="stPopover"] > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 18px 42px rgba(102,0,148,.34) !important;
            color: #FFFFFF !important;
        }

        /* BaseWeb popovers are shared by st.popover and select/multiselect menus. */
        div[data-baseweb="popover"] {
            z-index: 999999 !important;
        }

        /* -------- SELECT / MULTISELECT DROPDOWN FIX --------
           Keep dropdown lists compact. Do not force fixed/left positioning.
           BaseWeb will keep the menu under the input. */
        div[data-baseweb="popover"]:has([role="listbox"]) > div {
            width: auto !important;
            min-width: 0 !important;
            max-width: 240px !important;
            max-height: 280px !important;
            overflow: visible !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        div[data-baseweb="popover"] [role="listbox"] {
            width: 220px !important;
            min-width: 220px !important;
            max-width: 220px !important;
            max-height: 260px !important;
            padding: 6px !important;
            margin-top: 4px !important;
            background: #FFFFFF !important;
            border: 1px solid #E6E8EF !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 28px rgba(16,24,40,.18) !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        div[data-baseweb="popover"] [role="option"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            padding: 8px 10px !important;
            border-radius: 9px !important;
            font-size: 11.5px !important;
            font-weight: 750 !important;
            line-height: 1.25 !important;
            color: #344054 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }

        div[data-baseweb="popover"] [role="option"]:hover {
            background: rgba(102,0,148,.07) !important;
            color: #23152F !important;
        }

        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background: #F4EAF8 !important;
            color: #660094 !important;
            font-weight: 900 !important;
        }

        /* -------- COPILOT DRAWER ONLY --------
           Scope drawer styling to Streamlit popover content, but exclude listbox
           popovers used by select/multiselect widgets. */
        div[data-baseweb="popover"]:has([data-testid="stVerticalBlock"]):not(:has([role="listbox"])) > div {
            width: min(430px, calc(100vw - 32px)) !important;
            max-height: min(78vh, 720px) !important;
            overflow-y: auto !important;
            background: #FFFFFF !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-baseweb="popover"]:has([data-testid="stVerticalBlock"]):not(:has([role="listbox"])) > div > div,
        div[data-baseweb="popover"]:has([data-testid="stVerticalBlock"]):not(:has([role="listbox"])) [data-testid="stVerticalBlock"],
        div[data-baseweb="popover"]:has([data-testid="stVerticalBlock"]):not(:has([role="listbox"])) [data-testid="stElementContainer"] {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        div[data-baseweb="popover"]:has([data-testid="stVerticalBlock"]):not(:has([role="listbox"])) > div > div {
            padding: 0 !important;
            margin: 0 !important;
        }

        @media (max-width: 700px) {
            div[data-testid="stPopover"] {
                right: 14px !important;
                bottom: 72px !important;
            }

            div[data-testid="stPopover"] > button {
                min-height: 48px !important;
                padding: 0 15px !important;
                font-size: 12px !important;
            }

            div[data-baseweb="popover"] [role="listbox"] {
                width: min(220px, calc(100vw - 32px)) !important;
                min-width: min(220px, calc(100vw - 32px)) !important;
                max-width: min(220px, calc(100vw - 32px)) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_eusee_ai_copilot_body():
    st.markdown(
        """
        <div style="
            position:sticky;
            top:0;
            z-index:2;
            background:#FFFFFF;
            border-bottom:1px solid #EEF0F4;
            padding:14px 14px 12px 14px;
            margin:0;
            font-family:Arial,sans-serif;
        ">
            <div style="font-size:9px;font-weight:950;color:#660094;letter-spacing:.14em;text-transform:uppercase;">
                Dashboard assistant
            </div>
            <div style="font-size:16px;font-weight:950;color:#23152F;margin-top:4px;">
                🤖 EUSEE AI Copilot
            </div>
            <div style="font-size:11px;color:#667085;line-height:1.35;margin-top:5px;">
                Ask about the current filtered dashboard data. Answers and charts use the active dashboard context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_permission("use_ai_copilot"):
        st.info("AI Copilot is not enabled for your access level.")
        return

    load_user_chat_history()

    history_count = len(st.session_state.get("eusee_chat_messages", []))
    st.caption(f"Chat history: {history_count} saved message(s) for this user.")

    if st.button("Clear Chat Memory", use_container_width=True, key="eusee_ai_clear_chat_memory"):
        clear_user_chat_history()
        st.rerun()

    for i, msg in enumerate(st.session_state.eusee_chat_messages[-12:]):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        with st.chat_message(role):
            if role == "assistant":
                chart_key = f"chat_{i}_{msg.get('id', uuid.uuid4().hex)}"
                render_langflow_output(content, chart_instance_key=chart_key)
            else:
                st.markdown(content)

    with st.form("eusee_ai_popover_form", clear_on_submit=True):
        user_question = st.text_area(
            "Ask about the current dashboard data",
            placeholder="Example: summarise the negative alerts in Africa",
            height=90,
            label_visibility="collapsed",
            key="eusee_ai_popover_question",
        )

        submitted = st.form_submit_button("Ask Copilot", use_container_width=True)

    if submitted and user_question.strip():
        user_question = user_question.strip()

        append_user_chat_message("user", user_question)

        active_df = st.session_state.get("eusee_active_filtered_df", None)
        if active_df is None:
            active_df = filtered_global.copy()

        lookup_context = build_lookup_context(active_df, top_n=15)
        dashboard_context = build_dashboard_context(active_df, top_n=10)
        filter_summary = build_filter_summary(active_df)

        # Keep debug collapsed and available only during testing.
        with st.expander("DEBUG Copilot context", expanded=False):
            st.caption("If the answer is missing here, the Python context builder is the problem.")
            try:
                st.json(json.loads(lookup_context))
            except Exception:
                st.write(lookup_context[:3000])

        with st.spinner("Asking LangFlow..."):
            answer = ask_langflow(
                user_question=user_question,
                lookup_context=lookup_context,
                dashboard_context=dashboard_context,
                filter_summary=filter_summary,
            )

        append_user_chat_message("assistant", answer)

        st.rerun()


def render_eusee_ai_copilot_popover():
    """Render EUSEE Copilot only when enabled by Admin permissions."""

    if not has_permission("use_ai_copilot"):
        return

    inject_eusee_ai_popover_css()

    try:
        with st.popover("💬 EUSEE Copilot", use_container_width=False):
            _render_eusee_ai_copilot_body()
    except Exception:
        with st.expander("💬 EUSEE Copilot", expanded=False):
            _render_eusee_ai_copilot_body()
render_eusee_ai_copilot_popover()

# ---------------- FOOTER ----------------
# Feedback is rendered as a single collapsed responsive floating overlay near the dashboard header.


# OpenAI test UI is now integrated inside the AI Copilot drawer.

# --- Load image and convert to base64 ---
footer_image_path = "assets/footer_logo.png"
with open(footer_image_path, "rb") as f:
    data = f.read()
b64 = base64.b64encode(data).decode()

# --- Render fixed footer without reserving an extra Streamlit iframe spacer ---
st.markdown(f"""
<style>
.eusee-fixed-footer {{
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 4px 0 3px 0;
    margin: 0 !important;
    background: #FFFFFF;
    border-top: 1px solid rgba(230,232,239,.85);
    box-shadow: 0 -6px 18px rgba(16,24,40,.045);
    z-index: 9999;
}}
.eusee-fixed-footer img {{
    display: block;
    width: min(700px, 82vw);
    max-width: 82vw;
    height: auto;
    margin: 0 auto !important;
    padding: 0 !important;
}}
.eusee-fixed-footer-copy {{
    margin: 1px 0 0 0 !important;
    padding: 0 !important;
    color: #667085;
    font-family: var(--eusee-font, "Inter", "Segoe UI", Arial, sans-serif);
    font-size: 10px;
    line-height: 1.1;
    font-weight: 600;
}}
</style>
<div class="eusee-fixed-footer">
    <img src="data:image/png;base64,{b64}" alt="EU SEE footer logo">
    <div class="eusee-fixed-footer-copy">© 2025 EU SEE Dashboard. All rights reserved.</div>
</div>
""", unsafe_allow_html=True)

