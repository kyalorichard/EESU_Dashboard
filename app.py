import base64
import hashlib
import html
import json
import logging
import math
import os
import re
import tempfile
import textwrap
import uuid
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import paramiko
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import streamlit as st
import streamlit.components.v1 as components
from streamlit.elements.lib.policies import CachedWidgetWarning


warnings.filterwarnings(
    "ignore",
    category=CachedWidgetWarning
)

from auth import auth_ui, is_privileged, is_authenticated, init_session, restore_session, logout


st.set_page_config(page_title="EUSEE Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# EU SEE OFFICIAL TYPOGRAPHY
# Primary font: Anek Devanagari
# Fallback font: Arial
# ============================================================
EUSEE_FONT_FAMILY = '"Anek Devanagari", Arial, "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif'
PLOTLY_FONT_FAMILY = "Anek Devanagari, Arial, sans-serif"
CHART_FONT = PLOTLY_FONT_FAMILY


def inject_eusee_official_typography() -> None:
    """Apply EU SEE typography while preserving Streamlit and BaseWeb icons."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link
            href="https://fonts.googleapis.com/css2?family=Anek+Devanagari:wdth,wght@75..125,100..800&display=swap"
            rel="stylesheet"
        >

        <style>
        :root {
            --eusee-font:
                "Anek Devanagari",
                Arial,
                "Segoe UI Emoji",
                "Apple Color Emoji",
                "Noto Color Emoji",
                sans-serif;
        }

        /* Page-level typography */
        html,
        body,
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        .main,
        .main .block-container {
            font-family: var(--eusee-font);
        }

        /* Streamlit text and controls */
        h1, h2, h3, h4, h5, h6,
        p,
        label,
        small,
        strong,
        em,
        input,
        textarea,
        select,
        option,
        button,
        table,
        th,
        td,
        [role="option"],
        [role="combobox"],
        [role="gridcell"],
        [role="columnheader"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stMetric"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stTabs"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stAlert"],
        [data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] {
            font-family: var(--eusee-font) !important;
        }

        /* BaseWeb text-bearing controls */
        [data-baseweb="select"] input,
        [data-baseweb="select"] div:not([data-baseweb="icon"]),
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="radio"] label,
        [data-baseweb="checkbox"] label,
        [data-baseweb="tab"] {
            font-family: var(--eusee-font) !important;
        }

        /* Plotly text */
        .js-plotly-plot text,
        .plotly text,
        .svg-container text {
            font-family: var(--eusee-font) !important;
        }

        /*
        Preserve Streamlit's Material Symbols ligatures. Without this
        override, names such as arrow_drop_down can appear as visible text.
        */
        .material-icons,
        .material-icons-round,
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-symbols-sharp,
        [data-testid="stIconMaterial"],
        [data-testid="stIconMaterial"] *,
        span[class*="material-symbols"] {
            font-family:
                "Material Symbols Rounded",
                "Material Symbols Outlined",
                "Material Symbols Sharp",
                "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            font-feature-settings: "liga" !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
        }

        /* BaseWeb SVG/icon containers must not receive the dashboard text font. */
        [data-baseweb="icon"],
        [data-baseweb="icon"] * {
            font-family: initial !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def configure_eusee_plotly_typography() -> None:
    """Register a default Plotly template using the EU SEE font stack."""
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family=PLOTLY_FONT_FAMILY, color="#344054"),
            title=dict(font=dict(family=PLOTLY_FONT_FAMILY)),
            legend=dict(font=dict(family=PLOTLY_FONT_FAMILY)),
            hoverlabel=dict(font=dict(family=PLOTLY_FONT_FAMILY)),
            xaxis=dict(
                title_font=dict(family=PLOTLY_FONT_FAMILY),
                tickfont=dict(family=PLOTLY_FONT_FAMILY),
            ),
            yaxis=dict(
                title_font=dict(family=PLOTLY_FONT_FAMILY),
                tickfont=dict(family=PLOTLY_FONT_FAMILY),
            ),
        )
    )
    pio.templates["eusee_official"] = template
    pio.templates.default = "eusee_official"


def apply_eusee_plotly_font(fig):
    """Apply EU SEE typography to a Plotly figure and return it."""
    if fig is None:
        return fig

    fig.update_layout(
        font=dict(family=PLOTLY_FONT_FAMILY),
        title_font=dict(family=PLOTLY_FONT_FAMILY),
        legend_font=dict(family=PLOTLY_FONT_FAMILY),
        hoverlabel=dict(font_family=PLOTLY_FONT_FAMILY),
    )
    fig.update_xaxes(
        title_font=dict(family=PLOTLY_FONT_FAMILY),
        tickfont=dict(family=PLOTLY_FONT_FAMILY),
    )
    fig.update_yaxes(
        title_font=dict(family=PLOTLY_FONT_FAMILY),
        tickfont=dict(family=PLOTLY_FONT_FAMILY),
    )

    for annotation in fig.layout.annotations or []:
        annotation.font.family = PLOTLY_FONT_FAMILY

    return fig


inject_eusee_official_typography()
configure_eusee_plotly_typography()


init_session()

with st.spinner("Restoring secure session..."):
    restored = restore_session()

# Do not continue rendering dashboard as Guest while restore is pending
if not st.session_state.get("restored", False):
    st.stop()

# ------------------------------------------------------------------
# HEADER CLEAN-UP AND COLLAPSED-SIDEBAR SIGNPOST
# ------------------------------------------------------------------
st.markdown("""
<style>
/* Hide Streamlit menu, footer, deployment and source-code actions. */
#MainMenu,
footer,
[data-testid="stDecoration"],
[data-testid="stDeployButton"],
[data-testid="stHeaderActionElements"],
[data-testid="stToolbarActions"] {
    display: none !important;
    visibility: hidden !important;
}

/* Keep the header and native sidebar arrows available. */
header[data-testid="stHeader"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    background: rgba(247,248,251,0.95) !important;
}

button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
}

/* Hide GitHub and source links that may still be injected elsewhere. */
a[href*="github.com"],
a[href*="source"] {
    display: none !important;
}

.block-container {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Add a separate label beside Streamlit's native arrows. The label appears only
# when the sidebar is collapsed and is removed immediately when it opens.
components.html(
    r"""
    <script>
    (function () {
        const doc = window.parent.document;
        const win = window.parent;
        const LABEL_ID = "eusee-collapsed-sidebar-label";
        const STYLE_ID = "eusee-collapsed-sidebar-label-style";

        function ensureStyle() {
            if (doc.getElementById(STYLE_ID)) return;

            const style = doc.createElement("style");
            style.id = STYLE_ID;
            style.textContent = `
                #${LABEL_ID} {
                    position: fixed;
                    z-index: 1000002;
                    display: inline-flex;
                    align-items: center;
                    min-height: 30px;
                    padding: 0 11px;
                    border: 1px solid #E7D4F1;
                    border-radius: 999px;
                    background: #FFFFFF;
                    box-shadow: 0 3px 10px rgba(16,24,40,.08);
                    color: #660094;
                    font-family: "Anek Devanagari", Arial, sans-serif;
                    font-size: 12px;
                    line-height: 1;
                    font-weight: 800;
                    white-space: nowrap;
                    pointer-events: auto;
                    cursor: pointer;
                    user-select: none;
                    transition: border-color .15s ease, background .15s ease, box-shadow .15s ease, transform .15s ease;
                }
                #${LABEL_ID}:hover {
                    border-color: #660094;
                    background: #FBF7FD;
                    box-shadow: 0 5px 14px rgba(102,0,148,.14);
                    transform: translateY(-1px);
                }

                #${LABEL_ID}:focus-visible {
                    outline: 3px solid rgba(102,0,148,.18);
                    outline-offset: 2px;
                }

                @media (max-width: 700px) {
                    #${LABEL_ID} {
                        padding: 0 8px;
                        font-size: 11px;
                    }
                }
            `;
            doc.head.appendChild(style);
        }

        function findToggle() {
            const selectors = [
                '[data-testid="stSidebarCollapsedControl"] button',
                '[data-testid="stSidebarCollapsedControl"]',
                'button[data-testid="collapsedControl"]',
                '[data-testid="stSidebarCollapseButton"] button',
                '[data-testid="stSidebarCollapseButton"]',
                'button[aria-label*="sidebar" i]',
                'button[title*="sidebar" i]'
            ];

            for (const selector of selectors) {
                const nodes = Array.from(doc.querySelectorAll(selector));
                const visible = nodes.find((node) => {
                    const rect = node.getBoundingClientRect();
                    const css = win.getComputedStyle(node);
                    return rect.width > 0 && rect.height > 0 &&
                           css.display !== "none" &&
                           css.visibility !== "hidden";
                });
                if (visible) return visible;
            }
            return null;
        }

        function sidebarIsCollapsed() {
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) return true;

            const rect = sidebar.getBoundingClientRect();
            const css = win.getComputedStyle(sidebar);
            const ariaHidden = sidebar.getAttribute("aria-hidden") === "true";

            return ariaHidden ||
                   css.display === "none" ||
                   css.visibility === "hidden" ||
                   rect.width < 40 ||
                   rect.right <= 5 ||
                   rect.left < -(rect.width / 2);
        }

        function removeLabel() {
            const label = doc.getElementById(LABEL_ID);
            if (label) label.remove();
        }

        function updateLabel() {
            ensureStyle();

            const toggle = findToggle();
            if (!toggle || !sidebarIsCollapsed()) {
                removeLabel();
                return;
            }

            let label = doc.getElementById(LABEL_ID);
            if (!label) {
                label = doc.createElement("div");
                label.id = LABEL_ID;
                label.textContent = "Login & Global Filters";
                label.setAttribute("role", "button");
                label.setAttribute("tabindex", "0");
                label.setAttribute("aria-label", "Open Login and Global Filters sidebar");
                label.setAttribute("title", "Open Login and Global Filters");

                const openSidebar = function () {
                    const activeToggle = findToggle();
                    if (activeToggle && sidebarIsCollapsed()) {
                        activeToggle.click();
                    }
                };

                label.addEventListener("click", openSidebar);
                label.addEventListener("keydown", function (event) {
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openSidebar();
                    }
                });

                doc.body.appendChild(label);
            }

            const rect = toggle.getBoundingClientRect();
            const labelHeight = label.getBoundingClientRect().height || 30;
            label.style.left = `${Math.max(44, rect.right + 8)}px`;
            label.style.top = `${Math.max(5, rect.top + (rect.height - labelHeight) / 2)}px`;
        }

        updateLabel();

        const observer = new MutationObserver(updateLabel);
        observer.observe(doc.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["class", "style", "aria-expanded", "aria-hidden"]
        });

        win.addEventListener("resize", updateLabel);
        win.setInterval(updateLabel, 500);
    })();
    </script>
    """,
    height=0,
    width=0,
)


# Optional admin page integration. Firebase/Auth handles login;
# authz.py resolves guest/viewer/privileged/admin roles.
# ------------------------------------------------------------------
# AUTHORIZATION AND ADMIN MODULE LOADING
# ------------------------------------------------------------------
# Authorization is security-critical. Never continue with permissive
# defaults when authz.py is unavailable or fails during import.
try:
    from authz import (
        is_admin as admin_is_admin,
        get_current_role,
        get_current_email,
        has_permission,
        apply_data_scope,
    )
except Exception as authz_import_error:
    logging.exception(
        "Critical authorization module import failure",
        exc_info=authz_import_error,
    )
    st.error(
        "Authorization services are currently unavailable. "
        "For security, the dashboard has stopped instead of granting fallback access."
    )
    st.stop()

# The admin interface is optional. Failure to import it must not replace or
# weaken the valid authorization functions imported above.
try:
    # Preferred name used by the optimized admin script.
    from admin import render_admin_page, render_admin_sidebar_navigation
except Exception as admin_import_error:
    logging.warning(
        "Could not import admin.py; attempting admin_page.py fallback: %s",
        admin_import_error,
    )
    try:
        # Backward-compatible fallback if the file is still named admin_page.py.
        from admin_page import render_admin_page, render_admin_sidebar_navigation
    except Exception as admin_page_import_error:
        logging.exception(
            "Admin interface modules are unavailable",
            exc_info=admin_page_import_error,
        )

        def render_admin_page(data=None):
            st.error(
                "The admin page is unavailable. Confirm that admin.py or "
                "admin_page.py is deployed and imports successfully."
            )

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
        font-family: "Anek Devanagari", Arial, sans-serif !important; font-size: 11px !important; font-weight: 800 !important;
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
    .classic-filter-status .status-row { display:flex; justify-content:space-between; align-items:center; padding: 3px 0; font-family: "Anek Devanagari", Arial, sans-serif; font-size: 10.5px; color: var(--eusee-muted); }
    .classic-filter-status .status-value { color: var(--eusee-purple); font-weight: 900; }
    div[data-testid="stExpander"] { border: 1px solid var(--eusee-border) !important; border-radius: 16px !important; box-shadow: 0 8px 22px rgba(16,24,40,.06) !important; background: #FFFFFF !important; overflow: hidden !important; }
    div[data-testid="stExpander"] summary { font-family: "Anek Devanagari", Arial, sans-serif !important; font-size: 13px !important; font-weight: 900 !important; color: #23152F !important; background: linear-gradient(90deg, #FFFFFF 0%, #FAF7FC 100%) !important; border-bottom: 1px solid #EEF0F4 !important; padding: 10px 14px !important; }
    .data-preview-toolbar { display:flex; justify-content:space-between; align-items:center; gap:12px; background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); border: 1px solid #EEF0F4; border-radius: 14px; padding: 11px 13px; margin: 4px 0 12px 0; font-family: "Anek Devanagari", Arial, sans-serif; }
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
    .executive-table-status { display:flex; justify-content:space-between; align-items:center; gap:10px; background:#F9FAFB; border:1px solid #EEF0F4; border-radius:13px; padding:9px 11px; margin:9px 0 10px 0; font-size:11px; color:#344054; font-family:"Anek Devanagari", Arial, sans-serif; }
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

    /* Mobile-first dashboard consistency and overflow protection. */
    img, svg, canvas, video { max-width: 100% !important; height: auto; }
    [data-testid="stMetric"], [data-testid="stAlert"],
    div[data-testid="stVerticalBlockBorderWrapper"] { min-width: 0 !important; }
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container { width: 100% !important; }
    .modebar-container { max-width: 100% !important; }
    .modebar-group { flex-wrap: wrap !important; }
    table { max-width: 100%; }

    @media (max-width: 768px) {
        header[data-testid="stHeader"] { height: 44px !important; min-height: 44px !important; }
        .main .block-container {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: .65rem !important;
            padding-right: .65rem !important;
            padding-bottom: 6.5rem !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: .65rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 0 !important;
        }
        [data-testid="stPlotlyChart"] { min-height: 280px !important; }
        .js-plotly-plot .plotly .modebar {
            right: 4px !important;
            top: 4px !important;
            transform: scale(.86);
            transform-origin: top right;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            flex: 0 0 auto !important;
            min-width: max-content !important;
            white-space: nowrap !important;
        }
        .data-preview-toolbar, .executive-table-header,
        .executive-table-status, .last-updated-badge {
            flex-direction: column !important;
            align-items: flex-start !important;
        }
        .data-preview-pill-row { justify-content: flex-start !important; }
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
        <div class="classic-filter-title">🌍 Global Filters</div>        
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

def render_professional_data_preview(
    df,
    title="Search and export EU SEE alerts",
    key="summary_data_preview",
    remove_vertical_scroll=True,
):
    """Render a clean, searchable table with clickable report links."""

    if df is None or df.empty:
        st.info("No records are available for the current filter selection.")
        return

    DATA_PREVIEW_STANDARD_HEIGHT = 460

    # Use Streamlit's automatic table height when vertical scrolling is disabled.
    # Horizontal scrolling remains available when the table has many columns.
    dataframe_height = (
        "auto" if remove_vertical_scroll else DATA_PREVIEW_STANDARD_HEIGHT
    )

    display_df = df.copy()

    # ---------------------------------------------------------
    # FORMAT DATE COLUMNS
    # ---------------------------------------------------------
    for date_col in ["Date of submission", "creation_date"]:
        if date_col in display_df.columns:
            display_df[date_col] = pd.to_datetime(
                display_df[date_col],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    # ---------------------------------------------------------
    # CREATE CLICKABLE REPORT LINK
    # ---------------------------------------------------------
    # Detect the source column even when its case or spacing differs,
    # for example: Permalink, permalink, " Permalink ", or Report URL.
    permalink_col = next(
        (
            col for col in display_df.columns
            if str(col).strip().lower() in {
                "permalink",
                "permalink url",
                "report url",
                "report link",
            }
        ),
        None,
    )

    if permalink_col is not None:

        def clean_permalink(value):
            """Return a usable absolute URL or an empty string."""
            if pd.isna(value):
                return ""

            url = str(value).strip()

            if not url or url.lower() in {"nan", "none", "null"}:
                return ""

            # Convert relative EUSEE paths to full URLs.
            if url.startswith("/"):
                return f"https://eusee.hivos.org{url}"

            # Add HTTPS when a complete hostname is stored without a scheme.
            if not url.lower().startswith(("http://", "https://")):
                url = f"https://{url}"

            return url

        display_df["Open alert"] = (
            display_df[permalink_col]
            .apply(clean_permalink)
            .astype(str)
        )
    else:
        # Keep the function operational and make the missing source explicit.
        st.warning(
            "The report link column was not created because no Permalink "
            "column was found in the supplied dataframe."
        )

    # ---------------------------------------------------------
    # IDENTIFY ALERT-IMPACT COLUMN
    # ---------------------------------------------------------
    impact_col = None

    for candidate in [
        "alert-impact",
        "Impact of alert",
        "Alert impact",
    ]:
        if candidate in display_df.columns:
            impact_col = candidate
            break

    with st.expander(
        f"📋 {title}",
        expanded=False,
    ):
        st.markdown(
            """
            <style>
            .eusee-data-preview-note {
                background:
                    linear-gradient(
                        135deg,
                        #FFFFFF 0%,
                        #F8FAFC 100%
                    );
                border: 1px solid #E6E8EF;
                border-left: 4px solid #660094;
                border-radius: 14px;
                padding: 10px 12px;
                margin: 2px 0 12px 0;
                color: #667085;
                font-family:
                    var(--eusee-font);
                font-size: 11.5px;
                line-height: 1.42;
                font-weight: 550;
                box-shadow:
                    0 6px 16px
                    rgba(16,24,40,.045);
            }

            .eusee-data-preview-note strong {
                color: #23152F;
                font-weight: 900;
            }

            div[data-testid="stDataFrame"] {
                width: 100% !important;
                max-width: 100% !important;
                border:
                    1px solid #E6E8EF !important;
                border-radius: 16px !important;
                overflow: hidden !important;
                box-shadow:
                    0 10px 24px
                    rgba(16,24,40,.06) !important;
                background: #FFFFFF !important;
                font-family:
                    var(--eusee-font) !important;
            }

            div[data-testid="stDataFrame"] > div {
                width: 100% !important;
                max-width: 100% !important;
                overflow: hidden !important;
            }

            div[data-testid="stDataFrame"]
            div[role="grid"] {
                width: 100% !important;
                max-width: 100% !important;
                overflow: auto !important;
            }

            div[data-testid="stDataFrame"]
            [data-testid="stTable"] {
                overflow: visible !important;
            }

            div[data-testid="stDataFrame"]
            [role="columnheader"],
            div[data-testid="stDataFrame"]
            [role="columnheader"] * {
                background: #F4EAF8 !important;
                color: #23152F !important;
                font-family:
                    var(--eusee-font) !important;
                font-size: 11.5px !important;
                font-weight: 850 !important;
                border-bottom:
                    1px solid #E7D4F1 !important;
                line-height: 1.25 !important;
            }

            div[data-testid="stDataFrame"]
            [role="gridcell"],
            div[data-testid="stDataFrame"]
            [role="gridcell"] * {
                color: #344054 !important;
                font-family:
                    var(--eusee-font) !important;
                font-size: 11.5px !important;
                line-height: 1.35 !important;
                font-weight: 500 !important;
            }

            div[data-testid="stDataFrame"]
            div[role="grid"]::-webkit-scrollbar {
                width: 10px !important;
                height: 10px !important;
            }

            div[data-testid="stDataFrame"]
            div[role="grid"]::-webkit-scrollbar-thumb {
                background: #D6BBE5 !important;
                border-radius: 999px !important;
                border:
                    2px solid #FFFFFF !important;
            }

            div[data-testid="stDataFrame"]
            div[role="grid"]::-webkit-scrollbar-track {
                background: #F8FAFC !important;
                border-radius: 999px !important;
            }
            </style>

            """,
            unsafe_allow_html=True,
        )

        # -----------------------------------------------------
        # SEARCH
        # -----------------------------------------------------
        search_text = st.text_input(
            "Search table",
            value="",
            placeholder=(
                'Use Boolean search, e.g. Kenya AND negative, '
                'Uganda OR Kenya, '
                '"civil society" NOT positive.'
            ),
            key=f"{key}_search",
        )

        table_df = display_df.copy()
        active_filter_rows = len(table_df)

        search_text_clean = " ".join(
            str(search_text or "").split()
        ).strip()

        if search_text_clean:
            table_df = table_df.loc[
                _build_fast_table_search_mask(
                    table_df,
                    search_text_clean,
                )
            ].copy()

        table_view = table_df.copy()

        st.caption(
            f"Displaying {len(table_view):,} matching records "
            f"from {active_filter_rows:,} active-filter records. (Global filters selected do not affect the records displayed in the table below.)"
        )

        # -----------------------------------------------------
        # ALERT-IMPACT STYLING
        # -----------------------------------------------------
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

        if (
            impact_col
            and impact_col in table_view.columns
        ):
            try:
                table_to_render = table_view.style.map(
                    style_alert_impact,
                    subset=[impact_col],
                )
            except AttributeError:
                table_to_render = (
                    table_view.style.applymap(
                        style_alert_impact,
                        subset=[impact_col],
                    )
                )

        # -----------------------------------------------------
        # COLUMN ORDER
        # -----------------------------------------------------
        # Put the clickable link first so it is visible without horizontal scrolling.
        hidden_source_columns = {
            col for col in table_view.columns
            if str(col).strip().lower() in {
                "permalink",
                "permalink url",
                "report url",
                "report link",
            }
        }

        visible_columns = [
            col
            for col in table_view.columns
            if col != "Open alert" and col not in hidden_source_columns
        ]

        # Keep the clickable report link as the final visible column.
        if "Open alert" in table_view.columns:
            visible_columns.append("Open alert")

        column_config = {}

        if "Open alert" in table_view.columns:
            column_config["Open alert"] = (
                st.column_config.LinkColumn(
                    label="Full report",
                    help="Open the complete report in a new browser tab.",
                    display_text="Open alert ↗",
                    width="medium",
                )
            )

        # -----------------------------------------------------
        # DATAFRAME
        # -----------------------------------------------------
        st.dataframe(
            table_to_render,
            use_container_width=True,
            hide_index=True,
            height=dataframe_height,
            key=key,
            column_order=visible_columns,
            column_config=column_config,
        )

        # -----------------------------------------------------
        # EXCEL DOWNLOAD
        # -----------------------------------------------------
        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl",
        ) as writer:
            # Preserve the original Permalink in the download.
            export_df = table_df.drop(
                columns=["Open alert"],
                errors="ignore",
            )

            export_df.to_excel(
                writer,
                index=False,
                sheet_name="Filtered Data",
            )

        excel_buffer.seek(0)

        if has_permission("download_data"):
            st.download_button(
                label=(
                    "⬇️ Download filtered table as Excel"
                ),
                data=excel_buffer,
                file_name=f"{key}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                use_container_width=True,
                key=f"{key}_download_excel",
            )
        else:
            st.caption(
                "Excel download is disabled "
                "for your access level."
            )

        st.markdown(
            """
            <div class="data-preview-footnote">
                Interpretation note: this table reflects the
                active filters. Negative alerts are highlighted
                in red, positive alerts in green, and
                context-to-watch records in amber.
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        font-family: "Anek Devanagari", Arial, sans-serif;
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

  
    """, unsafe_allow_html=True)

    auth_ui()

    st.stop()


# ---------------- SHARED CONTINENT-TO-REGION HELPER ----------------
def continent_to_region(continent):
    """Map country metadata continents to the dashboard's regional groupings."""
    if continent == "Africa":
        return "Africa"
    elif continent in ["Asia", "Oceania"]:
        return "Asia and the Pacific"
    elif continent in ["Europe", "Middle East", "North Africa"]:
        return "Middle East and North Africa"
    elif continent in [
        "Americas",
        "North America",
        "South America",
        "Caribbean",
    ]:
        return "Americas and the Caribbean"
    else:
        return "Unknown"


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
    for col in [
        "alert-country",
        "alert-impact",
        "alert-type",
        "Actor of repression",
    ]:
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

    # --- Step 4: Basic country and alert cleaning ---
    df["alert-country"] = (
        df["alert-country"]
        .astype(str)
        .str.strip()
    )
    df = df[
            df["alert-country"].notna()
            & ~df["alert-country"].str.lower().isin([
                "",
                "nan",
                "none",
                "null",
                "na",
                "n/a",
            ])
        ].copy()

    df = df[
        df["alert-country"].str.lower() != "jose"
    ].copy()

    df["alert-impact"] = (
        df["alert-impact"]
        .astype(str)
        .str.strip()
    )

    df = df[
            df["alert-impact"].notna()
            & ~df["alert-impact"].str.lower().isin([
                "",
                "none",
                "nan",
                "null",
                "n/a",
                "na",
            ])
        ].copy()

    # Normalize country names before ISO mapping.
    COUNTRY_FIXES = {
        "Guinea-Bissau": "Guinea Bissau",
        "Democratic Republic of Congo":
            "Democratic Republic of the Congo",
        "Democratic Republic of Congo 2":
            "Democratic Republic of the Congo",
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

    # --- Step 5: Clean alert type and remove event rows ---
    df["alert-type"] = (
        df["alert-type"]
        .astype(str)
        .str.strip()
    )

    df = df[
            df["alert-type"].notna()
            & ~df["alert-type"].str.lower().isin([
                "",
                "none",
                "nan",
                "null",
                "n/a",
                "na",
            ])
        ].copy()

        # Remove Event records
    df = df[
        df["alert-type"].str.lower() != "event"
        ].copy()

    # Clean Actor of repression.
    df["Actor of repression"] = (
        df["Actor of repression"]
        .astype(str)
        .str.strip()
        .replace({
            "VNSAs": "Violent non-state actors",
        })
    )

    # --- Step 6: Map ISO codes and continent ---
    df["iso_alpha3"] = df["alert-country"].apply(
        lambda x: country_meta.get(
            x,
            {},
        ).get(
            "iso_alpha3",
            None,
        )
    )

    df["continent"] = df["alert-country"].apply(
        lambda x: country_meta.get(
            x,
            {},
        ).get(
            "continent",
            "Unknown",
        )
    )

    # --- Step 7: Map continent to dashboard region ---
    df["region"] = df["continent"].apply(
        continent_to_region
    )

    # --- Step 8: Warn about missing ISO codes ---
    missing_countries = (
        df.loc[
            df["iso_alpha3"].isna(),
            "alert-country",
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[
            lambda s:
                (s.str.lower() != "none")
                & (s.str.lower() != "nan")
                & (s != "")
        ]
        .unique()
    )

    if len(missing_countries) > 0:
        st.warning(
            "Countries missing ISO codes after metadata "
            "normalization: "
            + ", ".join(sorted(missing_countries))
        )

    # --- Step 9: Process dates ---
    if "creation_date" in df.columns:
        df["creation_date"] = pd.to_datetime(
            df["creation_date"],
            errors="coerce",
        )

        df["year"] = df["creation_date"].dt.year
        df["month_name"] = (
            df["creation_date"]
            .dt.strftime("%B")
        )

        latest_dataset_date = (
            df["creation_date"]
            .dropna()
            .max()
        )

        if pd.notna(latest_dataset_date):
            latest_dataset_date_display = (
                latest_dataset_date.strftime(
                    "%d %B %Y"
                )
            )

            latest_dataset_date_iso = (
                latest_dataset_date.strftime(
                    "%Y-%m-%d"
                )
            )
        else:
            latest_dataset_date_display = (
                "Not available"
            )
            latest_dataset_date_iso = ""

        st.session_state[
            "latest_dataset_date"
        ] = latest_dataset_date_display

        st.session_state[
            "latest_dataset_date_iso"
        ] = latest_dataset_date_iso

        st.session_state[
            "latest_dataset_date_source"
        ] = "Based on latest loaded dataset"

        df.attrs[
            "latest_dataset_date"
        ] = latest_dataset_date_display

        df.attrs[
            "latest_dataset_date_iso"
        ] = latest_dataset_date_iso

    else:
        st.session_state[
            "latest_dataset_date"
        ] = "Not available"

        st.session_state[
            "latest_dataset_date_iso"
        ] = ""

        st.session_state[
            "latest_dataset_date_source"
        ] = (
            "creation_date column not found "
            "in the loaded dataset"
        )

        st.warning(
            "No 'creation_date' column found in dataset."
        )

    # --- Step 10: Update alert-impact based on alert-type ---
    if (
        "alert-type" in df.columns
        and "alert-impact" in df.columns
    ):
        context_mask = (
            df["alert-type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("context to watch")
        )

        df.loc[
            context_mask,
            "alert-impact",
        ] = "Context to watch"

    # --- Step 11: Clean duplicated Event Summary headings ---
    if "summary" in df.columns:
        df["summary"] = (
            df["summary"]
            .fillna("")
            .astype(str)
            .str.replace(
                r"^\s*Event\s*Summary\s*",
                "",
                regex=True,
                flags=re.IGNORECASE,
            )
            .str.strip()
        )

    if "Permalink" in df.columns:
        df["Permalink"] = (
            df["Permalink"]
            .astype(str)
            .str.replace(
                "https://events-eusee.hivos.org/event/",
                "https://eusee.hivos.org/alerts/",
                regex=False,
            )
        )

    return df

# --- Load data safely ---
# IMPORTANT: keep this dataframe BEFORE any sidebar/global UI filters are applied.
# The AI Assistant reads this same cleaned dashboard dataset.
_data_loaded = load_data()

if isinstance(_data_loaded, pd.DataFrame):
    data = apply_data_scope(_data_loaded)
else:
    data = pd.DataFrame()

# Store the authoritative AI source dataframe immediately after loading.
# Do not use `filtered_global` or any sidebar-filtered dataframe here.
if isinstance(data, pd.DataFrame):
    st.session_state["eusee_full_dataset_df"] = data.copy()
else:
    st.session_state["eusee_full_dataset_df"] = pd.DataFrame()

#### --------prepare enabling principles to be ordered-------------------------------------------
ENABLING_PRINCIPLE_ORDER = [          
    "6. Access to a secure digital environment",
    "5. Supportive public culture and discourses on civil society",
    "4. Open and responsive State",
    "3. Accessible and sustainable resources",
    "2. Supportive legal and regulatory framework",
    "1. Respect and protection of fundamental freedoms"
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
        font-family: "Anek Devanagari", Arial, sans-serif;
    }


    .sidebar-access-shell {
        margin: 12px 0 10px 0;
        padding: 12px 12px 11px 12px;
        border-radius: 16px;
        background: linear-gradient(135deg, #FFFFFF 0%, #FCF7FF 100%);
        border: 1px solid rgba(102,0,148,.16);
        box-shadow: 0 10px 24px rgba(16,24,40,.065);
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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

    .sidebar-latest-update {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin: 2px 0 4px 0;
        padding: 9px 10px;
        border: 1px solid #E4E7EC;
        border-radius: 14px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        box-shadow: 0 5px 14px rgba(16,24,40,.045);
        font-family: "Anek Devanagari", Arial, sans-serif;
    }

    .sidebar-latest-update-left {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
    }

    .sidebar-latest-update-icon {
        width: 27px;
        height: 27px;
        min-width: 27px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        background: #F4EAF8;
        color: #660094;
        font-size: 15px;
        font-weight: 900;
    }

    .sidebar-latest-update-label {
        font-size: 9.5px;
        line-height: 1.15;
        color: #667085;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .045em;
    }

    .sidebar-latest-update-date {
        margin-top: 2px;
        color: #23152F;
        font-size: 11px;
        line-height: 1.2;
        font-weight: 900;
        white-space: nowrap;
    }

    .sidebar-latest-update-badge {
        flex: 0 0 auto;
        padding: 3px 7px;
        border-radius: 999px;
        background: #ECFDF3;
        border: 1px solid #ABEFC6;
        color: #067647;
        font-size: 8.5px;
        line-height: 1;
        font-weight: 900;
        letter-spacing: .06em;
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
        font-family: "Anek Devanagari", Arial, sans-serif !important;
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
    """Render the User Privilege Center with button-style workspace tabs."""

    signed_in = is_authenticated()
    is_admin_user = bool(signed_in and admin_is_admin())

    display_name = (
        st.session_state.get("name", "User")
        if signed_in
        else "Guest access"
    )

    # Initialise the current workspace.
    st.session_state.setdefault(
        "eusee_sidebar_workspace",
        "Dashboard",
    )

    # Non-admin users must always remain in the dashboard workspace.
    if not is_admin_user:
        st.session_state["eusee_sidebar_workspace"] = "Dashboard"

    # Styling limited to the sidebar privilege centre.
    st.sidebar.markdown(
        """
        <style>
        /* =========================================================
           EU SEE USER PRIVILEGE CENTER
        ========================================================= */

        section[data-testid="stSidebar"]
        div[data-testid="stVerticalBlock"]:has(.eusee-privilege-marker) {
            gap: 0.35rem;
        }

        .eusee-privilege-marker {
            display: none;
        }

        section[data-testid="stSidebar"] h3 {
            font-size: 12px !important;
            font-weight: 900 !important;
            color: #23152F !important;
            margin: 0 0 0.15rem 0 !important;
        }

        section[data-testid="stSidebar"] p {
            font-size: 10px !important;
            font-weight: 700 !important;
            color: #23152F !important;
            line-height: 1.4 !important;
            margin-bottom: 0 !important;
        }

        section[data-testid="stSidebar"]
        [data-testid="stCaptionContainer"] {
            font-size: 10px !important;
            font-family: "Anek Devanagari", Arial, sans-serif;
            line-height: 1.4 !important;
            color: #667085 !important;
            margin-bottom: 0.15rem !important;
        }

        /* Remove excessive spacing between workspace columns. */
        section[data-testid="stSidebar"]
        div[data-testid="stHorizontalBlock"]:has(.eusee-workspace-marker) {
            gap: 0.35rem !important;
        }

        .eusee-workspace-marker {
            display: none;
        }

        /* Workspace tab/button styling. */
        section[data-testid="stSidebar"]
        div[data-testid="stHorizontalBlock"]:has(.eusee-workspace-marker)
        div[data-testid="stButton"] button {
            min-height: 38px !important;
            border-radius: 9px !important;
            padding: 0.35rem 0.45rem !important;
            font-family: "Anek Devanagari", Arial, sans-serif !important;
            font-size: 11px !important;
            font-weight: 850 !important;
            line-height: 1.1 !important;
            box-shadow: none !important;
            transition:
                background-color 0.18s ease,
                border-color 0.18s ease,
                color 0.18s ease,
                transform 0.18s ease !important;
        }

        /* Inactive workspace button. */
        section[data-testid="stSidebar"]
        div[data-testid="stHorizontalBlock"]:has(.eusee-workspace-marker)
        div[data-testid="stButton"] button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 1px solid #D9DCE3 !important;
            color: #667085 !important;
        }

        section[data-testid="stSidebar"]
        div[data-testid="stHorizontalBlock"]:has(.eusee-workspace-marker)
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: #F9FAFB !important;
            border-color: #8C5A9F !important;
            color: #5E2A70 !important;
            transform: translateY(-1px);
        }

        /* Active workspace button. */
        section[data-testid="stSidebar"]
        div[data-testid="stHorizontalBlock"]:has(.eusee-workspace-marker)
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(
                135deg,
                #F9FAFB 0%,
                #F9FAFB 20%
            ) !important;
            border: 1px solid #23152F !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 10px rgba(35, 21, 47, 0.18) !important;
        }

        section[data-testid="stSidebar"]
        div[data-testid="stHorizontalBlock"]:has(.eusee-workspace-marker)
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(
                135deg,
                #F9FAFB 0%,
                #F9FAFB 20%
            ) !important;
            border-color: #301B40 !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        /* Sign-in and logout buttons. */
        section[data-testid="stSidebar"]
        div[data-testid="stExpander"]
        div[data-testid="stButton"] button {
            border-radius: 9px !important;
            font-family: "Anek Devanagari", Arial, sans-serif !important;
            font-size: 11px !important;
            font-weight: 850 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Marker used to isolate privilege-centre styling.
    st.sidebar.markdown(
        '<span class="eusee-privilege-marker"></span>',
        unsafe_allow_html=True,
    )

    with st.sidebar.expander(
        "🔐 Sign in / Register",
        expanded=True,
    ):
        st.markdown(f"**{display_name}**")

        st.caption(
            "Welcome."
            if signed_in
            else (
                "Sign in or register to access advanced features and "
                "analyses available to EU SEE partners."
            )
        )

        # ---------------------------------------------------------
        # ADMIN WORKSPACE BUTTON TABS
        # ---------------------------------------------------------
        if is_admin_user:
            current_workspace = st.session_state.get(
                "eusee_sidebar_workspace",
                "Dashboard",
            )

            workspace_col_1, workspace_col_2 = st.columns(
                2,
                gap="small",
            )

            with workspace_col_1:
                # Marker allows CSS to target only this button row.
                st.markdown(
                    '<span class="eusee-workspace-marker"></span>',
                    unsafe_allow_html=True,
                )

                dashboard_clicked = st.button(
                    "📊 Dashboard",
                    use_container_width=True,
                    type=(
                        "primary"
                        if current_workspace == "Dashboard"
                        else "secondary"
                    ),
                    key="eusee_dashboard_workspace_btn",
                )

            with workspace_col_2:
                st.markdown(
                    '<span class="eusee-workspace-marker"></span>',
                    unsafe_allow_html=True,
                )

                admin_clicked = st.button(
                    "⚙️ Admin",
                    use_container_width=True,
                    type=(
                        "primary"
                        if current_workspace == "Admin"
                        else "secondary"
                    ),
                    key="eusee_admin_workspace_btn",
                )

            if dashboard_clicked and current_workspace != "Dashboard":
                st.session_state["eusee_sidebar_workspace"] = "Dashboard"
                st.rerun()

            if admin_clicked and current_workspace != "Admin":
                st.session_state["eusee_sidebar_workspace"] = "Admin"
                st.rerun()

        st.divider()

        # ---------------------------------------------------------
        # AUTHENTICATION CONTROLS
        # ---------------------------------------------------------
        if signed_in:
            if st.button(
                "Logout",
                use_container_width=True,
                key="privilege_center_logout_btn",
            ):
                # Reset workspace before logging the user out.
                st.session_state["eusee_sidebar_workspace"] = "Dashboard"
                logout()

        else:
            if st.button(
                "🔐 Sign in / Register",
                use_container_width=True,
                key="privilege_center_signin_btn",
            ):
                st.session_state["auth_view"] = True
                st.rerun()
                
render_sidebar_access_settings_profile()

render_classic_filter_header()
inject_professional_sidebar_filter_css()

# Sidebar compact/responsive override removed to restore the previous sidebar layout.

regions_labels = [
    "Africa",
    "Middle East and North Africa",
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
        "Nature of Alert",
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

# ---------------- LATEST DATASET UPDATE TAG ----------------
# The date is calculated from the latest creation_date in the authoritative
# dataset during load_data(), then displayed here as the final sidebar item.
latest_update = st.session_state.get("latest_dataset_date", "Not available")

st.sidebar.markdown(
    f"""
    <div class="sidebar-latest-update" role="status" aria-label="Latest dataset update">
        <div class="sidebar-latest-update-left">
            <div class="sidebar-latest-update-icon">↻</div>
            <div>
                <div class="sidebar-latest-update-label">Latest update</div>
                <div class="sidebar-latest-update-date">{str(latest_update)}</div>
            </div>
        </div>
        <div class="sidebar-latest-update-badge">DATA</div>
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

# ---------------- FILTER DATA ----------------

def _filter_is_unrestricted(session_key):
    """
    Returns True when the user has not actively restricted this filter.

    safe_multiselect() uses an empty widget selection to represent
    'all available values'.
    """
    widget_key = f"{session_key}_widget"
    widget_selection = st.session_state.get(widget_key, [])

    return (
        widget_selection is None
        or len(widget_selection) == 0
        or "Select all" in widget_selection
    )


# Start with ALL records included.
filter_mask = pd.Series(True, index=data.index)


# ---------------- REGION ----------------
if not _filter_is_unrestricted("selected_regions"):
    filter_mask &= data["region"].isin(selected_regions)

# ---------------- COUNTRY ----------------
if not _filter_is_unrestricted("selected_countries"):
    filter_mask &= data["alert-country"].isin(selected_countries)

# ---------------- ALERT TYPE ----------------
if not _filter_is_unrestricted("selected_alert_types"):
    filter_mask &= data["alert-type"].isin(selected_alert_types)

# ---------------- ENABLING PRINCIPLE ----------------
if not _filter_is_unrestricted("selected_enabling_principle"):
    filter_mask &= data["enabling-principle"].apply(
        lambda x: contains_any(
            x,
            selected_enabling_principle
        )
    )

# ---------------- ALERT IMPACT ----------------
if not _filter_is_unrestricted("selected_alert_impacts"):
    filter_mask &= data["alert-impact"].isin(selected_alert_impacts)

# ---------------- MONTH ----------------
if not _filter_is_unrestricted("selected_months"):
    filter_mask &= data["month_name"].isin(selected_months)

# ---------------- YEAR ----------------
if not _filter_is_unrestricted("selected_years"):
    filter_mask &= data["year"].isin(selected_years)

# ---------------- FINAL FILTERED DATA ----------------
filtered_global = data.loc[filter_mask].copy()



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
    <br>Explore and analyse EU SEE data! This dashboard reflects the documentation of changes to the enabling environment for civil society, as reported by our network members in 80+ countries. We recommend complementing the data here with the primary source research available on the EU SEE website. Please consider: 
<br>•	Due to differences in partner reporting practices by country, direct country comparisons are not recommended when using this data.  
<br>•	Some alerts fall under more than one enabling environment principle. 

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
    font-family: "Anek Devanagari", Arial, sans-serif;
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
    font-family: "Anek Devanagari", Arial, sans-serif;
    color: #333333;
    margin-top: 0rem !important;
    margin-bottom: 3px !important;
    padding-bottom: 0px !important;
    max-width: 1100px;
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

# ---------------- CFR ANALYSIS MODULE ----------------

# ------------------------------------------------------------
# CFR CONFIGURATION
# ------------------------------------------------------------
CFR_SOURCE_FILENAME = "CFR_Export_2026_07_22.csv"

CFR_SCORE_MIN = 1.0
CFR_SCORE_MAX = 5.0

# Full source-column name -> compact dashboard name.
CFR_PRINCIPLES = {
    "Respect and protection of fundamental freedoms": "P1",
    "Supportive legal and regulatory framework": "P2",
    "Accessible and sustainable resources": "P3",
    "Open and responsive State": "P4",
    "Supportive public culture and discourses on civil society": "P5",
    "Access to a secure digital environment": "P6",
}

CFR_PRINCIPLE_NAMES = {
    "P1": "Respect and protection of fundamental freedoms",
    "P2": "Supportive legal and regulatory framework",
    "P3": "Accessible and sustainable resources",
    "P4": "Open and responsive State",
    "P5": "Supportive public culture and discourses on civil society",
    "P6": "Access to a secure digital environment",
}

# Keep these colours consistent in the graph, badges and tables.
CFR_PRINCIPLE_COLOURS = {
    "P1": "#5A0A78",
    "P2": "#FFB900",
    "P3": "#642814",
    "P4": "#008CAA",
    "P5": "#FF8C00",
    "P6": "#B40000",
}

CFR_PRINCIPLE_LIGHT_COLOURS = {
    "P1": "#5A0A78",
    "P2": "#FFB900",
    "P3": "#642814",
    "P4": "#008CAA",
    "P5": "#FF8C00",
    "P6": "#B40000",
}

CFR_TEXT = "#660094"
CFR_MUTED = "#E7EAF0"
CFR_GRID = "#E7EAF0"
CFR_PURPLE = "#660094"

CFR_REGION_ORDER = [
    "Africa",
    "Americas and the Caribbean",
    "Asia and the Pacific",
    "Middle East and North Africa",
    "Unknown",

]

# Use the same country normalisation applied to the alert dataset.
CFR_COUNTRY_FIXES = {
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


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
def _find_cfr_source():
    """Resolve the CFR CSV from an environment variable or exports directory."""
    configured = os.getenv("EUSEE_CFR_CSV")

    if configured:
        configured_path = Path(configured)
        if configured_path.exists() and configured_path.is_file():
            return configured_path

    default_path = EXPORT_DIR / CFR_SOURCE_FILENAME
    return default_path if default_path.exists() else None


def _load_country_metadata():
    """
    Load the same countries_metadata.json file used by load_data().

    Expected structure:
    {
        "Kenya": {
            "iso_alpha3": "KEN",
            "continent": "Africa"
        }
    }
    """
    metadata_path = EXPORT_DIR / "countries_metadata.json"

    if not metadata_path.exists():
        return {}, f"Country metadata file not found: {metadata_path}"

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        if not isinstance(metadata, dict):
            return {}, "countries_metadata.json does not contain a JSON object."

        return metadata, ""

    except Exception as exc:
        return {}, f"Could not load countries_metadata.json: {exc}"


@st.cache_data(show_spinner=False)
def load_cfr_data(
    source_path: str,
    source_mtime=None,
    metadata_mtime=None,
):
    """
    Load, validate and enrich CFR data.

    The modification-time arguments are intentionally included in the
    function signature so Streamlit invalidates the cache when either
    input file changes.
    """
    del source_mtime, metadata_mtime

    cfr = pd.read_csv(source_path)

    required_columns = [
        "Country",
        *CFR_PRINCIPLES.keys(),
        "Last Modified",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in cfr.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing CFR columns: " + ", ".join(missing_columns)
        )

    cfr = cfr.copy()

    # Country cleaning and normalisation.
    cfr["Country"] = (
        cfr["Country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(CFR_COUNTRY_FIXES)
    )

    cfr = cfr[
        cfr["Country"].ne("")
        & cfr["Country"].str.lower().ne("nan")
    ].copy()

    # Convert all six principle scores to numeric values.
    for source_column in CFR_PRINCIPLES:
        cfr[source_column] = pd.to_numeric(
            cfr[source_column],
            errors="coerce",
        )

    # Parse the modification date and derive the CFR reporting year.
    # The current export does not contain a separate Year field, so the
    # reporting year is taken from Last Modified.
    cfr["Last Modified"] = pd.to_datetime(
        cfr["Last Modified"],
        errors="coerce",
    )
    cfr["CFR Year"] = cfr["Last Modified"].dt.year.astype("Int64")

    # Permalink remains optional in the new simplified table.
    if "Permalink" not in cfr.columns:
        cfr["Permalink"] = ""
    else:
        cfr["Permalink"] = (
            cfr["Permalink"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Load the same country metadata used by the alerts dataset.
    country_metadata, metadata_error = _load_country_metadata()

    cfr["continent"] = cfr["Country"].apply(
        lambda country: country_metadata.get(
            country,
            {},
        ).get(
            "continent",
            "Unknown",
        )
    )

    # Explicitly use the existing continent_to_region() function.
    cfr["region"] = cfr["continent"].apply(
        continent_to_region
    )

    # Compact principle aliases used by the presentation layer.
    cfr = cfr.rename(columns=CFR_PRINCIPLES)

    principle_columns = list(CFR_PRINCIPLES.values())

    cfr["Overall CFR"] = cfr[principle_columns].mean(
        axis=1,
        skipna=True,
    )

    # Retain historical CFR rounds. If the same country has multiple export
    # rows within one reporting year, keep only the most recently modified
    # row for that country-year. This preserves first- and second-round CFRs
    # when they belong to different years.
    cfr = (
        cfr.sort_values(
            ["Country", "CFR Year", "Last Modified"],
            ascending=[True, True, True],
            na_position="first",
        )
        .drop_duplicates(
            subset=["Country", "CFR Year"],
            keep="last",
        )
        .sort_values(
            ["Country", "CFR Year"],
            ascending=[True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    cfr.attrs["metadata_error"] = metadata_error

    return cfr

# ------------------------------------------------------------
# CFR CSS
# ------------------------------------------------------------
def _inject_cfr_dashboard_css():
    st.markdown(
        """
        <style>
        .cfr-page {
            font-family:
                "Anek Devanagari",
                Arial,
                sans-serif;
            color: #101B57;
        }

        .cfr-page-title {
            margin: 1px 0 0 0;
            color: #101B57;
            font-size: clamp(28px, 3vw, 40px);
            line-height: 1.05;
            font-weight: 950;
            letter-spacing: -.02em;
        }

        .cfr-page-subtitle {
            margin: 8px 0 10px 0;
            color: black;
            font-size: 13px;
            line-height: 1.45;
            font-weight: 550;
        }

        .cfr-kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 18px;
            margin: 20px 0 18px 0;
        }

        .cfr-kpi-card {
            min-height: 118px;
            display: grid;
            grid-template-columns: 58px minmax(0, 1fr);
            align-items: center;
            column-gap: 16px;
            padding: 18px 20px;
            box-sizing: border-box;
            background: #FFFFFF;
            border: 1px solid #E1E5ED;
            border-radius: 14px;
            box-shadow: 0 5px 16px rgba(16, 24, 40, .035);
        }

        .cfr-kpi-content {
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start;
            text-align: left;
        }

        .cfr-kpi-icon {
            width: 56px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background:
                linear-gradient(
                    135deg,
                    #F7F2FF 0%,
                    #EDF7FF 100%
                );
            border: 1px solid #D9DDF8;
            color: #5F24F5;
            font-size: 28px;
            line-height: 1;
        }

        .cfr-kpi-title {
            color: #101B57;
            font-size: 13px;
            line-height: 1.15;
            font-weight: 900;
            white-space: normal;
        }

        .cfr-kpi-value {
            margin-top: 7px;
            color: #008CAA;
            font-size: 29px;
            line-height: 1;
            font-weight: 950;
            font-variant-numeric: tabular-nums;
        }

        .cfr-kpi-note {
            margin-top: 5px;
            color: #52628C;
            font-size: 11px;
            line-height: 1.2;
            font-weight: 550;
        }

        .cfr-panel {
            height: 100%;
            box-sizing: border-box;
            padding: 14px 16px 12px 16px;
            background: #FFFFFF;
            border: 1px solid #E1E5ED;
            border-radius: 14px;
            box-shadow: 0 5px 16px rgba(16, 24, 40, .035);
        }

        .cfr-panel-title {
            color: #101B57;
            font-size: 17px;
            line-height: 1.15;
            font-weight: 950;
        }

        .cfr-panel-note {
            margin-top: 3px;
            margin-bottom: 10px;
            color: #52628C;
            font-size: 11px;
            line-height: 1.3;
            font-weight: 550;
        }

        /*
        The chart panel uses Streamlit's real bordered container so the
        heading, selector and Plotly chart remain inside one DOM panel.
        The hidden marker limits these styles to the CFR chart panel.
        */
        .cfr-chart-panel-marker,
        .cfr-region-panel-marker {
            display: none;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
        .cfr-region-panel-marker
    ) {
        min-height: 430px !important;
        height: 430px !important;
        box-sizing: border-box;
        padding: 14px 18px 14px 18px !important;
        background: #FFFFFF !important;
        border: 1px solid #E1E5ED !important;
        border-radius: 14px !important;
        box-shadow: 0 5px 16px rgba(16, 24, 40, .035) !important;
        overflow: hidden !important;
    }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-region-panel-marker
        ) > div {
            gap: 0 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-region-panel-marker
        ) div[data-testid="stSelectbox"] label {
            display: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-region-panel-marker
        ) div[data-baseweb="select"] > div {
            min-height: 34px !important;
            height: 34px !important;
            border: 1px solid #D8DDE8 !important;
            border-radius: 7px !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(16, 24, 40, .04) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-region-panel-marker
        ) div[data-baseweb="select"] span {
            color: #344054 !important;
            font-size: 11px !important;
            font-weight: 750 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) {
            height: 100%;
            box-sizing: border-box;
            padding: 16px 18px 10px 18px !important;
            background: #FFFFFF !important;
            border: 1px solid #E1E5ED !important;
            border-radius: 14px !important;
            box-shadow: 0 5px 16px rgba(16, 24, 40, .035) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) > div {
            gap: 0 !important;
        }

        .cfr-chart-panel-heading {
            padding-top: 1px;
            padding-bottom: 4px;
        }

        /* Keep both upper CFR panels the same height. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) {
            min-height: 470px;
            height: 470px;
            overflow: hidden;
        }

        .cfr-top-panel-height {
            min-height: 470px;
            height: 470px;
            box-sizing: border-box;
            overflow: hidden;
        }

        .cfr-chart-panel-selector-label {
            margin-bottom: 4px;
            color: #101B57;
            font-size: 10px;
            line-height: 1.1;
            font-weight: 900;
            text-align: left;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-testid="stSelectbox"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-testid="stSelectbox"] label {
            display: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-baseweb="select"] > div {
            min-height: 38px !important;
            border: 1px solid #D8DDE8 !important;
            border-radius: 9px !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(16, 24, 40, .04) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-baseweb="select"] span {
            color: #344054 !important;
            font-size: 11px !important;
            font-weight: 750 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-testid="stPlotlyChart"] {
            margin-top: 2px !important;
        }

        /* =====================================================
           REFERENCE-MATCHED CFR UPPER PANELS
           ===================================================== */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) {
            min-height: 430px !important;
            height: auto !important;
            padding: 14px 18px 14px 18px !important;
            overflow: visible !important;
        }

        .cfr-top-panel-height {
            min-height: 430px !important;
            height: auto !important;
            padding: 14px 18px 14px 18px !important;
            overflow: visible !important;
        }

        .cfr-panel-title {
            font-size: 16px;
            line-height: 1.12;
            letter-spacing: -0.01em;
        }

        .cfr-panel-note {
            margin-top: 4px;
            margin-bottom: 10px;
            font-size: 10.5px;
            line-height: 1.25;
        }

        .cfr-chart-panel-heading {
            padding: 0;
        }

        .cfr-chart-panel-selector-label {
            margin: 0 0 3px 0;
            font-size: 9.5px;
            line-height: 1;
            text-align: left;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-baseweb="select"] > div {
            min-height: 34px !important;
            height: 34px !important;
            border-radius: 7px !important;
        }

        /* Left principle-score matrix. */
        .cfr-principle-matrix {
            width: 100%;
            margin-top: 3px;
            color: #101B57;
            font-family: "Anek Devanagari", Arial, sans-serif;
        }

        .cfr-principle-grid {
            display: grid;
            grid-template-columns:
                minmax(205px, 1.65fr)
                minmax(260px, 2.35fr)
                70px;
            align-items: center;
            column-gap: 12px;
        }

        .cfr-principle-header {
            min-height: 43px;
            padding-bottom: 4px;
            border-bottom: 0;
            font-size: 10px;
            font-weight: 900;
        }

        .cfr-principle-header-left {
            align-self: end;
            padding-bottom: 4px;
        }

        .cfr-scale-header {
            display: grid;
            grid-template-rows: 17px 20px;
            align-items: end;
        }

        .cfr-scale-ends {
            display: flex;
            justify-content: space-between;
            padding: 0 3px;
            font-size: 9.8px;
            font-weight: 900;
        }

        .cfr-scale-ticks {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            text-align: center;
            font-size: 10px;
            font-weight: 900;
        }

        .cfr-score-header {
            align-self: end;
            padding-bottom: 4px;
            text-align: center;
            white-space: nowrap;
        }

        .cfr-principle-row {
            min-height: 45px;
        }

        .cfr-principle-label {
            display: grid;
            grid-template-columns: 36px minmax(0, 1fr);
            align-items: center;
            column-gap: 10px;
            min-width: 0;
        }

        .cfr-principle-badge {
            width: 35px;
            height: 25px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            color: #FFFFFF;
            font-size: 13px;
            line-height: 1;
            font-weight: 950;
            box-shadow: inset 0 -1px 0 rgba(0,0,0,.08);
        }

        .cfr-principle-name {
            min-width: 0;
            color: #101B57;
            font-size: 10.8px;
            line-height: 1.14;
            font-weight: 850;
            white-space: normal;
        }

        .cfr-score-track-wrap {
            position: relative;
            height: 22px;
            display: flex;
            align-items: center;
        }

        .cfr-score-track {
            position: relative;
            width: 100%;
            height: 5px;
            border-radius: 999px;
            background: #E4E6E8;
            overflow: visible;
        }

        .cfr-score-progress {
            position: absolute;
            left: 0;
            top: 0;
            height: 5px;
            border-radius: 999px;
        }

        .cfr-score-dot {
            position: absolute;
            top: 50%;
            width: 13px;
            height: 13px;
            border: 2px solid #FFFFFF;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 1px 3px rgba(16, 24, 40, .20);
        }

        .cfr-score-value {
            color: #5A0A78;
            text-align: center;
            font-size: 15px;
            line-height: 1;
            font-weight: 950;
            font-variant-numeric: tabular-nums;
        }

        /* Reference-matched regional matrix. */
        .cfr-region-scroll {
            width: 100%;
            max-height: none !important;
            height: 400px;
            margin-top: 4px;
            border: 0 !important;
            border-bottom:1px solid #DDE4EE !important;
            border-radius: 0 !important;
            overflow: visible !important;
        }

        table.cfr-region-table{
            width:100%;
            table-layout:fixed;
            border-collapse:collapse;
            border:1px solid #DDE4EE !important;
            border-radius:12px;
            overflow:hidden;
        }

        table.cfr-region-table th,
        table.cfr-region-table td{
            border-right:1px solid #DDE4EE !important;
            border-bottom:1px solid #DDE4EE !important;
        }

        table.cfr-region-table tr:last-child td{
            border-bottom:1px solid #DDE4EE !important;
        }

        table.cfr-region-table th:last-child,
        table.cfr-region-table td:last-child{
            border-right:1px solid #DDE4EE !important;
        }

        table.cfr-region-table thead th {
            position: static !important;
            height: 48px;
            padding: 4px 7px 3px 7px !important;
            background: #FFFFFF !important;
            border-right: 0 !important;
            border-bottom: 1px solid #E3E8F0 !important;
            vertical-align: bottom;
        }

        table.cfr-region-table tbody td {
            height: 50px;
            padding: 7px 8px !important;
            background: #FFFFFF;
            border-right: 1px solid #DDE4EE !important;
            border-bottom: 1px solid #DDE4EE !important;
            color: #101B57;
            font-size: 11px !important;
            font-weight: 600;
            vertical-align: middle;
        }

        table.cfr-region-table tbody tr:last-child td {
            border-bottom: 1 !important;
        }

        table.cfr-region-table th:first-child,
        table.cfr-region-table td:first-child {
            width: 29% !important;
            padding-left: 10px !important;
            padding-right: 12px !important;
            text-align: left !important;
            white-space: normal !important;
            overflow-wrap: normal !important;
            color: #101B57;
            font-weight: 850;
        }

        table.cfr-region-table th:not(:first-child),
        table.cfr-region-table td:not(:first-child) {
            width: 11.83% !important;
            text-align: center !important;
            white-space: nowrap !important;
        }

        table.cfr-region-table th:last-child,
        table.cfr-region-table td:last-child {
            border-right: 1 !important;
        }

        table.cfr-region-table .cfr-principle-head {
            width: 64px !important;
            min-width: 0 !important;
            height: 30px !important;
            padding: 0 !important;
            border-radius: 5px !important;
            color: #FFFFFF !important;
            font-size: 14px !important;
            font-weight: 950 !important;
        }

        table.cfr-region-table tbody tr:hover {
            background: transparent !important;
            box-shadow: none !important;
        }

        /* Stretch both upper-column wrappers to equal height. */
        div[data-testid="stHorizontalBlock"]:has(
            .cfr-chart-panel-marker
        ) {
            align-items: stretch !important;
        }

        div[data-testid="stHorizontalBlock"]:has(
            .cfr-chart-panel-marker
        ) > div[data-testid="stColumn"] {
            display: flex !important;
            flex-direction: column !important;
        }

        div[data-testid="stHorizontalBlock"]:has(
            .cfr-chart-panel-marker
        ) > div[data-testid="stColumn"] > div {
            flex: 1 1 auto;
        }

        .cfr-country-table-panel {
            margin-top: 14px;
        }

        .cfr-table-scroll {
            width: 100%;
            overflow-x: auto;
            overflow-y: auto;
            border: 1px solid #EEF0F4;
            border-radius: 10px;
            background: #FFFFFF;
        }

        .cfr-region-scroll {
            width: 100%;
            max-height: 430px;
            overflow-x: hidden !important;
            overflow-y: auto;
        }

        table.cfr-region-table {
            width: 100% !important;
            min-width: 0 !important;
            table-layout: fixed !important;
        }

        table.cfr-region-table th,
        table.cfr-region-table td {
            padding: 9px 5px !important;
            font-size: 10.5px !important;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        table.cfr-region-table th:first-child,
        table.cfr-region-table td:first-child {
            width: 42%;
            padding-left: 9px !important;
            padding-right: 7px !important;
            white-space: normal;
            overflow-wrap: anywhere;
            text-overflow: clip;
        }

        table.cfr-region-table th:not(:first-child),
        table.cfr-region-table td:not(:first-child) {
            width: 9.66%;
            text-align: center;
            white-space: nowrap;
        }

        table.cfr-region-table .cfr-principle-head {
            min-width: 0;
            width: 27px;
            height: 24px;
            padding: 0 !important;
            border-radius: 6px;
            font-size: 9.5px;
        }

        .cfr-country-scroll {
            max-height: 485px;
        }

        table.cfr-table {
            width: 100%;
            min-width: 760px;
            border-collapse: separate;
            border-spacing: 0;
            table-layout: fixed;
            color: #101B57;
            background: #FFFFFF;
            font-family:
                "Anek Devanagari",
                Arial,
                sans-serif;
            font-size: 11.5px;
        }

        table.cfr-table th,
        table.cfr-table td {
            padding: 10px 11px;
            border-right: 1px solid #E9ECF2;
            border-bottom: 1px solid #E9ECF2;
            text-align: center;
            vertical-align: middle;
        }

        table.cfr-table th:last-child,
        table.cfr-table td:last-child {
            border-right: 0;
        }

        table.cfr-table tbody tr:last-child td {
            border-bottom: 0;
        }

        table.cfr-table thead th {
            position: sticky;
            top: 0;
            z-index: 3;
            background: #FAFBFC;
            color: #101B57;
            font-weight: 900;
        }

        table.cfr-table th.cfr-left,
        table.cfr-table td.cfr-left {
            width: 31%;
            text-align: left;
            font-weight: 800;
        }

        table.cfr-country-table th.cfr-left,
        table.cfr-country-table td.cfr-left {
            width: 25%;
        }

        table.cfr-country-table th,
        table.cfr-country-table td {
            padding-top: 8px;
            padding-bottom: 8px;
        }

        table.cfr-table tbody tr {
            transition:
                background-color .15s ease,
                box-shadow .15s ease;
        }

        table.cfr-table tbody tr:hover {
            background: #FAF8FF;
            box-shadow:
                inset 3px 0 0 #6C4CF1;
        }

        .cfr-principle-head {
            display: inline-flex;
            min-width: 54px;
            height: 28px;
            padding: 0 12px;
            align-items: center;
            justify-content: center;
            border-radius: 5px;
            color: #101B57;
            font-size: 13px;
            font-weight: 950;
        }

        .cfr-date-cell {
            color: #344054;
            white-space: nowrap;
        }

        .cfr-report-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 92px;
            min-height: 31px;
            padding: 0 12px;
            box-sizing: border-box;
            border: 1px solid #D8C6F0;
            border-radius: 8px;
            background: #F8F4FF;
            color: #5B21B6 !important;
            font-family:
                "Anek Devanagari",
                Arial,
                sans-serif;
            font-size: 10.5px;
            line-height: 1;
            font-weight: 900;
            text-decoration: none !important;
            white-space: nowrap;
            transition:
                background-color .15s ease,
                border-color .15s ease,
                transform .15s ease,
                box-shadow .15s ease;
        }

        .cfr-report-link:hover {
            background: #EEE6FF;
            border-color: #9B7AE0;
            box-shadow: 0 3px 8px rgba(91, 33, 182, .12);
            transform: translateY(-1px);
        }

        .cfr-report-unavailable {
            color: #98A2B3;
            font-size: 10.5px;
            font-weight: 700;
            white-space: nowrap;
        }

        .cfr-empty {
            padding: 28px 18px;
            border: 1px dashed #D6DAE4;
            border-radius: 12px;
            background: #FAFBFC;
            color: #667085;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
        }

        div[data-testid="stPlotlyChart"] {
            width: 100% !important;
            overflow: hidden !important;
        }

        .cfr-panel div[data-testid="stSelectbox"] {
            margin-top: -3px;
            margin-bottom: 2px;
        }

        .cfr-panel div[data-testid="stSelectbox"] label {
            color: #101B57 !important;
            font-size: 10px !important;
            font-weight: 900 !important;
        }

        .cfr-panel div[data-baseweb="select"] > div {
            min-height: 34px !important;
            border: 1px solid #D8DDE7 !important;
            border-radius: 8px !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }

        /* =====================================================
           CFR MOBILE RESPONSIVENESS
           ===================================================== */
        .cfr-page,
        .cfr-panel,
        .cfr-principle-matrix,
        .cfr-table-scroll,
        .cfr-region-scroll,
        .cfr-country-scroll {
            width: 100%;
            max-width: 100%;
            min-width: 0;
            box-sizing: border-box;
        }

        .cfr-table-scroll,
        .cfr-region-scroll,
        .cfr-country-scroll {
            overflow-x: auto !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch;
            overscroll-behavior-x: contain;
            scrollbar-width: thin;
        }

        .cfr-table-scroll::after,
        .cfr-region-scroll::after,
        .cfr-country-scroll::after {
            content: "Swipe horizontally to view all columns";
            display: none;
            padding: 7px 2px 1px;
            color: #667085;
            font-size: 9.5px;
            line-height: 1.2;
            font-weight: 700;
            text-align: left;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) div[data-testid="stPlotlyChart"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) .js-plotly-plot,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) .plot-container,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .cfr-chart-panel-marker
        ) .svg-container {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }

        @media (max-width: 900px) {
            .cfr-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                margin: 14px 0;
            }

            .cfr-kpi-card {
                min-height: 96px;
                grid-template-columns: 46px minmax(0, 1fr);
                column-gap: 11px;
                padding: 13px 14px;
            }

            .cfr-kpi-card:last-child {
                grid-column: 1 / -1;
            }

            .cfr-kpi-icon {
                width: 44px;
                height: 44px;
                font-size: 22px;
            }

            .cfr-kpi-value {
                font-size: 24px;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                .cfr-chart-panel-marker
            ),
            .cfr-top-panel-height,
            .cfr-panel {
                min-height: 0 !important;
                height: auto !important;
                padding: 13px !important;
                overflow: visible !important;
            }

            /* Stack the CFR panel heading and country selector. */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                .cfr-chart-panel-marker
            ) div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 8px !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                .cfr-chart-panel-marker
            ) div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                width: 100% !important;
                min-width: 0 !important;
                flex: 1 1 100% !important;
            }

            .cfr-chart-panel-selector-label {
                margin-top: 2px;
            }

            /* Convert every principle row into a compact two-line mobile card. */
            .cfr-principle-header {
                display: none !important;
            }

            .cfr-principle-matrix {
                display: grid;
                gap: 8px;
                margin-top: 10px;
            }

            .cfr-principle-grid.cfr-principle-row {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 48px;
                grid-template-areas:
                    "label value"
                    "track track";
                column-gap: 10px;
                row-gap: 8px;
                min-height: 0;
                padding: 10px;
                border: 1px solid #E7EAF0;
                border-radius: 10px;
                background: #FCFCFD;
            }

            .cfr-principle-row .cfr-principle-label {
                grid-area: label;
                grid-template-columns: 34px minmax(0, 1fr);
                column-gap: 8px;
            }

            .cfr-principle-row .cfr-score-track-wrap {
                grid-area: track;
                width: 100%;
                height: 18px;
                padding: 0 5px;
                box-sizing: border-box;
            }

            .cfr-principle-row .cfr-score-value {
                grid-area: value;
                align-self: center;
                justify-self: end;
                font-size: 15px;
            }

            .cfr-principle-name {
                font-size: 10.5px;
                line-height: 1.2;
                overflow-wrap: anywhere;
            }

            .cfr-principle-badge {
                width: 33px;
                height: 24px;
                font-size: 12px;
            }

            .cfr-score-track,
            .cfr-score-progress {
                height: 6px;
            }

            .cfr-score-dot {
                width: 14px;
                height: 14px;
            }

            .cfr-table-scroll,
            .cfr-region-scroll,
            .cfr-country-scroll {
                display: block;
                max-height: min(62vh, 480px) !important;
                border: 1px solid #E1E5ED !important;
                border-radius: 10px !important;
                overflow-x: auto !important;
                overflow-y: auto !important;
            }

            .cfr-table-scroll::after,
            .cfr-region-scroll::after,
            .cfr-country-scroll::after {
                display: block;
                position: sticky;
                left: 0;
                background: #FFFFFF;
            }

            table.cfr-region-table,
            table.cfr-country-table,
            table.cfr-table {
                width: max-content !important;
                min-width: 700px !important;
                table-layout: auto !important;
            }

            table.cfr-region-table th:first-child,
            table.cfr-region-table td:first-child,
            table.cfr-country-table th:first-child,
            table.cfr-country-table td:first-child {
                position: sticky;
                left: 0;
                z-index: 2;
                min-width: 145px !important;
                width: 145px !important;
                background: #FFFFFF;
                box-shadow: 1px 0 0 #E9ECF2;
            }

            table.cfr-region-table thead th:first-child,
            table.cfr-country-table thead th:first-child {
                z-index: 5;
                background: #FAFBFC;
            }

            table.cfr-region-table th,
            table.cfr-region-table td,
            table.cfr-country-table th,
            table.cfr-country-table td {
                padding: 8px 9px !important;
                font-size: 10px !important;
                white-space: nowrap;
            }

            table.cfr-region-table .cfr-principle-head,
            table.cfr-country-table .cfr-principle-head {
                min-width: 38px;
                height: 24px;
                padding: 0 7px !important;
                font-size: 10px;
            }

            .cfr-report-link {
                min-width: 82px;
                min-height: 29px;
                padding: 0 9px;
                font-size: 10px;
            }

            .cfr-country-table-panel {
                margin-top: 12px;
            }
        }

        @media (max-width: 600px) {
            .cfr-page-title {
                font-size: 25px;
            }

            .cfr-page-subtitle {
                margin-bottom: 16px;
                font-size: 11.5px;
            }

            .cfr-kpi-grid {
                grid-template-columns: 1fr;
            }

            .cfr-kpi-card:last-child {
                grid-column: auto;
            }

            .cfr-kpi-card {
                min-height: 84px;
            }

            .cfr-panel-title {
                font-size: 14px;
            }

            .cfr-panel-note {
                font-size: 10px;
                line-height: 1.3;
            }

            table.cfr-region-table,
            table.cfr-country-table,
            table.cfr-table {
                min-width: 660px !important;
            }

            /* Keep dashboard tabs usable without compressing labels. */
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                overflow-x: auto !important;
                overflow-y: hidden !important;
                flex-wrap: nowrap !important;
                scrollbar-width: thin;
                -webkit-overflow-scrolling: touch;
            }

            div[data-testid="stTabs"] [data-baseweb="tab"] {
                flex: 0 0 auto !important;
                min-width: max-content !important;
                white-space: nowrap !important;
                padding-left: 12px !important;
                padding-right: 12px !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                .cfr-chart-panel-marker
            ) .modebar-container {
                transform: scale(.82);
                transform-origin: top right;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    

# ------------------------------------------------------------
# HTML HELPERS
# ------------------------------------------------------------
def _safe_text(value):
    if value is None or pd.isna(value):
        return "—"
    return html.escape(str(value))

def _format_score(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f}"

def _principle_header_html(principle):
    background = CFR_PRINCIPLE_LIGHT_COLOURS[principle]
    return (
        f'<span class="cfr-principle-head" '
        f'style="background:{background}; color:#FFFFFF;">'
        f'{principle}'
        f'</span>'
    )


def _build_regional_table_html(regional_scores):
    """Build a compact regional table that fits its panel without scrolling."""
    if regional_scores is None or regional_scores.empty:
        return (
            '<div class="cfr-empty">'
            'No regional scores are available for the selected data.'
            '</div>'
        )

    principle_columns = list(CFR_PRINCIPLES.values())

    header_cells = [
        '<th class="cfr-left">Region</th>',
        *[
            f"<th>{_principle_header_html(principle)}</th>"
            for principle in principle_columns
        ],
    ]

    body_rows = []

    for _, row in regional_scores.iterrows():
        region_label = _safe_text(row["region"])
        region_label = (
            region_label
            .replace("Americas and the Caribbean", "Americas and the Caribbean")
            .replace("Asia and the Pacific", "Asia and the Pacific")
            .replace(
                "Middle East and North Africa",
                "Middle East and North Africa",
            )
        )

        cells = [
            f'<td class="cfr-left">{region_label}</td>'
        ]
        cells.extend(
            f"<td>{_format_score(row[principle])}</td>"
            for principle in principle_columns
        )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="cfr-table-scroll cfr-region-scroll">'
        '<table class="cfr-table cfr-region-table">'
        '<thead><tr>'
        + "".join(header_cells)
        + '</tr></thead>'
        '<tbody>'
        + "".join(body_rows)
        + '</tbody>'
        '</table>'
        '</div>'
    )


def _normalise_cfr_permalink(value):
    """Return a safe absolute CFR report URL or an empty string."""
    if value is None or pd.isna(value):
        return ""

    url = str(value).strip()

    if not url or url.lower() in {"nan", "none", "null"}:
        return ""

    if url.startswith("/"):
        url = f"https://eusee.hivos.org{url}"
    elif not url.lower().startswith(("http://", "https://")):
        url = f"https://{url}"

    return html.escape(url, quote=True)


def _build_country_table_html(country_scores):
    if country_scores is None or country_scores.empty:
        return (
            '<div class="cfr-empty">'
            'No country scores are available.'
            '</div>'
        )

    principle_columns = list(CFR_PRINCIPLES.values())

    header_cells = [
        '<th class="cfr-left">Country</th>',
        *[
            f"<th>{_principle_header_html(principle)}</th>"
            for principle in principle_columns
        ],
        "<th>Last modified</th>",
        "<th>Report</th>",
    ]

    body_rows = []

    for _, row in country_scores.iterrows():
        last_modified = (
            row["Last Modified"].strftime("%b %Y")
            if pd.notna(row["Last Modified"])
            else "—"
        )

        report_url = _normalise_cfr_permalink(
            row.get("Permalink", "")
        )

        if report_url:
            report_html = (
                '<a class="cfr-report-link" '
                f'href="{report_url}" '
                'target="_blank" '
                'rel="noopener noreferrer">'
                'Open report ↗'
                '</a>'
            )
        else:
            report_html = (
                '<span class="cfr-report-unavailable">'
                'Not available'
                '</span>'
            )

        cells = [
            f'<td class="cfr-left">{_safe_text(row["Country"])}</td>'
        ]

        cells.extend(
            f"<td>{_format_score(row[principle])}</td>"
            for principle in principle_columns
        )

        cells.append(
            f'<td class="cfr-date-cell">{_safe_text(last_modified)}</td>'
        )
        cells.append(f"<td>{report_html}</td>")

        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="cfr-table-scroll cfr-country-scroll">'
        '<table class="cfr-table cfr-country-table">'
        '<thead><tr>'
        + "".join(header_cells)
        + '</tr></thead>'
        '<tbody>'
        + "".join(body_rows)
        + '</tbody>'
        '</table>'
        '</div>'
    )


def _render_cfr_kpis(country_count):
    """
    Render the three aligned CFR KPI cards.

    HTML is assembled without leading indentation because indented HTML
    can be interpreted by Markdown as a code block and displayed literally.
    """
    kpi_html = (
        '<div class="cfr-kpi-grid">'

        '<div class="cfr-kpi-card">'
        '<div class="cfr-kpi-icon">🌐</div>'
        '<div class="cfr-kpi-content">'
        '<div class="cfr-kpi-title">Monitored countries</div>'
        f'<div class="cfr-kpi-value">+80</div>'
        '</div>'
        '</div>'

        '<div class="cfr-kpi-card">'
        '<div class="cfr-kpi-icon">◎</div>'
        '<div class="cfr-kpi-content">'
        '<div class="cfr-kpi-title">Monitored principles</div>'
        '<div class="cfr-kpi-value">6</div>'
        '</div>'
        '</div>'

        '<div class="cfr-kpi-card">'
        '<div class="cfr-kpi-icon">📏</div>'
        '<div class="cfr-kpi-content">'
        '<div class="cfr-kpi-title">Scoring scale</div>'
        '<div class="cfr-kpi-value">1–5</div>'
        '<div class="cfr-kpi-note">'
        'Most restricted to most enabling'
        '</div>'
        '</div>'
        '</div>'

        '</div>'
    )

    st.markdown(
        kpi_html,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# SINGLE CFR GRAPH
# ------------------------------------------------------------
def _build_principle_score_matrix_html(chart_data):
    """Build the reference-style CFR principle score matrix."""
    principle_columns = list(CFR_PRINCIPLES.values())

    averages = {
        principle: (
            float(chart_data[principle].mean())
            if chart_data[principle].notna().any()
            else np.nan
        )
        for principle in principle_columns
    }

    header_html = (
        '<div class="cfr-principle-grid cfr-principle-header">'
        '<div class="cfr-principle-header-left">Principle</div>'
        '<div class="cfr-scale-header">'
        '<div class="cfr-scale-ends">'
        '<span>Restricted</span>'
        '<span>Enabling</span>'
        '</div>'
        '<div class="cfr-scale-ticks">'
        '<span>1</span><span>2</span><span>3</span>'
        '<span>4</span><span>5</span>'
        '</div>'
        '</div>'
        '<div class="cfr-score-header">Mean score</div>'
        '</div>'
    )

    rows = []

    for principle in principle_columns:
        score = averages[principle]
        colour = CFR_PRINCIPLE_COLOURS[principle]
        principle_name = CFR_PRINCIPLE_NAMES[principle]

        if pd.notna(score):
            clipped_score = min(
                max(float(score), CFR_SCORE_MIN),
                CFR_SCORE_MAX,
            )
            position = (
                (clipped_score - CFR_SCORE_MIN)
                / (CFR_SCORE_MAX - CFR_SCORE_MIN)
                * 100
            )
            score_text = f"{score:.1f}"
        else:
            position = 0
            score_text = "—"

        rows.append(
            '<div class="cfr-principle-grid cfr-principle-row">'
            '<div class="cfr-principle-label">'
            f'<span class="cfr-principle-badge" '
            f'style="background:{colour};">{principle}</span>'
            f'<span class="cfr-principle-name">{principle_name}</span>'
            '</div>'
            '<div class="cfr-score-track-wrap">'
            '<div class="cfr-score-track">'
            f'<div class="cfr-score-progress" '
            f'style="width:{position:.2f}%;background:{colour};"></div>'
            f'<div class="cfr-score-dot" '
            f'style="left:{position:.2f}%;background:{colour};"></div>'
            '</div>'
            '</div>'
            f'<div class="cfr-score-value">{score_text}</div>'
            '</div>'
        )

    return (
        '<div class="cfr-principle-matrix">'
        + header_html
        + "".join(rows)
        + '</div>'
    )


def _build_aggregated_cfr_figure(chart_data):
    """
    Build a compact horizontal CFR score chart.

    The principle names are rendered as aligned y-axis labels, while the
    original principle colours remain on the score lines and markers.
    """
    principle_columns = list(CFR_PRINCIPLES.values())

    averages = {
        principle: (
            float(chart_data[principle].mean())
            if chart_data[principle].notna().any()
            else np.nan
        )
        for principle in principle_columns
    }

    # Plotly displays the final category at the top, so reverse the data.
    display_principles = list(reversed(principle_columns))
    y_positions = list(range(len(display_principles)))

    figure = go.Figure()

    for y_position, principle in zip(y_positions, display_principles):
        score = averages[principle]
        colour = CFR_PRINCIPLE_COLOURS[principle]

        # Neutral full scoring scale.
        figure.add_trace(
            go.Scatter(
                x=[CFR_SCORE_MIN, CFR_SCORE_MAX],
                y=[y_position, y_position],
                mode="lines",
                line=dict(color="#E5E7EB", width=7),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if pd.notna(score):
            clipped_score = min(
                max(float(score), CFR_SCORE_MIN),
                CFR_SCORE_MAX,
            )

            # Coloured achieved score.
            figure.add_trace(
                go.Scatter(
                    x=[CFR_SCORE_MIN, clipped_score],
                    y=[y_position, y_position],
                    mode="lines",
                    line=dict(color=colour, width=7),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Score marker.
            figure.add_trace(
                go.Scatter(
                    x=[clipped_score],
                    y=[y_position],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=colour,
                        line=dict(color="#FFFFFF", width=2),
                    ),
                    customdata=[[
                        principle,
                        CFR_PRINCIPLE_NAMES[principle],
                        score,
                    ]],
                    hovertemplate=(
                        "<b>%{customdata[0]} — %{customdata[1]}</b><br>"
                        "Score: %{customdata[2]:.2f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

            figure.add_annotation(
                x=5.18,
                y=y_position,
                text=f"<b>{score:.1f}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(
                    family=PLOTLY_FONT_FAMILY,
                    size=13,
                    color="#4020DD",
                ),
            )

    tick_labels = [
        (
            f"<b>{principle}</b>  "
            f"{CFR_PRINCIPLE_NAMES[principle]}"
        )
        for principle in display_principles
    ]

    figure.update_layout(
        height=390,
        margin=dict(l=205, r=44, t=54, b=34),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family=PLOTLY_FONT_FAMILY,
            color=CFR_TEXT,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D9DDE7",
            font=dict(
                family=PLOTLY_FONT_FAMILY,
                size=11,
                color=CFR_TEXT,
            ),
        ),
        showlegend=False,
        xaxis=dict(
            range=[0.92, 5.42],
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
            side="top",
            title=None,
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(size=10, color=CFR_TEXT),
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-0.55, len(display_principles) - 0.45],
            tickmode="array",
            tickvals=y_positions,
            ticktext=tick_labels,
            tickfont=dict(size=10.5, color=CFR_TEXT),
            ticks="",
            showgrid=False,
            zeroline=False,
            showline=False,
            automargin=True,
            fixedrange=True,
        ),
    )

    figure.add_annotation(
        x=1,
        y=1.12,
        xref="x",
        yref="paper",
        text="<b>Restricted</b>",
        showarrow=False,
        xanchor="center",
        font=dict(
            family=PLOTLY_FONT_FAMILY,
            size=10,
            color=CFR_MUTED,
        ),
    )

    figure.add_annotation(
        x=5,
        y=1.12,
        xref="x",
        yref="paper",
        text="<b>Enabling</b>",
        showarrow=False,
        xanchor="center",
        font=dict(
            family=PLOTLY_FONT_FAMILY,
            size=10,
            color=CFR_MUTED,
        ),
    )

    figure.add_annotation(
        x=5.18,
        y=1.12,
        xref="x",
        yref="paper",
        text="<b>Score</b>",
        showarrow=False,
        xanchor="left",
        font=dict(
            family=PLOTLY_FONT_FAMILY,
            size=10,
            color=CFR_MUTED,
        ),
    )

    return figure


def _build_cfr_over_time_figure(data):
    """Build annual CFR principle trends using year only — no monthly axis."""

    import plotly.graph_objects as go

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    principle_columns = list(CFR_PRINCIPLES.values())

    # Ensure year is numeric
    trend_data = data.copy()
    trend_data["CFR Year"] = pd.to_numeric(
        trend_data["CFR Year"],
        errors="coerce",
    )

    trend_data = trend_data.dropna(subset=["CFR Year"])

    if trend_data.empty:
        fig = go.Figure()
        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    # ---------------------------------------------------------
    # ANNUAL AGGREGATION ONLY
    # ---------------------------------------------------------
    annual_scores = (
        trend_data
        .groupby("CFR Year")[principle_columns]
        .mean()
        .reset_index()
        .sort_values("CFR Year")
    )

    fig = go.Figure()

    for principle in principle_columns:
        fig.add_trace(
            go.Scatter(
                x=annual_scores["CFR Year"],
                y=annual_scores[principle],
                mode="lines+markers",
                name=principle,
                line=dict(width=2),
                marker=dict(size=7),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Year: %{x}<br>"
                    "Mean score: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # ---------------------------------------------------------
    # AXIS / LAYOUT
    # ---------------------------------------------------------
    fig.update_xaxes(
        title_text="Year",
        type="linear",
        tickmode="linear",
        dtick=1,
        showgrid=True,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Mean CFR score",
        range=[CFR_SCORE_MIN, CFR_SCORE_MAX],
        dtick=1,
        showgrid=True,
        zeroline=False,
    )

    fig.update_layout(
        height=390,
        margin=dict(l=55, r=20, t=20, b=55),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        template="plotly_white",
    )

    return fig
# ------------------------------------------------------------
# CFR PAGE
# ------------------------------------------------------------

def render_cfr_matrix_zoom_controls():
    """Add accessible zoom controls for the aggregated CFR principle matrix."""
    components.html(
        r"""
        <script>
        (function () {
            const doc = window.parent.document;
            const ROOT_ID = "eusee-cfr-zoom-controls";
            const STYLE_ID = "eusee-cfr-zoom-style";
            const SCALE_MIN = 0.80;
            const SCALE_MAX = 1.40;
            const SCALE_STEP = 0.10;

            function getPanel() {
                const marker = doc.querySelector(".cfr-chart-panel-marker");
                return marker ? marker.closest('[data-testid="stVerticalBlockBorderWrapper"]') : null;
            }

            function getMatrix(panel) {
                return panel ? panel.querySelector(".cfr-principle-matrix") : null;
            }

            function ensureStyle() {
                if (doc.getElementById(STYLE_ID)) return;
                const style = doc.createElement("style");
                style.id = STYLE_ID;
                style.textContent = `
                    #${ROOT_ID} {
                        display: inline-flex;
                        align-items: center;
                        justify-content: flex-end;
                        gap: 6px;
                        width: 100%;
                        margin: 2px 0 8px 0;
                        font-family: "Anek Devanagari", Arial, sans-serif;
                    }
                    #${ROOT_ID} button {
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        min-width: 34px;
                        height: 32px;
                        padding: 0 10px;
                        border: 1px solid #D6BBE5;
                        border-radius: 9px;
                        background: #FFFFFF;
                        color: #660094;
                        font: 800 12px/1 "Anek Devanagari", Arial, sans-serif;
                        cursor: pointer;
                        box-shadow: 0 2px 6px rgba(16,24,40,.05);
                    }
                    #${ROOT_ID} button:hover { background: #FBF7FD; border-color: #660094; }
                    #${ROOT_ID} button:focus-visible { outline: 3px solid rgba(102,0,148,.18); outline-offset: 1px; }
                    #${ROOT_ID} .cfr-zoom-value {
                        min-width: 46px;
                        text-align: center;
                        color: #667085;
                        font-size: 11px;
                        font-weight: 800;
                    }
                    .cfr-principle-matrix {
                        transform-origin: top left;
                        transition: transform .16s ease;
                    }
                    @media (max-width: 700px) {
                        #${ROOT_ID} { justify-content: flex-start; overflow-x: auto; }
                        #${ROOT_ID} button { min-width: 32px; height: 30px; padding: 0 8px; }
                    }
                `;
                doc.head.appendChild(style);
            }

            function install() {
                const panel = getPanel();
                const matrix = getMatrix(panel);
                if (!panel || !matrix) return false;

                ensureStyle();
                const old = doc.getElementById(ROOT_ID);
                if (old) old.remove();

                const controls = doc.createElement("div");
                controls.id = ROOT_ID;
                controls.setAttribute("role", "group");
                controls.setAttribute("aria-label", "CFR chart zoom controls");
                controls.innerHTML = `
                    <button type="button" data-action="out" aria-label="Zoom out" title="Zoom out">−</button>
                    <span class="cfr-zoom-value" aria-live="polite">100%</span>
                    <button type="button" data-action="in" aria-label="Zoom in" title="Zoom in">+</button>
                    <button type="button" data-action="reset" aria-label="Reset zoom" title="Reset zoom">Reset</button>
                `;
                matrix.parentNode.insertBefore(controls, matrix);

                let scale = 1;
                const value = controls.querySelector(".cfr-zoom-value");
                function applyScale(next) {
                    scale = Math.min(SCALE_MAX, Math.max(SCALE_MIN, next));
                    matrix.style.transform = `scale(${scale})`;
                    matrix.style.width = `${100 / scale}%`;
                    value.textContent = `${Math.round(scale * 100)}%`;
                }
                controls.addEventListener("click", function (event) {
                    const button = event.target.closest("button[data-action]");
                    if (!button) return;
                    const action = button.dataset.action;
                    if (action === "in") applyScale(scale + SCALE_STEP);
                    if (action === "out") applyScale(scale - SCALE_STEP);
                    if (action === "reset") applyScale(1);
                });
                applyScale(1);
                return true;
            }

            let attempts = 0;
            const timer = window.setInterval(function () {
                attempts += 1;
                if (install() || attempts > 20) window.clearInterval(timer);
            }, 150);
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def render_cfr_analysis():
    _inject_cfr_dashboard_css()

    source = _find_cfr_source()

    if source is None:
        st.warning(
            f"CFR data file not found. Add `{CFR_SOURCE_FILENAME}` "
            "to the exports folder or define the EUSEE_CFR_CSV "
            "environment variable."
        )
        return

    metadata_path = EXPORT_DIR / "countries_metadata.json"
    metadata_mtime = (
        metadata_path.stat().st_mtime
        if metadata_path.exists()
        else None
    )

    try:
        cfr = load_cfr_data(
            str(source),
            source.stat().st_mtime,
            metadata_mtime,
        )
    except Exception as exc:
        st.error(f"Could not load CFR data: {exc}")
        return

    if cfr.empty:
        st.info("No CFR records are currently available.")
        return

    metadata_error = cfr.attrs.get("metadata_error", "")
    if metadata_error:
        st.warning(metadata_error)

    principle_columns = list(CFR_PRINCIPLES.values())

    # Notify administrators when the export is inconsistent with the
    # requested 1–5 scoring scale, but do not silently alter the data.
    score_values = cfr[principle_columns].stack().dropna()

    if not score_values.empty:
        below_scale = bool((score_values < CFR_SCORE_MIN).any())
        above_scale = bool((score_values > CFR_SCORE_MAX).any())

        if below_scale or above_scale:
            st.warning(
                "Some CFR values fall outside the configured 1–5 scale. "
                "The graph is displayed on the requested 1–5 axis; check "
                "the CFR export before publication."
            )

    available_years = sorted(
        cfr["CFR Year"].dropna().astype(int).unique().tolist(),
        reverse=True,
    )
    year_options = ["All years", *available_years]

    country_count = int(cfr["Country"].nunique())
    _render_cfr_kpis(country_count)

    chart_column, region_column = st.columns(
        [1.00, 1.00],
        gap="medium",
    )

    # --------------------------------------------------------
    # LEFT: TABBED AGGREGATED + OVER-TIME CFR CHARTS
    # --------------------------------------------------------
    with chart_column:
        with st.container(border=True):
            # Hidden marker used by the CFR-specific CSS selector.
            st.markdown(
                '<span class="cfr-chart-panel-marker"></span>',
                unsafe_allow_html=True,
            )

            aggregated_tab, over_time_tab = st.tabs(
                [
                    "Aggregated CFR scores by principle",
                    "Principles over time",
                ]
            )

            with aggregated_tab:
                st.markdown(
                    """
                    <div class="cfr-chart-panel-heading">
                        <div class="cfr-panel-title">
                            Aggregated CFR scores by principle
                        </div>
                        <div class="cfr-panel-note">
                            Mean score across the selected countries and year.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                country_selector_col, year_selector_col = st.columns(
                    [1.45, 0.85],
                    gap="small",
                )

                with country_selector_col:
                    st.markdown(
                        '<div class="cfr-chart-panel-selector-label">'
                        'Select country'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    selected_country = st.selectbox(
                        "Select country",
                        options=[
                            "All countries",
                            *sorted(
                                cfr["Country"]
                                .dropna()
                                .unique()
                                .tolist()
                            ),
                        ],
                        index=0,
                        key="cfr_country_selector",
                        label_visibility="collapsed",
                    )

                with year_selector_col:
                    st.markdown(
                        '<div class="cfr-chart-panel-selector-label">'
                        'Select Year'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    selected_year = st.selectbox(
                        "Select Year",
                        options=year_options,
                        index=0,
                        key="cfr_year_selector",
                        label_visibility="collapsed",
                    )

                chart_data = cfr.copy()

                if selected_country != "All countries":
                    chart_data = chart_data[
                        chart_data["Country"].eq(selected_country)
                    ]

                if selected_year != "All years":
                    chart_data = chart_data[
                        chart_data["CFR Year"].eq(int(selected_year))
                    ]

                render_cfr_matrix_zoom_controls()
                st.markdown(
                    _build_principle_score_matrix_html(chart_data),
                    unsafe_allow_html=True,
                )

            with over_time_tab:
                st.markdown(
                    """
                    <div class="cfr-chart-panel-heading">
                        <div class="cfr-panel-title">
                            Aggregated CFR Scores by Principle Over Time
                        </div>
                        <div class="cfr-panel-note">
                            Mean CFR principle scores by reporting year.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="cfr-chart-panel-selector-label">'
                    'Select country'
                    '</div>',
                    unsafe_allow_html=True,
                )
                time_country = st.selectbox(
                    "Select country for principle trends",
                    options=[
                        "All countries",
                        *sorted(
                            cfr["Country"]
                            .dropna()
                            .unique()
                            .tolist()
                        ),
                    ],
                    index=0,
                    key="cfr_time_country_selector",
                    label_visibility="collapsed",
                )

                time_data = (
                    cfr
                    if time_country == "All countries"
                    else cfr[cfr["Country"].eq(time_country)]
                )

                st.plotly_chart(
                    _build_cfr_over_time_figure(time_data),
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                    key="cfr_principles_over_time_chart",
                )

    # --------------------------------------------------------
    # RIGHT: REGIONAL TABLE + YEAR SELECTOR
    # --------------------------------------------------------
    with region_column:
        with st.container(border=True):
            st.markdown(
                '<span class="cfr-region-panel-marker"></span>',
                unsafe_allow_html=True,
            )

            regional_header_col, regional_selector_col = st.columns(
                [1.85, 0.75],
                gap="medium",
                vertical_alignment="bottom",
            )

            with regional_header_col:
                st.markdown(
                    """
                    <div class="cfr-chart-panel-heading">
                        <div class="cfr-panel-title">
                            Regional score patterns
                        </div>
                        <div class="cfr-panel-note">
                            Average score by principle and region.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with regional_selector_col:
                st.markdown(
                    '<div class="cfr-chart-panel-selector-label">'
                    'Select Year'
                    '</div>',
                    unsafe_allow_html=True,
                )
                regional_year = st.selectbox(
                    "Select Year for regional scores",
                    options=year_options,
                    index=0,
                    key="cfr_regional_year_selector",
                    label_visibility="collapsed",
                )

            regional_data = cfr.copy()
            if regional_year != "All years":
                regional_data = regional_data[
                    regional_data["CFR Year"].eq(int(regional_year))
                ]

            regional_scores = (
                regional_data.groupby(
                    "region",
                    dropna=False,
                )[principle_columns]
                .mean()
                .reset_index()
            )

            regional_scores["region"] = (
                regional_scores["region"]
                .fillna("Unknown")
                .astype(str)
            )

            regional_scores["_region_order"] = (
                regional_scores["region"]
                .apply(
                    lambda region: (
                        CFR_REGION_ORDER.index(region)
                        if region in CFR_REGION_ORDER
                        else len(CFR_REGION_ORDER)
                    )
                )
            )

            regional_scores = (
                regional_scores.sort_values(
                    ["_region_order", "region"],
                    ascending=[True, True],
                )
                .drop(columns=["_region_order"])
                .reset_index(drop=True)
            )

            st.markdown(
                _build_regional_table_html(regional_scores),
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # BOTTOM: COUNTRY TABLE
    # --------------------------------------------------------
    # Keep the existing country table as a latest-record snapshot so the
    # new historical rows do not duplicate countries in this table.
    country_table = (
        cfr.sort_values(
            ["Country", "Last Modified"],
            ascending=[True, True],
            na_position="first",
        )
        .drop_duplicates(subset=["Country"], keep="last")
        [
            [
                "Country",
                *principle_columns,
                "Last Modified",
                "Permalink",
            ]
        ]
        .sort_values("Country", ascending=True)
        .reset_index(drop=True)
    )

    st.markdown(
        f"""
        <div class="cfr-panel cfr-country-table-panel">
            <div class="cfr-panel-title">
                CFR scores per country
            </div>
            <div class="cfr-panel-note">
                Country-level scores across the six enabling environment
                principles.
            </div>
            {_build_country_table_html(country_table)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- MAIN TABS - PLACED IMMEDIATELY AFTER SUBTITLE ----------------
# This removes the visible blank space between the dashboard subtitle and the tabs.
#tab_map disabled

tab_overview = tab_negative = tab_map = tab_cfr = tab_manual = None

_dashboard_tab_specs = []

if has_permission("view_overview"):
    _dashboard_tab_specs.append(("overview", "Alert Overview"))

# CFR Score is placed immediately after Overview and before Negative Alerts Analysis.
# It follows the same dashboard-access permission as Overview so the existing
# authz/admin files do not need to change for this update.
if has_permission("view_overview"):
    _dashboard_tab_specs.append(("cfr", "Country Focus Report (CFR) Scores"))

if has_permission("view_negative_alerts"):
    _dashboard_tab_specs.append(("negative", "Negative Alerts Analysis"))

if has_permission("view_maps"):
    _dashboard_tab_specs.append(("map", "Visualization Map"))

if has_permission("view_user_manual"):
    _dashboard_tab_specs.append(("manual", "User Manual"))

if _dashboard_tab_specs:
    _dashboard_tabs = st.tabs([label for _, label in _dashboard_tab_specs])
    _dashboard_tab_lookup = {tab_id: tab for (tab_id, _), tab in zip(_dashboard_tab_specs, _dashboard_tabs)}

    tab_overview = _dashboard_tab_lookup.get("overview")
    tab_negative = _dashboard_tab_lookup.get("negative")
    tab_map = _dashboard_tab_lookup.get("map")
    tab_cfr = _dashboard_tab_lookup.get("cfr")
    tab_manual = _dashboard_tab_lookup.get("manual")
else:
    st.error("No dashboard tabs are enabled for your role. Please contact the dashboard administrator.")
    st.stop()


# ---------------- COLLAPSED RESPONSIVE FLOATING FEEDBACK OVERLAY ----------------
def render_top_feedback_bar():
    """Render a single-click floating link that opens the feedback form."""
    feedback_url = (
        "https://forms.office.com/pages/responsepage.aspx?"
        "id=aFcOUAlSoUeqnjS7rLiI3i2QH6350xBGsugTt9B-i59URUk5UEFTV0VKSDRaU0lXTEc1S1g1M0hYTi4u"
        "&route=shorturl"
    )

    components.html(
        f"""
        <script>
        (function () {{
            const doc = window.parent.document;
            const ROOT_ID = "eusee-feedback-floating-root";
            const STYLE_ID = "eusee-feedback-floating-style";

            const existingRoot = doc.getElementById(ROOT_ID);
            if (existingRoot) existingRoot.remove();
            const existingStyle = doc.getElementById(STYLE_ID);
            if (existingStyle) existingStyle.remove();

            const style = doc.createElement("style");
            style.id = STYLE_ID;
            style.textContent = `
                #${{ROOT_ID}} {{
                    position: fixed;
                    right: 20px;
                    bottom: 35px;
                    z-index: 1000001;
                    font-family: "Anek Devanagari", Arial, sans-serif;
                }}
                #${{ROOT_ID}} a {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 7px;
                    min-height: 42px;
                    padding: 0 15px;
                    border: 1px solid #660094;
                    border-radius: 999px;
                    background: #660094;
                    color: #FFFFFF;
                    text-decoration: none;
                    font-size: 12px;
                    font-weight: 850;
                    line-height: 1;
                    box-shadow: 0 8px 22px rgba(102,0,148,.24);
                    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
                }}
                #${{ROOT_ID}} a:hover {{
                    background: #4E0072;
                    box-shadow: 0 10px 26px rgba(102,0,148,.30);
                    transform: translateY(-1px);
                }}
                #${{ROOT_ID}} a:focus-visible {{
                    outline: 3px solid rgba(102,0,148,.22);
                    outline-offset: 3px;
                }}
                #${{ROOT_ID}} .feedback-plus {{
                    font-size: 17px;
                    font-weight: 700;
                    line-height: 1;
                }}
                @media (max-width: 700px) {{
                    #${{ROOT_ID}} {{ right: 12px; bottom: 12px; }}
                    #${{ROOT_ID}} a {{
                        min-height: 40px;
                        padding: 0 13px;
                        font-size: 11.5px;
                        box-shadow: 0 6px 18px rgba(102,0,148,.22);
                    }}
                }}
            `;
            doc.head.appendChild(style);

            const root = doc.createElement("div");
            root.id = ROOT_ID;
            root.innerHTML = `
                <a href="{feedback_url}" target="_blank" rel="noopener noreferrer"
                   aria-label="Open feedback form" title="Open feedback form">
                    <span aria-hidden="true">💬</span>
                    <span>Feedback</span>
                    <span class="feedback-plus" aria-hidden="true">+</span>
                </a>
            `;
            doc.body.appendChild(root);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


render_top_feedback_bar()  # One click opens the feedback form directly.
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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

    /* Match the CFR summary-card icon treatment. */
    .eusee-kpi-icon {
        width: 56px;
        height: 56px;
        min-width: 56px;
        border-radius: 999px;
        background: linear-gradient(135deg, #F7F2FF 0%, #EDF7FF 100%);
        color: #5F24F5;
        border: 1px solid #D9DDF8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 900;
        line-height: 1;
        box-shadow: none;
        flex: 0 0 56px;
    }

    .eusee-kpi-value {
        font-size: 36px;
        line-height: .92;
        font-weight: 950;
        margin-top: 9px;
        letter-spacing: -0.045em;
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
    }

    .eusee-donut-center .num {
        font-size: 14px;
        letter-spacing: -.03em;
    }

    .eusee-donut-center .lab {
        font-size: 7.8px;
        color: #667085;
        margin-top: 2px;
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
                    <div><div class="eusee-kpi-title">Total Alerts </div></div>
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
    /* Match the CFR summary-card icon treatment. */
    .negintel-icon {
        width:56px;
        height:56px;
        min-width:56px;
        border-radius:999px;
        background:linear-gradient(135deg, #F7F2FF 0%, #EDF7FF 100%);
        color:#5F24F5;
        border:1px solid #D9DDF8;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:28px;
        font-weight:900;
        line-height:1;
        flex:0 0 56px;
    }
    /* Preserve the original icon treatment for the Frequent Restriction Pattern card. */
    .negintel-icon-original {
        width:30px;
        height:30px;
        min-width:30px;
        border-radius:12px;
        background:linear-gradient(135deg, rgba(180,35,24,.12), rgba(255,219,88,.14));
        color:#B42318;
        border:1px solid rgba(180,35,24,.10);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:16px;
        font-weight:900;
        line-height:1;
        flex:0 0 30px;
    }
    .negintel-value { font-size:34px; line-height:.92; font-weight:950; margin-top:8px; letter-spacing:-0.045em; font-family:"Anek Devanagari", Arial, sans-serif; color:#FF6F61; }
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
    .negintel-row-pct { color:#101828; font-size:10.4px; font-weight:950; text-align:right; font-family:"Anek Devanagari", Arial, sans-serif; letter-spacing:-.035em; white-space:nowrap; padding-top:1px; }
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
                <div class="negintel-icon-original">⛓️</div>
            </div>
            <div class="negintel-row-list">
                <div class="negintel-row" title="Top restrictive actor: {top_actor}"><span class="negintel-row-label"><strong>Restrictive Actor:</strong> {top_actor}</span><span class="negintel-row-pct">{actor_pct}%</span><span class="negintel-row-count">{top_actor_count:,}</span></div>
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

CHART_FONT = PLOTLY_FONT_FAMILY
CHART_TITLE_COLOR = "#2D0055"
CHART_TEXT_COLOR = "#263238"
CHART_GRID_COLOR = "#EEF1F6"
CHART_AXIS_COLOR = "#D8DEE9"


DEFAULT_PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


def apply_classic_chart_theme(fig, title=None, height=None, horizontal=False, showlegend=True):
    """Apply the unified EUSEE premium chart style without changing chart data."""
    if fig is None:
        return fig

    current_title = title if title is not None else (fig.layout.title.text or "")

    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family=CHART_FONT,
            size=11,
            color=CHART_TEXT_COLOR,
        ),
        title=dict(
            text=current_title,
            x=0.018,
            xanchor="left",
            y=0.955,
            yanchor="top",
            pad=dict(t=0, b=10),
            font=dict(
                family=CHART_FONT,
                size=16,
                color="#16002B",
            ),
        ),
        margin=dict(
            l=145 if horizontal else 56,
            r=34,
            t=94,
            b=62,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E6E8EF",
            font=dict(
                family=CHART_FONT,
                size=12,
                color="#23152F",
            ),
            namelength=-1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.075,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#E8E2EF",
            borderwidth=1,
            font=dict(
                family=CHART_FONT,
                size=10,
                color="#23152F",
            ),
            title=None,
            itemsizing="constant",
            itemwidth=34,
            tracegroupgap=4,
        ),
        showlegend=showlegend,
        bargap=0.34,
        bargroupgap=0.12,
        uniformtext_minsize=9,
        uniformtext_mode="hide",
    )

    fig.update_xaxes(
        title=None,
        showgrid=True,
        gridwidth=1,
        gridcolor="#EEF2F6",
        griddash="solid",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#D8DEE9",
        ticks="",
        tickfont=dict(
            family=CHART_FONT,
            size=10,
            color="#4B5470",
        ),
        automargin=True,
        title_standoff=10,
    )

    fig.update_yaxes(
        title=None,
        showgrid=False if horizontal else True,
        gridwidth=1,
        gridcolor="#F1F4F8",
        griddash="solid",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#D8DEE9",
        ticks="",
        tickfont=dict(
            family=CHART_FONT,
            size=10.5,
            color="#23152F",
        ),
        automargin=True,
        title_standoff=10,
    )

    # Apply consistent trace polish while preserving all existing values and colors.
    for trace in fig.data:
        trace_type = getattr(trace, "type", "")

        if trace_type == "bar":
            try:
                trace.update(
                    marker_line_color="rgba(255,255,255,0.82)",
                    marker_line_width=0.7,
                    marker_cornerradius=8,
                    opacity=0.98,
                    cliponaxis=False,
                )
            except Exception:
                trace.update(
                    marker_line_color="rgba(255,255,255,0.82)",
                    marker_line_width=0.7,
                    opacity=0.98,
                    cliponaxis=False,
                )

        elif trace_type in {"scatter", "scattergl"}:
            mode = str(getattr(trace, "mode", "") or "")
            update_args = dict(
                line=dict(width=3),
                hoverlabel=dict(bgcolor="#FFFFFF"),
            )
            if "markers" in mode:
                update_args["marker"] = dict(
                    size=8,
                    line=dict(width=1.5, color="#FFFFFF"),
                )
            try:
                trace.update(**update_args)
            except Exception:
                pass

        elif trace_type == "pie":
            try:
                trace.update(
                    hole=getattr(trace, "hole", 0),
                    sort=False,
                    marker=dict(
                        line=dict(color="#FFFFFF", width=3),
                    ),
                    textfont=dict(
                        family=CHART_FONT,
                        size=11,
                        color="#23152F",
                    ),
                    hoverlabel=dict(bgcolor="#FFFFFF"),
                )
            except Exception:
                pass

        elif trace_type == "heatmap":
            try:
                trace.update(
                    xgap=2,
                    ygap=2,
                    hoverongaps=False,
                    hoverlabel=dict(bgcolor="#FFFFFF"),
                )
            except Exception:
                pass

    return fig

def render_chart_shell():
    """Apply the premium EUSEE card treatment to every Plotly chart."""
    st.markdown(
        """
        <style>
        div[data-testid="stPlotlyChart"] {
            position: relative;
            background:
                radial-gradient(circle at 8% 0%, rgba(102,0,148,.035), transparent 28%),
                linear-gradient(180deg, #FFFFFF 0%, #FEFEFF 100%);
            border: 1px solid #E8E2EF;
            border-radius: 22px;
            padding: 10px 12px 7px 12px;
            box-shadow:
                0 1px 2px rgba(16,24,40,.025),
                0 12px 34px rgba(45,0,85,.065);
            margin-bottom: 20px;
            overflow: hidden;
            transition:
                box-shadow .20s ease,
                transform .20s ease,
                border-color .20s ease;
        }


        div[data-testid="stPlotlyChart"]:hover {
            transform: translateY(-2px);
            border-color: #D8C7E6;
            box-shadow:
                0 2px 4px rgba(16,24,40,.035),
                0 18px 42px rgba(45,0,85,.105);
        }

        div[data-testid="stPlotlyChart"] .js-plotly-plot,
        div[data-testid="stPlotlyChart"] .plot-container,
        div[data-testid="stPlotlyChart"] .svg-container,
        div[data-testid="stPlotlyChart"] svg.main-svg {
            border-radius: 17px;
        }

        /* Keep the Plotly modebar fully hidden until the pointer enters the plot. */
        div[data-testid="stPlotlyChart"] .modebar {
            top: 10px !important;
            right: 10px !important;
            padding: 3px 5px !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            background: transparent !important;
            box-shadow: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
            transition:
                opacity .16s ease,
                visibility .16s ease,
                background-color .16s ease,
                border-color .16s ease,
                box-shadow .16s ease !important;
        }

        /* Reveal controls only while hovering anywhere inside the Plotly canvas. */
        div[data-testid="stPlotlyChart"] .js-plotly-plot:hover .modebar,
        div[data-testid="stPlotlyChart"] .plot-container:hover .modebar {
            opacity: 1 !important;
            visibility: visible !important;
            pointer-events: auto !important;
            border-color: #E6E8EF !important;
            background: rgba(255,255,255,.96) !important;
            box-shadow: 0 5px 16px rgba(16,24,40,.08) !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-btn {
            border-radius: 7px !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-btn:hover {
            background: #F4EAF8 !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-btn path {
            fill: #667085 !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-btn:hover path {
            fill: #660094 !important;
        }


        @media (max-width: 700px) {
            div[data-testid="stPlotlyChart"] {
                border-radius: 17px;
                padding: 7px 6px 4px 6px;
            }

        }
        </style>
        """,
        unsafe_allow_html=True,
    )

render_chart_shell()

# ---------------- PERCENT AXIS / STANDARD HEIGHT HELPERS ----------------
CHART_HEIGHT_VERTICAL = 410
CHART_HEIGHT_HORIZONTAL = 410


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


# ---------------- PERCENTAGE METHODOLOGY NOTE ----------------
PERCENTAGE_CHART_DISCLAIMER = (
    "Percentages are calculated from the category occurrences displayed. "
    "Where a record can be associated with more than one category, the same "
    "record may contribute to multiple categories; therefore, percentages "
    "may not sum to 100%."
)

def _mark_percentage_chart(fig, disclaimer=PERCENTAGE_CHART_DISCLAIMER):
    """Mark a figure so the central dashboard renderer displays the methodology badge."""
    if fig is None:
        return fig
    try:
        meta = dict(fig.layout.meta or {})
    except Exception:
        meta = {}
    meta["eusee_percentage_chart"] = True
    meta["eusee_percentage_disclaimer"] = disclaimer
    fig.update_layout(meta=meta)
    return fig


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
            lambda l: wrap_label_by_words(l, words_per_line=4)
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
        width=0.85,
        textposition=[
            "inside" if val >= (axis_max * 0.12) else "outside"
            for val in df["percent_value"]
        ],
        insidetextanchor="end",
        texttemplate="%{text}",
        textfont=dict(size=10, color="#1F2937", family=CHART_FONT),
        marker_line=dict(color="rgba(255,255,255,0.75)", width=0.8),
        hovertemplate=(
            "<b>%{y}</b><br>" if horizontal else "<b>%{x}</b><br>"
        ) + "Share: %{customdata[1]:.1f}%<br>Count: %{customdata[0]:,.0f}<extra></extra>",
    )

    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_xaxes(
            title="",
            ticksuffix="%",
            range=[0, axis_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_yaxes(
            title="",
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
        fig.update_xaxes(title="", ticksuffix="%", range=[0, axis_max])
    else:
        fig.update_yaxes(title="", ticksuffix="%", range=[0, axis_max])

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

# ---------------- STACKED BAR LABEL CONTRAST HELPER --------------
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
            width=0.85,
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
            title="",
            ticksuffix="%",
            range=[0, axis_max],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", title=None)
        fig.update_yaxes(
            title="",
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
        fig.update_xaxes(title="", ticksuffix="%", range=[0, axis_max])
    else:
        fig.update_yaxes(title="", ticksuffix="%", range=[0, axis_max])

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

    # Mark only stacked-bar charts for the percentage methodology note.
    fig = _mark_percentage_chart(fig)
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
        font-family: "Anek Devanagari", Arial, sans-serif;
        font-size: 15px;
        font-weight: 900;
        color: #2D0055;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }}
    .analytics-panel-subtitle {{
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
        font-size: 10.5px;
        font-weight: 800;
    }}
    .chart-card-caption {{
        font-family: var(--eusee-font);
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
    .flow-panel-eyebrow {font-family: "Anek Devanagari", Arial, sans-serif; font-size: 10.5px; font-weight: 900; letter-spacing: .10em; text-transform: uppercase; color: #008CAA; margin-bottom: 4px;}
    .flow-panel-title {font-family: "Anek Devanagari", Arial, sans-serif; font-size: 19px; font-weight: 950; color: #2D0055; margin-bottom: 4px;}
    .flow-panel-subtitle {font-family: "Anek Devanagari", Arial, sans-serif; font-size: 12px; color: #64748B; line-height: 1.45; max-width: 980px; margin-bottom: 12px;}
    .flow-panel-badges {display: flex; flex-wrap: wrap; gap: 7px; margin: 8px 0 4px 0;}
    .flow-panel-badge {background: #F3ECF8; border: 1px solid #E1D2EC; border-radius: 999px; padding: 5px 9px; font-family: "Anek Devanagari", Arial, sans-serif; font-size: 10.8px; font-weight: 800; color: #4B006E;}
    .flow-guide-card {background: #FFFFFF; border: 1px solid #E8EEF3; border-radius: 16px; padding: 11px 13px; min-height: 76px; box-shadow: 0 5px 16px rgba(15, 23, 42, 0.045);}
    .flow-guide-title {font-family: "Anek Devanagari", Arial, sans-serif; font-size: 11.8px; font-weight: 900; color: #2D0055; margin-bottom: 3px;}
    .flow-guide-text {font-family: "Anek Devanagari", Arial, sans-serif; font-size: 10.8px; color: #64748B; line-height: 1.35;}
    .flow-section-label {
        font-family: "Anek Devanagari", Arial, sans-serif;
        font-size: 15px;
        font-weight: 850;
        color: #101828;
        line-height: 1.25;
        letter-spacing: -0.015em;
        margin: 16px 0 6px 0;
    }
    .flow-section-note {
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: "Anek Devanagari", Arial, sans-serif;
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
        font-family: var(--eusee-font);
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
# only when its permission is enabled; restricted visuals are omitted completely.
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
    """Render a chart/widget only when permitted; otherwise hide it completely."""
    if can_render_feature(permission_key):
        return render_fn()

    # Restricted plots and widgets are intentionally omitted. Do not render a
    # "Restricted feature" placeholder because it adds visual clutter and
    # exposes unavailable dashboard components to restricted users.
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
        font-family: "Anek Devanagari", Arial, sans-serif !important;
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
        font-family: "Anek Devanagari", Arial, sans-serif !important;
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


# ---------------- FINAL MOBILE RESPONSIVE HARDENING ----------------
def inject_final_mobile_responsive_hardening() -> None:
    """Final cascade-level safeguards for phones, tablets, and narrow browser windows."""
    st.markdown(
        """
        <style>
        /* Global sizing and overflow safety */
        *, *::before, *::after {
            box-sizing: border-box;
        }

        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main, .main .block-container {
            width: 100%;
            max-width: 100%;
            overflow-x: clip !important;
        }

        .main .block-container {
            width: min(100%, 1500px) !important;
            padding-left: clamp(.70rem, 2.2vw, 1.50rem) !important;
            padding-right: clamp(.70rem, 2.2vw, 1.50rem) !important;
        }

        /* Prevent long labels, links and generated text from widening cards */
        .main p, .main span, .main div, .main a,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            overflow-wrap: anywhere;
            word-break: normal;
        }

        img, video, canvas, svg, iframe {
            max-width: 100% !important;
        }

        img, video {
            height: auto !important;
        }

        /* Streamlit rows and columns */
        div[data-testid="stHorizontalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
            align-items: stretch !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0 !important;
            max-width: 100% !important;
        }

        /* Forms and controls */
        [data-testid="stTextInput"],
        [data-testid="stNumberInput"],
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"],
        [data-testid="stDateInput"],
        [data-testid="stTextArea"],
        [data-testid="stFileUploader"],
        [data-testid="stDownloadButton"],
        [data-testid="stButton"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }

        [data-baseweb="select"],
        [data-baseweb="input"],
        [data-baseweb="textarea"],
        [data-baseweb="popover"] {
            max-width: 100% !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            max-width: 100% !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            min-height: 38px;
            height: auto !important;
        }

        /* Tabs remain usable without forcing the entire page wider */
        div[data-testid="stTabs"] {
            min-width: 0 !important;
            max-width: 100% !important;
        }

        div[data-testid="stTabs"] div[role="tablist"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            min-width: 0 !important;
            max-width: 100% !important;
        }

        /* Charts and maps */
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stDeckGlChart"],
        div[data-testid="stPydeckChart"],
        div[data-testid="stMap"],
        .stPlotlyChart,
        .js-plotly-plot,
        .plot-container,
        .svg-container {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }

        .js-plotly-plot .plotly .modebar {
            max-width: calc(100% - 8px) !important;
            flex-wrap: wrap !important;
        }

        /* Tables scroll inside their own container */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }

        div[data-testid="stDataFrame"] [role="grid"] {
            max-width: 100% !important;
            overflow: auto !important;
            -webkit-overflow-scrolling: touch;
        }

        /* Generic dashboard cards and custom HTML containers */
        .classic-filter-header,
        .classic-filter-status,
        .data-preview-toolbar,
        .executive-table-shell,
        .executive-table-header,
        .executive-table-status,
        .eusee-login-route-shell,
        .map-intel-hero,
        .eusee-kpi-card {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }

        .executive-table-header,
        .executive-table-status,
        .data-preview-toolbar {
            flex-wrap: wrap !important;
        }

        .executive-metric-grid {
            grid-template-columns: repeat(auto-fit, minmax(min(145px, 100%), 1fr)) !important;
        }

        /* Tablet */
        @media (max-width: 900px) {
            header[data-testid="stHeader"] {
                height: 44px !important;
                min-height: 44px !important;
            }

            .main .block-container {
                padding-top: .65rem !important;
            }

            .executive-table-header,
            .executive-table-status,
            .data-preview-toolbar {
                flex-direction: column !important;
                align-items: stretch !important;
            }

            .data-preview-pill-row {
                justify-content: flex-start !important;
            }

            .eusee-kpi-card {
                height: auto !important;
                min-height: 0 !important;
            }
        }

        /* Phones */
        @media (max-width: 640px) {
            .main .block-container {
                padding-left: .65rem !important;
                padding-right: .65rem !important;
                padding-bottom: 5.5rem !important;
            }

            section[data-testid="stSidebar"] {
                width: min(92vw, 360px) !important;
                max-width: 92vw !important;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                flex-wrap: nowrap !important;
                gap: .70rem !important;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
            }

            /* Collapse custom inline flex/grid layouts that lack dedicated classes */
            .main [style*="display:flex"],
            .main [style*="display: flex"] {
                max-width: 100% !important;
                flex-wrap: wrap !important;
            }

            .main [style*="display:grid"],
            .main [style*="display: grid"] {
                max-width: 100% !important;
                grid-template-columns: minmax(0, 1fr) !important;
            }

            .main [style*="width:"][style*="px"],
            .main [style*="min-width:"][style*="px"] {
                min-width: 0 !important;
                max-width: 100% !important;
            }

            div[data-testid="stTabs"] div[role="tablist"] {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }

            div[data-testid="stTabs"] button[role="tab"] {
                min-height: 36px !important;
                height: auto !important;
                max-height: none !important;
                padding: 6px 7px !important;
                white-space: normal !important;
                line-height: 1.12 !important;
                text-overflow: clip !important;
            }

            .animated-title {
                font-size: clamp(24px, 9vw, 34px) !important;
                line-height: 1.05 !important;
            }

            .animated-subtitle {
                font-size: 12px !important;
                line-height: 1.35 !important;
            }

            .executive-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }

            .executive-table-shell,
            .map-intel-hero,
            .eusee-login-route-shell {
                padding: 12px !important;
                border-radius: 14px !important;
            }

            #eusee-collapsed-sidebar-label {
                max-width: calc(100vw - 90px) !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
        }

        /* Very narrow phones */
        @media (max-width: 420px) {
            div[data-testid="stTabs"] div[role="tablist"] {
                grid-template-columns: minmax(0, 1fr) !important;
            }

            .executive-metric-grid {
                grid-template-columns: minmax(0, 1fr) !important;
            }

            .data-preview-pill-row {
                display: grid !important;
                grid-template-columns: minmax(0, 1fr) !important;
                width: 100% !important;
            }

            .data-preview-pill,
            .executive-table-badge {
                width: 100% !important;
                white-space: normal !important;
                text-align: center !important;
            }

            #eusee-collapsed-sidebar-label {
                font-size: 0 !important;
                width: 38px !important;
                min-width: 38px !important;
                padding: 0 !important;
                justify-content: center !important;
            }

            #eusee-collapsed-sidebar-label::after {
                content: "☰";
                font-size: 17px;
                line-height: 1;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_final_mobile_responsive_hardening()

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
        font-family: "Anek Devanagari", Arial, sans-serif !important;
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
        font-family: "Anek Devanagari", Arial, sans-serif !important;
        font-size: clamp(8px, .66vw, 9.2px) !important;
        font-weight: 760 !important;
        letter-spacing: -0.025em !important;
    }
    </style>
    """, unsafe_allow_html=True)
inject_plotly_legend_color_spacing_fix()


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
        # Hide restricted plots completely instead of displaying the
        # "Restricted feature" access panel.
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

    # Apply one consistent premium visual treatment to every dashboard plot.
    current_title = title or fig.layout.title.text or ""

    fig = apply_classic_chart_theme(
        fig,
        title=current_title,
        height=fig.layout.height,
        horizontal=False,
        showlegend=bool(fig.layout.showlegend)
        if fig.layout.showlegend is not None
        else any(getattr(trace, "showlegend", True) for trace in fig.data),
    )

    fig = apply_responsive_plotly_layout(fig)

    # Preserve the left-aligned premium title after responsive adjustments.
    if current_title:
        fig.update_layout(
            title=dict(
                text=current_title,
                x=0.018,
                xanchor="left",
                y=0.955,
                yanchor="top",
                pad=dict(t=0, b=10),
                font=dict(
                    family=CHART_FONT,
                    size=16,
                    color="#16002B",
                ),
            ),
            margin=dict(t=94),
        )

    # Percentage methodology: show a compact information icon inside the
    # chart. The full disclaimer appears only while the mouse is over the
    # icon, then disappears automatically when the mouse leaves.
    try:
        chart_meta = dict(fig.layout.meta or {})
    except Exception:
        chart_meta = {}

    if chart_meta.get("eusee_percentage_chart"):
        try:
            disclaimer = chart_meta.get(
                "eusee_percentage_disclaimer",
                PERCENTAGE_CHART_DISCLAIMER,
            )

            # Remove any previous percentage disclaimer annotation so reruns
            # never accumulate duplicate icons.
            existing_annotations = list(fig.layout.annotations or [])
            existing_annotations = [
                ann for ann in existing_annotations
                if "Percentage calculation" not in str(getattr(ann, "hovertext", "") or "")
            ]

            wrapped_disclaimer = _wrap_chart_tooltip_text(
                disclaimer,
                line_length=78,
            )

            # Small, unobtrusive information icon in the chart title area.
            # Hovering it opens the methodology note; moving away closes it.
            existing_annotations.append(
                dict(
                    x=0.985,
                    y=0.965,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="top",
                    text="ⓘ",
                    hovertext=(
                        "<b>Percentage calculation</b><br>"
                        + wrapped_disclaimer
                    ),
                    hoverlabel=dict(
                        bgcolor="#FFFFFF",
                        bordercolor="#D9DDE7",
                        font=dict(
                            family=CHART_FONT,
                            size=11,
                            color="#344054",
                        ),
                        align="left",
                    ),
                    showarrow=False,
                    bgcolor="rgba(244,234,248,0.96)",
                    bordercolor="#E7D4F1",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(
                        family=CHART_FONT,
                        size=12,
                        color="#660094",
                    ),
                    captureevents=True,
                )
            )

            fig.update_layout(annotations=existing_annotations)
        except Exception:
            pass

    final_config = DEFAULT_PLOTLY_CONFIG.copy()
    if config:
        final_config.update(config)

    # Keep the modebar hidden consistently unless a chart explicitly needs it.
    final_config["displayModeBar"] = "hover"
    final_config["scrollZoom"] = False
    final_config["displaylogo"] = False

    target.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config=final_config,
        key=key,
    )
# ---------------- TAB 1 ------------------------
if tab_overview is not None:
    with tab_overview:
        st.markdown(
               """
                <div class="cfr-page-subtitle">
                       Explore how alerts are distributed across alert types, enabling environment principles, and time. <br> 
                       Use the global filters to refine the view and update the dashboard.
                </div>
              
               """,
               unsafe_allow_html=True,
           )
        

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
                permission_key="view_chart_overview_enabling_principles",
                permission_label="Overview enabling-principle distribution",
            )
  
            #r1c2.plotly_chart(create_h_stacked_bar(a2,y="enabling-principle",x="count",color_col="alert-impact",title="Alert distribution across enabling principles", horizontal=True),use_container_width=True,  key="tab1_chart2")

            #if is_privileged():
            render_dashboard_plotly_chart(create_h_stacked_bar(a3,y="region",x="count",color_col="alert-impact",title="Alert distribution across regions", horizontal=False, normalize_labels=False), plot_df=a3, visual_type="stacked bar chart", x_col="region", group_col="alert-impact", dashboard_df=filtered_global, key="tab1_chart3", container=r2c1, permission_key="view_chart_overview_regions", permission_label="Overview regional distribution")
            render_dashboard_plotly_chart(create_h_stacked_bar(a4,y="alert-country",x="count",color_col="alert-impact",title="Alert distribution across countries", horizontal=False, normalize_labels=False), plot_df=a4, visual_type="stacked bar chart", x_col="alert-country", group_col="alert-impact", dashboard_df=filtered_global, key="tab1_chart4", container=r2c2, permission_key="view_chart_overview_countries", permission_label="Overview country distribution")

    
            cols_rename_map  = {
                "post_title": "Title of post",
                "summary": "Event Summary",
                "creation_date": "Date of submission",
                "alert-country": "Country",
                "enabling-principle": "Enabling principles",
                "alert-impact": "Impact of alert",
                "alert-type": "Type of alert",
                "Permalink": "Report Link"
            }
                # keep only existing columns, then rename
            overview_table_df = (
                data
                .loc[:, [c for c in cols_rename_map.keys() if c in data.columns]]
                .rename(columns=cols_rename_map)
            )
   
                # ---------------- Tab two data preview ------------------

            if has_permission("view_data_table"):
                render_professional_data_preview(
                    overview_table_df,
                    title="Search and export EU SEE alerts",
                    key="overview_summary_data_preview",
                    remove_vertical_scroll=True,
                )
            #else:
                #st.info("Sign in with an authorized account to unlock additional detailed and disaggregated data.")   
        
        # ---------------- Negative Events ----------------
        else:
            render_access_locked("Overview", "public-summary or viewer")

if tab_negative is not None:
    with tab_negative:
        
        st.markdown(
                    """
                        <div class="cfr-page-subtitle">
                            Explore patterns across negative alerts, including restrictive actors,mechanisms, and affected civil society actors.<br>
                            Use the global filters to refine the view and update the dashboard.
                        </div>
                    
                    """,
                    unsafe_allow_html=True,
           )
     

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
    

                # ---------------- NEGATIVE ALERTS FILTERS: ADMIN ONLY ----------------
                # Empty selections preserve the full Negative Alerts dataset for
                # viewers who are allowed to see the tab but not the filter panel.
                selected_actor_types = []
                selected_subject_types = []
                selected_mechanism_types = []
                selected_event_types = []

                if has_permission("view_negative_alert_filters"):
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

    
                render_dashboard_plotly_chart(create_bar_chart(m1, "Actor of repression", "count",title="Types of restrictive actors", horizontal=True, normalize_labels=True), plot_df=m1, visual_type="bar chart", x_col="Actor of repression", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart1", container=r1c1, permission_key="view_chart_negative_restrictive_actors", permission_label="Restrictive actors chart")
                render_dashboard_plotly_chart(create_bar_chart(m2, "Subject of repression", "count",title="Types of civil society actors affected", horizontal=True, normalize_labels=True), plot_df=m2, visual_type="bar chart", x_col="Subject of repression", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart2", container=r1c2, permission_key="view_chart_negative_affected_actors", permission_label="Civil society actors affected chart")
                render_dashboard_plotly_chart(create_bar_chart(m3, "Mechanism of repression", "count",title="Types of restrictive mechanisms", horizontal=True, normalize_labels=True), plot_df=m3, visual_type="bar chart", x_col="Mechanism of repression", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart3", container=r1c3, permission_key="view_chart_negative_restrictive_mechanisms", permission_label="Restrictive mechanisms chart")
                render_dashboard_plotly_chart(create_bar_chart(m4, "Type of event", "count",title="Types of negative events", horizontal=True, normalize_labels=True), plot_df=m4, visual_type="bar chart", x_col="Type of event", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart4", container=r2c1, permission_key="view_chart_negative_event_types", permission_label="Negative event types chart")
                render_dashboard_plotly_chart(create_bar_chart(m5, "alert-type", "count",title="Distribution of negative alert types", horizontal=True, normalize_labels=True), plot_df=m5, visual_type="bar chart", x_col="alert-type", group_col="alert-impact", dashboard_df=reactive_df_updated, key="tab2_chart5", container=r2c2, permission_key="view_chart_negative_alert_types", permission_label="Negative alert types chart")
                render_dashboard_plotly_chart(create_bar_chart(m6, "enabling-principle", "count", title="Negative alert distribution across enabling principles", horizontal=True, normalize_labels=False),plot_df=m6,visual_type="bar chart",x_col="enabling-principle",group_col="alert-impact",dashboard_df=reactive_df_updated,key="tab2_chart6",container=r2c3, permission_key="view_chart_negative_enabling_principles",permission_label="Negative enabling-principle distribution")
             

              
                # ---------------- ANALYTICAL FLOW PANEL ----------------
                if has_permission("view_analytical_flow_panel"):
                    render_analytical_flow_panel(filtered_df)
                #else:
                    #st.info("Analytical Flow Panel is disabled for your current access level.")

                cols_to_keep = {
                    "post_title": "Title of post",
                    "creation_date": "Date of submission",                    
                    "summary": "Event Summary",
                    "alert-country": "Country",
                    "enabling-principle": "Enabling principles",
                    "alert-impact": "Impact of alert",
                    "alert-type": "Type of alert",
                    "Actor of repression": "Types of restrictive actors",
                    "Subject of repression": "Types of civil society actors affected",
                    "Mechanism of repression": "Types of restrictive mechanisms",
                    "Type of event": "Types of negative events",
                    "Permalink": "Report Link"          
                }
                # Build the Negative Alerts table independently of all global and
                # tab-specific filters. The table always loads the full accessible
                # dataset and applies only the Negative alert-impact condition.
                negative_table_df = data.copy()
                if "alert-impact" in negative_table_df.columns:
                    negative_table_df = negative_table_df[
                        negative_table_df["alert-impact"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq("negative")
                    ].copy()
                else:
                    negative_table_df = negative_table_df.iloc[0:0].copy()

                negative_table_preview = (
                    negative_table_df
                    .loc[:, negative_table_df.columns.intersection(cols_to_keep.keys())]
                    .rename(columns=cols_to_keep)
                )
        
                # ---------------- Tab two data preview ----------------
                if has_permission("view_data_table"):
                    render_professional_data_preview(
                        negative_table_preview,
                        title="Search and export EU SEE alerts",
                        key="negative_summary_data_preview",
                    )
           
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

                MAP_FONT = PLOTLY_FONT_FAMILY

                st.markdown("""
                <style>
                .map-page-shell {
                    background: transparent;
                    border: 0;
                    border-radius: 0;
                    padding: 0;
                    margin: 2px 0 8px 0;
                    box-shadow: none;
                    font-family: "Anek Devanagari", Arial, sans-serif;
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
                    font-family: var(--eusee-font);
                    font-size: 9.5px;
                    font-weight: 850;
                    letter-spacing: .105em;
                    text-transform: uppercase;
                    color: #660094;
                    margin-bottom: 5px;
                    line-height: 1.15;
                }
                .map-intel-title {
                    font-family: var(--eusee-font);
                    font-size: 14px;
                    font-weight: 850;
                    color: #101828;
                    margin-bottom: 5px;
                    letter-spacing: -0.025em;
                    line-height: 1.15;
                }
                .map-intel-subtitle {
                    font-family: var(--eusee-font);
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
                    font-family: "Anek Devanagari", Arial, sans-serif;
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
                    font-family:"Anek Devanagari", Arial, sans-serif;
                }
                .map-panel-card {
                    background:#FFFFFF;
                    border:1px solid #E8EAF0;
                    border-radius:18px;
                    padding:12px 13px;
                    box-shadow:0 10px 22px rgba(17,24,39,.052);
                    margin: 6px 0 10px 0;
                    font-family:"Anek Devanagari", Arial, sans-serif;
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
                    font-family:"Anek Devanagari", Arial, sans-serif;
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
                    font-family:"Anek Devanagari", Arial, sans-serif;
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
                    font-family:"Anek Devanagari", Arial, sans-serif;
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
                            font-family: "Anek Devanagari", Arial, sans-serif;
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
                            font-family: "Anek Devanagari", Arial, sans-serif;
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
                            font-family: var(--eusee-font) !important;
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
                            mapbox_style="open-street-map",
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

# ---------------- CFR ANALYSIS TAB ----------------
if tab_cfr is not None:
    with tab_cfr:
        st.markdown(
               """
                <div class="cfr-page-subtitle">
                       Explore score patterns across the six EU SEE enabling
                       environment principles.<br>
                       Scores range from 1 (most restricted) to 5 (most enabling).
                </div>
              
               """,
               unsafe_allow_html=True,
           )

        if has_permission("view_overview"):
            render_cfr_analysis()
        else:
            render_access_locked("CFR Analysis", "viewer or privileged")


if tab_manual is not None:
    with tab_manual:
        if has_permission("view_user_manual"):
            footer_image_path = BASE_DIR / "assets" / "footer_logo.png"
            footer_b64 = ""

            if footer_image_path.exists():
                footer_b64 = base64.b64encode(
                    footer_image_path.read_bytes()
                ).decode("utf-8")

            st.html(
                textwrap.dedent("""
                <style>
                /* =========================================================
                   USER MANUAL — REFERENCE-STYLE QUICK GUIDE
                ========================================================= */
                .user-guide-shell {
                    width: 100%;
                    padding: 5px 4px 10px 4px;
                    font-family: "Anek Devanagari", Arial, sans-serif;
                }

                .user-guide-intro {
                    max-width: 760px;
                    margin: 8px 0 28px 0;
                    color: black;
                    font-size: 13px;
                    line-height: 1.45;
                    font-weight: 550;
                }

                .user-guide-grid {
                    display: grid;
                    grid-template-columns: minmax(0, 1.48fr) minmax(320px, 1fr);
                    gap: 20px;
                    align-items: start;
                }

                .user-guide-card {
                    background: #FFFFFF;
                    border: 1px solid #E7EAF1;
                    border-radius: 16px;
                    box-shadow:
                        0 8px 22px rgba(16, 24, 40, 0.045),
                        0 1px 3px rgba(16, 24, 40, 0.035);
                }

                .workflow-card {
                    padding: 14px 17px 12px 17px;
                    min-height: 414px;
                }

                .user-guide-card-title {
                    color: #182158;
                    font-size: 17px;
                    line-height: 1.15;
                    font-weight: 950;
                    margin: 0;
                }

                .user-guide-title-line {
                    width: 64px;
                    height: 3px;
                    margin: 7px 0 6px 0;
                    border-radius: 999px;
                    background: #660094;
                }

                .user-guide-card-note {
                    color: #6B7280;
                    font-size: 10.5px;
                    line-height: 1.35;
                    margin-bottom: 5px;
                    font-weight: 550;
                }

                .workflow-step {
                    display: grid;
                    grid-template-columns: 29px 42px minmax(0, 1fr);
                    gap: 11px;
                    align-items: center;
                    min-height: 64px;
                    padding: 7px 0;
                    border-bottom: 1px solid #ECEEF4;
                }

                .workflow-step:last-of-type {
                    border-bottom: 0;
                }

                .workflow-number {
                    width: 25px;
                    height: 25px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    color: #FFFFFF;
                    background: #660094;
                    font-size: 12px;
                    font-weight: 950;
                    box-shadow: 0 3px 8px rgba(102, 0, 148, 0.18);
                }

                .workflow-icon {
                    width: 38px;
                    height: 38px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    color: #7A22A1;
                    background: #F6F0FA;
                    border: 1px solid #EADCF2;
                    font-size: 18px;
                    line-height: 1;
                }

                .workflow-copy {
                    min-width: 0;
                }

                .workflow-title {
                    color: #222A57;
                    font-size: 12px;
                    line-height: 1.2;
                    font-weight: 900;
                    margin-bottom: 2px;
                }

                .workflow-text {
                    color: #667085;
                    font-size: 10.7px;
                    line-height: 1.32;
                    font-weight: 520;
                }

                .citation-box {
                    margin: 8px 73px 0 73px;
                    padding: 9px 11px;
                    border-radius: 8px;
                    background: #F5EEFA;
                    color: #344054;
                    font-size: 10.3px;
                    line-height: 1.35;
                }

                .citation-box strong {
                    color: #253B80;
                    font-weight: 900;
                }

                .user-guide-side {
                    display: grid;
                    gap: 14px;
                }

                .help-card {
                    display: grid;
                    grid-template-columns: 58px minmax(0, 1fr);
                    gap: 15px;
                    align-items: center;
                    min-height: 122px;
                    padding: 18px 20px;
                    background:
                        radial-gradient(circle at 8% 48%, rgba(63, 105, 255, 0.08), transparent 30%),
                        linear-gradient(135deg, #FFFFFF 0%, #F8FBFF 100%);
                }

                .help-icon-wrap {
                    width: 52px;
                    height: 52px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    background: #EEF4FF;
                    border: 1px solid #D8E5FF;
                    box-shadow: inset 0 0 0 8px rgba(255, 255, 255, 0.65);
                }

                .help-icon {
                    width: 27px;
                    height: 27px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    border: 2px solid #4263EB;
                    color: #4263EB;
                    font-size: 17px;
                    font-weight: 950;
                    line-height: 1;
                }

                .help-title {
                    color: #182158;
                    font-size: 15px;
                    line-height: 1.2;
                    font-weight: 950;
                    margin-bottom: 5px;
                }

                .help-text {
                    color: #667085;
                    font-size: 10.8px;
                    line-height: 1.4;
                }

                .brand-card {
                    min-height: 203px;
                    padding: 18px 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .brand-card img {
                    display: block;
                    width: 100%;
                    max-width: 470px;
                    height: auto;
                    max-height: 190px;
                    object-fit: contain;
                }

                .brand-fallback {
                    width: 100%;
                    min-height: 150px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #660094;
                    font-size: 22px;
                    font-weight: 950;
                    text-align: center;
                }

                @media (max-width: 980px) {
                    .user-guide-grid {
                        grid-template-columns: 1fr;
                    }

                    .workflow-card {
                        min-height: auto;
                    }

                    .user-guide-side {
                        grid-template-columns: 1fr 1fr;
                    }
                }

                @media (max-width: 700px) {
                    .user-guide-shell {
                        padding-top: 14px;
                    }

                    .user-guide-intro {
                        margin-bottom: 18px;
                        font-size: 12px;
                    }

                    .user-guide-side {
                        grid-template-columns: 1fr;
                    }

                    .workflow-step {
                        grid-template-columns: 27px 38px minmax(0, 1fr);
                        gap: 8px;
                    }

                    .workflow-icon {
                        width: 34px;
                        height: 34px;
                        font-size: 16px;
                    }

                    .citation-box {
                        margin-left: 0;
                        margin-right: 0;
                    }
                }

                @media (max-width: 430px) {
                    .workflow-card,
                    .help-card,
                    .brand-card {
                        padding-left: 12px;
                        padding-right: 12px;
                    }

                    .workflow-step {
                        grid-template-columns: 25px 1fr;
                    }

                    .workflow-icon {
                        display: none;
                    }

                    .help-card {
                        grid-template-columns: 48px 1fr;
                    }

                    .help-icon-wrap {
                        width: 44px;
                        height: 44px;
                    }
                }
                </style>
                """)
            )

            logo_html = (
                f'<img src="data:image/png;base64,{footer_b64}" '
                'alt="EU SEE and partner logos">'
                if footer_b64
                else '<div class="brand-fallback">EU SEE<br>Partner Network</div>'
            )

            st.html(
                textwrap.dedent(f"""
                <div class="user-guide-shell">
                    <div class="user-guide-intro">
                        A quick guide to help you navigate the dashboard, apply filters,
                        interpret the visualisations,<br> explore key patterns, review the
                        underlying data, and export results.
                    </div>

                    <div class="user-guide-grid">
                        <section class="user-guide-card workflow-card">
                            <div class="user-guide-card-title">Quick-start workflow</div>
                            <div class="user-guide-title-line"></div>
                            <div class="user-guide-card-note">
                                Recommended steps for first-time users.
                            </div>

                            <div class="workflow-step">
                                <div class="workflow-number">1</div>
                                <div class="workflow-icon">▽</div>
                                <div class="workflow-copy">
                                    <div class="workflow-title">Set your scope</div>
                                    <div class="workflow-text">
                                        Use the global filters to select the region, country,
                                        alert impact, nature of alert, enabling environment
                                        principle, year, and month.
                                    </div>
                                </div>
                            </div>

                            <div class="workflow-step">
                                <div class="workflow-number">2</div>
                                <div class="workflow-icon">▥</div>
                                <div class="workflow-copy">
                                    <div class="workflow-title">Start with the Alerts Overview</div>
                                    <div class="workflow-text">
                                        Review the main figures and charts to understand the
                                        main patterns in the filtered data.
                                    </div>
                                </div>
                            </div>

                            <div class="workflow-step">
                                <div class="workflow-number">3</div>
                                <div class="workflow-icon">⌁</div>
                                <div class="workflow-copy">
                                    <div class="workflow-title">Explore patterns in greater detail</div>
                                    <div class="workflow-text">
                                        Use the Alerts Overview, CFR Scores, and Negative Alerts
                                        Analysis sections to identify trends and better understand
                                        the filtered data.
                                    </div>
                                </div>
                            </div>

                            <div class="workflow-step">
                                <div class="workflow-number">4</div>
                                <div class="workflow-icon">▤</div>
                                <div class="workflow-copy">
                                    <div class="workflow-title">Review the data in detail, if available</div>
                                    <div class="workflow-text">
                                        Privileged users can use the Data Summary Preview and AI
                                        Assistant to search, review, further analyse, and export data.
                                    </div>
                                </div>
                            </div>

                            <div class="workflow-step">
                                <div class="workflow-number">5</div>
                                <div class="workflow-icon">◌</div>
                                <div class="workflow-copy">
                                    <div class="workflow-title">Cite the dashboard</div>
                                    <div class="workflow-text">
                                        When using data, charts, or findings from the dashboard,
                                        cite the EU SEE Dashboard and the relevant visualisation,
                                        including the date of access or consultation.
                                    </div>
                                </div>
                            </div>

                            <div class="citation-box">
                                <strong>Suggested citation:</strong>
                                EU SEE Dashboard, “[Name of visualisation]”, accessed [date].
                            </div>
                        </section>

                        <aside class="user-guide-side">
                            <section class="user-guide-card help-card">
                                <div class="help-icon-wrap">
                                    <div class="help-icon">?</div>
                                </div>
                                <div>
                                    <div class="help-title">Need help?</div>
                                    <div class="help-text">
                                        Use the Feedback button to share questions, suggestions,
                                        or report issues.
                                    </div>
                                </div>
                            </section>

                            <section class="user-guide-card brand-card">
                                {logo_html}
                            </section>
                        </aside>
                    </div>
                </div>
                """)
            )
        else:
            render_access_locked("User Manual", "guest or higher")

# ============================================================
# EU SEE OPENAI AI CHATBOT
# Lightweight, dataset-grounded Copilot with natural-language filtering.
# UI remains the existing EU SEE AI Assistant popover.
# ============================================================

OPENAI_CLIENT = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENAI_MODEL = str(
    st.secrets.get("openai", {}).get(
        "OPENAI_MODEL",
        "gpt-5.6-luna",
    )
).strip() or "gpt-5.6-luna"


@st.cache_resource(show_spinner=False)
def _get_eusee_openai_client():
    if OpenAI is None:
        return None

    api_key = str(
        st.secrets.get("openai", {}).get(
            "OPENAI_API_KEY",
            "",
        )
    ).strip()

    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ---------------- CHAT HISTORY PERSISTENCE ----------------
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_LIMIT = 100


def _current_chat_user_key() -> str:
    """Return a stable privacy-safe key for the active user chat history."""
    email = str(st.session_state.get("email") or "").lower().strip()

    if email:
        identity = f"user::{email}"
    else:
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
            "created_at": str(
                msg.get("created_at")
                or datetime.utcnow().isoformat(timespec="seconds") + "Z"
            ),
        })

    return clean_messages[-CHAT_HISTORY_LIMIT:]


def load_user_chat_history(force: bool = False) -> list[dict]:
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
            payload = json.loads(
                history_file.read_text(encoding="utf-8")
            )
            messages = _normalise_chat_messages(
                payload.get("messages", [])
            )
        except Exception:
            messages = []

    st.session_state.eusee_chat_user_key = user_key
    st.session_state.eusee_chat_messages = messages
    st.session_state.eusee_chat_history_loaded = True
    st.session_state.eusee_chat_session_id = user_key[:32]

    return messages


def save_user_chat_history() -> None:
    user_key = (
        st.session_state.get("eusee_chat_user_key")
        or _current_chat_user_key()
    )

    messages = _normalise_chat_messages(
        st.session_state.get(
            "eusee_chat_messages",
            [],
        )
    )

    st.session_state.eusee_chat_messages = messages

    payload = {
        "user_key": user_key,
        "email": str(
            st.session_state.get("email") or ""
        ).lower().strip(),
        "updated_at": datetime.utcnow().isoformat(
            timespec="seconds"
        ) + "Z",
        "message_count": len(messages),
        "messages": messages,
    }

    try:
        history_file = _chat_history_path(user_key)
        tmp_file = history_file.with_suffix(".tmp")
        tmp_file.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp_file.replace(history_file)
    except Exception as exc:
        if st.secrets.get("debug", {}).get(
            "show_chat_history_errors",
            False,
        ):
            st.warning(
                f"Chat history could not be saved: {exc}"
            )


def append_user_chat_message(role: str, content: str) -> None:
    load_user_chat_history()

    st.session_state.eusee_chat_messages.append({
        "id": uuid.uuid4().hex,
        "role": str(
            role or "assistant"
        ).lower().strip(),
        "content": str(
            content or ""
        ).strip(),
        "created_at": datetime.utcnow().isoformat(
            timespec="seconds"
        ) + "Z",
    })

    st.session_state.eusee_chat_messages = (
        _normalise_chat_messages(
            st.session_state.eusee_chat_messages
        )
    )

    save_user_chat_history()


def clear_user_chat_history() -> None:
    user_key = (
        st.session_state.get("eusee_chat_user_key")
        or _current_chat_user_key()
    )

    st.session_state.eusee_chat_messages = []
    st.session_state.eusee_chat_history_loaded = True

    try:
        history_file = _chat_history_path(user_key)
        if history_file.exists():
            history_file.unlink()
    except Exception:
        pass


load_user_chat_history(force=True)


# ============================================================
# DATA ACCESS
# ============================================================

def get_full_dashboard_dataframe() -> pd.DataFrame:
    """Return the single cleaned dataframe used by the dashboard."""

    full_df = st.session_state.get(
        "eusee_full_dataset_df"
    )

    if isinstance(full_df, pd.DataFrame):
        return full_df.copy()

    # Compatibility fallbacks.
    for key in (
        "eusee_original_df",
        "eusee_unfiltered_df",
        "eusee_full_df",
        "raw_eusee_df",
    ):
        candidate = st.session_state.get(key)
        if isinstance(candidate, pd.DataFrame):
            return candidate.copy()

    return pd.DataFrame()


# ============================================================
# DATASET INTELLIGENCE + NATURAL-LANGUAGE ANALYTICS ENGINE
# ============================================================

# The chatbot is intentionally dataset-grounded:
# - OpenAI interprets the user's natural language and proposes an analysis plan.
# - pandas performs every numerical/filtering operation deterministically.
# - OpenAI only turns the verified result into a professional response.
#
# This replaces the old fixed five-intent interpreter with a general analytical
# copilot capable of multi-step questions, profiles, comparisons, trends,
# cross-tabs, rankings, composition, relationships and exploratory analysis.

DEFAULT_AI_FILTER_STATE = {
    "regions": [],
    "countries": [],
    "alert_types": [],
    "alert_impacts": [],
    "enabling_principles": [],
    "years": [],
    "months": [],
    "date_from": None,
    "date_to": None,
}

AI_CANONICAL_COLUMNS = {
    "region": "region",
    "country": "alert-country",
    "alert_type": "alert-type",
    "alert_impact": "alert-impact",
    "enabling_principle": "enabling-principle",
    "actor": "Actor of repression",
    "subject": "Subject",
    "mechanism": "Mechanism",
    "event_type": "Type of event",
    "year": "year",
    "month": "month_name",
    "date": "creation_date",
}

ALIASES = {
    "negative": "Negative",
    "negative alerts": "Negative",
    "positive": "Positive",
    "positive alerts": "Positive",
}

MONTH_ALIASES = {
    "jan": "January", "feb": "February", "mar": "March",
    "apr": "April", "may": "May", "jun": "June",
    "jul": "July", "aug": "August", "sep": "September",
    "sept": "September", "oct": "October", "nov": "November",
    "dec": "December",
}

QUARTER_MONTHS = {
    1: ["January", "February", "March"],
    2: ["April", "May", "June"],
    3: ["July", "August", "September"],
    4: ["October", "November", "December"],
}

def get_full_dashboard_dataframe() -> pd.DataFrame:
    """Return the complete unfiltered EU SEE dataframe."""
    for key in (
        "eusee_full_dataset_df",
        "eusee_original_df",
        "eusee_unfiltered_df",
        "eusee_full_df",
        "raw_eusee_df",
    ):
        candidate = st.session_state.get(key)
        if isinstance(candidate, pd.DataFrame):
            return candidate.copy()
    return pd.DataFrame()


def _clean_ai_text(value) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "").strip().lower())


def _display_column_name(column: str) -> str:
    labels = {
        "alert-country": "Country",
        "alert-type": "Alert type",
        "alert-impact": "Alert impact",
        "enabling-principle": "Enabling principle",
        "Actor of repression": "Actor",
        "Type of event": "Event type",
        "creation_date": "Date",
        "month_name": "Month",
        "_analysis_period": "Analysis period",
        "analysis_period": "Analysis period",
    }
    return labels.get(column, str(column).replace("_", " ").replace("-", " ").title())


def _prepare_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create analysis-only temporal columns without modifying the source dataframe."""
    result = df.copy()

    if "creation_date" in result.columns:
        result["creation_date"] = pd.to_datetime(result["creation_date"], errors="coerce")
        if "year" not in result.columns:
            result["year"] = result["creation_date"].dt.year
        if "month_name" not in result.columns:
            result["month_name"] = result["creation_date"].dt.strftime("%B")

    if "year" in result.columns:
        result["year"] = pd.to_numeric(result["year"], errors="coerce")

    return result


def _actual_values(df: pd.DataFrame, column: str, limit: int = 500) -> list[str]:
    if column not in df.columns:
        return []
    values = df[column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return sorted(values.unique().tolist())[:limit]


def _dataset_schema(df: pd.DataFrame) -> dict:
    """Compact, complete schema used by the natural-language planner."""
    work = _prepare_analysis_dataframe(df)
    schema = {
        "record_count": int(len(work)),
        "columns": [],
        "date_range": None,
        "dimensions": {},
        "numeric_columns": [],
    }

    for column in work.columns:
        series = work[column]
        item = {
            "name": str(column),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            item["kind"] = "numeric"
            schema["numeric_columns"].append(str(column))
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if not clean.empty:
                item["min"] = float(clean.min())
                item["max"] = float(clean.max())
        elif pd.api.types.is_datetime64_any_dtype(series):
            item["kind"] = "date"
        else:
            item["kind"] = "categorical"
            if series.nunique(dropna=True) <= 500:
                item["values"] = _actual_values(work, column, 500)

        schema["columns"].append(item)

    for key, column in AI_CANONICAL_COLUMNS.items():
        if column in work.columns:
            schema["dimensions"][key] = {
                "column": column,
                "values": _actual_values(work, column, 500),
            }

    if "creation_date" in work.columns:
        dates = pd.to_datetime(work["creation_date"], errors="coerce").dropna()
        if not dates.empty:
            schema["date_range"] = {
                "min": dates.min().strftime("%Y-%m-%d"),
                "max": dates.max().strftime("%Y-%m-%d"),
            }

    if "year" in work.columns:
        years = pd.to_numeric(work["year"], errors="coerce").dropna().astype(int).unique()
        schema["dimensions"]["year"] = {
            "column": "year",
            "values": sorted(years.tolist()),
        }

    if "month_name" in work.columns:
        schema["dimensions"]["month"] = {
            "column": "month_name",
            "values": _actual_values(work, "month_name", 50),
        }

    return schema


def _principle_tokens(series: pd.Series) -> list[str]:
    """Return individual enabling-principle labels from multi-value cells."""
    if series is None:
        return []
    tokens = set()
    for value in series.dropna().astype(str):
        for part in re.split(r"[;,|\n]+", value):
            part = str(part).strip()
            if part:
                tokens.add(part)
    return sorted(tokens, key=lambda x: _clean_ai_text(x))


def _dataset_metadata(df: pd.DataFrame) -> dict:
    """Backward-compatible metadata wrapper with exploded principle values."""
    schema = _dataset_schema(df)
    if "enabling-principle" in df.columns:
        schema.setdefault("dimensions", {}).setdefault("enabling_principle", {})[
            "values"
        ] = _principle_tokens(df["enabling-principle"])
    return schema


def _get_ai_filter_state() -> dict:
    state = st.session_state.get("eusee_ai_filter_state")
    if not isinstance(state, dict):
        state = DEFAULT_AI_FILTER_STATE.copy()
        st.session_state["eusee_ai_filter_state"] = state
    result = DEFAULT_AI_FILTER_STATE.copy()
    result.update(state)
    return result


def _set_ai_filter_state(state: dict) -> None:
    st.session_state["eusee_ai_filter_state"] = state


def _clear_ai_filter_state() -> None:
    _set_ai_filter_state(DEFAULT_AI_FILTER_STATE.copy())


def _current_dashboard_filter_state() -> dict:
    state = DEFAULT_AI_FILTER_STATE.copy()
    state["regions"] = list(st.session_state.get("selected_regions", []) or [])
    state["countries"] = list(st.session_state.get("selected_countries", []) or [])
    state["alert_types"] = list(st.session_state.get("selected_alert_types", []) or [])
    state["alert_impacts"] = list(st.session_state.get("selected_alert_impacts", []) or [])
    state["enabling_principles"] = list(
        st.session_state.get("selected_enabling_principle", []) or []
    )
    state["years"] = list(st.session_state.get("selected_years", []) or [])
    state["months"] = list(st.session_state.get("selected_months", []) or [])
    return state


def _has_active_filter_state(state: dict) -> bool:
    return any(
        bool(state.get(k))
        for k in (
            "regions", "countries", "alert_types", "alert_impacts",
            "enabling_principles", "years", "months", "date_from", "date_to",
        )
    )


def _resolve_column(df: pd.DataFrame, requested: str | None) -> str | None:
    if not requested:
        return None

    raw = str(requested).strip()
    if raw in df.columns:
        return raw

    key = _clean_ai_text(raw)
    canonical = {
        "country": "alert-country",
        "countries": "alert-country",
        "nation": "alert-country",
        "region": "region",
        "regions": "region",
        "alert type": "alert-type",
        "alert types": "alert-type",
        "impact": "alert-impact",
        "alert impact": "alert-impact",
        "principle": "enabling-principle",
        "principles": "enabling-principle",
        "enabling principle": "enabling-principle",
        "actor": "Actor of repression",
        "actors": "Actor of repression",
        "event type": "Type of event",
        "date": "creation_date",
        "month": "month_name",
        "year": "year",
    }
    candidate = canonical.get(key, raw)

    if candidate in df.columns:
        return candidate

    normalized = {_clean_ai_text(c): c for c in df.columns}
    if key in normalized:
        return normalized[key]

    matches = [
        col for norm, col in normalized.items()
        if key in norm or norm in key
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_values(requested: list, available: list) -> tuple[list, list]:
    if not requested:
        return [], []

    lookup = {_clean_ai_text(v): v for v in available}
    resolved, unresolved = [], []

    for raw in requested:
        text = str(raw or "").strip()
        if not text:
            continue
        key = _clean_ai_text(ALIASES.get(_clean_ai_text(text), text))

        if key in lookup:
            resolved.append(lookup[key])
            continue

        matches = [
            actual for actual_key, actual in lookup.items()
            if key in actual_key or actual_key in key
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
        else:
            unresolved.append(text)

    return sorted(set(resolved)), sorted(set(unresolved))


def _normalise_request_filters(df: pd.DataFrame, requested: dict) -> tuple[dict, list[str]]:
    schema = _dataset_schema(df)
    dimensions = schema.get("dimensions", {})
    filters = DEFAULT_AI_FILTER_STATE.copy()
    warnings_out = []

    dimension_map = {
        "regions": "region",
        "countries": "country",
        "alert_types": "alert_type",
        "alert_impacts": "alert_impact",
        "enabling_principles": "enabling_principle",
    }

    for key, dimension in dimension_map.items():
        available = dimensions.get(dimension, {}).get("values", [])
        resolved, unresolved = _resolve_values(requested.get(key, []) or [], available)
        filters[key] = resolved
        if unresolved:
            warnings_out.append(
                f"Could not match {key.replace('_', ' ')}: {', '.join(unresolved)}"
            )

    valid_years = set(dimensions.get("year", {}).get("values", []))
    years = []
    for value in requested.get("years", []) or []:
        try:
            year = int(value)
        except Exception:
            warnings_out.append(f"Invalid year: {value}")
            continue
        if not valid_years or year in valid_years:
            years.append(year)
        else:
            warnings_out.append(f"Year {year} is not present in the dataset.")
    filters["years"] = sorted(set(years))

    available_months = dimensions.get("month", {}).get("values", [])
    resolved_months, unresolved_months = _resolve_values(
        requested.get("months", []) or [], available_months
    )
    month_lookup = {_clean_ai_text(v): v for v in available_months}
    for month in requested.get("months", []) or []:
        alias = MONTH_ALIASES.get(_clean_ai_text(month))
        if alias and _clean_ai_text(alias) in month_lookup:
            resolved_months.append(month_lookup[_clean_ai_text(alias)])
    filters["months"] = sorted(set(resolved_months))
    if unresolved_months:
        warnings_out.append(f"Could not match months: {', '.join(unresolved_months)}")

    for key in ("date_from", "date_to"):
        value = requested.get(key)
        filters[key] = None
        if value:
            try:
                filters[key] = str(pd.Timestamp(value).date())
            except Exception:
                warnings_out.append(f"Invalid {key.replace('_', ' ')}: {value}")

    return filters, warnings_out


def _apply_local_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = _prepare_analysis_dataframe(df)

    mapping = {
        "regions": "region",
        "countries": "alert-country",
        "alert_types": "alert-type",
        "alert_impacts": "alert-impact",
        "enabling_principles": "enabling-principle",
        "years": "year",
        "months": "month_name",
    }

    for filter_key, column in mapping.items():
        values = filters.get(filter_key) or []
        if not values or column not in result.columns:
            continue

        if filter_key == "years":
            result = result[pd.to_numeric(result[column], errors="coerce").isin([int(v) for v in values])]
        else:
            # Exact match is used for structured filters. This prevents accidental
            # inclusion of similarly named categories.
            result = result[result[column].astype(str).isin([str(v) for v in values])]

    if filters.get("date_from") and "creation_date" in result.columns:
        result = result[result["creation_date"] >= pd.Timestamp(filters["date_from"])]

    if filters.get("date_to") and "creation_date" in result.columns:
        # Date-to is inclusive through the end of the specified day.
        end = pd.Timestamp(filters["date_to"]) + pd.Timedelta(days=1)
        result = result[result["creation_date"] < end]

    return result


# ============================================================
# GENERAL ANALYTICAL PLAN
# ============================================================

OPENAI_ANALYSIS_TOOL = {
    "type": "function",
    "name": "plan_dataset_analysis",
    "description": (
        "Translate any natural-language question about the EU SEE dataset "
        "into a safe, structured analytical plan. Never invent dataset values "
        "or columns. Use only the supplied schema. Do not answer the user."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["analyze", "filter_and_analyze", "clear_filters"],
            },
            "intent": {
                "type": "string",
                "enum": [
                    "summary", "explore", "trend", "distribution", "compare",
                    "profile", "ranking", "composition", "cross_tab",
                    "relationship", "change", "records",
                ],
            },
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "column": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": [
                                "eq", "neq", "in", "not_in", "contains",
                                "gte", "lte", "gt", "lt", "between",
                            ],
                        },
                        "value": {},
                    },
                    "required": ["column", "operator", "value"],
                },
            },
            "group_by": {
                "type": "array",
                "items": {"type": "string"},
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "count", "nunique", "sum", "mean", "median",
                                "min", "max", "percentage", "rate",
                            ],
                        },
                        "column": {"type": ["string", "null"]},
                        "alias": {"type": "string"},
                    },
                    "required": ["operation", "column", "alias"],
                },
            },
            "compare_values": {
                "type": "array",
                "items": {"type": "string"},
            },
            "search_text": {"type": ["string", "null"]},
            "top_n": {"type": ["integer", "null"]},
            "sort": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "column": {"type": ["string", "null"]},
                    "direction": {
                        "type": "string",
                        "enum": ["ascending", "descending"],
                    },
                },
                "required": ["column", "direction"],
            },
            "time_granularity": {
                "type": "string",
                "enum": ["none", "month", "quarter", "year"],
            },
            "visualization": {
                "type": "string",
                "enum": [
                    "none", "auto", "bar", "line", "area", "pie",
                    "donut", "stacked_bar", "heatmap", "table",
                ],
            },
            "limit": {"type": "integer"},
        },
        "required": [
            "action", "intent", "filters", "group_by", "metrics",
            "compare_values", "search_text", "top_n", "sort",
            "time_granularity", "visualization", "limit",
        ],
    },
    "strict": True,
}


OPENAI_ANALYSIS_INSTRUCTIONS = """
You are the analytical planning layer for an EU SEE dataset assistant.

The supplied DATASET_SCHEMA is the only source of truth about the dataset.
Translate the user's natural language into an analytical plan.

CORE PRINCIPLES
1. Understand ordinary natural language, not just predefined question wording.
2. Never invent a column, category, country, region, principle, year, actor or value.
3. Use exact dataset column names from DATASET_SCHEMA.
4. Use values only when they are present in the schema.
5. If the user asks an open-ended question such as "what are the main patterns",
   choose intent=explore and create several useful analytical dimensions.
6. If the user asks "tell me about <country/region>", use intent=profile and
   group/compare across useful dimensions.
7. If the user asks about "trends", use time_granularity=month or year as
   appropriate to the wording and available date information.
8. If the user asks "top", "largest", "most common", "leading", or "highest",
   use ranking/distribution and descending sorting.
9. If the user asks "compare", preserve every explicitly named comparison value.
10. If the user asks "how did X change", use intent=change and compare time periods.
11. If the user asks for "by X and Y", use both columns in group_by.
12. If the user asks for percentages/shares/proportions, include percentage.
13. If the user asks for a relationship/association between categorical variables,
    use cross_tab. For numeric variables use relationship where supported.
14. If the user asks for records/examples/incidents, use intent=records.
15. Choose a visualization automatically when it adds value.
16. "clear", "reset", or "remove filters" means clear_filters.
17. Follow-up questions inherit relevant context from CONVERSATION_CONTEXT unless
    the new request clearly replaces that context.
18. Do not write Python, SQL, URLs, web searches, or prose answers.
19. Keep the plan computationally safe: aggregation, filtering, grouping,
    sorting and descriptive statistics only.
20. A single user question may require multiple dimensions and metrics. Build a
    useful plan rather than forcing it into one simplistic analysis type.

EXAMPLES OF INTERPRETATION
"What are the main trends in West Africa?"
 -> filter region, intent=trend/explore, time series plus useful composition.

"Which principles are most associated with negative alerts in West Africa?"
 -> filters region and negative impact, group_by enabling-principle, count, descending.

"Tell me about Kenya."
 -> filter country=Kenya, intent=profile, summarize time, impacts, alert types,
    principles and actors where those columns exist.

"Compare Kenya and Uganda from 2023 to 2025."
 -> filter countries, years, intent=compare/change, group by country and year.

"Show me the records about freedom of expression in Kenya."
 -> filter country, search_text if applicable, intent=records.

"Give me the main patterns in the dataset."
 -> intent=explore, no arbitrary filter, multiple descriptive dimensions.

"How has negative activity changed by region?"
 -> filter negative impact, group by region and time, intent=change/trend.
"""


def _build_conversation_context() -> list[dict]:
    return [
        {"role": m.get("role"), "content": m.get("content")}
        for m in st.session_state.get("eusee_chat_messages", [])[-10:]
        if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
    ]


def _infer_nl_filter_candidates(question: str, df: pd.DataFrame) -> list[dict]:
    """Infer high-confidence dataset filters directly from natural language.

    This is a safety fallback, not a replacement for the LLM planner. It only
    creates filters when a phrase has a strong match to an actual dataset value
    or when a broad geographic term can be represented safely with `contains`.
    """
    q = _clean_ai_text(question)
    work = _prepare_analysis_dataframe(df)
    filters: list[dict] = []

    dimension_columns = [
        ("region", "region"),
        ("country", "alert-country"),
        ("alert type", "alert-type"),
        ("alert impact", "alert-impact"),
        ("impact", "alert-impact"),
        ("enabling principle", "enabling-principle"),
        ("principle", "enabling-principle"),
        ("actor", "Actor of repression"),
        ("event type", "Type of event"),
    ]

    used_columns = set()

    # Explicit negative/positive aliases.
    if "negative alert" in q or "negative alerts" in q:
        if "alert-impact" in work.columns:
            filters.append({"column": "alert-impact", "operator": "eq", "value": "Negative"})
            used_columns.add("alert-impact")
    elif "positive alert" in q or "positive alerts" in q:
        if "alert-impact" in work.columns:
            filters.append({"column": "alert-impact", "operator": "eq", "value": "Positive"})
            used_columns.add("alert-impact")

    # First look for exact/phrase category mentions, longest first.
    for _, column in dimension_columns:
        if column in used_columns or column not in work.columns:
            continue

        values = _actual_values(work, column, 1000)
        matches = []
        for value in values:
            v = _clean_ai_text(value)
            if len(v) < 2:
                continue
            # Avoid treating generic words such as "type", "region", etc.
            if v in {"unknown", "none", "other", "all", "total"}:
                continue
            if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", q):
                matches.append(value)

        if matches:
            # Keep the most specific matches. Multiple explicit values are
            # represented as an IN filter.
            matches = sorted(set(matches), key=lambda x: (-len(str(x)), str(x)))
            filters.append({
                "column": column,
                "operator": "in" if len(matches) > 1 else "eq",
                "value": matches[:20],
            })
            used_columns.add(column)

    # Broad geographic wording such as "in Africa" should also work when the
    # dataset stores sub-regions such as "West Africa", "East Africa", etc.
    geographic_terms = [
        "africa", "europe", "asia", "americas", "north america",
        "south america", "latin america", "middle east",
    ]
    if "region" in work.columns and "region" not in used_columns:
        for term in geographic_terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", q):
                region_values = _actual_values(work, "region", 1000)
                containing = [
                    v for v in region_values
                    if term in _clean_ai_text(v)
                ]
                if len(containing) > 1:
                    filters.append({
                        "column": "region",
                        "operator": "contains",
                        "value": term,
                    })
                    used_columns.add("region")
                elif len(containing) == 1:
                    filters.append({
                        "column": "region",
                        "operator": "eq",
                        "value": containing[0],
                    })
                    used_columns.add("region")
                break

    # Years mentioned in the question.
    if "year" in work.columns:
        available_years = set(
            pd.to_numeric(work["year"], errors="coerce").dropna().astype(int).tolist()
        )
        mentioned_years = [
            int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", q)
            if int(y) in available_years
        ]
        if mentioned_years:
            filters.append({
                "column": "year",
                "operator": "in" if len(set(mentioned_years)) > 1 else "eq",
                "value": sorted(set(mentioned_years)),
            })

    return filters


def _infer_nl_grouping(question: str, df: pd.DataFrame) -> list[str]:
    """Infer grouping dimensions from phrases such as 'by region' or 'across countries'."""
    q = _clean_ai_text(question)
    work = _prepare_analysis_dataframe(df)
    candidates = [
        (["region", "regions", "by region", "across regions", "each region"], "region"),
        (["country", "countries", "by country", "across countries", "each country"], "alert-country"),
        (["alert type", "alert types", "by alert type"], "alert-type"),
        (["impact", "impacts", "by impact", "alert impact"], "alert-impact"),
        (["principle", "principles", "by principle", "enabling principle"], "enabling-principle"),
        (["actor", "actors", "by actor"], "Actor of repression"),
        (["event type", "event types", "by event type"], "Type of event"),
        (["month", "monthly", "by month"], "month_name"),
        (["year", "years", "yearly", "annual", "by year"], "year"),
    ]
    result = []
    for phrases, column in candidates:
        if column not in work.columns:
            continue
        if any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", q) for p in phrases):
            if column not in result:
                result.append(column)
    return result[:3]


def _fallback_natural_language_plan(user_question: str, df: pd.DataFrame) -> dict:
    """Create a safe analytical plan when the LLM planner is unavailable."""
    q = _clean_ai_text(user_question)

    # Intent classification is deliberately broad.
    if any(k in q for k in ["record", "records", "incident", "incidents", "example", "examples", "show me"]):
        intent = "records"
    elif any(k in q for k in ["correlation", "relationship", "associated", "association", "related to"]):
        intent = "relationship"
    elif any(k in q for k in ["cross tab", "cross-tab", "crosstab", "contingency"]):
        intent = "cross_tab"
    elif any(k in q for k in ["change", "changed", "increase", "decrease", "growth", "decline", "difference over"]):
        intent = "change"
    elif any(k in q for k in ["trend", "trends", "over time", "evolution", "monthly", "quarterly", "yearly", "annual"]):
        intent = "trend"
    elif any(k in q for k in ["top ", "highest", "lowest", "largest", "smallest", "most common", "least common", "leading"]):
        intent = "ranking"
    elif any(k in q for k in ["compare", "comparison", "versus", " vs ", "difference between"]):
        intent = "compare"
    elif any(k in q for k in ["distribution", "breakdown", "share", "proportion", "percentage", "composition"]):
        intent = "distribution"
    elif any(k in q for k in ["tell me about", "profile", "overview of"]):
        intent = "profile"
    elif any(k in q for k in ["main patterns", "patterns", "overview", "explore", "key findings", "what stands out", "summarise", "summarize"]):
        intent = "explore"
    else:
        intent = "summary"

    filters = _infer_nl_filter_candidates(user_question, df)
    group_by = _infer_nl_grouping(user_question, df)

    # Open-ended questions should expose multiple dimensions rather than
    # forcing a single arbitrary grouping.
    if intent in {"explore", "profile"} and not group_by:
        group_by = ["region"] if "region" in df.columns else []

    # Time semantics.
    if any(k in q for k in ["quarter", "q1", "q2", "q3", "q4"]):
        time_granularity = "quarter"
    elif any(k in q for k in ["month", "monthly"]):
        time_granularity = "month"
    elif any(k in q for k in ["year", "yearly", "annual", "over time"]):
        time_granularity = "year"
    else:
        time_granularity = "year" if intent in {"trend", "change"} and "creation_date" in df.columns else "none"

    if intent in {"trend", "change"} and not group_by:
        group_by = ["year"] if "year" in df.columns else []

    # Metric selection.
    if any(k in q for k in ["average", "mean", "avg"]):
        metric_op = "mean"
        metric_col = next(
            (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])),
            None,
        )
    elif "median" in q:
        metric_op = "median"
        metric_col = next(
            (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])),
            None,
        )
    elif any(k in q for k in ["percentage", "percent", "share", "proportion", "%"]):
        metric_op = "percentage"
        metric_col = None
    else:
        metric_op = "count"
        metric_col = None

    metrics = [{
        "operation": metric_op,
        "column": metric_col,
        "alias": "records" if metric_op == "count" else metric_op,
    }]

    # Rankings need descending order and a finite top-N.
    top_n = 10 if intent == "ranking" else None
    sort = {
        "column": metrics[0]["alias"],
        "direction": "descending",
    }

    if intent == "records":
        visualization = "table"
    elif intent in {"trend", "change"}:
        visualization = "line"
    elif intent in {"distribution", "ranking", "compare", "summary"}:
        visualization = "bar"
    else:
        visualization = "auto"

    return {
        "action": "filter_and_analyze" if filters else "analyze",
        "intent": intent,
        "filters": filters,
        "group_by": group_by,
        "metrics": metrics,
        "compare_values": [],
        "search_text": None,
        "top_n": top_n,
        "sort": sort,
        "time_granularity": time_granularity,
        "visualization": visualization,
        "limit": 20,
    }


def _parse_json_object(value) -> dict | None:
    """Parse a JSON object from model output, tolerating fenced JSON."""
    if isinstance(value, dict):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        # Recover the first JSON object if the model added prose.
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _plan_dashboard_analysis(user_question: str, df: pd.DataFrame) -> dict | None:
    """Translate natural language into a validated-capable analytical plan.

    The planner deliberately has multiple paths. A transient model/API/tool
    failure must not make an otherwise answerable dataset question fail.
    """
    client = _get_eusee_openai_client()
    schema = _dataset_schema(df)

    payload = {
        "DATASET_SCHEMA": schema,
        "CURRENT_SIDEBAR_FILTERS": _current_dashboard_filter_state(),
        "CURRENT_ANALYSIS_CONTEXT": _get_ai_filter_state(),
        "CONVERSATION_CONTEXT": _build_conversation_context(),
        "USER_REQUEST": user_question,
    }

    if client is not None:
        # Path 1: Responses API + strict function tool.
        try:
            response = client.responses.create(
                model=OPENAI_MODEL,
                instructions=OPENAI_ANALYSIS_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False, default=str),
                tools=[OPENAI_ANALYSIS_TOOL],
                tool_choice={"type": "function", "name": "plan_dataset_analysis"},
                max_output_tokens=1800,
            )
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "function_call":
                    parsed = _parse_json_object(getattr(item, "arguments", None))
                    if parsed:
                        return parsed
        except Exception as exc:
            if st.secrets.get("debug", {}).get("show_chat_ai_errors", False):
                st.warning(f"AI analytical planner (Responses API) failed: {exc}")

        # Path 2: Chat Completions JSON mode. This is intentionally independent
        # of function-tool support and handles environments where the Responses
        # tool interface is unavailable or behaves differently.
        try:
            chat_prompt = (
                OPENAI_ANALYSIS_INSTRUCTIONS
                + "\n\nReturn ONLY one JSON object matching this exact plan shape:\n"
                + json.dumps({
                    "action": "analyze",
                    "intent": "explore",
                    "filters": [],
                    "group_by": [],
                    "metrics": [{"operation": "count", "column": None, "alias": "records"}],
                    "compare_values": [],
                    "search_text": None,
                    "top_n": None,
                    "sort": {"column": None, "direction": "descending"},
                    "time_granularity": "none",
                    "visualization": "auto",
                    "limit": 20,
                }, ensure_ascii=False)
            )
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": chat_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
                ],
                response_format={"type": "json_object"},
                max_tokens=1800,
            )
            content = response.choices[0].message.content if response.choices else ""
            parsed = _parse_json_object(content)
            if parsed:
                return parsed
        except Exception as exc:
            if st.secrets.get("debug", {}).get("show_chat_ai_errors", False):
                st.warning(f"AI analytical planner (Chat Completions) failed: {exc}")

    # Path 3: deterministic natural-language planner. This is always available
    # and is schema-aware, so questions remain dataset-grounded even when the
    # LLM service is temporarily unavailable.
    return _fallback_natural_language_plan(user_question, df)


# ============================================================
# GENERIC PLAN VALIDATION + EXECUTION
# ============================================================

def _validate_plan(df: pd.DataFrame, plan: dict) -> tuple[dict, list[str]]:
    work = _prepare_analysis_dataframe(df)
    warnings_out = []
    validated = dict(plan or {})

    valid_columns = set(work.columns)

    # Validate filters and resolve category values against the real dataframe.
    clean_filters = []
    for item in plan.get("filters", []) or []:
        if not isinstance(item, dict):
            continue

        column = _resolve_column(work, item.get("column"))
        if not column:
            warnings_out.append(f"Unknown filter column: {item.get('column')}")
            continue

        operator = str(item.get("operator", "eq")).lower().strip()
        value = item.get("value")

        if operator == "contains" and column in work.columns:
            # "contains" is intentionally free-text/semantic.  It must NOT
            # require the requested phrase to be an existing category.
            value = str(value or "").strip()
            if not value:
                warnings_out.append(f"Empty contains filter for {column}.")
                continue

        elif operator in {"eq", "neq", "in", "not_in"} and column in work.columns:
            available = _actual_values(work, column, 2000)
            values = value if isinstance(value, list) else [value]

            if operator in {"eq", "neq"}:
                values = values[:1]

            resolved, unresolved = _resolve_values(values, available)
            if unresolved:
                # Numeric/date filters should not use categorical matching.
                if pd.api.types.is_numeric_dtype(work[column]):
                    try:
                        resolved = [float(values[0])]
                        unresolved = []
                    except Exception:
                        pass

            if unresolved:
                warnings_out.append(
                    f"Could not match {column}: {', '.join(map(str, unresolved))}"
                )
                continue

            if operator in {"eq", "neq"}:
                value = resolved[0] if resolved else value
            else:
                value = resolved

        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                warnings_out.append(f"Invalid between filter for {column}.")
                continue

        clean_filters.append({
            "column": column,
            "operator": operator,
            "value": value,
        })

    validated["filters"] = clean_filters

    group_by = []
    for col in plan.get("group_by", []) or []:
        resolved = _resolve_column(work, col)
        if resolved and resolved in valid_columns:
            if resolved not in group_by:
                group_by.append(resolved)
        else:
            warnings_out.append(f"Unknown analysis column: {col}")
    validated["group_by"] = group_by

    metrics = []
    for metric in plan.get("metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        operation = str(metric.get("operation", "count")).lower()
        column = metric.get("column")
        resolved_column = None if column in (None, "", "null") else _resolve_column(work, column)

        if operation not in {
            "count", "nunique", "sum", "mean", "median",
            "min", "max", "percentage", "rate",
        }:
            continue

        if operation not in {"count"} and resolved_column is None:
            warnings_out.append(f"Metric {operation} requires a valid column.")
            continue

        metrics.append({
            "operation": operation,
            "column": resolved_column,
            "alias": str(metric.get("alias") or operation),
        })
    if not metrics:
        metrics = [{"operation": "count", "column": None, "alias": "records"}]
    validated["metrics"] = metrics

    sort = plan.get("sort") or {}
    sort_column = _resolve_column(work, sort.get("column")) if sort.get("column") else None
    validated["sort"] = {
        "column": sort_column,
        "direction": sort.get("direction", "descending"),
    }

    try:
        validated["limit"] = max(1, min(int(plan.get("limit", 20)), 100))
    except Exception:
        validated["limit"] = 20

    try:
        top_n = plan.get("top_n")
        validated["top_n"] = None if top_n is None else max(1, min(int(top_n), 100))
    except Exception:
        validated["top_n"] = None

    return validated, warnings_out


def _apply_generic_plan_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    result = _prepare_analysis_dataframe(df)

    for item in filters:
        column = item["column"]
        operator = item["operator"]
        value = item["value"]

        if column not in result.columns:
            continue

        series = result[column]

        if operator in {"eq", "neq", "in", "not_in"}:
            values = value if isinstance(value, list) else [value]

            if column == "enabling-principle":
                wanted = {_clean_ai_text(v) for v in values if str(v).strip()}

                def principle_match(cell) -> bool:
                    if pd.isna(cell):
                        return False
                    cell_tokens = {
                        _clean_ai_text(part)
                        for part in re.split(r"[;,|\n]+", str(cell))
                        if str(part).strip()
                    }
                    return bool(cell_tokens & wanted)

                mask = series.apply(principle_match)
            elif pd.api.types.is_numeric_dtype(series):
                converted = pd.to_numeric(
                    pd.Series(values), errors="coerce"
                ).dropna().tolist()
                mask = pd.to_numeric(series, errors="coerce").isin(converted)
            else:
                compare_values = {_clean_ai_text(v) for v in values}
                mask = series.astype(str).map(_clean_ai_text).isin(compare_values)

            if operator in {"neq", "not_in"}:
                mask = ~mask
            result = result[mask]

        elif operator == "contains":
            needle = str(value or "").strip().lower()
            result = result[
                series.fillna("").astype(str).str.lower().str.contains(
                    needle, regex=False
                )
            ]

        elif operator in {"gte", "lte", "gt", "lt"}:
            numeric = pd.to_numeric(series, errors="coerce")
            try:
                target = float(value)
            except Exception:
                try:
                    target = pd.Timestamp(value)
                    numeric = pd.to_datetime(series, errors="coerce")
                except Exception:
                    continue

            if operator == "gte":
                result = result[numeric >= target]
            elif operator == "lte":
                result = result[numeric <= target]
            elif operator == "gt":
                result = result[numeric > target]
            else:
                result = result[numeric < target]

        elif operator == "between":
            values = value if isinstance(value, list) else []
            if len(values) != 2:
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            if pd.api.types.is_numeric_dtype(series):
                try:
                    lo, hi = float(values[0]), float(values[1])
                    result = result[numeric.between(lo, hi)]
                except Exception:
                    continue
            else:
                dates = pd.to_datetime(series, errors="coerce")
                try:
                    lo, hi = pd.Timestamp(values[0]), pd.Timestamp(values[1])
                    result = result[dates.between(lo, hi)]
                except Exception:
                    continue

    return result


def _metric_series(group: pd.DataFrame, metric: dict, denominator: int | None = None):
    op = metric["operation"]
    column = metric.get("column")

    if op == "count":
        return len(group)
    if op == "nunique":
        return group[column].nunique(dropna=True)
    if op == "sum":
        return pd.to_numeric(group[column], errors="coerce").sum()
    if op == "mean":
        return pd.to_numeric(group[column], errors="coerce").mean()
    if op == "median":
        return pd.to_numeric(group[column], errors="coerce").median()
    if op == "min":
        return pd.to_numeric(group[column], errors="coerce").min()
    if op == "max":
        return pd.to_numeric(group[column], errors="coerce").max()
    if op in {"percentage", "rate"}:
        numerator = len(group)
        denom = denominator if denominator is not None else max(len(group), 1)
        return 100.0 * numerator / max(denom, 1)
    return len(group)


def _choose_visualization(plan: dict, result_df: pd.DataFrame) -> str:
    requested = str(plan.get("visualization", "auto")).lower()
    if requested != "auto":
        if requested in {"line", "area"} and result_df is not None and not result_df.empty:
            temporal_cols = [
                c for c in result_df.columns
                if str(c).strip().lower() in {"analysis period", "_analysis_period"}
            ]
            if temporal_cols and result_df[temporal_cols[0]].nunique(dropna=True) <= 1:
                return "bar"
        return requested

    if result_df.empty:
        return "none"

    group_by = plan.get("group_by", []) or []
    granularity = plan.get("time_granularity", "none")

    if granularity != "none" or any(
        _clean_ai_text(c) in {"creation_date", "year", "month_name"}
        for c in group_by
    ):
        temporal_cols = [
            c for c in result_df.columns
            if str(c).strip().lower() in {"analysis period", "_analysis_period"}
        ]
        if temporal_cols and result_df[temporal_cols[0]].nunique(dropna=True) <= 1:
            return "bar"
        return "line"

    if len(group_by) >= 2:
        return "heatmap"

    if len(result_df) <= 8 and result_df.shape[1] >= 2:
        return "donut" if plan.get("intent") == "composition" else "bar"

    return "bar"


def _execute_grouped_analysis(filtered: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, dict]:
    work = filtered.copy()
    group_by = list(plan.get("group_by", []) or [])
    metrics = list(plan.get("metrics", []) or [])
    granularity = plan.get("time_granularity", "none")

    # Automatic temporal grouping when the user asks for a trend/change.
    if granularity != "none" and "creation_date" in work.columns:
        dates = pd.to_datetime(work["creation_date"], errors="coerce")
        if granularity == "month":
            work["_analysis_period"] = dates.dt.to_period("M").astype(str)
        elif granularity == "quarter":
            work["_analysis_period"] = dates.dt.to_period("Q").astype(str)
        elif granularity == "year":
            work["_analysis_period"] = dates.dt.year.astype("Int64").astype(str)

        if "_analysis_period" not in group_by:
            group_by = ["_analysis_period"] + group_by

    if not group_by:
        values = {}
        denominator = len(filtered)
        for metric in metrics:
            values[metric["alias"]] = _metric_series(
                filtered, metric, denominator=denominator
            )
        out = pd.DataFrame([values])
        return out, {"group_by": [], "metrics": metrics}

    rows = []
    grouped = work.groupby(group_by, dropna=False, sort=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {}
        for col, key in zip(group_by, keys):
            row[_display_column_name(col)] = "Unknown" if pd.isna(key) else key

        for metric in metrics:
            value = _metric_series(
                group,
                metric,
                denominator=len(filtered),
            )
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            row[metric["alias"]] = value
        rows.append(row)

    return pd.DataFrame(rows), {"group_by": group_by, "metrics": metrics}


def _execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    filtered = _apply_generic_plan_filters(df, plan.get("filters", []))

    search_text = str(plan.get("search_text") or "").strip()
    if search_text:
        preferred_searchable = [
            "alert-country", "region", "alert-type", "alert-impact",
            "Actor of repression", "Subject", "Mechanism", "Type of event",
            "Alert title", "Description", "enabling-principle",
        ]
        searchable = [c for c in preferred_searchable if c in filtered.columns]
        if not searchable:
            searchable = [
                c for c in filtered.columns
                if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])
            ]
        if searchable:
            mask = pd.Series(False, index=filtered.index)
            needle = search_text.lower()
            for col in searchable:
                mask |= filtered[col].fillna("").astype(str).str.lower().str.contains(
                    needle, regex=False
                )
            filtered = filtered[mask]

    intent = plan.get("intent", "summary")
    grouped_df, grouped_meta = _execute_grouped_analysis(filtered, plan)

    sort = plan.get("sort") or {}
    sort_column = sort.get("column")
    if sort_column:
        display_sort = _display_column_name(sort_column)
        if display_sort in grouped_df.columns:
            grouped_df = grouped_df.sort_values(
                display_sort,
                ascending=sort.get("direction") != "descending",
            )

    if plan.get("top_n"):
        grouped_df = grouped_df.head(plan["top_n"])

    limit = int(plan.get("limit", 20))
    grouped_df = grouped_df.head(limit)

    # Automatic exploration: assemble several deterministic summaries.
    exploration = {}
    if intent == "explore":
        exploration["dataset_records"] = int(len(df))
        exploration["matching_records"] = int(len(filtered))

        for label, column in (
            ("regions", "region"),
            ("countries", "alert-country"),
            ("alert_impacts", "alert-impact"),
            ("alert_types", "alert-type"),
            ("enabling_principles", "enabling-principle"),
        ):
            if column in filtered.columns:
                temp = (
                    filtered[column]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                temp.columns = ["category", "records"]
                exploration[label] = temp.to_dict("records")

        if "creation_date" in filtered.columns:
            dates = pd.to_datetime(filtered["creation_date"], errors="coerce")
            trend = (
                pd.DataFrame({"date": dates})
                .dropna()
                .assign(year=lambda x: x["date"].dt.year)
                .groupby("year")
                .size()
                .reset_index(name="records")
            )
            exploration["yearly_trend"] = trend.to_dict("records")

    # Profile: deterministic overview across the main analytical dimensions.
    profile = {}
    if intent == "profile":
        profile["records"] = int(len(filtered))
        for label, column in (
            ("countries", "alert-country"),
            ("regions", "region"),
            ("alert_impacts", "alert-impact"),
            ("alert_types", "alert-type"),
            ("enabling_principles", "enabling-principle"),
            ("actors", "Actor of repression"),
        ):
            if column in filtered.columns:
                temp = (
                    filtered[column]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                temp.columns = ["category", "records"]
                profile[label] = temp.to_dict("records")

        if "creation_date" in filtered.columns:
            dates = pd.to_datetime(filtered["creation_date"], errors="coerce")
            trend = (
                pd.DataFrame({"date": dates})
                .dropna()
                .assign(year=lambda x: x["date"].dt.year)
                .groupby("year")
                .size()
                .reset_index(name="records")
            )
            profile["yearly_trend"] = trend.to_dict("records")

    # Records request.
    records = None
    if intent == "records":
        selected = [
            c for c in [
                "creation_date", "alert-country", "region", "alert-type",
                "alert-impact", "Actor of repression", "enabling-principle",
                "Subject", "Mechanism", "Type of event",
            ] if c in filtered.columns
        ]
        records_df = filtered[selected].head(limit).copy()
        if "creation_date" in records_df.columns:
            records_df["creation_date"] = pd.to_datetime(
                records_df["creation_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        records = records_df.where(pd.notna(records_df), None).to_dict("records")

    # Cross-tab / relationship analysis.
    cross_tab = None
    group_by = plan.get("group_by", []) or []
    if intent == "cross_tab" and len(group_by) >= 2:
        a, b = group_by[:2]
        if a in filtered.columns and b in filtered.columns:
            ct = pd.crosstab(
                filtered[a].fillna("Unknown").astype(str),
                filtered[b].fillna("Unknown").astype(str),
            )
            ct = ct.iloc[:50, :50]
            cross_tab = {
                "row_dimension": _display_column_name(a),
                "column_dimension": _display_column_name(b),
                "data": ct.reset_index().to_dict("records"),
            }

    # Change analysis: calculate first-to-last period difference when possible.
    change = None
    if intent == "change" and not grouped_df.empty:
        period_candidates = [
            c for c in grouped_df.columns
            if c.lower() in {"date", "year", "month", "period"} or "period" in c.lower()
        ]
        if period_candidates:
            pcol = period_candidates[0]
            numeric_cols = [
                c for c in grouped_df.columns
                if c != pcol and pd.api.types.is_numeric_dtype(grouped_df[c])
            ]
            if numeric_cols:
                temp = grouped_df.copy()
                temp = temp.sort_values(pcol)
                first = temp.iloc[0]
                last = temp.iloc[-1]
                change = {
                    "from_period": first[pcol],
                    "to_period": last[pcol],
                    "changes": {
                        c: (
                            None if pd.isna(first[c]) or pd.isna(last[c])
                            else float(last[c] - first[c])
                        )
                        for c in numeric_cols
                    },
                }

    chart = None
    chart_df = grouped_df.copy()

    if cross_tab:
        ct_df = pd.DataFrame(cross_tab["data"])
        if not ct_df.empty:
            first_col = ct_df.columns[0]
            value_cols = list(ct_df.columns[1:])
            chart = {
                "type": "heatmap",
                "x": value_cols,
                "y": first_col,
                "title": "Cross-dimensional distribution",
                "data": cross_tab["data"],
            }
    elif not chart_df.empty and len(chart_df.columns) >= 2:
        chart_type = _choose_visualization(plan, chart_df)

        # CRITICAL CHART INTEGRITY RULE:
        # Never select a numeric grouping column (for example `year`) as the
        # y-axis merely because it happens to be numeric. The y-axis must come
        # from the metric(s) explicitly requested/computed by the plan.
        metric_aliases = [
            str(m.get("alias", "")).strip()
            for m in (plan.get("metrics") or [])
            if isinstance(m, dict) and str(m.get("alias", "")).strip()
        ]
        metric_columns = [
            c for c in metric_aliases
            if c in chart_df.columns and pd.api.types.is_numeric_dtype(chart_df[c])
        ]
        if not metric_columns:
            metric_columns = [
                c for c in chart_df.columns
                if c not in {
                    _display_column_name(g) for g in (plan.get("group_by") or [])
                }
                and c not in {_display_column_name("_analysis_period"), "Analysis period"}
                and pd.api.types.is_numeric_dtype(chart_df[c])
            ]

        # Temporal analyses should use the generated analysis period as x.
        # For a multi-region/multi-category trend, retain the second grouping
        # dimension as a colour series rather than silently discarding it.
        group_columns = [
            _display_column_name(c)
            for c in (plan.get("group_by") or [])
            if _display_column_name(c) in chart_df.columns
        ]
        period_columns = [
            c for c in chart_df.columns
            if str(c).strip().lower() in {"analysis period", "_analysis_period"}
        ]

        if chart_type != "none" and metric_columns:
            y_col = metric_columns[0]
            if plan.get("time_granularity") != "none" and period_columns:
                x_col = period_columns[0]
            elif group_columns:
                x_col = group_columns[0]
            else:
                x_col = chart_df.columns[0]

            chart = {
                "type": chart_type,
                "x": x_col,
                "y": y_col,
                "title": _analysis_title(plan, x_col),
                "x_label": _display_column_name(x_col),
                "y_label": _display_column_name(y_col),
                "data": chart_df.to_dict("records"),
            }

            # Preserve a meaningful comparison dimension for time-series charts.
            color_candidates = [
                c for c in group_columns
                if c != x_col and c in chart_df.columns
            ]
            if chart_type in {"line", "area"} and color_candidates:
                chart["color"] = color_candidates[0]

    analysis = {
        "intent": intent,
        "matching_records": int(len(filtered)),
        "visualization_requested": bool(plan.get("visualization_requested", False)),
        "result_rows": int(len(grouped_df)),
        "grouped_data": grouped_df.to_dict("records"),
        "exploration": exploration,
        "profile": profile,
        "records": records,
        "cross_tab": cross_tab,
        "change": change,
        "chart": chart,
        "applied_filters": plan.get("filters", []),
    }

    return {
        "records_count": int(len(filtered)),
        "analysis_type": intent,
        "metric": plan.get("metrics", [{"alias": "records"}])[0].get("alias", "records"),
        "filters": plan.get("filters", []),
        "analysis": analysis,
        "chart": chart,
        "records": records,
    }


def _analysis_title(plan: dict, x_col: str) -> str:
    intent = str(plan.get("intent", "analysis")).replace("_", " ").title()
    return f"{intent} by {_display_column_name(x_col)}"


# ============================================================
# DASHBOARD FILTER SYNCHRONISATION
# ============================================================

def _sync_ai_filters_to_dashboard(filters: dict) -> list[str]:
    unsupported = []

    mapping = {
        "regions": "selected_regions",
        "countries": "selected_countries",
        "alert_types": "selected_alert_types",
        "alert_impacts": "selected_alert_impacts",
        "enabling_principles": "selected_enabling_principle",
        "years": "selected_years",
        "months": "selected_months",
    }

    for ai_key, dashboard_key in mapping.items():
        values = list(filters.get(ai_key) or [])
        if values:
            st.session_state[dashboard_key] = values

    if filters.get("date_from") or filters.get("date_to"):
        unsupported.append(
            "The exact date range is applied to the AI analysis; "
            "the existing dashboard has no dedicated date-range control."
        )

    return unsupported


def _clear_dashboard_filters() -> None:
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
        st.session_state[key] = []


# ============================================================
# NATURAL-LANGUAGE PROCESSOR
# ============================================================

def _extract_simple_filters_for_dashboard(plan: dict, df: pd.DataFrame) -> dict:
    """Map validated structured filters to the existing sidebar state."""
    filters = DEFAULT_AI_FILTER_STATE.copy()

    for item in plan.get("filters", []) or []:
        column = item.get("column")
        op = item.get("operator")
        value = item.get("value")

        if op not in {"eq", "in"}:
            continue

        values = value if isinstance(value, list) else [value]

        if column == "region":
            filters["regions"] = values
        elif column == "alert-country":
            filters["countries"] = values
        elif column == "alert-type":
            filters["alert_types"] = values
        elif column == "alert-impact":
            filters["alert_impacts"] = values
        elif column == "enabling-principle":
            filters["enabling_principles"] = values
        elif column == "year":
            filters["years"] = [int(v) for v in values if str(v).isdigit()]
        elif column == "month_name":
            filters["months"] = values

    return filters


def _explicit_visualization_requested(question: str) -> bool:
    """Only return True when the user explicitly asks for a chart/visual."""
    q = _clean_ai_text(question)
    patterns = [
        r"\bshow\s+(?:me\s+)?(?:a\s+)?(?:chart|graph|plot|visualization)\b",
        r"\b(?:chart|graph|plot|visualization)\s+(?:of|for|showing)\b",
        r"\bplot\b",
        r"\bvisuali[sz]e\b",
        r"\bvisuali[sz]ation\b",
        r"\bgraphical(?:ly)?\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _explicit_record_request(question: str) -> bool:
    """Detect requests for actual alert/case examples."""
    q = _clean_ai_text(question)
    patterns = [
        r"\bshow\s+(?:me\s+)?(?:some\s+)?(?:alerts|cases|records|examples|incidents)\b",
        r"\bshare\s+(?:some\s+)?(?:alerts|cases|records|examples|incidents)\b",
        r"\blist\s+(?:some\s+)?(?:alerts|cases|records|examples|incidents)\b",
        r"\bgive\s+(?:me\s+)?(?:some\s+)?(?:alerts|cases|records|examples|incidents)\b",
        r"\bexamples?\s+of\b",
        r"\b(?:cases?|alerts?|records?|incidents?)\s+related\s+to\b",
        r"\bslapp\b",
        r"\bstrategic lawsuits? against public participation\b",
        r"\bfind\s+(?:some\s+)?(?:alerts|cases|records)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _infer_explicit_filters(question: str, df: pd.DataFrame) -> list[dict]:
    """Infer only high-confidence filters from values present in the dataset."""
    work = _prepare_analysis_dataframe(df)
    q = _clean_ai_text(question)
    filters = []

    for column in ("region", "alert-country", "alert-type", "alert-impact"):
        if column not in work.columns:
            continue
        matches = []
        for value in _actual_values(work, column, 2000):
            norm = _clean_ai_text(value)
            if len(norm) < 2 or norm in {"unknown", "none", "all", "total"}:
                continue
            if re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", q):
                matches.append(value)
        if matches:
            filters.append({
                "column": column,
                "operator": "in" if len(matches) > 1 else "eq",
                "value": sorted(set(matches), key=lambda x: (-len(str(x)), str(x)))[:20],
            })

    if "alert-impact" in work.columns and not any(
        f.get("column") == "alert-impact" for f in filters
    ):
        if re.search(r"\bnegative\s+alerts?\b", q):
            filters.append({"column": "alert-impact", "operator": "eq", "value": "Negative"})
        elif re.search(r"\bpositive\s+alerts?\b", q):
            filters.append({"column": "alert-impact", "operator": "eq", "value": "Positive"})

    if "enabling-principle" in work.columns:
        principles = _principle_tokens(work["enabling-principle"])
        matches = []
        for value in principles:
            norm = _clean_ai_text(value)
            if norm and re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", q):
                matches.append(value)

        for number in re.findall(r"\bprinciple\s*(\d{1,2})\b", q):
            target = f"principle {number}"
            matches.extend(
                value for value in principles
                if _clean_ai_text(value) == target
            )

        if matches:
            filters.append({
                "column": "enabling-principle",
                "operator": "in",
                "value": sorted(set(matches)),
            })

    if "year" in work.columns:
        available = set(
            pd.to_numeric(work["year"], errors="coerce").dropna().astype(int)
        )
        years = [
            int(y) for y in re.findall(r"\b(?:19|20|21)\d{2}\b", q)
            if int(y) in available
        ]
        if years:
            filters.append({
                "column": "year",
                "operator": "in" if len(set(years)) > 1 else "eq",
                "value": sorted(set(years)),
            })

    return filters


def _merge_chatbot_safety_controls(
    question: str, plan: dict, df: pd.DataFrame
) -> dict:
    """Apply deterministic high-confidence user intent/filter semantics."""
    plan = dict(plan or {})

    explicit_filters = _infer_explicit_filters(question, df)
    current = list(plan.get("filters", []) or [])

    for item in explicit_filters:
        column = item["column"]
        current = [f for f in current if f.get("column") != column]
        current.append(item)

    plan["filters"] = current

    if _explicit_record_request(question):
        plan["intent"] = "records"
        plan["visualization"] = "none"
        if re.search(
            r"\bslapp\b|\bstrategic lawsuits? against public participation\b",
            _clean_ai_text(question),
        ):
            plan["search_text"] = "slapp"

    requested_visual = _explicit_visualization_requested(question)
    plan["visualization_requested"] = requested_visual
    if not requested_visual:
        plan["visualization"] = "none"

    return plan


def _process_eusee_ai_request(user_question: str) -> dict:
    df = get_full_dashboard_dataframe()

    if df.empty:
        return {
            "answer": "The complete EU SEE dashboard dataset is not available for the AI Assistant.",
            "analysis": None,
        }

    plan = _plan_dashboard_analysis(user_question, df)

    if not plan:
        return {
            "answer": (
                "I could not create a reliable analytical plan for that request. "
                "Please rephrase the question in terms of the EU SEE dataset."
            ),
            "analysis": None,
        }

    if plan.get("action") == "clear_filters":
        _clear_ai_filter_state()
        _clear_dashboard_filters()
        return {
            "answer": "The dashboard filters have been cleared.",
            "analysis": None,
            "filter_updated": True,
            "clear_filters": True,
        }

    plan = _merge_chatbot_safety_controls(user_question, plan, df)

    validated_plan, plan_warnings = _validate_plan(df, plan)

    # Deterministic Q1-Q4/H1-H2 handling. This is deliberately outside the LLM.
    # IMPORTANT: month_name is often a derived analysis column, so always work
    # from the prepared dataframe rather than requiring it to exist in the raw
    # source dataframe. This guarantees that Q1/Q2/Q3/Q4 and H1/H2 filters are
    # actually applied to the records used for the answer and chart.
    q = user_question.lower()
    temporal_months = None
    temporal_label = None
    for quarter, months in QUARTER_MONTHS.items():
        if re.search(rf"\bq{quarter}\b|\bquarter\s*{quarter}\b", q):
            temporal_months = months
            temporal_label = f"Q{quarter}"
            break
    if temporal_months is None and re.search(r"\b(h1|first half|first half-year)\b", q):
        temporal_months = QUARTER_MONTHS[1] + QUARTER_MONTHS[2]
        temporal_label = "H1"
    if temporal_months is None and re.search(r"\b(h2|second half|second half-year)\b", q):
        temporal_months = QUARTER_MONTHS[3] + QUARTER_MONTHS[4]
        temporal_label = "H2"

    if temporal_months:
        analysis_df = _prepare_analysis_dataframe(df)
        if "month_name" in analysis_df.columns:
            available = set(_actual_values(analysis_df, "month_name", 50))
            matched_months = [m for m in temporal_months if m in available]
            if matched_months:
                temporal_filter = {
                    "column": "month_name",
                    "operator": "in",
                    "value": matched_months,
                }
                validated_plan["filters"] = [
                    f for f in validated_plan.get("filters", [])
                    if f.get("column") != "month_name"
                ] + [temporal_filter]

                # A quarter/half-year named together with a specific year is a
                # bounded period, not a request for the whole multi-period trend.
                # The filter above is combined with the year filter when present.
                # If the planner omitted the year, recover it from the question.
                mentioned_years = [
                    int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", q)
                ]
                if mentioned_years and "year" in analysis_df.columns:
                    year_values = [
                        y for y in mentioned_years
                        if y in set(pd.to_numeric(analysis_df["year"], errors="coerce").dropna().astype(int))
                    ]
                    if year_values:
                        validated_plan["filters"] = [
                            f for f in validated_plan.get("filters", [])
                            if f.get("column") != "year"
                        ] + [{
                            "column": "year",
                            "operator": "in" if len(set(year_values)) > 1 else "eq",
                            "value": sorted(set(year_values)),
                        }]

    # A named quarter/half-year used with a trend/change request is a bounded
    # window. Plot the periods inside that window (e.g. Jul-Aug-Sep for Q3)
    # rather than accidentally showing Q1-Q3 for the full year.
    if temporal_label and validated_plan.get("intent") in {"trend", "change"}:
        validated_plan["time_granularity"] = "month"

    analysis = _execute_plan(df, validated_plan)
    analysis.setdefault("analysis", {})["warnings"] = plan_warnings
    analysis["warnings"] = plan_warnings
    analysis["visualization_requested"] = bool(
        validated_plan.get("visualization_requested", False)
    )
    if not analysis["visualization_requested"]:
        analysis["chart"] = None

    # Keep conversational state synchronized for follow-up questions.
    dashboard_like_filters = _extract_simple_filters_for_dashboard(validated_plan, df)
    if _has_active_filter_state(dashboard_like_filters):
        _set_ai_filter_state(dashboard_like_filters)

    filter_updated = plan.get("action") == "filter_and_analyze"
    if filter_updated:
        unsupported = _sync_ai_filters_to_dashboard(dashboard_like_filters)
        analysis["warnings"].extend(unsupported)

    answer = _generate_eusee_answer(user_question, analysis)

    return {
        "answer": answer,
        "analysis": analysis,
        "filter_updated": filter_updated,
    }


# ============================================================
# OPENAI FINAL RESPONSE
# ============================================================

def _generate_eusee_answer(user_question: str, analysis: dict) -> str:
    client = _get_eusee_openai_client()

    if client is None:
        return _deterministic_eusee_answer(analysis)

    compact_result = {
        "records_count": analysis.get("records_count", 0),
        "analysis_type": analysis.get("analysis_type"),
        "filters": analysis.get("filters", {}),
        "analysis": analysis.get("analysis", {}),
        "records": analysis.get("records"),
        "warnings": analysis.get("warnings", []),
        "visualization_requested": bool(analysis.get("visualization_requested", False)),
    }

    prompt = json.dumps(
        {
            "USER_QUESTION": user_question,
            "VERIFIED_DATASET_RESULT": compact_result,
        },
        ensure_ascii=False,
        default=str,
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions="""
            You are the final response writer for a professional EU SEE dataset analytics assistant.
            
            Use ONLY VERIFIED_DATASET_RESULT. Never invent or infer a number, record, category,
            country, region, principle, cause, or explanation.
            
            COUNT RULES:
            - records_count is the number of records remaining AFTER every requested filter and
              any topic/search filter has been applied.
            - Never substitute the complete dataset count for a narrower country, region,
              principle, impact, alert-type, or topic count.
            - If the result is zero, say clearly that no matching alerts were found.
            - If a country and principle were requested, report their intersection, not the country total.
            - If actual records are supplied, answer with those records/examples rather than replacing
              them with a distribution or overall summary.
            
            RECORD/EXAMPLE REQUESTS:
            - Requests such as "share examples", "show cases", "list alerts", or SLAPP examples
              must be answered from the supplied records.
            - Keep the answer concise and do not fabricate examples.
            
            STYLE:
            - Answer the user's actual question directly in 1-4 concise paragraphs or bullets.
            - Prefer natural wording such as: "In our data, there are 27 alerts in Argentina."
            - For principle intersections, use wording such as: "In our data, there are 8 alerts related
              to Principle 6 in Argentina."
            - Do not say "matching records" when a more natural phrase is available.
            - Do not mention OpenAI, APIs, Python, tools, prompts, schemas, or internal implementation.
            - Do not use external knowledge.
            - Do not make political judgments or recommendations.
            - Do not mention a chart unless visualization_requested is true.
            - If visualization_requested is false, do not describe or request a chart.
            
            End every answer with:
            "For more information, please visit the EU SEE website: https://eusee.hivos.org/"
            """,
            input=prompt,
            reasoning={"effort": "none"},
            max_output_tokens=900,
        )

        output = str(response.output_text or "").strip()
        if output:
            return output
    except Exception as exc:
        if st.secrets.get("debug", {}).get("show_chat_ai_errors", False):
            st.warning(f"AI response generation failed: {exc}")

    return _deterministic_eusee_answer(analysis)


def _format_plan_filters(filters: list[dict]) -> str:
    parts = []
    for item in filters or []:
        column = _display_column_name(item.get("column", ""))
        op = item.get("operator", "eq")
        value = item.get("value")
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        parts.append(f"{column} {op} {value}")
    return "; ".join(parts)


def _deterministic_eusee_answer(analysis: dict) -> str:
    """Safe fallback response using only verified local results."""
    count = int(analysis.get("records_count", 0) or 0)
    intent = analysis.get("analysis_type", "analysis")
    filters = analysis.get("filters") or []

    countries, regions, principles, impacts, alert_types = [], [], [], [], []
    for item in filters:
        column = item.get("column")
        value = item.get("value")
        values = value if isinstance(value, list) else [value]
        if column == "alert-country":
            countries = [str(v) for v in values]
        elif column == "region":
            regions = [str(v) for v in values]
        elif column == "enabling-principle":
            principles = [str(v) for v in values]
        elif column == "alert-impact":
            impacts = [str(v) for v in values]
        elif column == "alert-type":
            alert_types = [str(v) for v in values]

    if count == 0:
        answer = "In our data, there are no alerts matching that request."
    elif principles and countries:
        answer = (
            f"In our data, there are {count:,} alerts related to "
            f"{', '.join(principles)} in {', '.join(countries)}."
        )
    elif principles:
        answer = (
            f"In our data, there are {count:,} alerts related to "
            f"{', '.join(principles)}."
        )
    elif countries:
        answer = f"In our data, there are {count:,} alerts in {', '.join(countries)}."
    elif regions:
        answer = f"In our data, there are {count:,} alerts in {', '.join(regions)}."
    elif impacts:
        answer = f"In our data, there are {count:,} {', '.join(impacts).lower()} alerts."
    elif alert_types:
        answer = (
            f"In our data, there are {count:,} alerts of type "
            f"{', '.join(alert_types)}."
        )
    elif intent in {"trend", "change"}:
        answer = f"In our data, there are {count:,} alerts in the requested period."
    elif intent == "records":
        returned = len(analysis.get("records") or [])
        answer = f"In our data, there are {count:,} alerts matching that request."
        if returned:
            answer += f" I have included {returned:,} examples below."
    else:
        answer = f"In our data, there are {count:,} alerts matching the request."

    warnings = analysis.get("warnings") or []
    if warnings:
        answer += "\n\nNote: " + " ".join(map(str, warnings[:2]))

    return answer + (
        "\n\nFor more information, please visit the EU SEE website: "
        "https://eusee.hivos.org/"
    )


# ============================================================
# GENERAL OUTPUT RENDERER
# ============================================================

def render_openai_output(result: dict, chart_instance_key: str | None = None):
    if not isinstance(result, dict):
        return

    answer = str(result.get("answer", "")).strip()
    if answer:
        st.markdown(answer)

    analysis = result.get("analysis")
    if not isinstance(analysis, dict):
        return

    # Exploration/profile summaries.
    payload = analysis.get("analysis", {})
    if isinstance(payload, dict):
        for section_key in ("exploration", "profile"):
            section = payload.get(section_key)
            if not isinstance(section, dict):
                continue

            for key, rows in section.items():
                if key in {"dataset_records", "matching_records", "records"}:
                    continue
                if isinstance(rows, list) and rows:
                    with st.expander(
                        key.replace("_", " ").title(),
                        expanded=False,
                    ):
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

        cross_tab = payload.get("cross_tab")
        if isinstance(cross_tab, dict) and cross_tab.get("data"):
            st.dataframe(
                pd.DataFrame(cross_tab["data"]),
                use_container_width=True,
                hide_index=True,
            )

        change = payload.get("change")
        if isinstance(change, dict):
            st.caption(
                f"Change from {change.get('from_period')} "
                f"to {change.get('to_period')}: "
                + ", ".join(
                    f"{k}={v}" for k, v in (change.get("changes") or {}).items()
                )
            )

    chart = analysis.get("chart")
    if not analysis.get("visualization_requested", False):
        chart = None
    if not isinstance(chart, dict) or not chart:
        records = analysis.get("records")
        if isinstance(records, list) and records:
            st.dataframe(
                pd.DataFrame(records),
                use_container_width=True,
                hide_index=True,
            )
        return

    chart_type = str(chart.get("type", "")).lower().strip()
    chart_data = chart.get("data", [])

    # Heatmap is rendered separately because it requires a matrix.
    if chart_type == "heatmap":
        try:
            chart_df = pd.DataFrame(chart_data)
            if chart_df.empty or len(chart_df.columns) < 2:
                return

            z_df = chart_df.set_index(chart_df.columns[0])
            z_df = z_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            fig = px.imshow(
                z_df,
                text_auto=True,
                aspect="auto",
                title=chart.get("title", "Cross-dimensional distribution"),
                labels={"x": z_df.columns.name or "Category", "y": z_df.index.name or "Category"},
            )
            fig.update_layout(height=480)

            try:
                fig = apply_classic_chart_theme(
                    fig,
                    title=fig.layout.title.text,
                    height=480,
                )
            except Exception:
                pass

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"eusee_openai_chart_{chart_instance_key or uuid.uuid4().hex}",
                config=DEFAULT_PLOTLY_CONFIG,
            )
        except Exception as exc:
            st.warning(f"The AI chart could not be rendered: {exc}")
        return

    chart_df = pd.DataFrame(chart_data)
    x_col = chart.get("x")
    y_col = chart.get("y")

    if chart_df.empty or not x_col or not y_col:
        return
    if x_col not in chart_df.columns or y_col not in chart_df.columns:
        return

    try:
        chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
        chart_df = chart_df.dropna(subset=[y_col])

        title = chart.get("title", "Dashboard analysis")
        labels = {
            x_col: chart.get("x_label", x_col),
            y_col: chart.get("y_label", y_col),
        }
        color_col = chart.get("color")
        if color_col not in chart_df.columns:
            color_col = None

        if chart_type == "line":
            fig = px.line(
                chart_df, x=x_col, y=y_col,
                color=color_col,
                title=title, markers=True, labels=labels,
            )
        elif chart_type == "pie":
            fig = px.pie(
                chart_df, names=x_col, values=y_col, title=title,
            )
        elif chart_type == "donut":
            fig = px.pie(
                chart_df, names=x_col, values=y_col, title=title, hole=0.45,
            )
        elif chart_type == "area":
            fig = px.area(
                chart_df, x=x_col, y=y_col, color=color_col,
                title=title, labels=labels,
            )
        else:
            fig = px.bar(
                chart_df, x=x_col, y=y_col, title=title,
                text=y_col, labels=labels,
            )

        fig.update_layout(height=430)
        try:
            fig = apply_classic_chart_theme(
                fig, title=fig.layout.title.text, height=430
            )
        except Exception:
            pass

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"eusee_openai_chart_{chart_instance_key or uuid.uuid4().hex}",
            config=DEFAULT_PLOTLY_CONFIG,
        )
    except Exception as exc:
        st.warning(f"The AI chart could not be rendered: {exc}")


# ============================================================
# DASHBOARD FILTER SYNCHRONISATION
# ============================================================

def _sync_ai_filters_to_dashboard(filters: dict) -> list[str]:
    """Update the dashboard's existing sidebar filter session keys."""

    unsupported = []

    mapping = {
        "regions": "selected_regions",
        "countries": "selected_countries",
        "alert_types": "selected_alert_types",
        "alert_impacts": "selected_alert_impacts",
        "enabling_principles": "selected_enabling_principle",
        "years": "selected_years",
        "months": "selected_months",
    }

    for ai_key, dashboard_key in mapping.items():
        values = list(
            filters.get(ai_key) or []
        )
        if values:
            st.session_state[
                dashboard_key
            ] = values

    if filters.get("date_from") or filters.get("date_to"):
        unsupported.append(
            "The exact date range is applied to the AI analysis, "
            "but the existing dashboard has no dedicated date-range filter."
        )

    return unsupported


def _clear_dashboard_filters() -> None:
    """Clear only the existing global filter controls."""
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
        st.session_state[key] = []


    return text


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

# ============================================================
# EXISTING EU SEE AI POPOVER BODY
# Appearance intentionally preserved.
# ============================================================

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
            font-family:"Anek Devanagari", Arial, sans-serif;
        ">
            <div style="font-size:9px;font-weight:950;color:#660094;letter-spacing:.14em;text-transform:uppercase;">
                AI assistant
            </div>
            <div style="font-size:16px;font-weight:950;color:#23152F;margin-top:4px;">
                🤖 AI Assistant
            </div>
            <div style="font-size:11px;color:#667085;line-height:1.35;margin-top:5px;">
                Ask me about EU SEE data! For example: What are the trends in digital rights in Southern Africa over the last 3 months?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_permission("use_ai_copilot"):
        st.info(
            "AI Copilot is not enabled for your access level."
        )
        return

    if OpenAI is None:
        st.error(
            "The OpenAI package is not installed. Add `openai` to requirements.txt."
        )
        return

    if _get_eusee_openai_client() is None:
        st.error(
            "OpenAI is not configured. Add [openai] API_KEY to Streamlit secrets."
        )
        return

    load_user_chat_history()

    for i, msg in enumerate(
        st.session_state.eusee_chat_messages[-12:]
    ):
        if not isinstance(msg, dict):
            continue

        role = msg.get(
            "role",
            "assistant",
        )
        content = msg.get(
            "content",
            "",
        )

        with st.chat_message(role):
            if role == "assistant":
                try:
                    stored_result = json.loads(
                        content
                    )
                    render_openai_output(
                        stored_result,
                        chart_instance_key=(
                            f"history_{i}_"
                            f"{msg.get('id', uuid.uuid4().hex)}"
                        ),
                    )
                except Exception:
                    st.markdown(content)
            else:
                st.markdown(content)

    with st.form(
        "eusee_ai_popover_form",
        clear_on_submit=True,
    ):
        user_question = st.text_area(
            "Ask about the complete EUSEE dataset",
            placeholder=(
                "Example: summarise the negative alerts in Africa"
            ),
            height=90,
            label_visibility="collapsed",
            key="eusee_ai_popover_question",
        )

        submitted = st.form_submit_button(
            "Ask ",
            use_container_width=True,
        )

    if submitted and user_question.strip():
        user_question = user_question.strip()

        append_user_chat_message(
            "user",
            user_question,
        )

        with st.spinner(
            "Analysing dashboard data..."
        ):
            result = _process_eusee_ai_request(
                user_question
            )

        # Store the complete local analytical result with the assistant message.
        # This guarantees that reopening the chatbot reproduces the same answer
        # and chart without querying OpenAI again.
        assistant_payload = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

        append_user_chat_message(
            "assistant",
            assistant_payload,
        )

        # If a natural-language filtering request changed the dashboard filters,
        # rerun the dashboard so its existing sidebar and visualisations update.
        if result.get("filter_updated"):
            st.rerun()

        st.rerun()

    st.markdown(
        "<div style='height:10px'></div>",
        unsafe_allow_html=True,
    )

    with st.expander(
        "⚙️ Chat settings",
        expanded=False,
    ):
        st.caption(
            "Conversation history is saved automatically for the signed-in account. "
            "Clearing it cannot be undone."
        )

        if st.button(
            "Clear conversation history",
            use_container_width=True,
            key="eusee_ai_clear_chat_memory",
            type="secondary",
        ):
            clear_user_chat_history()
            _clear_ai_filter_state()
            st.rerun()


def render_eusee_ai_copilot_popover():
    """Render the existing EU SEE Copilot launcher exactly once per run.

    The previous implementation caught ALL exceptions raised inside the
    popover body and then rendered the body again in an expander. If the first
    body render had already registered a Streamlit form, the second render
    reused the same form key and triggered a duplicate-form exception.

    The fallback below is therefore limited to creation of the popover
    container. Exceptions raised by the chatbot body are allowed to surface
    normally instead of causing a second widget tree to be rendered.
    """
    if not has_permission("use_ai_copilot"):
        return

    inject_eusee_ai_popover_css()

    popover = None
    try:
        popover = st.popover(
            "💬 AI assistant",
            use_container_width=False,
        )
    except Exception:
        popover = None

    if popover is not None:
        with popover:
            _render_eusee_ai_copilot_body()
    else:
        with st.expander(
            "💬 AI assistant",
            expanded=False,
        ):
            _render_eusee_ai_copilot_body()


render_eusee_ai_copilot_popover()

