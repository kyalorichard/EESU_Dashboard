"""
EU SEE Dashboard — Premium UX Refactor
=================================================
Production-oriented Streamlit architecture aligned to the final deployment wireframe:
1) Header + live context strip
2) Grouped sidebar filters
3) KPI intelligence row
4) Analytical Flow Panel: Actor → Mechanism → Subject heatmaps + Sankey
5) Map Intelligence Panel with country drill-down and AI-style local insight
6) Negative Alerts deep analytics
7) Global AI Copilot shell
8) Evidence & Records table

Drop this file in your Streamlit app root as app.py.
Expected project structure:
    app.py
    auth.py                       optional, if you already use authentication
    assets/eu-see-logo.png        optional
    assets/footer_logo.png        optional
    exports/output_final.parquet  required data file, or data/output_final.parquet fallback
    exports/countries_metadata.json or data/countries_metadata.json optional
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
from pathlib import Path
from textwrap import shorten
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# Optional authentication layer: uses your existing auth.py when available.
# -----------------------------------------------------------------------------
try:
    from auth import auth_ui, is_authenticated, is_privileged
except Exception:  # pragma: no cover - safe fallback for local testing
    def is_authenticated() -> bool:
        return False

    def is_privileged() -> bool:
        return True

    def auth_ui():
        st.info("Authentication module not available in this environment.")

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    plotly_events = None
    HAS_PLOTLY_EVENTS = False

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EU SEE Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = Path("/exports") if Path("/exports").exists() else BASE_DIR / "exports"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Theme constants
# -----------------------------------------------------------------------------
PURPLE = "#660094"
PURPLE_DARK = "#2D0055"
TEAL = "#008CAA"
YELLOW = "#FFDB58"
NEGATIVE = "#D92D20"
POSITIVE = PURPLE
CONTEXT = TEAL
BG = "#F7F8FB"
BORDER = "#E6E8EF"
MUTED = "#667085"
TEXT = "#232633"
GRID = "#EEF1F6"
FONT = "Inter, Arial, sans-serif"

IMPACT_COLORS = {
    "Negative": NEGATIVE,
    "Positive": POSITIVE,
    "Postive": POSITIVE,
    "Context to watch": CONTEXT,
    "Mixed": "#71717A",
}

REQUIRED_COLUMNS = [
    "alert-country",
    "alert-impact",
    "alert-type",
    "enabling-principle",
    "creation_date",
    "Actor of repression",
    "Subject of repression",
    "Mechanism of repression",
    "Type of event",
]

ENABLING_PRINCIPLE_LABEL_MAP = {
    "association": "Freedom of association",
    "assembly": "Freedom of peaceful assembly",
    "expression": "Freedom of expression",
    "access to information": "Access to information",
    "participation": "Participation in public affairs",
    "resources": "Access to resources",
    "protection": "Protection from interference",
}

# -----------------------------------------------------------------------------
# Global CSS
# -----------------------------------------------------------------------------
def inject_premium_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --purple: {PURPLE};
            --purple-dark: {PURPLE_DARK};
            --teal: {TEAL};
            --negative: {NEGATIVE};
            --yellow: {YELLOW};
            --bg: {BG};
            --border: {BORDER};
            --text: {TEXT};
            --muted: {MUTED};
        }}

        html, body, [class*="css"] {{ font-family: {FONT}; }}
        .stApp {{ background: radial-gradient(circle at top left, rgba(102,0,148,.055), transparent 28%), {BG}; }}
        .main .block-container {{ max-width: 1560px; padding-top: 1.2rem; padding-bottom: 8rem; }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] label {{
            font-size: 11px !important; font-weight: 850 !important; color: #344054 !important;
        }}
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="input"] {{
            border-radius: 12px !important; border: 1px solid #D0D5DD !important;
            background: #FFFFFF !important; box-shadow: 0 1px 2px rgba(16,24,40,.04) !important;
        }}
        section[data-testid="stSidebar"] [data-baseweb="tag"] {{
            background: #F4EAF8 !important; color: var(--purple) !important;
            border-radius: 999px !important; border: 1px solid #E7D4F1 !important;
            font-size: 10px !important; font-weight: 800 !important;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px !important; font-weight: 850 !important;
            border: 1px solid #D0D5DD !important;
            box-shadow: 0 1px 2px rgba(16,24,40,.05) !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--purple) !important; color: var(--purple) !important;
            transform: translateY(-1px); transition: all .16s ease;
        }}

        .app-header {{
            display: flex; justify-content: space-between; align-items: flex-start; gap: 18px;
            padding: 8px 2px 14px 2px; margin-bottom: 6px;
        }}
        .app-title {{ font-size: 36px; line-height: 1.05; font-weight: 950; color: #121527; letter-spacing: -.035em; }}
        .app-title span {{ color: var(--purple); }}
        .app-subtitle {{ font-size: 13px; color: #475467; margin-top: 7px; max-width: 780px; line-height: 1.45; }}
        .context-strip {{
            display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end;
            background:#FFFFFF; border:1px solid #E9DDF2; border-radius:999px; padding:8px 12px;
            box-shadow: 0 8px 20px rgba(45,0,85,.06); color: var(--purple-dark);
            font-size: 11px; font-weight: 900; white-space: nowrap;
        }}
        .context-dot {{ width:4px; height:4px; border-radius:50%; background: var(--purple); opacity:.55; }}

        .section-title-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin: 18px 0 10px 0; }}
        .section-kicker {{ font-size: 10px; font-weight: 950; color: var(--purple); letter-spacing:.13em; text-transform:uppercase; }}
        .section-title {{ font-size: 17px; font-weight: 950; color: #111827; line-height:1.15; }}
        .section-subtitle {{ font-size: 11.5px; color: var(--muted); margin-top: 4px; line-height: 1.35; }}
        .section-badge {{
            border: 1px solid #E7D4F1; color: var(--purple); background:#FBF7FD; border-radius:999px;
            padding: 6px 10px; font-size:10px; font-weight:900; white-space:nowrap;
        }}
        .panel-card {{
            background:#FFFFFF; border:1px solid #E9E2F2; border-radius:20px;
            box-shadow: 0 14px 34px rgba(45,0,85,.055); padding: 14px; margin-bottom: 18px;
        }}
        .mini-note {{ font-size:10.5px; color: var(--muted); line-height:1.35; }}
        .filter-header {{
            background: linear-gradient(135deg,#FFFFFF 0%,#F5EAF8 100%); border:1px solid #E7D4F1;
            border-radius:16px; padding:13px; margin:8px 0 12px 0; box-shadow:0 8px 20px rgba(102,0,148,.08);
        }}
        .filter-eyebrow {{ font-size:9.5px; color:var(--purple); font-weight:950; letter-spacing:.12em; text-transform:uppercase; }}
        .filter-title {{ font-size:15px; font-weight:950; color:#23152F; margin-top:3px; }}
        .filter-note {{ font-size:10.5px; color:var(--muted); margin-top:5px; line-height:1.35; }}
        .active-filter-box {{
            background:#FFFFFF; border:1px solid var(--border); border-radius:14px; padding:10px 11px;
            margin: 10px 0 12px 0; box-shadow:0 4px 12px rgba(16,24,40,.05);
        }}
        .active-filter-line {{ font-size:10.5px; color:#475467; line-height:1.4; }}
        .active-filter-line strong {{ color:var(--purple); }}
        .chip-row {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
        .chip {{ background:#F4EAF8; color:var(--purple); border:1px solid #E7D4F1; border-radius:999px; padding:4px 8px; font-size:9.5px; font-weight:900; }}

        .kpi-card {{
            height: 158px; min-height: 158px; background: radial-gradient(circle at 100% 0%, rgba(102,0,148,.06), transparent 32%), linear-gradient(180deg,#FFFFFF 0%,#FCFAFF 100%);
            border:1px solid rgba(102,0,148,.12); border-radius:18px; padding:14px;
            box-shadow:0 12px 26px rgba(17,24,39,.07), inset 0 1px 0 rgba(255,255,255,.95);
            display:flex; flex-direction:column; justify-content:space-between; position:relative; overflow:hidden;
        }}
        .kpi-card::before {{ content:""; position:absolute; top:0; left:0; right:0; height:4px; background:linear-gradient(90deg,var(--purple),var(--teal),var(--yellow)); }}
        .kpi-top {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
        .kpi-eyebrow {{ color:#8A6AA0; font-size:9px; font-weight:950; letter-spacing:.11em; text-transform:uppercase; }}
        .kpi-title {{ color:#2D0055; font-size:13px; font-weight:950; line-height:1.08; margin-top:4px; }}
        .kpi-icon {{ width:34px; height:34px; border-radius:13px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,rgba(102,0,148,.12),rgba(0,140,170,.10)); color:var(--purple); border:1px solid rgba(102,0,148,.10); font-size:17px; }}
        .kpi-value {{ font-size:36px; font-weight:950; letter-spacing:-.045em; line-height:1; color:#111827; margin-top:8px; }}
        .kpi-note {{ font-size:10.5px; color:var(--muted); font-weight:700; line-height:1.25; }}
        .kpi-pill {{ display:inline-flex; width:fit-content; align-items:center; border-radius:999px; padding:5px 9px; font-size:10px; font-weight:900; margin-top:6px; background:#FBF7FD; border:1px solid #E7D4F1; color:var(--purple); }}

        div[data-testid="stPlotlyChart"] {{
            background:#FFFFFF; border:1px solid #E9E2F2; border-radius:18px; padding:8px 10px 4px 10px;
            box-shadow:0 10px 28px rgba(45,0,85,.055); margin-bottom:14px;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius:16px !important; overflow:hidden !important; border:1px solid #E6E8EF !important;
            box-shadow:0 8px 20px rgba(16,24,40,.055) !important;
        }}
        div[data-testid="stTabs"] [role="tab"] {{
            border-radius: 999px !important; padding: 8px 16px !important; font-weight: 900 !important;
            border: 1px solid #E7D4F1 !important; background:#FFFFFF !important; color:var(--purple-dark) !important;
        }}
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            background: linear-gradient(90deg,var(--purple),#7A1FA2) !important; color:#FFFFFF !important;
            box-shadow: inset 0 -3px 0 var(--yellow) !important;
        }}
        .country-panel {{
            background:linear-gradient(180deg,#FFFFFF 0%,#FCFAFF 100%); border:1px solid #E9DDF2;
            border-radius:18px; padding:15px; min-height:460px; box-shadow:0 12px 28px rgba(45,0,85,.06);
        }}
        .country-row {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; border-bottom:1px solid #EEF0F4; padding:10px 0; font-size:12px; color:#475467; }}
        .country-row strong {{ color:#111827; font-size:13px; text-align:right; }}
        .ai-box {{
            background:#FBF7FD; border:1px solid #E7D4F1; border-radius:15px; padding:12px; margin-top:14px;
            color:#344054; font-size:11.5px; line-height:1.45;
        }}
        .ai-box-title {{ color:var(--purple); font-weight:950; margin-bottom:6px; font-size:12px; }}
        .footer-spacer {{ height: 120px; }}
        @media (max-width: 900px) {{
            .app-header {{ flex-direction:column; }}
            .context-strip {{ justify-content:flex-start; }}
            .kpi-card {{ height:auto; min-height:140px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def title_case_soft(value: object) -> str:
    value = clean_text(value)
    if not value:
        return "Not available"
    return value[:1].upper() + value[1:]


def compact(value: object, width: int = 42) -> str:
    return shorten(clean_text(value) or "Not available", width=width, placeholder="…")


def split_multi(value: object, protected_label: str = "Journalists, media and influencers") -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    text = text.replace("VNSAs", "Violent non-state actors")
    placeholder = "Journalists__MEDIA__and__influencers"
    text = text.replace(protected_label, placeholder)
    parts = [p.strip().replace(placeholder, protected_label) for p in text.split(",")]
    return [p for p in parts if p and p.lower() not in {"nan", "none", "error"}]


def contains_any(value: object, selected_values: Sequence[str]) -> bool:
    if not selected_values:
        return True
    selected_clean = {clean_text(v).lower() for v in selected_values if clean_text(v)}
    if not selected_clean:
        return True
    items = {i.lower() for i in split_multi(value)} or {clean_text(value).lower()}
    return bool(items.intersection(selected_clean))


def count_unique(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique())


def top_value(df: pd.DataFrame, col: str, exploded: bool = False) -> Tuple[str, int, float]:
    if df is None or df.empty or col not in df.columns:
        return "Not available", 0, 0.0
    if exploded:
        values = df[col].dropna().apply(split_multi).explode().dropna().astype(str).str.strip()
    else:
        values = df[col].dropna().astype(str).str.strip()
    values = values[(values != "") & (values.str.lower() != "nan") & (values.str.lower() != "none")]
    if values.empty:
        return "Not available", 0, 0.0
    counts = values.value_counts()
    label, count = str(counts.index[0]), int(counts.iloc[0])
    pct = round((count / max(len(df), 1)) * 100, 1)
    return label, count, pct


def metric_pct(part: int, total: int) -> float:
    return round((part / total) * 100, 1) if total else 0.0

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    parquet_candidates = [EXPORT_DIR / "output_final.parquet", DATA_DIR / "output_final.parquet"]
    meta_candidates = [EXPORT_DIR / "countries_metadata.json", DATA_DIR / "countries_metadata.json"]

    parquet_file = next((p for p in parquet_candidates if p.exists()), None)
    if parquet_file is None:
        st.error("Data file not found. Expected exports/output_final.parquet or data/output_final.parquet.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    try:
        df = pd.read_parquet(parquet_file)
    except Exception as exc:
        st.error(f"Failed to read Parquet file: {exc}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["alert-country"] = df["alert-country"].astype(str).str.strip().replace({
        "Lebanon NAR": "Lebanon",
        "Democratic Republic of Congo 2": "Democratic Republic of the Congo",
        "jose": np.nan,
        "Jose": np.nan,
    })
    df = df[df["alert-country"].notna()]
    df["alert-impact"] = df["alert-impact"].astype(str).str.strip()
    df["alert-type"] = df["alert-type"].astype(str).str.strip()
    df = df[(df["alert-impact"] != "") & (df["alert-type"].str.lower() != "event") & (df["alert-type"] != "")]

    df["Actor of repression"] = df["Actor of repression"].astype(str).str.strip().str.replace(r"\bVNSAs\b", "Violent non-state actors", regex=True)

    country_meta: Dict[str, Dict[str, str]] = {}
    meta_file = next((p for p in meta_candidates if p.exists()), None)
    if meta_file is not None:
        try:
            country_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            st.warning(f"Could not load country metadata: {exc}")

    df["iso_alpha3"] = df["alert-country"].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3"))
    df["continent"] = df["alert-country"].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))

    def continent_to_region(continent: str) -> str:
        if continent == "Africa":
            return "Africa"
        if continent in ["Asia", "Oceania"]:
            return "Asia and the Pacific"
        if continent in ["Europe", "Middle East"]:
            return "The Middle East"
        if continent in ["Americas", "North America", "South America", "Caribbean"]:
            return "Americas and the Caribbean"
        return "Unknown"

    df["region"] = df["continent"].apply(continent_to_region)

    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["year"] = df["creation_date"].dt.year
    df["month_name"] = df["creation_date"].dt.strftime("%B")

    context_mask = df["alert-type"].astype(str).str.strip().str.lower().eq("context to watch")
    df.loc[context_mask, "alert-impact"] = "Context to watch"

    return df.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Plot theme
# -----------------------------------------------------------------------------
def apply_chart_theme(fig: go.Figure, title: str = "", height: int = 360, horizontal: bool = False, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=TEXT),
        title=dict(text=title, x=0.02, xanchor="left", y=0.96, font=dict(size=14, color=PURPLE_DARK, family=FONT)),
        margin=dict(l=130 if horizontal else 48, r=28, t=58, b=56),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(color=TEXT, family=FONT, size=12)),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", title=None, font=dict(size=10), bgcolor="rgba(255,255,255,.84)", bordercolor="#EEF1F6", borderwidth=1),
        showlegend=showlegend,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=True, linecolor="#D8DEE9", ticks="", title=None)
    fig.update_yaxes(showgrid=False if horizontal else True, gridcolor=GRID, zeroline=False, showline=True, linecolor="#D8DEE9", ticks="", title=None)
    return fig


def add_watermark(fig: go.Figure) -> go.Figure:
    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper", yref="paper", x=.5, y=.5, showarrow=False,
        font=dict(size=20, color="black"), opacity=.035, xanchor="center", yanchor="middle",
    )
    return fig

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
def sidebar_multiselect(label: str, options: Iterable, key: str) -> List:
    options = [o for o in sorted(pd.Series(list(options)).dropna().unique().tolist(), key=lambda x: str(x)) if clean_text(o)]
    options_with_all = ["Select all"] + options
    default = st.session_state.get(key, ["Select all"])
    selected = st.multiselect(label, options_with_all, default=[v for v in default if v in options_with_all], key=f"widget_{key}")
    if not selected or "Select all" in selected:
        st.session_state[key] = ["Select all"]
        return options
    st.session_state[key] = selected
    return selected


def render_sidebar(data: pd.DataFrame) -> Dict[str, List]:
    logo_path = ASSETS_DIR / "eu-see-logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.markdown("### EU SEE Dashboard")

    st.sidebar.markdown(
        """
        <div class="filter-header">
            <div class="filter-eyebrow">Dashboard controls</div>
            <div class="filter-title">Global Filters</div>
            <div class="filter-note">Refine the analytical view by geography, alert characteristics, enabling principle, and time period.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("🌍 Geography", expanded=True):
        regions = ["Africa", "The Middle East", "Asia and the Pacific", "Americas and the Caribbean"]
        selected_regions = sidebar_multiselect("Region", [r for r in regions if r in data["region"].unique()], "selected_regions")
        country_scope = data[data["region"].isin(selected_regions)] if selected_regions else data
        selected_countries = sidebar_multiselect("Country", country_scope["alert-country"].dropna().unique(), "selected_countries")

    with st.sidebar.expander("⚠️ Alert Structure", expanded=True):
        selected_impacts = sidebar_multiselect("Impact", data["alert-impact"].dropna().unique(), "selected_alert_impacts")
        selected_types = sidebar_multiselect("Type", data["alert-type"].dropna().unique(), "selected_alert_types")
        principles = data["enabling-principle"].dropna().astype(str).str.split(",").explode().str.strip()
        principles = principles.replace(ENABLING_PRINCIPLE_LABEL_MAP).dropna().unique()
        selected_principles = sidebar_multiselect("Enabling Principle", principles, "selected_enabling_principles")

    with st.sidebar.expander("📅 Time", expanded=True):
        selected_years = sidebar_multiselect("Year", data["year"].dropna().astype(int).unique(), "selected_years")
        month_scope = data[data["year"].isin(selected_years)] if selected_years else data
        month_order = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
        months = sorted(month_scope["month_name"].dropna().unique(), key=lambda m: month_order.get(m, 99))
        selected_months = sidebar_multiselect("Month", months, "selected_months")

    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
        for k in ["selected_regions", "selected_countries", "selected_alert_impacts", "selected_alert_types", "selected_enabling_principles", "selected_years", "selected_months"]:
            st.session_state[k] = ["Select all"]
        st.rerun()

    return {
        "regions": selected_regions,
        "countries": selected_countries,
        "impacts": selected_impacts,
        "types": selected_types,
        "principles": selected_principles,
        "years": selected_years,
        "months": selected_months,
    }


def filter_data(data: pd.DataFrame, filters: Dict[str, List]) -> pd.DataFrame:
    if data.empty:
        return data

    def principle_matches(value: object) -> bool:
        selected = filters["principles"]
        if not selected:
            return True
        raw_items = split_multi(value)
        mapped = [ENABLING_PRINCIPLE_LABEL_MAP.get(i.lower(), title_case_soft(i)) for i in raw_items]
        return bool(set(mapped).intersection(set(selected)))

    mask = (
        data["region"].isin(filters["regions"]) &
        data["alert-country"].isin(filters["countries"]) &
        data["alert-impact"].isin(filters["impacts"]) &
        data["alert-type"].isin(filters["types"]) &
        data["year"].isin(filters["years"]) &
        data["month_name"].isin(filters["months"]) &
        data["enabling-principle"].apply(principle_matches)
    )
    return data.loc[mask].copy()


def render_sidebar_status(df: pd.DataFrame, filters: Dict[str, List]) -> None:
    chips = []
    for label, vals in [("Region", filters["regions"]), ("Country", filters["countries"]), ("Impact", filters["impacts"]), ("Year", filters["years"]), ("Month", filters["months"] )]:
        visible = "All" if len(vals) > 4 else ", ".join(map(str, vals[:4]))
        chips.append(f"<span class='chip'>{label}: {compact(visible, 28)}</span>")
    st.sidebar.markdown(
        f"""
        <div class="active-filter-box">
            <div class="active-filter-line"><strong>{len(df):,}</strong> active records</div>
            <div class="active-filter-line"><strong>{count_unique(df, 'alert-country'):,}</strong> countries · <strong>{count_unique(df, 'year'):,}</strong> years</div>
            <div class="chip-row">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔐 Login / Access**")
    if is_authenticated():
        st.sidebar.success(f"Signed in: {st.session_state.get('name', 'User')}")
        if st.sidebar.button("Logout", use_container_width=True):
            try:
                from auth import logout
                logout()
            except Exception:
                st.session_state.clear()
            st.rerun()
    else:
        st.sidebar.caption("Sign in only when privileged access is needed.")
        if st.sidebar.button("🔐 Sign in / Access", use_container_width=True):
            st.session_state.auth_view = True
            st.rerun()

# -----------------------------------------------------------------------------
# Header and cards
# -----------------------------------------------------------------------------
def render_header(df: pd.DataFrame, filters: Dict[str, List]) -> None:
    region_label = "Multiple regions" if len(filters["regions"]) != 1 else str(filters["regions"][0])
    year_label = "Multiple years" if len(filters["years"]) != 1 else str(int(filters["years"][0]))
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-title"><span>EU SEE</span> Dashboard</div>
                <div class="app-subtitle">Interactive monitoring of civic space alerts across countries, enabling rapid exploration of alert patterns, restrictive pathways, geographic signals, and supporting evidence.</div>
            </div>
            <div class="context-strip">
                <span>{compact(region_label, 26)}</span><span class="context-dot"></span>
                <span>{year_label}</span><span class="context-dot"></span>
                <span>{len(df):,} alerts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", badge: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-title-row">
            <div>
                <div class="section-kicker">EU SEE Intelligence</div>
                <div class="section-title">{title}</div>
                {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
            {f'<div class="section-badge">{badge}</div>' if badge else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(df: pd.DataFrame) -> None:
    total = len(df)
    countries = count_unique(df, "alert-country")
    negative = int((df["alert-impact"] == "Negative").sum()) if not df.empty else 0
    positive = int((df["alert-impact"] == "Positive").sum()) if not df.empty else 0
    context = int((df["alert-impact"] == "Context to watch").sum()) if not df.empty else 0
    top_country, top_country_count, _ = top_value(df, "alert-country")

    c1, c2, c3 = st.columns(3)
    with c1:
        value = f"{countries:,}" if is_privileged() else "On request"
        st.markdown(f"""
        <div class="kpi-card">
            <div><div class="kpi-top"><div><div class="kpi-eyebrow">Coverage</div><div class="kpi-title">Monitored Countries</div></div><div class="kpi-icon">🌍</div></div>
            <div class="kpi-value" style="color:{TEAL};">{value}</div><div class="kpi-pill">Active geographic scope</div></div>
            <div class="kpi-note">Countries represented by the current filters</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div><div class="kpi-top"><div><div class="kpi-eyebrow">Monitoring volume</div><div class="kpi-title">Total Alerts</div></div><div class="kpi-icon">⚠️</div></div>
            <div class="kpi-value" style="color:{PURPLE};">{total:,}</div><div class="kpi-pill">Top country: {compact(top_country, 25)} ({top_country_count:,})</div></div>
            <div class="kpi-note">Filtered records after geography, alert structure, and time filters</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div><div class="kpi-top"><div><div class="kpi-eyebrow">Composition</div><div class="kpi-title">Alerts Breakdown</div></div><div class="kpi-icon">◔</div></div>
            <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                <div class="country-row" style="padding:4px 0;"><span>Negative</span><strong style="color:{NEGATIVE};">{negative:,} ({metric_pct(negative,total)}%)</strong></div>
                <div class="country-row" style="padding:4px 0;"><span>Positive</span><strong style="color:{POSITIVE};">{positive:,} ({metric_pct(positive,total)}%)</strong></div>
                <div class="country-row" style="padding:4px 0;border-bottom:0;"><span>Context</span><strong style="color:{CONTEXT};">{context:,} ({metric_pct(context,total)}%)</strong></div>
            </div></div>
            <div class="kpi-note">Composition by alert impact category</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Chart builders
# -----------------------------------------------------------------------------
def grouped_bar(df: pd.DataFrame, group_col: str, color_col: str, title: str, top_n: int = 15, horizontal: bool = True) -> go.Figure:
    if df.empty or group_col not in df.columns or color_col not in df.columns:
        fig = go.Figure()
        return apply_chart_theme(fig, title, height=360, horizontal=horizontal)
    base_counts = df[group_col].dropna().astype(str).value_counts().head(top_n).index.tolist()
    d = df[df[group_col].isin(base_counts)].groupby([group_col, color_col]).size().reset_index(name="count")
    fig = px.bar(
        d,
        x="count" if horizontal else group_col,
        y=group_col if horizontal else "count",
        color=color_col,
        orientation="h" if horizontal else "v",
        color_discrete_map=IMPACT_COLORS,
        text="count",
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:,}", marker_line=dict(color="rgba(255,255,255,.8)", width=.6), hovertemplate="%{y}<br>%{x:,}<extra></extra>")
    fig.update_layout(barmode="stack")
    return add_watermark(apply_chart_theme(fig, title, height=max(360, min(560, 140 + 28 * len(base_counts))), horizontal=horizontal))


def make_heatmap(df: pd.DataFrame, row_col: str, col_col: str, title: str, top_n: int = 8) -> go.Figure:
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        return apply_chart_theme(go.Figure(), title, height=310)
    xdf = explode_columns(df, [row_col, col_col])
    row_top = xdf[row_col].value_counts().head(top_n).index.tolist()
    col_top = xdf[col_col].value_counts().head(top_n).index.tolist()
    xdf = xdf[xdf[row_col].isin(row_top) & xdf[col_col].isin(col_top)]
    pivot = pd.crosstab(xdf[row_col], xdf[col_col]).reindex(index=row_top, columns=col_top, fill_value=0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[compact(c, 22) for c in pivot.columns],
        y=[compact(r, 24) for r in pivot.index],
        colorscale=[[0, "#F7ECFA"], [.45, "#B48AD0"], [1, PURPLE]],
        hovertemplate="<b>%{y}</b><br>%{x}<br>Count: %{z:,}<extra></extra>",
        colorbar=dict(title="Alerts", thickness=10, len=.72),
    ))
    return apply_chart_theme(fig, title, height=330, horizontal=True, showlegend=False)


def explode_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = "Not available"
        out[col] = out[col].apply(split_multi)
        out = out.explode(col)
        out[col] = out[col].astype(str).str.strip()
        out = out[(out[col] != "") & (out[col].str.lower() != "nan") & (out[col].str.lower() != "none")]
    return out


def make_sankey(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    required = ["Actor of repression", "Mechanism of repression", "Subject of repression"]
    if df.empty or any(c not in df.columns for c in required):
        return apply_chart_theme(go.Figure(), "Actor → Mechanism → Subject Sankey Flow", height=420, showlegend=False)
    xdf = explode_columns(df, required)
    if xdf.empty:
        return apply_chart_theme(go.Figure(), "Actor → Mechanism → Subject Sankey Flow", height=420, showlegend=False)

    actor_top = xdf["Actor of repression"].value_counts().head(top_n).index.tolist()
    mech_top = xdf["Mechanism of repression"].value_counts().head(top_n).index.tolist()
    subj_top = xdf["Subject of repression"].value_counts().head(top_n).index.tolist()
    xdf = xdf[xdf["Actor of repression"].isin(actor_top) & xdf["Mechanism of repression"].isin(mech_top) & xdf["Subject of repression"].isin(subj_top)]

    actor_nodes = [f"Actor: {compact(x, 28)}" for x in actor_top]
    mech_nodes = [f"Mechanism: {compact(x, 28)}" for x in mech_top]
    subj_nodes = [f"Subject: {compact(x, 28)}" for x in subj_top]
    nodes = actor_nodes + mech_nodes + subj_nodes
    node_idx = {node: i for i, node in enumerate(nodes)}

    sources, targets, values, labels = [], [], [], []
    a_m = xdf.groupby(["Actor of repression", "Mechanism of repression"]).size().reset_index(name="count")
    for _, row in a_m.iterrows():
        s = f"Actor: {compact(row['Actor of repression'], 28)}"
        t = f"Mechanism: {compact(row['Mechanism of repression'], 28)}"
        if s in node_idx and t in node_idx:
            sources.append(node_idx[s]); targets.append(node_idx[t]); values.append(int(row["count"])); labels.append(f"{s} → {t}")

    m_s = xdf.groupby(["Mechanism of repression", "Subject of repression"]).size().reset_index(name="count")
    for _, row in m_s.iterrows():
        s = f"Mechanism: {compact(row['Mechanism of repression'], 28)}"
        t = f"Subject: {compact(row['Subject of repression'], 28)}"
        if s in node_idx and t in node_idx:
            sources.append(node_idx[s]); targets.append(node_idx[t]); values.append(int(row["count"])); labels.append(f"{s} → {t}")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=16, thickness=18, line=dict(color="#FFFFFF", width=.8),
            label=nodes,
            color=[PURPLE] * len(actor_nodes) + [TEAL] * len(mech_nodes) + [NEGATIVE] * len(subj_nodes),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels,
            color="rgba(102,0,148,.18)",
            hovertemplate="%{label}<br>Alerts: %{value:,}<extra></extra>",
        ),
    )])
    fig.update_layout(title="Actor → Mechanism → Subject Sankey Flow")
    return apply_chart_theme(fig, "Actor → Mechanism → Subject Sankey Flow", height=430, showlegend=False)

# -----------------------------------------------------------------------------
# Analytical panels
# -----------------------------------------------------------------------------
def render_analytical_flow_panel(df: pd.DataFrame, top_n: int = 8) -> None:
    section_header(
        "Analytical Flow Panel",
        "Core pathway engine showing how restrictive actors, mechanisms, and affected subjects relate under the active filters.",
        badge="Actor → Mechanism → Subject",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(make_heatmap(df, "Actor of repression", "Mechanism of repression", "Actor × Mechanism Heatmap", top_n=top_n), use_container_width=True)
    with c2:
        st.plotly_chart(make_heatmap(df, "Mechanism of repression", "Subject of repression", "Mechanism × Subject Heatmap", top_n=top_n), use_container_width=True)
    with c3:
        st.plotly_chart(make_heatmap(df, "Actor of repression", "Subject of repression", "Actor × Subject Heatmap", top_n=top_n), use_container_width=True)
    st.plotly_chart(make_sankey(df, top_n=top_n), use_container_width=True)


def render_overview_charts(df: pd.DataFrame) -> None:
    section_header("Overview Diagnostics", "High-level distribution of alerts by type, enabling principle, region, and country.", badge="Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grouped_bar(df, "alert-type", "alert-impact", "Alert Type Distribution", top_n=12, horizontal=True), use_container_width=True)
    with c2:
        edf = df.copy()
        edf["enabling-principle-clean"] = edf["enabling-principle"].apply(split_multi)
        edf = edf.explode("enabling-principle-clean")
        edf["enabling-principle-clean"] = edf["enabling-principle-clean"].str.lower().map(ENABLING_PRINCIPLE_LABEL_MAP).fillna(edf["enabling-principle-clean"].apply(title_case_soft))
        st.plotly_chart(grouped_bar(edf, "enabling-principle-clean", "alert-impact", "Alerts Across Enabling Principles", top_n=10, horizontal=True), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(grouped_bar(df, "region", "alert-impact", "Alert Distribution Across Regions", top_n=10, horizontal=False), use_container_width=True)
    with c4:
        st.plotly_chart(grouped_bar(df, "alert-country", "alert-impact", "Alert Distribution Across Countries", top_n=15, horizontal=True), use_container_width=True)

# -----------------------------------------------------------------------------
# Map intelligence
# -----------------------------------------------------------------------------
def map_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["alert-country", "iso_alpha3", "total_alerts", "negative", "positive", "context", "negative_share"])
    d = df.groupby(["alert-country", "iso_alpha3"], dropna=False).agg(
        total_alerts=("alert-country", "size"),
        negative=("alert-impact", lambda s: int((s == "Negative").sum())),
        positive=("alert-impact", lambda s: int((s == "Positive").sum())),
        context=("alert-impact", lambda s: int((s == "Context to watch").sum())),
    ).reset_index()
    d = d[d["iso_alpha3"].notna()]
    d["negative_share"] = d.apply(lambda r: metric_pct(int(r["negative"]), int(r["total_alerts"])), axis=1)
    return d


def make_map(df: pd.DataFrame) -> go.Figure:
    md = map_data(df)
    if md.empty:
        fig = go.Figure()
        fig.add_annotation(text="No geocoded country data available", x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
        return apply_chart_theme(fig, "Global Map Intelligence", height=540, showlegend=False)
    fig = px.choropleth(
        md,
        locations="iso_alpha3",
        color="total_alerts",
        hover_name="alert-country",
        hover_data={"total_alerts": ":,", "negative_share": ":.1f", "negative": ":,", "positive": ":,", "context": ":,", "iso_alpha3": False},
        color_continuous_scale=[[0, "#F4EAF8"], [.5, "#B48AD0"], [1, PURPLE]],
        projection="natural earth",
    )
    fig.update_geos(showframe=False, showcoastlines=True, coastlinecolor="#CDD5DF", showcountries=True, countrycolor="#D0D5DD", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(coloraxis_colorbar=dict(title="Alerts", thickness=12, len=.62))
    return apply_chart_theme(fig, "Global Alerts Density Map", height=560, showlegend=False)


def country_insight(df: pd.DataFrame, country: str) -> str:
    cdf = df[df["alert-country"].astype(str).eq(country)] if country else df
    if cdf.empty:
        return "No records are available for this country under the current filters."
    total = len(cdf)
    neg = int((cdf["alert-impact"] == "Negative").sum())
    actor, _, actor_pct = top_value(cdf, "Actor of repression", exploded=True)
    mech, _, mech_pct = top_value(cdf, "Mechanism of repression", exploded=True)
    subj, _, subj_pct = top_value(cdf, "Subject of repression", exploded=True)
    return (
        f"Recent alerts in {country} are concentrated around **{compact(actor, 42)}** as the dominant actor "
        f"({actor_pct}% of records), mainly through **{compact(mech, 42)}** ({mech_pct}%), "
        f"affecting **{compact(subj, 42)}** ({subj_pct}%). Negative alerts represent "
        f"**{metric_pct(neg, total)}%** of this country view."
    )


def render_map_intelligence(df: pd.DataFrame) -> None:
    section_header("Map Intelligence Panel", "Geographic distribution of alerts with country-level diagnostic context.", badge="Clickable country view")
    md = map_data(df)
    default_country = md.sort_values("total_alerts", ascending=False)["alert-country"].iloc[0] if not md.empty else ""

    map_col, panel_col = st.columns([2.4, 1])
    with map_col:
        fig = make_map(df)
        selected_country = default_country
        clicked = None
        if HAS_PLOTLY_EVENTS:
            clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=560, key="country_map_click")
            if clicked:
                loc = clicked[0].get("location")
                match = md.loc[md["iso_alpha3"].eq(loc), "alert-country"]
                if not match.empty:
                    selected_country = match.iloc[0]
        else:
            st.plotly_chart(fig, use_container_width=True, key="map_no_click")

    with panel_col:
        countries = md.sort_values("total_alerts", ascending=False)["alert-country"].tolist() if not md.empty else []
        if countries:
            selected_country = st.selectbox("Country drill-down", countries, index=countries.index(default_country) if default_country in countries else 0, key="country_drilldown")
        cdf = df[df["alert-country"].astype(str).eq(selected_country)] if selected_country else df.iloc[0:0]
        total = len(cdf)
        neg = int((cdf["alert-impact"] == "Negative").sum()) if not cdf.empty else 0
        actor, _, _ = top_value(cdf, "Actor of repression", exploded=True)
        mech, _, _ = top_value(cdf, "Mechanism of repression", exploded=True)
        subj, _, _ = top_value(cdf, "Subject of repression", exploded=True)
        st.markdown(f"""
        <div class="country-panel">
            <div class="section-kicker">Country intelligence</div>
            <div class="section-title" style="margin-top:4px;">{selected_country or 'No country selected'}</div>
            <div class="country-row"><span>Total alerts</span><strong>{total:,}</strong></div>
            <div class="country-row"><span>Negative share</span><strong style="color:{NEGATIVE};">{metric_pct(neg,total)}%</strong></div>
            <div class="country-row"><span>Top actor</span><strong>{compact(actor, 24)}</strong></div>
            <div class="country-row"><span>Top mechanism</span><strong>{compact(mech, 24)}</strong></div>
            <div class="country-row"><span>Top subject</span><strong>{compact(subj, 24)}</strong></div>
            <div class="ai-box"><div class="ai-box-title">✦ AI Insight</div>{country_insight(df, selected_country)}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Negative alerts deep analytics
# -----------------------------------------------------------------------------
def render_negative_filters(negative_df: pd.DataFrame) -> pd.DataFrame:
    if negative_df.empty:
        return negative_df
    xdf = explode_columns(negative_df, ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"])
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        actors = st.multiselect("Restrictive actors", sorted(xdf["Actor of repression"].dropna().unique()), key="neg_actor_filter")
    with c2:
        mechanisms = st.multiselect("Restrictive mechanisms", sorted(xdf["Mechanism of repression"].dropna().unique()), key="neg_mechanism_filter")
    with c3:
        subjects = st.multiselect("Affected actors", sorted(xdf["Subject of repression"].dropna().unique()), key="neg_subject_filter")
    with c4:
        events = st.multiselect("Negative event types", sorted(xdf["Type of event"].dropna().unique()), key="neg_event_filter")
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = negative_df.copy()
    if actors:
        filtered = filtered[filtered["Actor of repression"].apply(lambda x: contains_any(x, actors))]
    if mechanisms:
        filtered = filtered[filtered["Mechanism of repression"].apply(lambda x: contains_any(x, mechanisms))]
    if subjects:
        filtered = filtered[filtered["Subject of repression"].apply(lambda x: contains_any(x, subjects))]
    if events:
        filtered = filtered[filtered["Type of event"].apply(lambda x: contains_any(x, events))]
    return filtered


def render_negative_kpis(negative_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    total = len(negative_df)
    all_total = len(all_df)
    countries = count_unique(negative_df, "alert-country")
    actor, actor_count, actor_pct = top_value(negative_df, "Actor of repression", exploded=True)
    mech, mech_count, mech_pct = top_value(negative_df, "Mechanism of repression", exploded=True)
    subj, subj_count, subj_pct = top_value(negative_df, "Subject of repression", exploded=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card"><div><div class="kpi-top"><div><div class="kpi-eyebrow">Coverage</div><div class="kpi-title">Countries with Negative Alerts</div></div><div class="kpi-icon">🌍</div></div>
        <div class="kpi-value" style="color:{TEAL};">{countries:,}</div><div class="kpi-pill">Negative-alert scope</div></div><div class="kpi-note">Countries represented by current negative filters</div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card"><div><div class="kpi-top"><div><div class="kpi-eyebrow">Negative alert scope</div><div class="kpi-title">Negative Alerts</div></div><div class="kpi-icon">⚠️</div></div>
        <div class="kpi-value" style="color:{NEGATIVE};">{total:,}</div><div class="kpi-pill">{metric_pct(total, all_total)}% of filtered alerts</div></div><div class="kpi-note">Focused diagnostic view for restrictive events only</div></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card"><div><div class="kpi-top"><div><div class="kpi-eyebrow">Dominant pathway</div><div class="kpi-title">Actor → Mechanism → Subject</div></div><div class="kpi-icon">⛓️</div></div>
        <div style="margin-top:10px;display:flex;flex-direction:column;gap:5px;">
            <div class="country-row" style="padding:3px 0;"><span>Actor</span><strong>{compact(actor, 22)} · {actor_pct}%</strong></div>
            <div class="country-row" style="padding:3px 0;"><span>Mechanism</span><strong>{compact(mech, 22)} · {mech_pct}%</strong></div>
            <div class="country-row" style="padding:3px 0;border-bottom:0;"><span>Subject</span><strong>{compact(subj, 22)} · {subj_pct}%</strong></div>
        </div></div><div class="kpi-note">Use the flow panel below to inspect relationships</div></div>
        """, unsafe_allow_html=True)


def render_negative_deep_analytics(df: pd.DataFrame) -> None:
    negative_df = df[df["alert-impact"] == "Negative"].copy()
    section_header("Negative Alerts Deep Analytics", "Dedicated restrictive-events workspace with local filters and pathway diagnostics.", badge="Deep analytics")
    if negative_df.empty:
        st.warning("No negative alerts are available under the current global filters.")
        return
    filtered_negative = render_negative_filters(negative_df)
    render_negative_kpis(filtered_negative, df)
    render_analytical_flow_panel(filtered_negative, top_n=8)
    render_map_intelligence(filtered_negative)
    render_evidence_table(filtered_negative, key="negative_evidence", title="Negative Evidence & Records")

# -----------------------------------------------------------------------------
# AI Copilot local layer
# -----------------------------------------------------------------------------
def local_ai_answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    total = len(df)
    if total == 0:
        return "No records are available under the current filters. Adjust filters and try again."
    top_country, country_count, _ = top_value(df, "alert-country")
    top_actor, _, actor_pct = top_value(df, "Actor of repression", exploded=True)
    top_mech, _, mech_pct = top_value(df, "Mechanism of repression", exploded=True)
    top_subj, _, subj_pct = top_value(df, "Subject of repression", exploded=True)
    negative_count = int((df["alert-impact"] == "Negative").sum())

    if "country" in q or "where" in q or "high" in q:
        return f"The strongest geographic signal is **{top_country}** with **{country_count:,}** alerts. Negative alerts make up **{metric_pct(negative_count,total)}%** of the current view."
    if "driver" in q or "why" in q or "pathway" in q:
        return f"The dominant pathway is **{top_actor} → {top_mech} → {top_subj}**. Actor concentration is {actor_pct}%, mechanism concentration is {mech_pct}%, and subject concentration is {subj_pct}% within the active view."
    if "compare" in q or "region" in q:
        region_counts = df["region"].value_counts().head(4)
        return "Regional comparison: " + "; ".join([f"**{idx}**: {val:,} alerts" for idx, val in region_counts.items()]) + "."
    return (
        f"Current view contains **{total:,}** alerts across **{count_unique(df, 'alert-country')}** countries. "
        f"Top country: **{top_country}**. Dominant pathway: **{top_actor} → {top_mech} → {top_subj}**. "
        f"Negative share: **{metric_pct(negative_count,total)}%**."
    )


def render_ai_copilot(df: pd.DataFrame) -> None:
    section_header("AI Copilot", "Ask questions about the current filtered view, request explanations, or generate quick interpretive summaries.", badge="Context-aware")
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("**Ask the dashboard**")
        examples = ["Why is the top country high?", "What are the top drivers of repression?", "Compare civic space trends across regions", "Which countries need attention?"]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{abs(hash(ex))}"):
                st.session_state.ai_messages.append((ex, local_ai_answer(ex, df)))
                st.rerun()
        question = st.text_input("Ask anything about the current data", placeholder="e.g., Explain the dominant restrictive pathway", key="ai_question")
        if st.button("Ask Copilot", type="primary", use_container_width=True):
            if question.strip():
                st.session_state.ai_messages.append((question, local_ai_answer(question, df)))
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("**Latest insight**")
        if st.session_state.ai_messages:
            q, a = st.session_state.ai_messages[-1]
            st.markdown(f"**Q:** {q}")
            st.markdown(a)
        else:
            st.markdown(local_ai_answer("summarize", df))
        st.download_button("Download current AI summary", data=local_ai_answer("summarize", df), file_name="eusee_ai_summary.txt", mime="text/plain", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Evidence table and footer
# -----------------------------------------------------------------------------
def render_evidence_table(df: pd.DataFrame, key: str = "evidence", title: str = "Evidence & Records") -> None:
    section_header(title, "Search, inspect, and export the filtered records supporting the dashboard visuals.", badge="Live filtered view")
    if df.empty:
        st.info("No records are available for the current selection.")
        return

    rename_map = {
        "post_title": "Title",
        "summary": "Event summary",
        "creation_date": "Date",
        "alert-country": "Country",
        "alert-impact": "Impact",
        "alert-type": "Type",
        "Actor of repression": "Actor",
        "Mechanism of repression": "Mechanism",
        "Subject of repression": "Subject",
        "enabling-principle": "Enabling principles",
    }
    cols = [c for c in rename_map if c in df.columns]
    display_df = df[cols].rename(columns=rename_map).copy()
    if "Date" in display_df.columns:
        display_df["Date"] = pd.to_datetime(display_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.4, 1.2, .8])
    with c1:
        search = st.text_input("Search records", placeholder="Search country, actor, mechanism, subject...", key=f"{key}_search")
    with c2:
        selected_cols = st.multiselect("Columns", display_df.columns.tolist(), default=display_df.columns.tolist(), key=f"{key}_cols")
    with c3:
        row_limit = st.selectbox("Rows", [25, 50, 100, 250, 500, "All"], index=1, key=f"{key}_rows")

    table_df = display_df[selected_cols] if selected_cols else display_df
    if search.strip():
        q = search.strip().lower()
        table_df = table_df[table_df.astype(str).apply(lambda r: r.str.lower().str.contains(q, na=False).any(), axis=1)]
    view_df = table_df if row_limit == "All" else table_df.head(int(row_limit))

    st.caption(f"Showing {len(view_df):,} rows from {len(table_df):,} matching records and {len(df):,} active records.")
    st.dataframe(view_df, use_container_width=True, hide_index=True, height=min(560, max(280, 36 * min(len(view_df), 12) + 80)))
    st.download_button("⬇️ Download filtered table as CSV", data=table_df.to_csv(index=False).encode("utf-8"), file_name=f"{key}.csv", mime="text/csv", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_feedback_and_footer() -> None:
    feedback_url = "https://forms.office.com/pages/responsepage.aspx?id=aFcOUAlSoUeqnjS7rLiI3i2QH6350xBGsugTt9B-i59URUk5UEFTV0VKSDRaU0lXTEc1S1g1M0hYTi4u&route=shorturl"
    st.markdown(
        f"""
        <div class="panel-card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
            <div><strong style="color:{PURPLE_DARK};">Help improve the EUSEE Dashboard</strong><br><span class="mini-note">Share feedback on usability, insights, and deployment readiness.</span></div>
            <a href="{feedback_url}" target="_blank" style="background:linear-gradient(90deg,{PURPLE},{TEAL});color:white;text-decoration:none;border-radius:999px;padding:9px 14px;font-size:12px;font-weight:900;">Formular ausfüllen</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer_path = ASSETS_DIR / "footer_logo.png"
    if footer_path.exists():
        try:
            b64 = base64.b64encode(footer_path.read_bytes()).decode()
            components.html(
                f"""
                <div style="position:fixed;bottom:0;left:0;width:100%;text-align:center;padding:8px 0;background:white;border-top:1px solid #E6E8EF;z-index:9999;">
                    <img src="data:image/png;base64,{b64}" style="max-width:900px;width:80%;height:auto;">
                </div>
                """,
                height=80,
            )
        except Exception:
            pass
    st.markdown("<div class='footer-spacer'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:gray;font-size:11px;'>© 2026 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    inject_premium_css()

    st.session_state.setdefault("auth_view", False)
    if is_authenticated():
        st.session_state.auth_view = False
    if st.session_state.get("auth_view", False) and not is_authenticated():
        auth_ui()
        st.stop()

    data = load_data()
    if data.empty:
        st.stop()

    filters = render_sidebar(data)
    filtered = filter_data(data, filters)
    render_sidebar_status(filtered, filters)

    render_header(filtered, filters)
    render_kpi_row(filtered)

    tabs = st.tabs([
        "1  Main Dashboard",
        "2  Negative Alerts",
        "3  Map Intelligence",
        "4  AI Copilot",
        "5  Evidence & Records",
    ])

    with tabs[0]:
        render_analytical_flow_panel(filtered, top_n=8)
        render_overview_charts(filtered)
        render_map_intelligence(filtered)
        render_evidence_table(filtered, key="overview_evidence", title="Evidence & Records")

    with tabs[1]:
        render_negative_deep_analytics(filtered)

    with tabs[2]:
        render_map_intelligence(filtered)
        render_overview_charts(filtered)

    with tabs[3]:
        render_ai_copilot(filtered)

    with tabs[4]:
        render_evidence_table(filtered, key="all_evidence", title="Evidence & Records")

    render_feedback_and_footer()


if __name__ == "__main__":
    main()
