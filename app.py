import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import streamlit.components.v1 as components
import plotly.graph_objects as go
import base64
from auth import auth_ui, is_privileged
import math
import paramiko
import logging
import tempfile  
import os
import re

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

# ---------------- DASHBOARD TITLE WITH ANIMATED DIVIDER AND TITLE ----------------
st.markdown("""
<!-- Container for animations -->
<div style="overflow: hidden;">

<h1 class="animated-title">
    EU SEE Dashboard
</h1>

<!-- Animated divider -->
<div class="animated-divider"></div>

<div class="animated-subtitle">
    This interactive dashboard allows exploration and analysis of data produced by the EU SEE project.
    It aggregates information reported by Network Members across 86 countries to document trends 
    in the enabling environment for civil society.
</div>

</div>

<style>
/* ---------------- Title ---------------- */
.animated-title {
    margin: 0 0 6px 0;
    line-height: 1.1;
    color: #660094;
    font-size: 48px;
    font-family: Arial, sans-serif;
    font-weight: 700;
    opacity: 0;
    transform: translateY(-20px);
    animation: titleFadeSlide 0.8s ease-out forwards;
    animation-delay: 0.2s;
}

/* Title animation */
@keyframes titleFadeSlide {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------------- Divider ---------------- */
.animated-divider {
    width: 15%;
    max-width: 120px;
    height: 4px;
    background: linear-gradient(to right, #FFDB58, #660094);
    border-radius: 2px;
    margin-bottom: 16px;
    opacity: 0;
    transform: translateX(-120%);
    animation: dividerSlide 1s ease-out forwards;
    animation-delay: 0.6s;
}

@keyframes dividerSlide {
    from { transform: translateX(-120%); opacity: 0; }
    to   { transform: translateX(0); opacity: 1; }
}

/* ---------------- Subtitle ---------------- */
.animated-subtitle {
    font-size: 14px;
    font-family: Arial, sans-serif;
    color: #333333;
    margin-bottom: 20px;
    max-width: 900px;
    line-height: 1.5;
    opacity: 0;
    animation: subtitleFade 0.8s ease-out forwards;
    animation-delay: 1.0s;
}

@keyframes subtitleFade {
    from { opacity: 0; }
    to   { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=0)
def load_data():
    parquet_file = EXPORT_DIR / "output_final.parquet"
    
    meta_file = EXPORT_DIR / "countries_metadata.json"

   # parquet_file = Path.cwd() / "data" / "output_final.parquet"
    #meta_file = Path.cwd() / "data" / "countries_metadata.json"

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

    # --- Step 2: Basic cleaning ---
    for col in ['alert-country', 'alert-impact', 'Actor of repression']:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in dataset.")
            df[col] = ""

    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'].str.lower() != "jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Clean country names
    
    df['alert-country'] = df['alert-country'].replace({
        "Lebanon NAR": "Lebanon",
        "Democratic Republic of Congo 2": "Democratic Republic of the Congo"
    })

    # ❗ REMOVE alert-type == "event"    
    df['alert-type'] = df['alert-type'].astype(str).str.strip()

    df = df[
        (df['alert-type'].str.lower() != "event") & 
        (df['alert-type'] != "")
    ]

    # Clean Actor of repression
    df['Actor of repression'] = df['Actor of repression'].astype(str).str.strip()
    df['Actor of repression'] = df['Actor of repression'].replace({"VNSAs": "Violent non-state actors"})

    # --- Step 3: Load metadata ---
    country_meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, encoding="utf-8") as f:
                country_meta = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load countries metadata: {e}")
    else:
        st.warning(f"Countries metadata JSON not found: {meta_file}")

    # --- Step 4: Map ISO codes and continent ---
    df['iso_alpha3'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3", None))
    df['continent'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))

    # --- Step 5: Map continent to region ---
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

    df['region'] = df['continent'].apply(continent_to_region)

    # --- Step 6: Warn about missing ISO codes ---
    missing_countries = (
        df.loc[df['iso_alpha3'].isna(), 'alert-country']
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.lower() != "none"]
        .unique()
    )

    if len(missing_countries) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing_countries)}")

    # --- Step 7: Process dates ---
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    else:
        st.warning("No 'creation_date' column found in dataset.")
    
    # --- Step 8: Update alert-impact based on alert-type ---
    
    if 'alert-type' in df.columns and 'alert-impact' in df.columns:
        mask = df['alert-type'].astype(str).str.strip().str.lower() == 'context to watch'
        df.loc[mask, 'alert-impact'] = 'Context to watch'

    return df

# --- Load data safely ---
data = load_data()


# ---------------- MULTISELECT WITH SELECT ALL ----------------
def safe_multiselect(label, options, session_key, sidebar=True):
    options = sorted(list(options))
    
    # Always keep "Select All" as first dropdown option
    options_with_all = ["Select all"] + options

    # Initialize session_state if not present
    if session_key not in st.session_state:
        st.session_state[session_key] = options.copy()  # internally select all

    # Determine what to display in the widget
    displayed_default = []  # show nothing selected in the dropdown
    try:
        if sidebar:
            selected = st.sidebar.multiselect(label, options_with_all, default=displayed_default)
        else:
            selected = st.multiselect(label, options_with_all, default=displayed_default)
    except Exception:
        selected = []

    # If user selects "Select All" or nothing, internally select all options
    if "Select all" in selected or len(selected) == 0:
        st.session_state[session_key] = options.copy()
        return options
    else:
        st.session_state[session_key] = selected
        return selected
        
# ---------------- GLOBAL FILTERS (COMPACT SIDEBAR) ----------------
st.sidebar.image("assets/eu-see-logo.png", width=400)

# Global Filters
st.sidebar.markdown(
    '<div style="font-family: Arial; font-size: 14px; font-weight: bold; color: purple;">🌍 Global Filters</div>',
    unsafe_allow_html=True
)

# Apply global CSS to sidebar to remove spacing between label and dropdown
st.markdown("""
    <style>
    /* Sidebar dropdown text */
    .css-1hwfws3 {  /* The selected item in multiselect/selectbox */
        font-family: Arial !important;
        font-size: 12px !important;
        color: purple !important;
    }

    /* Dropdown options list */
    .css-1n76uvr div[role="option"] {
        font-family: Arial !important;
        font-size: 12px !important;
        color: purple !important;
    }

    /* Placeholder text in dropdown */
    .css-1wa3eu0-placeholder {
        font-family: Arial !important;
        font-size: 12px !important;
        color: purple !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar filters (no separate markdown)
regions_labels = ["Africa", "The Middle East", "Asia and the Pacific", "Americas and the Caribbean"]
selected_regions = safe_multiselect("Select region", regions_labels, "selected_regions")

filtered_countries = data[data['region'].isin(selected_regions)] if "Select all" not in selected_regions else data
selected_countries = safe_multiselect("Select country", filtered_countries['alert-country'].dropna().unique(), "selected_countries")

selected_alert_impacts = safe_multiselect("Select nature of event/alert", data['alert-impact'].dropna().unique(), "selected_alert_impacts")
selected_alert_types = safe_multiselect("Select impact of alert", data['alert-type'].dropna().unique(), "selected_alert_types")

selected_enabling_principle = safe_multiselect(
    "Select enabling principle", 
    data['enabling-principle'].dropna().str.split(",").explode().str.strip().str.capitalize().unique(),
    "selected_enabling_principle"
)
selected_years = safe_multiselect("Select year", sorted(data['year'].dropna().unique()), "selected_years")

available_months = sorted(
    data['month_name'].dropna().unique(),
    key=lambda m: pd.to_datetime(m, format='%B').month
) if "Select All" in selected_years else sorted(
    data[data['year'].isin(selected_years)]['month_name'].dropna().unique(),
    key=lambda m: pd.to_datetime(m, format='%B').month
)
selected_months = safe_multiselect("Select month", available_months, "selected_months")
# Reset button
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_regions","selected_countries","selected_alert_types","selected_enabling_principle",
                "selected_alert_impacts","selected_months","selected_years"]:
        st.session_state[key] = ["Select all"]

# ---------------- FILTER DATA ----------------
def contains_any(cell_value, selected_values):
    if pd.isna(cell_value): return False
    return any(sel in str(cell_value) for sel in selected_values)

filtered_global = data[
    (data['region'].isin(selected_regions)) &
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# Login / Access
st.sidebar.markdown(
    '<div style="font-family: Arial; font-size: 14px; font-weight: bold; color: purple;">🔐 Login / Access</div>',
    unsafe_allow_html=True
)

auth_ui()

if st.session_state.get("user") and st.session_state.get("email_verified"):
    st.write(f"Hello, {st.session_state.name}!")
   
else:
    st.info("")

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
def render_summary_cards(df, base_bar_height=25,show_breakdown=True):
    """
    Render three summary cards with gradient background:
    1. Monitored Countries
    2. Total Alerts
    3. Alerts Breakdown (Negative vs Positive vs Context to watch)
    
    Parameters:
        df (DataFrame): Filtered data
        base_bar_height (int): Base height of the horizontal bar
    """
    total_countries = df['alert-country'].nunique() if not df.empty else 0
    total_alerts = len(df) if not df.empty else 0
    negative = (df['alert-impact'] == "Negative").sum() if not df.empty else 0
    positive = (df['alert-impact'] == "Positive").sum() if not df.empty else 0
    context = (df['alert-impact'] == "Context to watch").sum() if not df.empty else 0
    total_np = negative + positive + context

    # Percentages
    neg_pct = round((negative / total_np) * 100, 1) if total_np else 0
    pos_pct = round((positive / total_np) * 100, 1) if total_np else 0
    context_pct = round((context / total_np) * 100, 1) if total_np else 0

    # Adjust bar height and font size based on total alerts
    bar_height = max(base_bar_height, min(50, total_alerts // 10 + 20))
    font_size = max(12, min(16, 14 - int(total_alerts/100)))

   
    bar_height = 30
    font_size = 12

    col1, col2, col3 = st.columns(3)

    # Base card style
    card_style = """
    background: #FFFFFF;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin: 5px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s;
    """

    icon_style = """
    width: 50px; 
    height: 50px; 
    border-radius: 50%; 
    background: #008CAA; 
    color: white; 
    display: flex; 
    align-items:center; 
    justify-content:center; 
    font-size:24px; 
    font-weight:bold;
    margin-bottom:10px;
    """

    # --- Monitored Countries ---
    if is_privileged():
        with col1:
            st.markdown(f"""
        <div style="{card_style}">
            <div style="{icon_style}">🌍</div>
            <span style="font-size:16px; font-weight:600; color:#555;">Monitored Countries</span>
            <span style="font-size:36px; font-weight:bold; color:#008CAA; margin-top:5px;">{total_countries}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(f"""
        <div style="{card_style}">
            <div style="{icon_style}">🌍</div>
            <span style="font-size:16px; font-weight:600; color:#555;">Monitored Countries</span>
            <span style="font-size:20px; font-weight:bold; color:#008CAA; margin-top:5px;">Available on Request</span>
        </div>
        """, unsafe_allow_html=True)
            

    # --- Total Alerts ---
    with col2:
        st.markdown(f"""
    <style>
    .tooltip-box {{
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: #008CAA;
        font-weight: bold;
    }}
    .tooltip-box .tooltiptext {{
        visibility: hidden;
        width: 260px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 130%;
        left: 50%;
        margin-left: -130px;
        opacity: 0;
        transition: opacity 0.3s;
        font-family: Arial;
        font-size: 12px;
    }}
    .tooltip-box:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>

    <div style="{card_style}">
        <div style="{icon_style}">⚠️</div>
        <span style="font-size:16px; font-weight:600; color:#555;">
            Total Alerts
            <span class="tooltip-box">?
                <span class="tooltiptext">
                    Higher numbers of alerts do not always indicate a worse situation; 
                    they may reflect better reporting or different thresholds across countries.
                </span>
            </span>
        </span>
        <span style="font-size:36px; font-weight:bold; color:#FF6F61; margin-top:5px;">{total_alerts}</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Alerts Breakdown ----------------
    with col3:
        st.markdown(f'''
    <div style="{card_style} ; padding:2px 10px;">
        <svg width="120" height="120">
            <circle cx="60" cy="60" r="50" stroke="#e0e0e0" stroke-width="12" fill="none"/>
            <circle cx="60" cy="60" r="50" stroke="#FFDB58" stroke-width="12" fill="none"
                stroke-dasharray="{2*3.1416*50}" 
                stroke-dashoffset="{2*3.1416*50*(1-(neg_pct/100))}"
                stroke-linecap="round" transform="rotate(-90 60 60)">
                <title>Negative Alerts: {negative} ({neg_pct}%)</title>
            </circle>            
            <circle cx="60" cy="60" r="40" stroke="#008CAA" stroke-width="12" fill="none"
                stroke-dasharray="{2*3.1416*40}" 
                stroke-dashoffset="{2*3.1416*40*(1-(context_pct/100))}"
                stroke-linecap="round" transform="rotate(-90 60 60)">
                <title>Context to watch Alerts: {context} ({context_pct}%)</title>
            </circle>
            <circle cx="60" cy="60" r="30" stroke="#660094" stroke-width="12" fill="none"
                stroke-dasharray="{2*3.1416*30}" 
                stroke-dashoffset="{2*3.1416*30*(1-(pos_pct/100))}"
                stroke-linecap="round" transform="rotate(-90 60 60)">
                <title>Positive Alerts: {positive} ({pos_pct}%)</title>
            </circle>
            <text x="40" y="50" text-anchor="middle" font-size="12" font-weight="bold" fill="#660094">
                {pos_pct}%
            </text>
            <text x="40" y="80" text-anchor="middle" font-size="12" font-weight="bold" fill="#FFDB58">
                {neg_pct}%
            </text>
            <text x="70" y="65" text-anchor="middle" font-size="12" font-weight="bold" fill="#008CAA">
                {context_pct}%
            </text>
        </svg>
        <div style="margin-top:10px; font-size:16px; font-weight:600; color:#555;">
            Alerts Breakdown
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; padding:6px 2px; font-size:12px; font-weight:700; width:100%; gap:20px;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:10px; height:10px; background:#FFDB58; border-radius:30%; display:inline-block;"></span>
                <span style="color:#555;">Negative:</span>
                <span style="color:#FFDB58;">{negative}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:10px; height:10px; background:#660094; border-radius:30%; display:inline-block;"></span>
                <span style="color:#555;">Positive:</span>
                <span style="color:#660094;">{positive}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:10px; height:10px; background:#008CAA; border-radius:40%; display:inline-block;"></span>
                <span style="color:#555;">Context to Watch:</span>
                <span style="color:#008CAA;">{context}</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
 
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

# ---------------- DYNAMIC BAR CHART ----------------
def create_bar_chart(df, x, y, title=None, horizontal=False, color_col=None,normalize_labels=True):
   
    df = df.copy()

    # ---------------- Safe numeric conversion for y ----------------
    #df[y] = pd.to_numeric(df[y], errors='coerce').fillna(0)

    num_bars = df.shape[0]
    height = max(350, num_bars * 25)  # Auto height based on number of bars
    font_size = max(12, 14 - int(num_bars / 5))  # Dynamic font size

    # Optional: wrap labels (assuming wrap_label_by_words exists)
    
    # ---------------- Label handling ----------------
    if normalize_labels:
        df[x] = df[x].apply(
            lambda l: wrap_label_by_words(
                normalize_label(l) if x not in ["alert-country", "region"] else str(l),
                words_per_line=3
            )
        )
    else:
        df[x] = df[x].astype(str).apply(
            lambda l: wrap_label_by_words(l, words_per_line=3)
        )
        
    # Move "Other" category to the end if present
    if "Other" in df[x].values:
        df_other = df[df[x] == "Other"]
        df_main = df[df[x] != "Other"]
        df = pd.concat([df_main, df_other], ignore_index=True)
        
    # For horizontal charts, reverse order so "Other" is at bottom
        if horizontal:
            df = df[::-1].reset_index(drop=True)
    # Create bar chart
    fig = px.bar(
        df,
        x=x if not horizontal else y,
        y=y if not horizontal else x,
        orientation='h' if horizontal else 'v',
        color=color_col,
        #color_discrete_map=COLOR_MAPPING if color_col else None,
        color_discrete_sequence=['#FFDB58'],  # yellow color for all bars
        text=y
    )

    # Text positions (inside if large enough, otherwise outside)
    fig.update_traces(
        textposition=['inside' if val > 25 else 'outside' for val in df[y]],
        insidetextanchor='end',
        textfont=dict(size=10, color='black', family="Arial black")
    )

    # Bold axis lines
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')

    # Grid and axis
    fig.update_xaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Layout
    fig.update_layout(
        height=height,
        margin=dict(l=120 if horizontal else 20, r=20, t=40, b=20),
        title=dict(text=title, x=0.5, xanchor='center',font=dict(color="#660094",family="Arial black", size=12))
    )

    # ---------------- Dynamic download-only source ----------------
    if horizontal:
        # For horizontal bars, find max x for positioning
        max_val = df[x].sum()
    else:
        # For vertical bars, find max y for positioning
        max_val = df[x].sum()
   
    # ---------------- WATERMARK ----------------
    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=20,
            color="black"
        ),
        #textangle=-30,
        opacity=0.05,
        xanchor="center",
        yanchor="middle"
    )  
    return fig

# ---------------- HORIZONTAL STACKED BAR ----------------
def create_h_stacked_bar(df, y, x="count", color_col="alert-impact",title=None, horizontal=False, normalize_labels=True):
    categories = sorted(df[color_col].unique())
    #color_sequence = ['#008CAA','#660094','#FFDB58']

    # ---------------- Define category-to-color mapping ----------------
    category_colors = {
        "Context to watch": "#008CAA",
        "Positive": "#660094",
        "Postive": "#660094",
        "Negative": "#FFDB58"
    }
    
    categories = sorted(df[color_col].unique())
   
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat].copy()
         # ---------------- Label handling ----------------
        if normalize_labels:
            df_cat[y] = df_cat[y].apply(lambda l: wrap_label_by_words(normalize_label(l), words_per_line=4))
        else:
            df_cat[y] = df_cat[y].apply(
                lambda l: wrap_label_by_words(l, words_per_line=4)
            )                  
        
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=category_colors.get(cat, "#660094"),  # fallback color if category missing
            text=df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if category_colors.get(cat)=="#FFDB58" else 'white', size=10, family="Arial black"),
            hovertemplate=f"%{{y}}<br>{cat}: %{{x}}<extra></extra>"
        ))
    num_bars = df.shape[0]
    height = 350
    # Bold axis line
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')     
        fig.update_xaxes(tickfont=dict(family="Arial",size=11, color="black"))  
        fig.update_xaxes(tickfont=dict(family="Arial",size=11, color="black"))  
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
              
    fig.update_layout(barmode='stack', height=height, margin=dict(l=120 if horizontal else 20, r=20, t=20, b=20))
    fig.update_xaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(
        barmode='stack',
        height=height,
        margin=dict(l=120 if horizontal else 20, r=20, t=45, b=20),
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(
                family="Arial Black",
                size=12,
                color="#660094"
            )
        ),
        font=dict(
            family="Arial",
            size=12,
            color="black"
        )
    )

    # ---------------- Dynamic download-only source ----------------
    if horizontal:
        # For horizontal bars, find max x for positioning
        max_val = df[x].sum()
    else:
        # For vertical bars, find max y for positioning
        max_val = df[x].sum()
  
    # ---------------- WATERMARK ----------------
    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=20,
            color="black"
        ),
        #textangle=-30,
        opacity=0.05,
        xanchor="center",
        yanchor="middle"
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

# ---------------- FORMATTED HEATMAP ----------------
def create_heatmap(pivot_df, title="Heatmap"):
    """
    Creates a Plotly heatmap from a pivot table with formatted labels and hover info.
    """
    if pivot_df.empty:
        # Placeholder chart if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    # Wrap labels for better readability
    pivot_df.index = [wrap_label_by_words(str(i), words_per_line=3) for i in pivot_df.index]
    pivot_df.columns = [wrap_label_by_words(str(i), words_per_line=3) for i in pivot_df.columns]

    # Define traffic-light colorscale
    colorscale=[
        [0, "green"],
        [0.5, "yellow"],
        [1, "red"]
    ]

    # Normalize data between 0 and 1 for the colorscale
    z_values = pivot_df.values.astype(float)
    z_min, z_max = z_values.min(), z_values.max()
    z_norm = (z_values - z_min) / (z_max - z_min + 1e-6)  # avoid division by zero

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=colorscale,
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count", tickfont=dict(size=12))
        )
    )

    fig.update_layout(
        title=title,
        title_font=dict(size=18, color="#660094"),
        xaxis_title="",
        yaxis_title="",
        xaxis_tickangle=-45,
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=80, r=20, t=50, b=120),
        height=max(350, len(pivot_df)*35)
    )
   
    return fig
# ---------------- HELPER: Get Top-N Items ----------------
def get_top_n_items(df, col, top_n):
    """
    Returns a list of top-N items in a column based on frequency.
    If top_n is None, returns all items.
    """
    counts = df[col].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return counts.index.tolist()

# ---------------- UPDATED HEATMAP RENDER FUNCTION ----------------
def render_heatmaps(df, top_n=5):
    """
    Renders three heatmaps for Negative Events tab, handling multi-valued fields safely:
    - Actor → Mechanism
    - Subject → Mechanism
    - Actor → Subject

    Parameters:
        df (DataFrame): Filtered data (Negative Events)
        top_n (int or None): Number of top items to show per axis. Use None for all.
    """
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
        parts = [p.replace(placeholder, protected_label) for p in parts]
        return parts

    # Explode multi-valued columns safely
    df_exploded = df.copy()

    explode_cols = [
        "Actor of repression",
        "Subject of repression",
        "Mechanism of repression"
    ]

    for col in explode_cols:
        df_exploded[col] = df_exploded[col].apply(safe_split)
        df_exploded = df_exploded.explode(col)
        df_exploded[col] = df_exploded[col].astype(str).str.strip()

    # Remove blanks created during splitting/exploding
    df_exploded = df_exploded[
        (df_exploded["Actor of repression"] != "") &
        (df_exploded["Subject of repression"] != "") &
        (df_exploded["Mechanism of repression"] != "")
    ].copy()

    # Determine Top-N items
    top_actors = get_top_n_items(df_exploded, "Actor of repression", top_n)
    top_subjects = get_top_n_items(df_exploded, "Subject of repression", top_n)
    top_mechanisms = get_top_n_items(df_exploded, "Mechanism of repression", top_n)

    # Filter to Top-N items
    df_top = df_exploded[
        df_exploded["Actor of repression"].isin(top_actors) &
        df_exploded["Subject of repression"].isin(top_subjects) &
        df_exploded["Mechanism of repression"].isin(top_mechanisms)
    ].copy()

    if df_top.empty:
        st.warning("No heatmap data available after applying Top-N filters.")
        return

    # Create pivot tables
    actor_mechanism_pivot = filter_top_n(
        df_top, "Actor of repression", "Mechanism of repression", top_n
    )
    subject_mechanism_pivot = filter_top_n(
        df_top, "Subject of repression", "Mechanism of repression", top_n
    )
    actor_subject_pivot = filter_top_n(
        df_top, "Actor of repression", "Subject of repression", top_n
    )

    # Consistent color scale
    all_values = pd.concat([
        actor_mechanism_pivot.stack(),
        subject_mechanism_pivot.stack(),
        actor_subject_pivot.stack()
    ])
    zmax = all_values.max() if not all_values.empty else 1

    # Render heatmaps in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = create_heatmap(
            actor_mechanism_pivot,
            title="What are the mechanisms used<br>by restrictive actors?"
        )
        fig1.update_traces(zmin=0, zmax=zmax)
        fig1.update_layout(
            xaxis=dict(tickfont=dict(size=10, family="Arial")),
            yaxis=dict(tickfont=dict(size=10, family="Arial")),
            title=dict(
                text=fig1.layout.title.text,
                x=0.5,
                xanchor="center",
                font=dict(size=12, family="Arial black")
            )
        )
        st.plotly_chart(fig1, use_container_width=True, key="heatmap_actor_mechanism")

    with col2:
        fig2 = create_heatmap(
            subject_mechanism_pivot,
            title="What are the restrictive mechanisms<br>affecting civil society actors?"
        )
        fig2.update_traces(zmin=0, zmax=zmax)
        fig2.update_layout(
            xaxis=dict(tickfont=dict(size=10, family="Arial")),
            yaxis=dict(tickfont=dict(size=10, family="Arial")),
            title=dict(
                text=fig2.layout.title.text,
                x=0.5,
                xanchor="center",
                font=dict(size=12, family="Arial black")
            )
        )
        st.plotly_chart(fig2, use_container_width=True, key="heatmap_subject_mechanism")

    with col3:
        fig3 = create_heatmap(
            actor_subject_pivot,
            title="Who are the actors restricting<br>civil society?"
        )
        fig3.update_traces(zmin=0, zmax=zmax)
        fig3.update_layout(
            xaxis=dict(tickfont=dict(size=10, family="Arial")),
            yaxis=dict(tickfont=dict(size=10, family="Arial")),
            title=dict(
                text=fig3.layout.title.text,
                x=0.5,
                xanchor="center",
                font=dict(size=12, family="Arial black")
            )
        )
        st.plotly_chart(fig3, use_container_width=True, key="heatmap_actor_subject")

# ---------------- HELPER: Get Top-N Items ----------------
def get_top_n_items(df, col, top_n):
    """
    Returns a list of top-N items in a column based on frequency.
    If top_n is None, returns all items.
    """
    counts = df[col].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return counts.index.tolist()

# ---------------- UPDATED HEATMAP RENDER FUNCTION ----------------
def render_sankey(df, top_n=None, width=900, wrap_width=25):
    """
    Render a Sankey diagram for Negative Events:
    Actor → Mechanism → Subject
    Wraps long labels to display fully.
    
    Parameters:
        df (DataFrame): Filtered negative events data
        top_n (int or None): Number of top items to show per axis. Use None for all.
        wrap_width (int): Maximum characters per line before wrapping
    """
    if df.empty:
        st.warning("No data available for Sankey")
        return go.Figure()

    # Helper: wrap long labels
    def wrap_label(label):
        words = str(label).split()
        lines = []
        line = ""
        for word in words:
            if len(line + " " + word) <= wrap_width:
                line = (line + " " + word).strip()
            else:
                lines.append(line)
                line = word
        lines.append(line)
        return "<br>".join(lines)
    

    # Get top-N nodes
    def get_top_nodes(col):
        counts = df[col].value_counts()
        if top_n is not None:
            counts = counts.head(top_n)
        return counts.index.tolist()

    top_actors = get_top_nodes("Actor of repression")
    top_mechanisms = get_top_nodes("Mechanism of repression")
    top_subjects = get_top_nodes("Subject of repression")

    # Build node labels (wrapped)
    actor_nodes = [wrap_label(f"Actor: {a}") for a in top_actors]
    mechanism_nodes = [wrap_label(f"Mechanism: {m}") for m in top_mechanisms]
    subject_nodes = [wrap_label(f"Subject: {s}") for s in top_subjects]

    nodes = actor_nodes + mechanism_nodes + subject_nodes
    node_index = {name: i for i, name in enumerate(nodes)}

    node_colors = (
        ["#FF5733"] * len(actor_nodes) +
        ["#33C1FF"] * len(mechanism_nodes) +
        ["#33FF8A"] * len(subject_nodes)
    )

    links = []

    # Actor → Mechanism
    df_am = df[df["Actor of repression"].isin(top_actors) &
               df["Mechanism of repression"].isin(top_mechanisms)]
    for _, r in df_am.groupby(["Actor of repression", "Mechanism of repression"]).size().reset_index(name="value").iterrows():
        links.append(dict(
            source=node_index[wrap_label(f"Actor: {r['Actor of repression']}")],
            target=node_index[wrap_label(f"Mechanism: {r['Mechanism of repression']}")],
            value=r["value"]
        ))

    # Mechanism → Subject
    df_ms = df[df["Mechanism of repression"].isin(top_mechanisms) &
               df["Subject of repression"].isin(top_subjects)]
    for _, r in df_ms.groupby(["Mechanism of repression", "Subject of repression"]).size().reset_index(name="value").iterrows():
        links.append(dict(
            source=node_index[wrap_label(f"Mechanism: {r['Mechanism of repression']}")],
            target=node_index[wrap_label(f"Subject: {r['Subject of repression']}")],
            value=r["value"]
        ))

    # Figure height scales with number of nodes
    fig_height = max(500, len(nodes) * 40)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=40,
            thickness=35,
            line=dict(color="black", width=0.5),
            label=actor_nodes + mechanism_nodes + subject_nodes,  # plain text
            color=node_colors,
            hovertemplate="%{label}<extra></extra>"
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            hovertemplate="%{value} alerts<extra></extra>"
        )
    ))

    fig.update_layout(font=dict(family="Arial Black", size=12, color="black"))  # text color

    # Optional legend as scatter
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="#FF5733"),
        name="Restrictive actor "
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="#33C1FF"),
        name="Restrictive mechanism "
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="#33FF8A"),
        name="Civil society actor affected "
    ))

    # Layout with consistent fonts
    fig.update_layout(
        title=dict(
            text="Flow of Negative Events",
            x=0.5,
            xanchor="center",
            font=dict(size=12, family="Arial Black", color="#660094")  # Title font
        ),
        font=dict(size=10, family="Arial", color="black"),  # Axis, legend, hover font
        height=fig_height,
        width=width,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )


    return fig
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



# ---------------- AI ASSISTANT HELPERS v2 ----------------
def _clean_text_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def _safe_series_counts(df, col, top=5):
    if df is None or df.empty or col not in df.columns:
        return {}
    return (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(top)
        .to_dict()
    )


def _safe_exploded_counts(df, col, top=5):
    if df is None or df.empty or col not in df.columns:
        return {}
    s = df[col].dropna().astype(str)
    if col == "Actor of repression":
        s = s.str.replace(r"\bVNSAs\b", "Violent non-state actors", regex=True)
    return (
        s.str.split(",")
        .explode()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(top)
        .to_dict()
    )


def _format_ranked(items, label="alerts"):
    if not items:
        return "No matching records are available under the current filters."
    return "\n".join([f"{i}. {k} — {v} {label}" for i, (k, v) in enumerate(items.items(), start=1)])


def _month_trend(df):
    if df is None or df.empty or "creation_date" not in df.columns:
        return pd.DataFrame(columns=["month", "total", "negative", "positive", "context"])
    tmp = df.copy()
    tmp["creation_date"] = pd.to_datetime(tmp["creation_date"], errors="coerce")
    tmp = tmp.dropna(subset=["creation_date"])
    if tmp.empty:
        return pd.DataFrame(columns=["month", "total", "negative", "positive", "context"])
    tmp["month"] = tmp["creation_date"].dt.to_period("M").astype(str)
    out = (
        tmp.groupby("month")
        .agg(
            total=("alert-impact", "size"),
            negative=("alert-impact", lambda x: int((x == "Negative").sum())),
            positive=("alert-impact", lambda x: int((x == "Positive").sum())),
            context=("alert-impact", lambda x: int((x == "Context to watch").sum())),
        )
        .reset_index()
        .sort_values("month")
    )
    return out


def _trend_sentence(df):
    trend = _month_trend(df)
    if trend.shape[0] < 2:
        return "A monthly trend cannot be calculated because fewer than two time periods are available under the current filters."
    first = int(trend.iloc[0]["total"])
    last = int(trend.iloc[-1]["total"])
    diff = last - first
    pct = round((diff / first) * 100, 1) if first else 0
    direction = "increased" if diff > 0 else "decreased" if diff < 0 else "remained stable"
    return f"Total alerts {direction} from {first} in {trend.iloc[0]['month']} to {last} in {trend.iloc[-1]['month']} ({diff:+d}, {pct:+.1f}%)."


def summarize_for_ai(df):
    if df is None or df.empty:
        return {
            "total_alerts": 0,
            "negative": 0,
            "positive": 0,
            "context": 0,
            "negative_pct": 0,
            "positive_pct": 0,
            "context_pct": 0,
            "countries_count": 0,
            "regions_count": 0,
            "top_countries": {},
            "top_negative_countries": {},
            "top_regions": {},
            "top_alert_types": {},
            "top_principles": {},
            "top_actors": {},
            "top_mechanisms": {},
            "trend_sentence": "No data are available under the current filters.",
        }

    total = len(df)
    negative = int((df["alert-impact"] == "Negative").sum()) if "alert-impact" in df.columns else 0
    positive = int((df["alert-impact"] == "Positive").sum()) if "alert-impact" in df.columns else 0
    context = int((df["alert-impact"] == "Context to watch").sum()) if "alert-impact" in df.columns else 0
    neg_df = df[df["alert-impact"] == "Negative"] if "alert-impact" in df.columns else df.iloc[0:0]

    return {
        "total_alerts": int(total),
        "negative": negative,
        "positive": positive,
        "context": context,
        "negative_pct": round((negative / total) * 100, 1) if total else 0,
        "positive_pct": round((positive / total) * 100, 1) if total else 0,
        "context_pct": round((context / total) * 100, 1) if total else 0,
        "countries_count": int(df["alert-country"].nunique()) if "alert-country" in df.columns else 0,
        "regions_count": int(df["region"].nunique()) if "region" in df.columns else 0,
        "top_countries": _safe_series_counts(df, "alert-country", 5),
        "top_negative_countries": _safe_series_counts(neg_df, "alert-country", 5),
        "top_regions": _safe_series_counts(df, "region", 5),
        "top_alert_types": _safe_series_counts(df, "alert-type", 5),
        "top_principles": _safe_exploded_counts(df, "enabling-principle", 5),
        "top_actors": _safe_exploded_counts(neg_df, "Actor of repression", 5),
        "top_mechanisms": _safe_exploded_counts(neg_df, "Mechanism of repression", 5),
        "trend_sentence": _trend_sentence(df),
    }



# ---------------- EUSEE WEBSITE REDIRECT ----------------
# Update this URL if the official EUSEE website uses a different domain.
try:
    EUSEE_URL = st.secrets.get("eusee", {}).get("website_url", "https://eusee.org")
except Exception:
    EUSEE_URL = "https://eusee.org"

EUSEE_REDIRECT_TEXT = (
    "For a broader overview and additional qualitative insights, "
    "please visit the EUSEE website."
)

def append_eusee_redirect(response: str) -> str:
    """Append a standard EUSEE website redirect to every chatbot answer."""
    response = str(response or "").strip()
    if EUSEE_REDIRECT_TEXT in response:
        return response
    return (
        f"{response}\n\n"
        f"🌐 {EUSEE_REDIRECT_TEXT}\n"
        f"Open EUSEE: {EUSEE_URL}"
    )

def generate_ai_executive_summary(df):
    s = summarize_for_ai(df)
    if s["total_alerts"] == 0:
        return "EU SEE AI Assistant Summary\n\nNo data are available under the current filters."

    return f"""EU SEE AI Assistant Summary

Filtered dashboard scope
- Total alerts: {s['total_alerts']}
- Countries represented: {s['countries_count']}
- Regions represented: {s['regions_count']}
- Negative alerts: {s['negative']} ({s['negative_pct']}%)
- Positive alerts: {s['positive']} ({s['positive_pct']}%)
- Context to watch: {s['context']} ({s['context_pct']}%)

Trend signal
{s['trend_sentence']}

Top countries
{_format_ranked(s['top_countries'])}

Top countries by negative alerts
{_format_ranked(s['top_negative_countries'])}

Top alert types
{_format_ranked(s['top_alert_types'])}

Top enabling principles
{_format_ranked(s['top_principles'])}

Top restrictive actors among negative alerts
{_format_ranked(s['top_actors'])}

Top restrictive mechanisms among negative alerts
{_format_ranked(s['top_mechanisms'])}

Interpretation note
These results reflect the currently selected filters and reporting coverage. Higher counts may reflect either more incidents, stronger reporting intensity, broader monitoring coverage, or a combination of these factors.
"""


def _local_ai_response_core(question, df):
    q = (question or "").lower().strip()
    s = summarize_for_ai(df)
    if s["total_alerts"] == 0:
        return "No data are available under the current filters. Please adjust the filters and try again."

    if any(k in q for k in ["negative countr", "highest negative", "top negative countr"]):
        return "Top countries by negative alerts under the current filters:\n\n" + _format_ranked(s["top_negative_countries"])

    if any(k in q for k in ["country", "countries"]):
        return "Top countries under the current filters:\n\n" + _format_ranked(s["top_countries"])

    if any(k in q for k in ["compare region", "regional comparison", "regions compare"]):
        return "Regional comparison under the current filters:\n\n" + _format_ranked(s["top_regions"])

    if any(k in q for k in ["region", "regions"]):
        return "Regional distribution under the current filters:\n\n" + _format_ranked(s["top_regions"])

    if any(k in q for k in ["trend", "over time", "time", "increase", "decrease"]):
        return s["trend_sentence"]

    if "negative" in q:
        return f"There are {s['negative']} negative alerts out of {s['total_alerts']} total alerts under the current filters. This represents {s['negative_pct']}% of the filtered records."

    if "positive" in q:
        return f"There are {s['positive']} positive alerts out of {s['total_alerts']} total alerts under the current filters. This represents {s['positive_pct']}% of the filtered records."

    if any(k in q for k in ["alert type", "type", "types"]):
        return "Main alert types under the current filters:\n\n" + _format_ranked(s["top_alert_types"])

    if any(k in q for k in ["principle", "principles", "enabling"]):
        return "Top enabling principles represented in the current filters:\n\n" + _format_ranked(s["top_principles"])

    if any(k in q for k in ["actor", "actors", "repression"]):
        return "Top restrictive actors among negative alerts in the current filtered records:\n\n" + _format_ranked(s["top_actors"])

    if any(k in q for k in ["mechanism", "mechanisms", "restriction"]):
        return "Top restrictive mechanisms among negative alerts in the current filtered records:\n\n" + _format_ranked(s["top_mechanisms"])

    if any(k in q for k in ["policy brief", "briefing note"]):
        return generate_ai_policy_brief(df)

    if any(k in q for k in ["data quality", "missing", "completeness"]):
        return ai_data_quality_report(df)

    if any(k in q for k in ["next step", "recommend", "action"]):
        return ai_recommended_next_steps(df)

    if any(k in q for k in ["summary", "summarise", "summarize", "overview", "brief"]):
        return generate_ai_executive_summary(df)

    if any(k in q for k in ["interpret", "meaning", "explain", "insight"]):
        return (
            f"Main interpretation from the current filters:\n\n"
            f"The filtered dataset contains {s['total_alerts']} alerts across {s['countries_count']} countries. "
            f"Negative alerts account for {s['negative_pct']}% of records. "
            f"The most represented countries are:\n{_format_ranked(s['top_countries'])}\n\n"
            f"Important caution: alert counts should be interpreted alongside reporting intensity and monitoring coverage."
        )

    return (
        "I can answer questions from the currently filtered dashboard data. Try asking about:\n\n"
        "- top countries\n- top countries with negative alerts\n- regional comparison\n- trends over time\n- alert types\n- enabling principles\n- restrictive actors\n- restrictive mechanisms\n- executive summary"
    )



def local_ai_response(question, df):
    """Chatbot-facing response wrapper with EUSEE website redirect."""
    return append_eusee_redirect(_local_ai_response_core(question, df))

def render_ai_trend_chart(df):
    trend = _month_trend(df)
    if trend.empty or trend.shape[0] < 2:
        st.info("Trend chart unavailable for the current filter selection.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["month"], y=trend["total"], mode="lines+markers", name="Total"))
    fig.add_trace(go.Scatter(x=trend["month"], y=trend["negative"], mode="lines+markers", name="Negative"))
    fig.update_layout(
        title=dict(text="Filtered Alert Trend", font=dict(size=12, color="#660094")),
        height=240,
        margin=dict(l=10, r=10, t=40, b=20),
        font=dict(size=10),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True, key="ai_trend_chart")



# ---------------- AI ASSISTANT PROFESSIONAL UX HELPERS v3 ----------------
def _pct(part, total):
    return round((part / total) * 100, 1) if total else 0


def ai_priority_level(df):
    """Returns a simple UX priority label based on negative-alert share."""
    s = summarize_for_ai(df)
    neg_pct = s.get("negative_pct", 0)
    if s.get("total_alerts", 0) == 0:
        return "No data", "#999999", "No records match the current filters."
    if neg_pct >= 70:
        return "High attention", "#B42318", "Negative alerts dominate the current filtered view."
    if neg_pct >= 45:
        return "Moderate attention", "#B54708", "Negative alerts are substantial and should be reviewed with country and actor breakdowns."
    return "Watch", "#027A48", "Negative-alert share is comparatively lower under the current filters."


def ai_data_quality_report(df):
    """Compact data-quality report for user trust and interpretation."""
    if df is None or df.empty:
        return "No records are available under the current filters."

    cols = [
        "creation_date", "alert-country", "region", "alert-impact", "alert-type",
        "enabling-principle", "Actor of repression", "Subject of repression",
        "Mechanism of repression", "Type of event"
    ]
    available = [c for c in cols if c in df.columns]
    lines = [f"Records assessed: {len(df):,}"]
    for c in available:
        missing = int(df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum())
        lines.append(f"- {c}: {missing:,} blank/missing values")
    return "\n".join(lines)


def ai_recommended_next_steps(df):
    s = summarize_for_ai(df)
    if s["total_alerts"] == 0:
        return "Adjust the filters to include at least one region, country, year, or alert type."

    steps = []
    if s["negative_pct"] >= 50:
        steps.append("Prioritize the Negative Alerts tab to inspect restrictive actors and mechanisms.")
    if s["top_negative_countries"]:
        top_country = next(iter(s["top_negative_countries"].keys()))
        steps.append(f"Drill down into {top_country} to validate whether the high count reflects risk, reporting intensity, or both.")
    if s["top_mechanisms"]:
        top_mech = next(iter(s["top_mechanisms"].keys()))
        steps.append(f"Review the mechanism pathway for '{top_mech}' using the heatmaps and Sankey flow.")
    steps.append("Export the filtered summary for reporting and include the interpretation note on reporting coverage.")
    return "\n".join([f"{i}. {x}" for i, x in enumerate(steps, start=1)])


def generate_ai_policy_brief(df):
    s = summarize_for_ai(df)
    if s["total_alerts"] == 0:
        return "EU SEE AI Policy Brief\n\nNo records are available under the current filters."

    level, _, level_note = ai_priority_level(df)
    return f"""EU SEE AI Policy Brief

Priority signal: {level}
{level_note}

Scope
The current filtered view contains {s['total_alerts']} alerts across {s['countries_count']} countries and {s['regions_count']} regions. Negative alerts account for {s['negative']} records ({s['negative_pct']}%), positive alerts account for {s['positive']} records ({s['positive_pct']}%), and context-to-watch alerts account for {s['context']} records ({s['context_pct']}%).

Most represented countries
{_format_ranked(s['top_countries'])}

Most represented countries by negative alerts
{_format_ranked(s['top_negative_countries'])}

Dominant issue areas
Top alert types:\n{_format_ranked(s['top_alert_types'])}

Top enabling principles:\n{_format_ranked(s['top_principles'])}

Negative-alert pathways
Top restrictive actors:\n{_format_ranked(s['top_actors'])}

Top restrictive mechanisms:\n{_format_ranked(s['top_mechanisms'])}

Recommended analytical next steps
{ai_recommended_next_steps(df)}

Interpretation caveat
Counts should be interpreted with care because they may reflect incident frequency, monitoring intensity, reporting coverage, network activity, or a combination of these factors.
"""


def render_ai_metric(label, value, note="", color="#660094"):
    st.markdown(
        f"""
        <div class="ai-kpi">
            <div class="ai-kpi-label">{label}</div>
            <div class="ai-kpi-value" style="color:{color};">{value}</div>
            <div class="ai-kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _render_chat_content_html(content):
    """Escape chatbot text while rendering the EUSEE redirect as a trusted clickable link."""
    import html as _html
    text = str(content or "")
    redirect_line = f"Open EUSEE: {EUSEE_URL}"
    if redirect_line in text:
        main_text = text.replace(f"\n{redirect_line}", "").replace(redirect_line, "").strip()
        main_html = _html.escape(main_text).replace("\n", "<br>")
        redirect_html = f"""
        <div style="margin-top:10px;padding:10px 12px;border-left:4px solid #660094;background:#fbf8ff;border-radius:10px;">
            <div style="font-size:12px;line-height:1.35;color:#333;">🌐 <b>{EUSEE_REDIRECT_TEXT}</b></div>
            <a href="{EUSEE_URL}" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin-top:8px;background:#660094;color:#ffffff;padding:6px 10px;border-radius:8px;text-decoration:none;font-weight:800;font-size:12px;">
                Open EUSEE →
            </a>
        </div>
        """
        return main_html + redirect_html
    return _html.escape(text).replace("\n", "<br>")

def _ai_stream_response_text(text: str):
    """Yield words to create a ChatGPT-style streaming response."""
    words = str(text or "").split(" ")
    for word in words:
        yield word + " "


def _save_ai_answer(question, df):
    """Generate and store an answer using the local assistant engine."""
    q = str(question or "").strip()
    if not q:
        return
    st.session_state.ai_messages.append({"role": "user", "content": q})
    q_lower = q.lower()
    if "data quality" in q_lower or "missing" in q_lower:
        answer = append_eusee_redirect(ai_data_quality_report(df))
    elif "next step" in q_lower or "recommend" in q_lower:
        answer = append_eusee_redirect(ai_recommended_next_steps(df))
    elif "policy brief" in q_lower or "briefing" in q_lower:
        answer = append_eusee_redirect(generate_ai_policy_brief(df))
    else:
        answer = local_ai_response(q, df)
    st.session_state.ai_pending_answer = answer
    st.session_state.ai_is_typing = True


def render_ai_assistant_panel(df):
    """Render a professional floating ChatGPT-style assistant."""
    st.markdown("""
        <style>
        .eusee-ai-shell {position: fixed; right: 22px; bottom: 22px; width: min(430px, calc(100vw - 34px)); height: min(720px, calc(100vh - 44px)); background:#fff; border:1px solid #eadff8; border-radius:24px; box-shadow:0 24px 70px rgba(45,0,85,.28); z-index:999999; overflow:hidden; display:flex; flex-direction:column;}
        .eusee-ai-mini {position:fixed; right:24px; bottom:24px; z-index:999999; background:linear-gradient(135deg,#660094,#7b2cff); color:#fff; border-radius:999px; padding:12px 18px; box-shadow:0 16px 38px rgba(45,0,85,.28); font-family:Arial,sans-serif; font-weight:900; font-size:14px;}
        .eusee-ai-head {background:linear-gradient(135deg,#2d0055,#660094 55%,#7b2cff); color:#fff; padding:14px 16px 12px 16px;}
        .eusee-ai-title {font-family:Arial,sans-serif; font-size:17px; font-weight:900;}
        .eusee-ai-badge {font-size:10px; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.25); padding:3px 8px; border-radius:999px; margin-left:6px;}
        .eusee-ai-sub {font-size:12px; line-height:1.35; color:#f1e8ff; margin-top:4px;}
        .eusee-ai-context {display:flex; gap:6px; flex-wrap:wrap; margin-top:10px;}
        .eusee-ai-pill {background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.18); padding:4px 8px; border-radius:999px; font-size:11px; font-weight:800; color:#fff;}
        .eusee-ai-body {padding:12px; background:linear-gradient(180deg,#ffffff 0%,#fbf8ff 100%); height:calc(100% - 96px); overflow-y:auto;}
        .eusee-ai-card {background:#fff; border:1px solid #eee5ff; border-radius:16px; padding:12px; box-shadow:0 6px 18px rgba(45,0,85,.07); margin-bottom:10px;}
        .eusee-ai-msg {background:#f7f2ff; border:1px solid #eee5ff; border-radius:16px 16px 16px 5px; padding:10px 12px; font-size:13px; color:#222; margin:8px 28px 10px 0; line-height:1.45; white-space:pre-wrap;}
        .eusee-ai-user {background:linear-gradient(135deg,#660094,#7b2cff); color:#fff; border-radius:16px 16px 5px 16px; padding:10px 12px; font-size:13px; margin:8px 0 10px 42px; line-height:1.45; white-space:pre-wrap;}
        .eusee-ai-kpi {background:#fff; border:1px solid #eee5ff; border-radius:14px; padding:10px; margin-bottom:8px;}
        .eusee-ai-kpi-label {font-size:11px; color:#666; font-weight:700; text-transform:uppercase; letter-spacing:.03em;}
        .eusee-ai-kpi-value {font-size:22px; font-weight:900; margin-top:2px;}
        .eusee-ai-kpi-note {font-size:11px; color:#777; line-height:1.3;}
        .eusee-ai-status {display:inline-block; font-size:11px; font-weight:900; color:#fff; padding:5px 9px; border-radius:999px;}
        .eusee-typing {display:inline-flex; gap:4px; align-items:center; padding:6px 0;}
        .eusee-typing span {width:6px; height:6px; background:#660094; border-radius:50%; display:block; animation:euseeTyping 1.1s infinite ease-in-out;}
        .eusee-typing span:nth-child(2){animation-delay:.15s}.eusee-typing span:nth-child(3){animation-delay:.3s}
        @keyframes euseeTyping {0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-3px)}}
        .eusee-ai-guide li {font-size:12px; margin-bottom:6px; color:#333;}
        .eusee-ai-shell div[data-testid="stButton"] button, .eusee-ai-mini-wrap div[data-testid="stButton"] button {border-radius:999px !important; font-weight:800 !important; min-height:34px !important;}
        .eusee-ai-shell textarea {font-size:13px !important;}
        .eusee-ai-shell div[data-testid="stTabs"] button {font-size:12px !important; font-weight:800 !important; padding:7px 0 !important;}
        @media (max-width:900px){.eusee-ai-shell{right:10px;bottom:10px;width:calc(100vw - 20px);height:min(720px, calc(100vh - 20px));}}
        </style>
        """, unsafe_allow_html=True)

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [{"role": "assistant", "content": "Hello. I am your EU SEE AI assistant. Ask me about alerts, countries, regions, actors, mechanisms, trends, enabling principles, or data quality."}]
    if "ai_floating_open" not in st.session_state:
        st.session_state.ai_floating_open = True
    if "ai_is_typing" not in st.session_state:
        st.session_state.ai_is_typing = False
    if "ai_pending_answer" not in st.session_state:
        st.session_state.ai_pending_answer = None

    s = summarize_for_ai(df)
    level, level_color, level_note = ai_priority_level(df)

    if not st.session_state.ai_floating_open:
        st.markdown('<div class="eusee-ai-mini-wrap"><div class="eusee-ai-mini">💬 EUSEE AI Assistant</div></div>', unsafe_allow_html=True)
        if st.button("Open AI Assistant", key="ai_open_floating", use_container_width=False):
            st.session_state.ai_floating_open = True
            st.rerun()
        return

    st.markdown('<div class="eusee-ai-shell">', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="eusee-ai-head">
            <div class="eusee-ai-title">🤖 EUSEE AI Assistant <span class="eusee-ai-badge">Pro</span></div>
            <div class="eusee-ai-sub">ChatGPT-style assistant linked to current dashboard filters.</div>
            <div class="eusee-ai-context">
                <span class="eusee-ai-status" style="background:{level_color};">{level}</span>
                <span class="eusee-ai-pill">{s['total_alerts']:,} alerts</span>
                <span class="eusee-ai-pill">{s['countries_count']:,} countries</span>
                <span class="eusee-ai-pill">{s['negative_pct']}% negative</span>
            </div>
        </div><div class="eusee-ai-body">
        ''', unsafe_allow_html=True)

    top_controls = st.columns([1, 1, 1])
    with top_controls[0]:
        if st.button("Minimize", key="ai_minimize_float", use_container_width=True):
            st.session_state.ai_floating_open = False
            st.rerun()
    with top_controls[1]:
        if st.button("Clear", key="ai_clear_float_top", use_container_width=True):
            st.session_state.ai_messages = [{"role": "assistant", "content": "Chat cleared. Ask me about the currently filtered EU SEE dashboard data."}]
            st.session_state.ai_pending_answer = None
            st.session_state.ai_is_typing = False
            st.rerun()
    with top_controls[2]:
        if st.button("Brief", key="ai_brief_float_top", use_container_width=True):
            _save_ai_answer("Generate a policy brief.", df)
            st.rerun()

    chat_tab, insight_tab, export_tab, guide_tab = st.tabs(["Chat", "Insights", "Export", "Guide"])

    with chat_tab:
        st.markdown('<div class="eusee-ai-card"><b style="color:#2d0055;">Quick prompts</b>', unsafe_allow_html=True)
        quick_questions = {
            "Summary": "Generate an executive summary.", "Top countries": "Which countries have the highest number of alerts?",
            "Negative": "Which countries have the highest negative alerts?", "Regions": "Compare regions under the current filters.",
            "Trend": "Show trend of alerts over time.", "Types": "What are the main alert types?",
            "Mechanisms": "What are the top restrictive mechanisms?", "Actors": "What are the top restrictive actors?",
            "Quality": "Show the data quality report.", "Next steps": "What are the recommended next analytical steps?",
        }
        qcols = st.columns(2)
        for idx, (label, prompt) in enumerate(quick_questions.items()):
            with qcols[idx % 2]:
                if st.button(label, key=f"ai_float_quick_{label}", use_container_width=True):
                    _save_ai_answer(prompt, df)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        for msg in st.session_state.ai_messages[-10:]:
            css_class = "eusee-ai-user" if msg["role"] == "user" else "eusee-ai-msg"
            safe_content = _render_chat_content_html(msg["content"])
            st.markdown(f'<div class="{css_class}">{safe_content}</div>', unsafe_allow_html=True)

        if st.session_state.ai_is_typing and st.session_state.ai_pending_answer:
            st.markdown('<div class="eusee-ai-msg"><div class="eusee-typing"><span></span><span></span><span></span></div><br><b>AI is preparing the answer...</b></div>', unsafe_allow_html=True)
            stream_box = st.empty()
            streamed = ""
            for chunk in _ai_stream_response_text(st.session_state.ai_pending_answer):
                streamed += chunk
                stream_box.markdown(f'<div class="eusee-ai-msg">{_render_chat_content_html(streamed)}</div>', unsafe_allow_html=True)
            st.session_state.ai_messages.append({"role": "assistant", "content": st.session_state.ai_pending_answer})
            st.session_state.ai_pending_answer = None
            st.session_state.ai_is_typing = False
            st.rerun()

        with st.form("ai_assistant_floating_form", clear_on_submit=True):
            user_q = st.text_area("Ask EUSEE AI", placeholder="Ask about countries, regions, trends, actors, mechanisms, or data quality...", height=78, label_visibility="collapsed")
            submitted = st.form_submit_button("Send message", use_container_width=True)
        if submitted and user_q.strip():
            _save_ai_answer(user_q, df)
            st.rerun()

    with insight_tab:
        st.markdown(f'<div class="eusee-ai-card"><b style="color:#2d0055;">Priority signal</b><br><span class="eusee-ai-status" style="background:{level_color};margin-top:8px;">{level}</span><div style="font-size:12px;color:#666;margin-top:8px;">{level_note}</div></div>', unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        with k1:
            render_ai_metric("Total alerts", f"{s['total_alerts']:,}", "Filtered records")
            render_ai_metric("Negative share", f"{s['negative_pct']}%", f"{s['negative']:,} negative alerts", color=level_color)
        with k2:
            render_ai_metric("Countries", f"{s['countries_count']:,}", "Covered in current filter")
            render_ai_metric("Regions", f"{s['regions_count']:,}", "Regional coverage")
        with st.expander("AI interpretation", expanded=True):
            st.markdown(_render_chat_content_html(local_ai_response("interpret the current view", df)), unsafe_allow_html=True)
        with st.expander("Mini trend chart", expanded=True):
            render_ai_trend_chart(df)
        with st.expander("Recommended next analytical steps", expanded=False):
            st.text(ai_recommended_next_steps(df))
        with st.expander("Data quality / completeness check", expanded=False):
            st.text(ai_data_quality_report(df))

    with export_tab:
        summary_text = generate_ai_executive_summary(df)
        policy_text = generate_ai_policy_brief(df)
        chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.ai_messages])
        st.download_button("Download executive summary (.txt)", data=summary_text, file_name="eusee_ai_executive_summary.txt", mime="text/plain", use_container_width=True)
        st.download_button("Download policy brief (.txt)", data=policy_text, file_name="eusee_ai_policy_brief.txt", mime="text/plain", use_container_width=True)
        st.download_button("Download chat transcript (.txt)", data=chat_text, file_name="eusee_ai_chat_transcript.txt", mime="text/plain", use_container_width=True)
        if df is not None and not df.empty:
            cols = [c for c in ["creation_date", "alert-country", "region", "alert-impact", "alert-type", "enabling-principle", "Actor of repression", "Subject of repression", "Mechanism of repression"] if c in df.columns]
            csv_data = df[cols].to_csv(index=False).encode("utf-8") if cols else df.to_csv(index=False).encode("utf-8")
            st.download_button("Download filtered records (.csv)", data=csv_data, file_name="eusee_filtered_records.csv", mime="text/csv", use_container_width=True)

    with guide_tab:
        st.markdown("""
            <div class="eusee-ai-card eusee-ai-guide"><b style="color:#2d0055;">How to use this assistant</b>
            <ul><li>Use dashboard filters first; answers are generated from the filtered view.</li>
            <li>Use Chat for natural-language questions and quick prompts.</li>
            <li>Use Insights for priority signal, interpretation, trend and data-quality checks.</li>
            <li>Use Export for executive summaries, policy briefs, transcripts and filtered records.</li>
            <li>Every answer links users back to the EUSEE website for broader qualitative context.</li></ul></div>
            """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)
# ---------------- MAIN DASHBOARD + AI ASSISTANT LAYOUT ----------------


# ---------------- MAP INTELLIGENCE HELPERS ----------------
def classify_map_risk(total_alerts, negative_alerts, negative_share):
    """Classify country-level map risk using volume + negative-alert share."""
    try:
        total_alerts = int(total_alerts or 0)
        negative_alerts = int(negative_alerts or 0)
        negative_share = float(negative_share or 0)
    except Exception:
        return "No signal"

    if total_alerts == 0:
        return "No signal"
    if negative_alerts >= 10 and negative_share >= 70:
        return "Critical"
    if negative_alerts >= 5 and negative_share >= 50:
        return "High"
    if negative_alerts >= 2 or negative_share >= 35:
        return "Moderate"
    return "Watch"


def prepare_map_intelligence_data(df, countries_gj):
    """Create country-level analytical layer for the Visualization Map tab."""
    if df is None or df.empty:
        return pd.DataFrame()

    geo_countries = [f.get('properties', {}).get('name') for f in countries_gj.get('features', [])]
    geo_countries = [c for c in geo_countries if c]

    for col in ['alert-country', 'alert-impact', 'region']:
        if col not in df.columns:
            df[col] = "Unknown"

    stats = (
        df.groupby(['alert-country', 'region'], dropna=False)
        .agg(
            total_alerts=('alert-impact', 'size'),
            negative_alerts=('alert-impact', lambda x: int((x == 'Negative').sum())),
            positive_alerts=('alert-impact', lambda x: int((x == 'Positive').sum())),
            context_to_watch_alerts=('alert-impact', lambda x: int((x == 'Context to watch').sum())),
        )
        .reset_index()
    )

    stats = stats[stats['alert-country'].isin(geo_countries)].copy()
    if stats.empty:
        return stats

    stats['negative_share'] = ((stats['negative_alerts'] / stats['total_alerts'].replace(0, np.nan)) * 100).round(1).fillna(0)
    stats['positive_share'] = ((stats['positive_alerts'] / stats['total_alerts'].replace(0, np.nan)) * 100).round(1).fillna(0)
    stats['context_share'] = ((stats['context_to_watch_alerts'] / stats['total_alerts'].replace(0, np.nan)) * 100).round(1).fillna(0)
    stats['risk_level'] = stats.apply(lambda r: classify_map_risk(r['total_alerts'], r['negative_alerts'], r['negative_share']), axis=1)
    risk_order = {'Critical': 4, 'High': 3, 'Moderate': 2, 'Watch': 1, 'No signal': 0}
    stats['risk_score'] = stats['risk_level'].map(risk_order).fillna(0).astype(int)
    stats['risk_label'] = stats['risk_level'] + ' | ' + stats['negative_share'].astype(str) + '% negative'
    return stats.sort_values(['risk_score', 'negative_alerts', 'total_alerts'], ascending=False)


def map_intelligence_summary(map_df):
    """Short narrative for map insights and exports."""
    if map_df is None or map_df.empty:
        return "No country-level map data are available under the current filters."

    total_countries = int(map_df['alert-country'].nunique())
    total_alerts = int(map_df['total_alerts'].sum())
    negative = int(map_df['negative_alerts'].sum())
    neg_share = round((negative / total_alerts) * 100, 1) if total_alerts else 0
    top_total = map_df.sort_values('total_alerts', ascending=False).head(3)
    top_risk = map_df.sort_values(['risk_score', 'negative_alerts', 'total_alerts'], ascending=False).head(3)

    top_total_txt = '; '.join([f"{r['alert-country']} ({int(r['total_alerts'])})" for _, r in top_total.iterrows()])
    top_risk_txt = '; '.join([f"{r['alert-country']} ({r['risk_level']}, {int(r['negative_alerts'])} negative)" for _, r in top_risk.iterrows()])

    return (
        f"Map intelligence summary: the current filtered map covers {total_countries} countries and {total_alerts} alerts. "
        f"Negative alerts account for {neg_share}% of mapped records. Highest alert volumes: {top_total_txt}. "
        f"Highest priority country signals: {top_risk_txt}. Interpret these as monitoring signals, not direct prevalence estimates, because reporting intensity and coverage vary by country."
    )


def render_map_intelligence_cards(map_df):
    """Render compact map intelligence KPI cards."""
    if map_df is None or map_df.empty:
        st.info("No map intelligence indicators are available for the selected filters.")
        return

    mapped_countries = map_df['alert-country'].nunique()
    total_alerts = int(map_df['total_alerts'].sum())
    negative_alerts = int(map_df['negative_alerts'].sum())
    critical_high = int(map_df['risk_level'].isin(['Critical', 'High']).sum())
    neg_share = round((negative_alerts / total_alerts) * 100, 1) if total_alerts else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mapped countries", mapped_countries)
    c2.metric("Mapped alerts", f"{total_alerts:,}")
    c3.metric("Negative-alert share", f"{neg_share}%")
    c4.metric("High/Critical signals", critical_high)


def create_intelligent_choropleth(map_df, countries_gj, map_mode='Risk score'):
    """Create map layer with switchable intelligence mode."""
    if map_df is None or map_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No map data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    coords = []
    selected = set(map_df['alert-country'].astype(str))
    for feature in countries_gj.get('features', []):
        if feature.get('properties', {}).get('name') in selected:
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'Polygon':
                coords.extend(geometry.get('coordinates', [[]])[0])
            elif geometry.get('type') == 'MultiPolygon':
                for poly in geometry.get('coordinates', []):
                    if poly:
                        coords.extend(poly[0])
    if coords:
        lons, lats = zip(*coords)
        center = {"lat": float(np.mean(lats)), "lon": float(np.mean(lons))}
        lon_span = max(lons) - min(lons)
        zoom = max(1, min(5, 2 / (lon_span + 0.01)))
    else:
        center, zoom = {"lat": 10, "lon": 0}, 2

    if map_mode == 'Negative share':
        color_col, scale, title = 'negative_share', 'YlOrRd', '% negative alerts'
    elif map_mode == 'Total alerts':
        color_col, scale, title = 'total_alerts', 'YlOrBr', 'Total alerts'
    else:
        color_col, scale, title = 'risk_score', [[0, '#e5f4f7'], [0.25, '#c7e9ef'], [0.5, '#ffdb58'], [0.75, '#f59e0b'], [1, '#dc2626']], 'Risk score'

    fig = px.choropleth_mapbox(
        map_df,
        geojson=countries_gj,
        locations='alert-country',
        featureidkey='properties.name',
        color=color_col,
        hover_name='alert-country',
        color_continuous_scale=scale,
        mapbox_style='open-street-map',
        zoom=zoom,
        center=center,
        opacity=0.82,
        custom_data=['region', 'total_alerts', 'negative_alerts', 'positive_alerts', 'context_to_watch_alerts', 'negative_share', 'risk_level']
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Region: %{customdata[0]}<br>"
            "Total alerts: %{customdata[1]}<br>"
            "Negative: %{customdata[2]}<br>"
            "Positive: %{customdata[3]}<br>"
            "Context to watch: %{customdata[4]}<br>"
            "Negative share: %{customdata[5]}%<br>"
            "Priority signal: <b>%{customdata[6]}</b><extra></extra>"
        ),
        hoverlabel=dict(bgcolor='#2D0055', font_size=13, font_family='Arial', font_color='white'),
        marker_line_width=0.7,
        marker_line_color='white'
    )
    fig.update_layout(
        height=560,
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=title, thickness=12, len=0.62),
    )
    return fig


def render_country_intelligence(map_df, selected_country):
    """Country drill-down card for map tab."""
    if map_df is None or map_df.empty or not selected_country:
        return
    row = map_df[map_df['alert-country'] == selected_country]
    if row.empty:
        st.info("Selected country is not available under the current filters.")
        return
    r = row.iloc[0]
    st.markdown(f"""
    <div style="border:1px solid #eadff5;border-radius:16px;padding:14px;background:#fbf9ff;margin-top:8px;">
      <div style="font-size:15px;font-weight:900;color:#2d0055;margin-bottom:6px;">🧭 Country intelligence: {r['alert-country']}</div>
      <div style="font-size:13px;line-height:1.55;color:#333;">
        <b>Region:</b> {r['region']}<br>
        <b>Total alerts:</b> {int(r['total_alerts'])}<br>
        <b>Negative alerts:</b> {int(r['negative_alerts'])} ({r['negative_share']}%)<br>
        <b>Positive alerts:</b> {int(r['positive_alerts'])} ({r['positive_share']}%)<br>
        <b>Context to watch:</b> {int(r['context_to_watch_alerts'])} ({r['context_share']}%)<br>
        <b>Priority signal:</b> <span style="font-weight:900;color:#660094;">{r['risk_level']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
# ---------------- TABS ----------------
tab_overview, tab_negative, tab_map, tab_manual = st.tabs(
    [
        "📊 Overview",
        "⚠️ Negative Alerts",
        "🗺️ Visualization Map",
        "📘 User Manual",
    ]
)
st.markdown(
    """
    <style>
    /* Tabs container */
    div[data-testid="stTabs"] {
        display: flex !important;       /* flex container */
        width: 100%;
        gap: 0px;                        /* no extra gap */
        background-color: #ffffff;
        border-bottom: 1px solid #e6e6e6;
    }

    /* Ensure each tab wrapper div expands */
    div[data-testid="stTabs"] > div {
        flex: 1 !important;             /* make wrapper div take equal space */
    }

    /* Tabs list: buttons inside wrapper div */
    div[data-testid="stTabs"] button {
        width: 100% !important;          /* fill the wrapper div completely */
        font-size: 15px;
        font-weight: 700;                /* bold text */
        font-family: "Arial Black", Arial, sans-serif;
        color: #444444;
        padding: 10px 0;
        margin: 0;                        /* remove default margin */
        border-radius: 6px 6px 0 0;
        background-color: #f8f9fa;
        text-align: center;
        border-bottom: 3px solid transparent; /* reserve space for hover underline */
        transition: all 0.2s ease;
    }

    /* Hover effect: thicker underline */
    div[data-testid="stTabs"] button:hover {
        background-color: #660094;
        color: white;
        border-bottom: 5px solid #2d0055;
    }

    /* Active tab */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #660094;
        color: white;
        font-weight: 700;
        border-bottom: 5px solid #2d0055;
    }

    /* Tab content spacing */
    div[data-testid="stTabs"] > div[role="tabpanel"] {
        padding-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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


# ---------------- TAB 1 ------------------------
with tab_overview:
    #st.subheader("Overview Metrics")
    render_summary_cards(filtered_global)
    a1 = filtered_global.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
    df_clean = filtered_global.assign(**{"enabling-principle": filtered_global["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().map(ENABLING_PRINCIPLE_LABEL_MAP)
    df_clean["enabling-principle"] = pd.Categorical(df_clean["enabling-principle"],categories=ENABLING_PRINCIPLE_ORDER,ordered=True)
    a2 = df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count').sort_values("enabling-principle",ascending=False)
    a3 = filtered_global.groupby(["region","alert-impact"]).size().reset_index(name='count')
    a4 = filtered_global.groupby(["alert-country","alert-impact"]).size().reset_index(name='count').sort_values(by='count', ascending=False).head(20)
    r1c1,r1c2 = st.columns(2)
    r2c1,r2c2 = st.columns(2)


    r1c1.plotly_chart(create_h_stacked_bar(a1,y="alert-type",x="count",color_col="alert-impact",title="Alert type distribution", horizontal=True, normalize_labels=True),use_container_width=True,  key="tab1_chart1")

    fig12 = create_h_stacked_bar(
        a2,
        y="enabling-principle",
        x="count",
        color_col="alert-impact",
        title="Alert distribution across enabling principles", 
        horizontal=True,
        normalize_labels=False
    )

        # Add "?" tooltip icon immediately after title
    fig12.add_annotation(
        xref='paper', yref='paper',
        x=0.42,  # adjust so it sits right after the title
        y=1.05,
        text="❔",  # unicode "?" inside a circle
        showarrow=False,
        font=dict(color="white", size=10, family="Arial", weight="bold"),
        align="center",
        bordercolor="black",
        borderwidth=0.8,
        borderpad=3,
        bgcolor="#660094",
        opacity=0.9,
        hovertext=(
            "Alerts may be classified under more than one enabling principle "
            "<br>and can therefore be counted in multiple principles."
        ),
        hoverlabel=dict(bgcolor="black", font_color="white", font_size=12)
    )

    # Add source line if needed
    fig12 = add_source_line(fig12)

    # Render chart in Streamlit
    r1c2.plotly_chart(fig12, use_container_width=True, key="tab1_chart2")
  
    #r1c2.plotly_chart(create_h_stacked_bar(a2,y="enabling-principle",x="count",color_col="alert-impact",title="Alert distribution across enabling principles", horizontal=True),use_container_width=True,  key="tab1_chart2")

    #if is_privileged():
    r2c1.plotly_chart(create_h_stacked_bar(a3,y="region",x="count",color_col="alert-impact",title="Alert distribution across regions", horizontal=False, normalize_labels=False),use_container_width=True,  key="tab1_chart3")
    r2c2.plotly_chart(create_h_stacked_bar(a4,y="alert-country",x="count",color_col="alert-impact",title="Alert distribution across countries", horizontal=False, normalize_labels=False),use_container_width=True,  key="tab1_chart4")

    
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

    with st.expander("Summary Data preview"):
        st.write(filtered_global_prev)  
    #else:
        #st.info("Sign in with an authorized account to unlock additional detailed and disaggregated data.")   
        
# ---------------- Negative Events ----------------
with tab_negative:
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
    

        # ---------------- INLINE FILTERS ----------------
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            selected_actor_types = safe_multiselect(
                "Types of restrictive actors",
                formatted_options(df_exploded["Actor of repression"]),
                "selected_actor_types",
                sidebar=False
            )

        with col2:
            selected_subject_types = safe_multiselect(
                "Types of civil society actors affected",
                formatted_options(df_exploded["Subject of repression"]),
                "selected_subject_types",
                sidebar=False
            )

        with col3:
            selected_mechanism_types = safe_multiselect(
                "Types of restrictive mechanisms",
                formatted_options(df_exploded["Mechanism of repression"]),
                "selected_mechanism_types",
                sidebar=False
            )

        with col4:
            selected_event_types = safe_multiselect(
                "Types of negative events",
                formatted_options(df_exploded["Type of event"]),
                "selected_event_types",
                sidebar=False
            )

        reactive_df_updated = reactive_df[
            (reactive_df['Actor of repression'].apply(lambda x: contains_any(x, selected_actor_types))) &
            (reactive_df['Subject of repression'].apply(lambda x: contains_any(x, selected_subject_types))) &
            (reactive_df['Mechanism of repression'].apply(lambda x: contains_any(x, selected_mechanism_types))) &
            (reactive_df['Type of event'].apply(lambda x: contains_any(x, selected_event_types)))
        ]
        render_summary_cards(reactive_df_updated, show_breakdown=False)

        filtered_df = df_exploded[
            (df_exploded['Actor of repression'].apply(lambda x: contains_any(x, selected_actor_types))) &
            (df_exploded['Subject of repression'].apply(lambda x: contains_any(x, selected_subject_types))) &
            (df_exploded['Mechanism of repression'].apply(lambda x: contains_any(x, selected_mechanism_types))) &
            (df_exploded['Type of event'].apply(lambda x: contains_any(x, selected_event_types)))
        ]

        tab2_actor = reactive_df_updated.assign(**{"Actor of repression": reactive_df_updated["Actor of repression"].str.split(",")}).explode("Actor of repression")
        tab2_actor["Actor of repression"] = tab2_actor["Actor of repression"].str.strip()
        m1 = tab2_actor.groupby(["Actor of repression", "alert-impact"]).size().reset_index(name='count')

        tab2_subj = reactive_df_updated.assign(**{"Subject of repression": reactive_df_updated["Subject of repression"].apply(safe_split)}).explode("Subject of repression")
        tab2_subj["Subject of repression"] = tab2_subj["Subject of repression"].str.strip()
        m2 = tab2_subj.groupby(["Subject of repression", "alert-impact"]).size().reset_index(name='count')

        tab2_mech = reactive_df_updated.assign(**{"Mechanism of repression": reactive_df_updated["Mechanism of repression"].str.split(",")}).explode("Mechanism of repression")
        tab2_mech["Mechanism of repression"] = tab2_mech["Mechanism of repression"].str.strip()
        m3 = tab2_mech.groupby(["Mechanism of repression", "alert-impact"]).size().reset_index(name='count')

        tab2_type = reactive_df_updated.assign(**{"Type of event": reactive_df_updated["Type of event"].str.split(",")}).explode("Type of event")
        tab2_type["Type of event"] = tab2_type["Type of event"].str.strip()
        m4 = tab2_type.groupby(["Type of event", "alert-impact"]).size().reset_index(name='count')

        tab2_alert = reactive_df_updated.assign(**{"alert-type": reactive_df_updated["alert-type"].str.split(",")}).explode("alert-type")
        tab2_alert["alert-type"] = tab2_alert["alert-type"].str.strip()
        m5 = tab2_alert.groupby(["alert-type", "alert-impact"]).size().reset_index(name='count')

        tab2_enabling_principle = reactive_df_updated.assign(**{"enabling-principle": reactive_df_updated["enabling-principle"].str.split(",")}).explode("enabling-principle")
        tab2_enabling_principle["enabling-principle"] = tab2_enabling_principle["enabling-principle"].str.strip().map(ENABLING_PRINCIPLE_LABEL_MAP)
        tab2_enabling_principle["enabling_principle"] = pd.Categorical(tab2_enabling_principle["enabling-principle"], categories=ENABLING_PRINCIPLE_ORDER, ordered=True)
        m6 = tab2_enabling_principle.groupby(["enabling-principle", "alert-impact"]).size().reset_index(name='count').sort_values("enabling-principle", ascending=False)

        # ---------------- BAR CHARTS ----------------
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)

        r1c1.plotly_chart(create_bar_chart(m1, "Actor of repression", "count", title="Types of restrictive actors", normalize_labels=True), use_container_width=True, key="tab2_chart1")
        r1c2.plotly_chart(create_bar_chart(m2, "Subject of repression", "count", title="Types of civil society actors affected", normalize_labels=True), use_container_width=True, key="tab2_chart2")
        r1c3.plotly_chart(create_bar_chart(m3, "Mechanism of repression", "count", title="Types of restrictive mechanisms", normalize_labels=True), use_container_width=True, key="tab2_chart3")
        r2c1.plotly_chart(create_bar_chart(m4, "Type of event", "count", title="Types of negative events", horizontal=True, normalize_labels=True), use_container_width=True, key="tab2_chart4")
        r2c2.plotly_chart(create_bar_chart(m5, "alert-type", "count", title="Distribution of negative alert types", horizontal=True, normalize_labels=True), use_container_width=True, key="tab2_chart5")

        fig23 = create_bar_chart(m6, "enabling-principle", "count", title="Negative alert distribution across enabling principle", horizontal=True, normalize_labels=False)
        fig23.add_annotation(
            xref='paper', yref='paper', x=0.42, y=1.05, text="❔", showarrow=False,
            font=dict(color="white", size=10, family="Arial black", weight="bold"),
            align="center", bordercolor="black", borderwidth=1.3, borderpad=3, bgcolor="#660094", opacity=1.0,
            hovertext="Alerts may be classified under more than one enabling principle <br>and can therefore be counted in multiple principles.",
            hoverlabel=dict(bgcolor="black", font_color="white", font_size=12)
        )
        fig23 = add_source_line(fig23)
        r2c3.plotly_chart(fig23, use_container_width=True, key="tab2_chart6")

        # ---------------- TOP-N CONFIG ----------------
        if "top_n_option" not in st.session_state:
            st.session_state.top_n_option = "Top 5"
            st.session_state.top_n = 5

        st.markdown(
            """
            <style>
            #top-n-select div[data-baseweb="select"] > div {
                font-size: 30px;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div id="top-n-select">', unsafe_allow_html=True)
        top_n_map = {"Top 2": 2, "Top 3": 3, "Top 4": 4, "Top 5": 5, "All": None}
        selected = st.selectbox(
            "Select a value from the drop-down menu to view the top mechanism used by restrictive actor, \n"
            "restrictive mechanism affecting cicil society actors, and who are the actors restricting civil society",
            options=list(top_n_map.keys()),
            index=list(top_n_map.keys()).index(st.session_state.get("top_n_option", "Top 5"))
        )
        st.session_state.top_n_option = selected
        st.session_state.top_n = top_n_map[selected]
        st.markdown('</div>', unsafe_allow_html=True)
        top_n = st.session_state.top_n

        # ---------------- HEATMAPS ----------------
        #with st.expander("Show Heatmaps"):
        #filtered_df['Subject of repression']= filtered_df['Subject of repression'].apply(safe_split)

        render_heatmaps(filtered_df, top_n=top_n)
    
        # ---------------- SANKEY DIAGRAM ----------------
        #with st.expander("Show Flowchart (Sankey Diagram)"):
        st.plotly_chart(render_sankey(filtered_df, top_n=top_n), use_container_width=True)

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
        #if is_privileged():        
        with st.expander("Summary Data preview"):
            st.write(reactive_df_updated_prev)
        #else:
            #st.info("Sign in with an authorized account to unlock additional detailed and disaggregated data.")      
    
        # ---------------- TAB 3 (MAP) ----------------
with tab_map:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#2d0055,#660094);color:white;padding:16px;border-radius:18px;margin-bottom:14px;">
        <div style="font-size:22px;font-weight:900;">🗺️ Map Intelligence</div>
        <div style="font-size:13px;opacity:.92;">Country-level spatial signals, risk categories, drill-down summaries, and AI-ready interpretation from the current filters.</div>
    </div>
    """, unsafe_allow_html=True)

    render_summary_cards(filtered_global)

    geo_file = EXPORT_DIR / "countriess.geojson"
    if not geo_file.exists():
        geo_file = Path.cwd() / "exports" / "countriess.geojson"

    if geo_file.exists():
        with open(geo_file, encoding="utf-8") as f:
            countries_gj = json.load(f)

        map_intel_df = prepare_map_intelligence_data(filtered_global.copy(), countries_gj)

        if map_intel_df.empty:
            st.warning("No mapped country records are available under the current filters.")
        else:
            render_map_intelligence_cards(map_intel_df)

            ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1])
            with ctrl1:
                map_mode = st.selectbox(
                    "Map layer",
                    ["Risk score", "Negative share", "Total alerts"],
                    index=0,
                    help="Switch between composite priority signal, negative-alert intensity, and total reporting volume."
                )
            with ctrl2:
                selected_country = st.selectbox(
                    "Country drill-down",
                    ["Select country"] + sorted(map_intel_df["alert-country"].dropna().astype(str).unique().tolist()),
                    index=0
                )
            with ctrl3:
                top_n_map = st.slider("Top priority countries", 3, 15, 8)

            fig = create_intelligent_choropleth(map_intel_df, countries_gj, map_mode=map_mode)
            st.plotly_chart(fig, use_container_width=True, key="map_intelligence_choropleth")

            left, right = st.columns([1.15, 1])
            with left:
                st.markdown("### 🚦 Priority country signals")
                priority_cols = [
                    "alert-country", "region", "total_alerts", "negative_alerts", "negative_share", "risk_level"
                ]
                priority_df = (
                    map_intel_df[priority_cols]
                    .sort_values(["risk_score", "negative_alerts", "total_alerts"], ascending=False)
                    .head(top_n_map)
                    .rename(columns={
                        "alert-country": "Country",
                        "region": "Region",
                        "total_alerts": "Total alerts",
                        "negative_alerts": "Negative alerts",
                        "negative_share": "% negative",
                        "risk_level": "Priority signal"
                    })
                )
                st.dataframe(priority_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download map intelligence CSV",
                    map_intel_df.to_csv(index=False).encode("utf-8"),
                    file_name="eusee_map_intelligence.csv",
                    mime="text/csv"
                )

            with right:
                if selected_country != "Select country":
                    render_country_intelligence(map_intel_df, selected_country)
                else:
                    st.markdown("### 🧠 AI map interpretation")
                    map_summary = map_intelligence_summary(map_intel_df)
                    st.info(map_summary)

                map_summary_txt = map_intelligence_summary(map_intel_df)
                if st.button("💬 Send map insight to AI Copilot", key="send_map_intel_to_ai"):
                    if "ai_messages" not in st.session_state:
                        st.session_state.ai_messages = []
                    st.session_state.ai_messages.append({"role": "assistant", "content": append_eusee_redirect(map_summary_txt)})
                    st.success("Map insight sent to the AI Copilot chat history.")

            with st.expander("How to interpret this map intelligence layer"):
                st.markdown("""
                - **Risk score** combines negative-alert volume and negative-alert share into a simple priority signal.
                - **Negative share** highlights countries where a larger proportion of mapped alerts are negative.
                - **Total alerts** shows reporting volume and should not be interpreted alone as severity.
                - Country signals are affected by reporting coverage, monitoring intensity, and partner submission patterns.
                """)
    else:
        st.warning("GeoJSON file not found for map visualization.")

# -----------------USER MANUAL TAB-----------------
    
with tab_manual:
    def display_pdf_link(title, description, pdf_path: Path):
        st.markdown(f"""
            <div style="font-family: Arial; color: #660094; font-size: 14px;">
                <h2 style="font-size: 20px;">{title}</h2>
                <p style="font-size: 12px;">{description}</p>
            </div>
        """, unsafe_allow_html=True)

        if pdf_path.exists():
            # Download button
            st.download_button(
                f"Download {title} (PDF)",
                pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf"
            )

            # Open in new tab
            st.markdown(
                f'<a href="{pdf_path.as_posix()}" target="_blank">Open {title} in new tab</a>',
                unsafe_allow_html=True
            )
        else:
            st.warning(f"{title} PDF not found.")


    # --- Dashboard Header ---
    st.markdown("""
    <div style="font-family: Arial; color: #660094; font-size: 14px;">
        <h1 style="font-size: 24px;">EU SEE Dashboard – Quick Start</h1>
        <p>This section provides concise, decision-ready documentation for executives,
        donors, and policy stakeholders.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Executive Brief ---
    display_pdf_link(
        "Executive Brief (1 Page)",
        "For senior leadership, donors, and policy reporting.",
        EXEC_BRIEF_PATH
    )

    st.divider()

    # --- Full User Manual ---
    display_pdf_link(
        "Full User Manual",
        "<em>Detailed guidance for analysts and advanced users</em>",
        USER_MANUAL_PATH
    )


# ---------------- AI ASSISTANT v5: POLISHED UX + CHATBOT PLOT BUILDER ----------------
def _ai_get_available_plot_dimensions(df):
    dims = []
    candidates = [
        ("alert-country", "Country"), ("region", "Region"), ("alert-impact", "Alert impact"),
        ("alert-type", "Alert type"), ("enabling-principle", "Enabling principle"),
        ("Actor of repression", "Restrictive actor"), ("Subject of repression", "Affected civil society actor"),
        ("Mechanism of repression", "Restrictive mechanism"), ("Type of event", "Negative event type"),
        ("year", "Year"), ("month_name", "Month"),
    ]
    for col, label in candidates:
        if df is not None and not df.empty and col in df.columns:
            dims.append((label, col))
    return dims


def _ai_clean_count_df(df, col, top_n=10):
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    tmp = df.copy()
    multi_cols = ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event", "enabling-principle", "alert-type"]
    if col in multi_cols:
        tmp[col] = tmp[col].fillna("").astype(str).str.replace(r"\bVNSAs\b", "Violent non-state actors", regex=True)
        tmp = tmp.assign(**{col: tmp[col].str.split(",")}).explode(col)
    tmp[col] = tmp[col].fillna("").astype(str).str.strip()
    tmp = tmp[tmp[col] != ""]
    out = tmp[col].value_counts().head(top_n).reset_index()
    out.columns = [col, "count"]
    return out


def _ai_make_plot(df, dimension_col, chart_type="Horizontal bar", top_n=10, title=None):
    plot_df = _ai_clean_count_df(df, dimension_col, top_n=top_n)
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for this plot under the current filters.", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    title = title or f"Top {min(top_n, len(plot_df))} records by {dimension_col}"
    if chart_type == "Horizontal bar":
        plot_df = plot_df.sort_values("count", ascending=True)
        fig = px.bar(plot_df, x="count", y=dimension_col, orientation="h", text="count", title=title)
        fig.update_traces(marker_color="#660094", textposition="outside")
    elif chart_type == "Donut":
        fig = px.pie(plot_df, values="count", names=dimension_col, hole=0.55, title=title)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    elif chart_type == "Treemap":
        fig = px.treemap(plot_df, path=[dimension_col], values="count", title=title)
    else:
        fig = px.bar(plot_df, x=dimension_col, y="count", text="count", title=title)
        fig.update_traces(marker_color="#008CAA", textposition="outside")
        fig.update_xaxes(tickangle=-35)
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=55, b=65),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=13, family="Arial Black", color="#2d0055"), x=0.02),
        font=dict(family="Arial", size=11, color="#222"), showlegend=True,
    )
    fig.add_annotation(text="EUSEE Dashboard | filtered view", xref="paper", yref="paper", x=0.5, y=-0.22, showarrow=False, font=dict(size=10, color="#777"))
    return fig


def _ai_plot_intent_to_dimension(question, df):
    q = str(question).lower()
    mapping = [
        (["country", "countries"], "alert-country", "Horizontal bar"),
        (["region", "regional"], "region", "Bar"),
        (["impact", "negative", "positive"], "alert-impact", "Donut"),
        (["alert type", "type"], "alert-type", "Horizontal bar"),
        (["principle", "enabling"], "enabling-principle", "Horizontal bar"),
        (["actor", "actors"], "Actor of repression", "Horizontal bar"),
        (["subject", "affected", "civil society"], "Subject of repression", "Horizontal bar"),
        (["mechanism", "mechanisms"], "Mechanism of repression", "Horizontal bar"),
        (["event"], "Type of event", "Horizontal bar"),
        (["year", "annual"], "year", "Bar"),
        (["month", "monthly"], "month_name", "Bar"),
    ]
    for keys, col, ctype in mapping:
        if any(k in q for k in keys) and col in getattr(df, "columns", []):
            return col, ctype
    dims = _ai_get_available_plot_dimensions(df)
    return (dims[0][1], "Horizontal bar") if dims else (None, "Bar")


def _save_ai_answer(question, df):
    q = str(question).strip()
    st.session_state.ai_messages.append({"role": "user", "content": q})
    answer = local_ai_response(q, df)
    plot_words = ["plot", "chart", "graph", "visual", "visualize", "draw", "show me a chart"]
    if any(w in q.lower() for w in plot_words):
        dim, ctype = _ai_plot_intent_to_dimension(q, df)
        if dim:
            st.session_state.ai_last_plot = {"dimension_col": dim, "chart_type": ctype, "top_n": 10, "title": f"Chatbot-generated plot: {dim}"}
            answer += "\n\n📊 I generated an additional plot from the current filtered dashboard data. Open the **Plot** tab in the assistant to view or modify it."
    st.session_state.ai_pending_answer = answer
    st.session_state.ai_streaming = True


def render_ai_assistant_panel(df):
    """Professional sidebar AI cockpit. It does not interfere with dashboard layout."""
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [{"role": "assistant", "content": "Hello. I am your EU SEE AI assistant. Ask me for insights or type: plot top countries, chart alert impacts, or graph mechanisms."}]
    if "ai_streaming" not in st.session_state:
        st.session_state.ai_streaming = False
    if "ai_pending_answer" not in st.session_state:
        st.session_state.ai_pending_answer = ""
    if "ai_last_plot" not in st.session_state:
        st.session_state.ai_last_plot = None

    s = summarize_for_ai(df)
    level, level_color, level_note = ai_priority_signal(s)

    st.sidebar.markdown("""
    <style>
    section[data-testid="stSidebar"] {background: linear-gradient(180deg,#ffffff 0%,#fbf8ff 100%);}
    .eusee-ai-shell{border:1px solid #eadff5;border-radius:18px;padding:14px;background:#fff;box-shadow:0 12px 30px rgba(45,0,85,.10);margin:12px 0 18px 0;}
    .eusee-ai-brand{background:linear-gradient(135deg,#2d0055,#660094 55%,#008CAA);color:white;padding:14px;border-radius:16px;margin-bottom:12px;}
    .eusee-ai-brand-title{font-size:17px;font-weight:900;line-height:1.15;}
    .eusee-ai-brand-sub{font-size:11px;opacity:.88;margin-top:4px;}
    .eusee-ai-chip-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;}
    .eusee-ai-chip{font-size:10px;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);padding:4px 7px;border-radius:20px;}
    .eusee-ai-metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0;}
    .eusee-ai-metric-card{border:1px solid #eee6f5;border-radius:14px;padding:9px;background:#fbf9ff;}
    .eusee-ai-metric-label{font-size:10px;color:#666;font-weight:700;text-transform:uppercase;letter-spacing:.03em;}
    .eusee-ai-metric-value{font-size:18px;color:#2d0055;font-weight:900;}
    .eusee-ai-msg{background:#f6f2ff;border-left:4px solid #660094;padding:10px;border-radius:12px;margin:8px 0;font-size:12px;line-height:1.45;}
    .eusee-ai-user{background:#2d0055;color:white;padding:10px;border-radius:12px;margin:8px 0;font-size:12px;line-height:1.45;}
    .eusee-ai-note{font-size:11px;color:#555;background:#fff9dc;border-left:4px solid #FFDB58;padding:8px;border-radius:10px;margin-top:8px;}
    .eusee-ai-section-title{font-size:12px;color:#2d0055;font-weight:900;margin:8px 0 4px 0;}
    div[data-testid="stSidebar"] div[data-testid="stTabs"] button{font-size:11px!important;font-weight:800!important;padding:7px 2px!important;}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="eusee-ai-shell">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="eusee-ai-brand">
            <div class="eusee-ai-brand-title">🤖 EU SEE AI Copilot</div>
            <div class="eusee-ai-brand-sub">Professional assistant linked to current dashboard filters</div>
            <div class="eusee-ai-chip-row"><span class="eusee-ai-chip">Chat</span><span class="eusee-ai-chip">Insights</span><span class="eusee-ai-chip">Plots</span><span class="eusee-ai-chip">Export</span></div>
        </div>
        <div class="eusee-ai-metric-grid">
            <div class="eusee-ai-metric-card"><div class="eusee-ai-metric-label">Alerts</div><div class="eusee-ai-metric-value">{s['total_alerts']:,}</div></div>
            <div class="eusee-ai-metric-card"><div class="eusee-ai-metric-label">Negative</div><div class="eusee-ai-metric-value">{s['negative_pct']}%</div></div>
            <div class="eusee-ai-metric-card"><div class="eusee-ai-metric-label">Countries</div><div class="eusee-ai-metric-value">{s['countries_count']:,}</div></div>
            <div class="eusee-ai-metric-card"><div class="eusee-ai-metric-label">Priority</div><div class="eusee-ai-metric-value" style="color:{level_color};">{level}</div></div>
        </div>
        """, unsafe_allow_html=True)

        chat_tab, plot_tab, insight_tab, export_tab = st.tabs(["Chat", "Plot", "Insights", "Export"])

        with chat_tab:
            st.caption("Ask questions, or request a chart: `plot top countries`, `chart alert impacts`, `graph mechanisms`.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Summarise", key="ai_v5_q_summary", use_container_width=True):
                    _save_ai_answer("summarise the current view", df); st.rerun()
                if st.button("Top countries", key="ai_v5_q_countries", use_container_width=True):
                    _save_ai_answer("plot top countries", df); st.rerun()
            with c2:
                if st.button("Impacts chart", key="ai_v5_q_impacts", use_container_width=True):
                    _save_ai_answer("chart alert impact", df); st.rerun()
                if st.button("Mechanisms", key="ai_v5_q_mechanisms", use_container_width=True):
                    _save_ai_answer("plot mechanisms", df); st.rerun()
            if st.button("Clear chat", key="ai_v5_clear", use_container_width=True):
                st.session_state.ai_messages = [{"role": "assistant", "content": "Chat cleared. Ask me about the filtered EU SEE data or request a plot."}]
                st.session_state.ai_last_plot = None; st.rerun()

            with st.container(height=330):
                for msg in st.session_state.ai_messages[-8:]:
                    css = "eusee-ai-user" if msg["role"] == "user" else "eusee-ai-msg"
                    st.markdown(f'<div class="{css}">{_render_chat_content_html(msg["content"])}</div>', unsafe_allow_html=True)
                if st.session_state.ai_streaming and st.session_state.ai_pending_answer:
                    st.markdown('<div class="eusee-ai-msg">▌ AI is preparing a response...</div>', unsafe_allow_html=True)
                    st.session_state.ai_messages.append({"role": "assistant", "content": st.session_state.ai_pending_answer})
                    st.session_state.ai_pending_answer = ""; st.session_state.ai_streaming = False; st.rerun()

            with st.form("ai_v5_chat_form", clear_on_submit=True):
                user_q = st.text_area("Message", placeholder="Example: Plot top 10 countries by negative alerts", height=70, label_visibility="collapsed")
                submitted = st.form_submit_button("Send", use_container_width=True)
            if submitted and user_q.strip():
                _save_ai_answer(user_q, df); st.rerun()

        with plot_tab:
            st.markdown('<div class="eusee-ai-section-title">Create additional plots from chatbot data</div>', unsafe_allow_html=True)
            dims = _ai_get_available_plot_dimensions(df)
            dim_labels = [d[0] for d in dims]
            dim_map = {label: col for label, col in dims}
            if dim_labels:
                selected_label = st.selectbox("Dimension", dim_labels, index=0)
                chart_type = st.selectbox("Chart type", ["Horizontal bar", "Bar", "Donut", "Treemap"], index=0)
                top_n = st.slider("Top N", 3, 30, 10)
                selected_col = dim_map[selected_label]
                fig = _ai_make_plot(df, selected_col, chart_type=chart_type, top_n=top_n, title=f"{selected_label} distribution")
                st.plotly_chart(fig, use_container_width=True, key="ai_v5_plot_builder")
                plot_df = _ai_clean_count_df(df, selected_col, top_n=top_n)
                st.download_button("Download plot data (.csv)", data=plot_df.to_csv(index=False).encode("utf-8"), file_name="eusee_ai_plot_data.csv", mime="text/csv", use_container_width=True)
                if st.button("Insert this plot into chat", key="ai_v5_insert_plot", use_container_width=True):
                    st.session_state.ai_last_plot = {"dimension_col": selected_col, "chart_type": chart_type, "top_n": top_n, "title": f"{selected_label} distribution"}
                    st.session_state.ai_messages.append({"role": "assistant", "content": f"📊 Added a {chart_type.lower()} plot for **{selected_label}** using the current filters."})
                    st.rerun()
            else:
                st.info("No suitable fields are available for plotting under the current filters.")
            if isinstance(st.session_state.ai_last_plot, dict):
                st.markdown('<div class="eusee-ai-section-title">Last chatbot-generated plot</div>', unsafe_allow_html=True)
                lp = st.session_state.ai_last_plot
                st.plotly_chart(_ai_make_plot(df, lp["dimension_col"], lp.get("chart_type", "Horizontal bar"), lp.get("top_n", 10), lp.get("title")), use_container_width=True, key="ai_v5_last_plot")

        with insight_tab:
            st.markdown(f"**Priority signal:** :violet[{level}]  ")
            st.caption(level_note)
            st.markdown('<div class="eusee-ai-note">Use Plot to generate extra visuals without changing the main dashboard layout.</div>', unsafe_allow_html=True)
            with st.expander("Mini trend chart", expanded=True):
                render_ai_trend_chart(df)
            with st.expander("Interpret current view", expanded=False):
                st.markdown(_render_chat_content_html(local_ai_response("interpret the current view", df)), unsafe_allow_html=True)
            with st.expander("Data quality report", expanded=False):
                st.text(ai_data_quality_report(df))

        with export_tab:
            summary_text = generate_ai_executive_summary(df)
            policy_text = generate_ai_policy_brief(df)
            chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.ai_messages])
            st.download_button("Executive summary (.txt)", data=summary_text, file_name="eusee_ai_executive_summary.txt", mime="text/plain", use_container_width=True)
            st.download_button("Policy brief (.txt)", data=policy_text, file_name="eusee_ai_policy_brief.txt", mime="text/plain", use_container_width=True)
            st.download_button("Chat transcript (.txt)", data=chat_text, file_name="eusee_ai_chat_transcript.txt", mime="text/plain", use_container_width=True)
            if df is not None and not df.empty:
                st.download_button("Filtered data (.csv)", data=df.to_csv(index=False).encode("utf-8"), file_name="eusee_filtered_dashboard_data.csv", mime="text/csv", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

render_ai_assistant_panel(filtered_global)

# ---------------- FOOTER ----------------
# Footer image
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
