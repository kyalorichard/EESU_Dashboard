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


st.set_page_config(page_title="EUSEE Dashboard", layout="wide")

if is_privileged():
   
    st.write("Welcome to the privileged dashboard!")
else:
    if st.session_state.get("user"):
        st.warning("⚠️ Your email is not verified. Please verify your email to access the dashboard.")
    else:
        st.info("Please log in or register using the sidebar to access the dashboard.")


BASE_DIR = Path(__file__).resolve().parent

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
    parquet_file = Path.cwd() / "data" / "output_final.parquet"
    meta_file = Path.cwd() / "data" / "countries_metadata.json"

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
    df['alert-country'] = df['alert-country'].replace({"Lebanon NAR": "Lebanon"})

    # Clean Actor of repression
    df['Actor of repression'] = df['Actor of repression'].astype(str).str.strip()
    df['Actor of repression'] = df['Actor of repression'].replace({"VNSAs": "Violent Non-State Actors"})

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
    if 'alert-type' in df.columns:
        df.loc[df['alert-type'].str.strip().str.lower() == 'Context to watch', 'alert-impact'] = 'Context to watch'

    return df

# --- Load data safely ---
data = load_data()

# ---------------- MULTISELECT WITH SELECT ALL ----------------
def safe_multiselect(label, options, session_key, sidebar=True):
    options = sorted(list(options))
    
    # Always keep "Select All" as first dropdown option
    options_with_all = ["Select All"] + options

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
    if "Select All" in selected or len(selected) == 0:
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

filtered_countries = data[data['region'].isin(selected_regions)] if "Select All" not in selected_regions else data
selected_countries = safe_multiselect("Select country", filtered_countries['alert-country'].dropna().unique(), "selected_countries")

selected_alert_impacts = safe_multiselect("Select Nature of event/alert", data['alert-impact'].dropna().unique(), "selected_alert_impacts")
selected_alert_types = safe_multiselect("Select Impact of alert", data['alert-type'].dropna().unique(), "selected_alert_types")

selected_enabling_principle = safe_multiselect(
    "Select enabling principle", 
    data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique(),
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
selected_months = safe_multiselect("Select Month", available_months, "selected_months")
# Reset button
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_regions","selected_countries","selected_alert_types","selected_enabling_principle",
                "selected_alert_impacts","selected_months","selected_years"]:
        st.session_state[key] = ["Select All"]

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
    3. Alerts Breakdown (Negative vs Positive)
    
    Parameters:
        df (DataFrame): Filtered data
        base_bar_height (int): Base height of the horizontal bar
    """
    total_countries = df['alert-country'].nunique() if not df.empty else 0
    total_alerts = len(df) if not df.empty else 0
    negative = (df['alert-impact'] == "Negative").sum() if not df.empty else 0
    positive = (df['alert-impact'] == "Positive").sum() if not df.empty else 0
    total_np = negative + positive

    # Percentages
    neg_pct = round((negative / total_np) * 100, 1) if total_np else 0
    pos_pct = round((positive / total_np) * 100, 1) if total_np else 0

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
            <circle cx="60" cy="60" r="40" stroke="#660094" stroke-width="12" fill="none"
                stroke-dasharray="{2*3.1416*40}" 
                stroke-dashoffset="{2*3.1416*40*(1-(pos_pct/100))}"
                stroke-linecap="round" transform="rotate(-90 60 60)">
                <title>Positive Alerts: {positive} ({pos_pct}%)</title>
            </circle>
            <text x="60" y="40" text-anchor="middle" font-size="12" font-weight="bold" fill="#660094">
                {pos_pct}%
            </text>
            <text x="40" y="80" text-anchor="middle" font-size="12" font-weight="bold" fill="#FFDB58">
                {neg_pct}%
            </text>
            <text x="60" y="65" text-anchor="middle" font-size="2" font-weight="bold" color="white",fill="#333">
                {total_alerts}
            </text>
        </svg>
        <div style="margin-top:10px; font-size:16px; font-weight:600; color:#555;">
            Alerts Breakdown
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; padding:6px 2px; font-size:15px; font-weight:700; width:100%; gap:40px;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:10px; height:10px; background:#FFDB58; border-radius:50%; display:inline-block;"></span>
                <span style="color:#555;">Negative:</span>
                <span style="color:#FFDB58;">{negative}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:10px; height:10px; background:#660094; border-radius:50%; display:inline-block;"></span>
                <span style="color:#555;">Positive:</span>
                <span style="color:#660094;">{positive}</span>
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

# Define a consistent color mapping for your dashboard
COLOR_MAPPING = {
    "positive": "#660094",
    "negative": "#FFDB58"
}

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
        
    # Add hidden annotation just below plot
    fig.add_annotation(
        text="Source: EUSEE Dashboard. Data compiled by EUSEE Network.",
        xref="paper", yref="paper",
        x=0, y=-0.12,  # fixed slightly below chart for all cases
        showarrow=False,
        font=dict(size=10, color="gray"),
        opacity=0  # invisible on-screen
    )

    # ---------------- WATERMARK ----------------
    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.02,
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
    color_sequence = ['#FFDB58', '#660094']
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
            marker_color=color_sequence[i % len(color_sequence)],
            text=df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if color_sequence[i]=="#FFDB58" else 'white', size=10, family="Arial black"),
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
        
    # Add hidden annotation just below plot
    fig.add_annotation(
        text="Source: EUSEE Dashboard. Data compiled by EUSEE Network.",
        xref="paper", yref="paper",
        x=0, y=-0.12,  # fixed slightly below chart for all cases
        showarrow=False,
        font=dict(size=10, color="gray"),
        opacity=0  # invisible on-screen
    )
    # ---------------- WATERMARK ----------------
    fig.add_annotation(
        text="EUSEE Dashboard<br>Data compiled by EUSEE Network",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.02,
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
    Renders three heatmaps for Negative Events tab, handling multi-valued fields:
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

    # Explode multi-valued columns
    df_exploded = explode_multi_valued_columns(df, [
        "Actor of repression",
        "Subject of repression",
        "Mechanism of repression"
    ])

    # Determine Top-N items
    top_actors = get_top_n_items(df_exploded, "Actor of repression", top_n)
    top_subjects = get_top_n_items(df_exploded, "Subject of repression", top_n)
    top_mechanisms = get_top_n_items(df_exploded, "Mechanism of repression", top_n)

    # Filter to Top-N items
    df_top = df_exploded[
        df_exploded['Actor of repression'].isin(top_actors) &
        df_exploded['Subject of repression'].isin(top_subjects) &
        df_exploded['Mechanism of repression'].isin(top_mechanisms)
    ].copy()

    # Create pivot tables
    actor_mechanism_pivot = filter_top_n(df_top, 'Actor of repression', 'Mechanism of repression', top_n)
    subject_mechanism_pivot = filter_top_n(df_top, 'Subject of repression', 'Mechanism of repression', top_n)
    actor_subject_pivot = filter_top_n(df_top, 'Actor of repression', 'Subject of repression', top_n)

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
        fig1.update_layout(
            xaxis=dict(tickfont=dict(size=10, family="Arial")),
            yaxis=dict(tickfont=dict(size=10, family="Arial")),
            title=dict(
                text=fig1.layout.title.text,
                x=0.5,
                xanchor="center",
                font=dict(size=12, family="Arial")
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
    a4 = filtered_global.groupby(["alert-country","alert-impact"]).size().reset_index(name='count').sort_values(by='count', ascending=False).head(21)  # select top 20
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
    
    if is_privileged():
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
    else:
        st.info("Sign in with an authorized account to unlock additional detailed and disaggregated data.")   
            
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
        reactive_df['Actor of repression'] = reactive_df['Actor of repression'].replace("VNSAs", "Violent Non-State Actors")
            
        # ---------------- SUMMARY CARDS ----------------
        # Show totals BEFORE exploding multi-valued columns
            
        # ---------------- EXPLODE MULTI-VALUED COLUMNS ----------------
        cols_to_explode = ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]
        df_exploded = explode_multi_valued_columns(reactive_df, cols_to_explode)
            
        df_exploded = reactive_df.copy()
        for col in cols_to_explode:
            df_exploded[col] = df_exploded[col].str.split(",")
            df_exploded = df_exploded.explode(col)
            df_exploded[col] = df_exploded[col].str.strip()
            
        # ---------------- INLINE FILTERS ----------------
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_actor_types = safe_multiselect(
                "Types of restrictive actors",
                df_exploded['Actor of repression'].dropna().unique(),
                "selected_actor_types", sidebar=False
            )
        with col2:
            selected_subject_types = safe_multiselect(
                "Types of civil society actors affected",
                df_exploded['Subject of repression'].dropna().unique(),
                "selected_subject_types", sidebar=False
            )
        with col3:
            selected_mechanism_types = safe_multiselect(
                "Types of restrictive mechanisms",
                df_exploded['Mechanism of repression'].dropna().unique(),
                "selected_mechanism_types", sidebar=False
            )
        with col4:
            selected_event_types = safe_multiselect(
                "Types of negative events",
                df_exploded['Type of event'].dropna().unique(),
                "selected_event_types", sidebar=False
            )
        ##### -------- Tab 2 Summary card totals--------------------------
        reactive_df_updated= reactive_df[(reactive_df['Actor of repression'].apply(lambda x: contains_any(x, selected_actor_types))) &
            (reactive_df['Subject of repression'].apply(lambda x: contains_any(x, selected_subject_types))) &
            (reactive_df['Mechanism of repression'].apply(lambda x: contains_any(x, selected_mechanism_types))) &
            (reactive_df['Type of event'].apply(lambda x: contains_any(x, selected_event_types)))
        ]
        render_summary_cards(reactive_df_updated,show_breakdown=False)

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

        tab2_subj = reactive_df_updated.assign(**{"Subject of repression": reactive_df_updated["Subject of repression"].str.split(",")}).explode("Subject of repression")
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

        
        r1c1.plotly_chart(create_bar_chart(m1, "Actor of repression", "count",title="Types of restrictive actors", normalize_labels=True), use_container_width=True, key="tab2_chart1")
        r1c2.plotly_chart(create_bar_chart(m2, "Subject of repression", "count",title="Types of civil society actors affected", normalize_labels=True), use_container_width=True, key="tab2_chart2")
        r1c3.plotly_chart(create_bar_chart(m3, "Mechanism of repression", "count",title="Types of restrictive mechanisms", normalize_labels=True), use_container_width=True, key="tab2_chart3")
        r2c1.plotly_chart(create_bar_chart(m4, "Type of event", "count",title="Types of negative events", horizontal=True, normalize_labels=True), use_container_width=True, key="tab2_chart4")
        r2c2.plotly_chart(create_bar_chart(m5, "alert-type", "count",title="Distribution of negative alert types", horizontal=True, normalize_labels=True), use_container_width=True, key="tab2_chart5")
              
        fig23= (create_bar_chart(m6, "enabling-principle", "count", title="Negative alert distribution across enabling principle", horizontal=True, normalize_labels=False))

              
        # Add the "?" tooltip icon immediately after the title
        fig23.add_annotation(
            xref='paper', yref='paper',
            x=0.42,         # adjust so it's at the end of the title
            y=1.05,         # same vertical alignment as title
            text="❔",       # Unicode circle with question mark
            showarrow=False,
            font=dict(color="white", size=10, family="Arial black", weight="bold"),
            align="center",
            bordercolor="black",
            borderwidth=1.3,
            borderpad=3,
            bgcolor="#660094",
            opacity=1.0,
            hovertext=(
                "Alerts may be classified under more than one enabling principle "
                "<br>and can therefore be counted in multiple principles."
            ),
            hoverlabel=dict(bgcolor="black", font_color="white", font_size=12)
        )

        # Add source line if needed
        fig23 = add_source_line(fig23)

        # Render the chart in Streamlit
        r2c3.plotly_chart(fig23, use_container_width=True, key="tab2_chart6")

        #r2c3.plotly_chart(create_bar_chart(m6, "enabling-principle", "count",title="Negative alert distribution across enabling principles", horizontal=True), use_container_width=True, key="tab2_chart6")

        # ---------------- TOP-N CONFIG ----------------
        if "top_n_option" not in st.session_state:
            st.session_state.top_n_option = "Top 5"
            st.session_state.top_n = 5
            
        def update_top_n():
            st.session_state.top_n = {
                "Top 2": 2, "Top 3": 3, "Top 4": 4, "Top 5": 5, "All": None
            }[st.session_state.top_n_option]

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
        top_n_map = {
            "Top 2": 2,
            "Top 3": 3,
            "Top 4": 4,
            "Top 5": 5,
            "All": None
        }
        
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
        if is_privileged():        
            with st.expander("Summary Data preview"):
                st.write(reactive_df_updated_prev)
        else:
            st.info("Sign in with an authorized account to unlock additional detailed and disaggregated data.")      
        
        # ---------------- TAB 3 (MAP) ----------------
with tab_map:
    #st.subheader("Visualization Map")
    render_summary_cards(filtered_global)
    geo_file = Path.cwd() / "data" / "countriess.geojson"
    if geo_file.exists():
        with open(geo_file) as f: 
            countries_gj = json.load(f)
        
        # Base map data
        df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
        map_df = filtered_global.groupby(["alert-country","iso_alpha3"]).size().reset_index(name="count")

        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]

        # ----- Dynamic center & zoom -----
        if not df_map.empty:
            coords = []
            for feature in countries_gj['features']:
                if feature['properties']['name'] in df_map['alert-country'].values:
                    geometry = feature['geometry']
                    if geometry['type'] == "Polygon":
                        coords.extend(geometry['coordinates'][0])
                    elif geometry['type'] == "MultiPolygon":
                        for poly in geometry['coordinates']:
                            coords.extend(poly[0])

            if coords:
                lons, lats = zip(*coords)
                center = {"lat": np.mean(lats), "lon": np.mean(lons)}
                zoom = max(1, min(5, 2 / (max(lons)-min(lons) + 0.01)))
            else:
                center = {"lat":10,"lon":0}
                zoom = 2
        else:
            center = {"lat":10,"lon":0}
            zoom = 2

        # ----- Add advanced hover stats -----
        stats = (
            filtered_global
            .groupby("alert-country")
            .agg(
                total_alerts=("alert-impact", "size"),
                negative_alerts=("alert-impact", lambda x: (x == "Negative").sum()),
                positive_alerts=("alert-impact", lambda x: (x == "Positive").sum())
            )
            .reset_index()
        )

        df_map = df_map.merge(stats, on="alert-country", how="left")

        # Ensure numeric
        df_map["negative_alerts"] = pd.to_numeric(df_map["negative_alerts"], errors="coerce")
        df_map["total_alerts"] = pd.to_numeric(df_map["total_alerts"], errors="coerce")
        
        # Compute percentage safely
        df_map["perc_negative"] = ((df_map["negative_alerts"] / df_map["total_alerts"]) * 100).round(1)
        
        # Optional: handle NaN values if total_alerts was 0
        df_map["perc_negative"] = df_map["perc_negative"].fillna(0)
                
        # ----- Main choropleth -----
        map_height = max(400, len(df_map)*20)

        fig = px.choropleth_mapbox(
            df_map,
            geojson=countries_gj,
            locations="alert-country",
            featureidkey="properties.name",
            color="count",
            hover_name="alert-country",
            hover_data={
                "count": False,
                "total_alerts": False,
                "negative_alerts": False,
                "positive_alerts": False,
                "perc_negative": False
            },
            color_continuous_scale="YlOrBr",
            mapbox_style="open-street-map",
            zoom=zoom,
            center=center,
            opacity=0.8
        )

        # ----- Card-style hover tooltip -----
        fig.update_traces(
            hovertemplate=(
                "<b>%{location}</b><br>"
                "<span style='color:#FFD700'>●</span> Total Alerts: %{customdata[0]}<br>"
                "<span style='color:#FF4C4C'>●</span> Negative: %{customdata[1]}<br>"
                "<span style='color:#00FFAA'>●</span> Positive: %{customdata[2]}<br>"
                "% Negative: %{customdata[3]}%<extra></extra>"
            ),
            customdata=df_map[["total_alerts","negative_alerts","positive_alerts","perc_negative"]].values,
            hoverlabel=dict(
                bgcolor="#2D0055",
                font_size=13,
                font_family="Arial",
                font_color="white",
                bordercolor="#ffffff"
            ),
            marker_line_width=1,
            marker_line_color="black"
        )

        # ----- Bubble density overlay -----
        

        # ----- Final layout -----
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            height=map_height
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("GeoJSON file not found for map visualization.") 

# -----------------USER MANUAL TAB-----------------
        
with tab_manual:
    st.markdown("""
    <div style="font-family: Arial; color: #660094; font-size: 14px;">
        <h1 style="font-size: 24px;">EU SEE Dashboard – Quick Start</h1>
        <p>This section provides concise, decision-ready documentation for executives,
        donors, and policy stakeholders.</p>
        <h2 style="font-size: 20px;">Executive Brief (1 Page)</h2>
        <p>For senior leadership, donors, and policy reporting.</p>

        
    </div>
    """, unsafe_allow_html=True)

    if EXEC_BRIEF_PATH.exists():
        pdf_bytes = EXEC_BRIEF_PATH.read_bytes()

        st.download_button(
            "Download Executive Brief (PDF)",
            pdf_bytes,
            file_name="EU_SEE_Dashboard_Quick_Start_Executive.pdf",
            mime="application/pdf"
        )

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f"""
            <div style="font-family: Arial; color: #660094; font-size: 12px;">
                <iframe
                    src="data:application/pdf;base64,{pdf_base64}"
                    width="100%"
                    height="550px"
                    style="border:none;"
                ></iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Executive Brief PDF not found.")

    st.divider()

    # Full User Manual
    st.markdown("""
    <div style="font-family: Arial; color: #660094; font-size: 14px;">
        <h2 style="font-size: 20px;">Full User Manual</h2>
        <p style="font-size: 12px;"><em>Detailed guidance for analysts and advanced users</em></p>
    </div>
    """, unsafe_allow_html=True)

    if USER_MANUAL_PATH.exists():
        pdf_bytes = USER_MANUAL_PATH.read_bytes()

        st.download_button(
            "Download Full User Manual (PDF)",
            pdf_bytes,
            file_name="EU SEE Dashboard user manual.pdf",
            mime="application/pdf"
        )

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f"""
            <div style="font-family: Arial; color: #660094; font-size: 14px;">
                <iframe
                    src="data:application/pdf;base64,{pdf_base64}"
                    width="100%"
                    height="700px"
                    style="border:none;"
                ></iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("User Manual PDF not found.")

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
