import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style='margin-top:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<hr style='margin:5px 0'>
""", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Multi-select styling */
.css-1wa3eu0 .css-1d391kg {background-color:#660094 !important; color:white !important;}
.css-1wa3eu0 input {color:#660094 !important;}
.css-1gtu0r7 {background-color:#f2e6ff !important; color:#660094 !important;}
.css-1gtu0r7 div[role="option"]:hover {background-color:#b266ff !important; color:white !important;}
.css-1wa3eu0 {border-color:#660094 !important;}

/* Remove default top spacing */
.css-18e3th9 {padding-top:0rem;}
.css-1d391kg {padding-top:0rem; padding-bottom:0rem;}

/* Summary cards & tabs */
.summary-card {background:linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%); color:white; padding:5px; border-radius:12px; text-align:center; margin:5px; box-shadow:0 4px 8px rgba(0,0,0,0.2);}
.summary-card h2 {font-size:22px; margin:5px 0;}
.summary-card p {font-size:12px; margin:0; opacity:0.9;}
.stTabs [role="tab"] button {font-size:20px; font-weight:bold;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD GEOJSON & METADATA ----------------
geojson_file = Path.cwd() / "data" / "countriess.geojson"
countries_gj = None
if geojson_file.exists():
    try:
        with open(geojson_file) as f:
            countries_gj = json.load(f)
    except json.JSONDecodeError:
        st.error("Invalid countries.geojson format")
else:
    st.error("❌ countries.geojson file missing inside /data folder")

meta_file = Path.cwd() / "data" / "countries_metadata.json"
country_meta = {}
if meta_file.exists():
    with open(meta_file, encoding="utf-8") as f:
        country_meta = json.load(f)
else:
    st.error("❌ countries_metadata.json file missing inside /data folder")

# ---------------- DATA LOADING ----------------
@st.cache_data(ttl=3600)
def load_data():
    parquet_file = Path.cwd() / "data" / "output_final.parquet"
    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame()
    
    df = pd.read_parquet(parquet_file)
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'] != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Map ISO and continent
    df['iso_alpha3'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3"))
    df['continent'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))
    missing_countries = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing_countries) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing_countries)}")

    # Extract year/month
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    return df

data = load_data()

# ---------------- SESSION-STATE SAFE MULTISELECT ----------------
def reactive_multiselect(label, options, session_key):
    if session_key not in st.session_state:
        st.session_state[session_key] = options
    default = [v for v in st.session_state[session_key] if v in options]
    selected = st.multiselect(label, options, default=default)
    st.session_state[session_key] = selected if selected else options
    return st.session_state[session_key]

# ---------------- GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

selected_continents = reactive_multiselect(
    "Select Continent",
    sorted(data['continent'].dropna().unique()),
    "selected_continents"
)

# Filter countries by selected continent(s)
if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique())

selected_countries = reactive_multiselect("Select Country", country_options, "selected_countries")
selected_alert_types = reactive_multiselect("Select Alert Type", sorted(data['alert-type'].dropna().unique()), "selected_alert_types")

enabling_principle_options = sorted(
    data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique().tolist()
)
selected_enablinge_principle = reactive_multiselect(
    "Select Enabling Principle", enabling_principle_options, "selected_enablinge_principle"
)

selected_alert_impacts = reactive_multiselect(
    "Select Alert Impact", sorted(data['alert-impact'].dropna().unique()), "selected_alert_impacts"
)

month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
selected_months = reactive_multiselect("Select Month", month_options, "selected_months")

year_options = sorted(data['year'].dropna().unique())
selected_years = reactive_multiselect("Select Year", year_options, "selected_years")

# Global reset
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_continents","selected_countries","selected_alert_types",
                "selected_enablinge_principle","selected_alert_impacts",
                "selected_months","selected_years"]:
        st.session_state[key] = ["Select All"]

# ---------------- FILTER DATA ----------------
def contains_any(cell, selected):
    if pd.isna(cell): return False
    return any(sel in str(cell) for sel in selected)

filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enablinge_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# ---------------- CHART & CARD FUNCTIONS ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)

def render_summary_cards(df):
    total = df.shape[0]
    neg = df[df['alert-impact'] == "Negative"].shape[0]
    pos = df[df['alert-impact'] == "Positive"].shape[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='summary-card'><p>Total Value</p><h1>{total}</h1></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='summary-card'><p>Negative Alerts</p><h1>{neg}</h1></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='summary-card'><p>Positive Alerts</p><h1>{pos}</h1></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='summary-card'><p>Max Value</p><h1>{total}</h1></div>", unsafe_allow_html=True)

def create_bar_chart(df, x, y, horizontal=False, height=350):
    fig = px.bar(df, x=x if not horizontal else y, y=y if not horizontal else x,
                 orientation='h' if horizontal else 'v',
                 color_discrete_sequence=['#660094'], text=y)
    fig.update_traces(textposition='inside', insidetextanchor='end', textfont=dict(size=13, color='white', family="Arial Black"))
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=20,b=20), xaxis_title=None, yaxis_title=None, uniformtext_mode='hide', bargap=0.2)
    return fig

def create_h_stacked_bar(df, y, x, color_col, horizontal=False, height=350):
    categories = sorted(df[color_col].unique())
    color_seq = ['#FFDB58', '#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col] == cat]
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=color_seq[i % len(color_seq)],
            text=df_cat[x] if not horizontal else df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if color_seq[i] == '#FFDB58' else 'white', size=13)
        ))
    fig.update_layout(barmode='stack', height=height, margin=dict(l=120,r=20,t=20,b=20))
    return fig

# ---------------- TAB DATA ----------------
def get_summary_data(tab, **kwargs):
    df = filtered_global.copy()
    if tab == "Tab 2":
        for key, val in kwargs.items():
            df = df[df[key].isin(val)]
    return df

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others", "Visualization map"])

# TAB 1: Overview
with tab1:
    summary_data = get_summary_data("Tab 1")
    render_summary_cards(summary_data)
    a1 = summary_data.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
    df_clean = summary_data.assign(**{"enabling-principle": summary_data["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x))
    a2 = df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count')
    a3 = summary_data.groupby(["continent","alert-impact"]).size().reset_index(name='count')
    a4 = summary_data.groupby(["alert-country","alert-impact"]).size().reset_index(name='count')
    r1c1,r1c2 = st.columns(2,gap="large")
    r2c1,r2c2 = st.columns(2,gap="large")
    r1c1.plotly_chart(create_h_stacked_bar(a1,'alert-type','count','alert-impact',True),use_container_width=True)
    r1c2.plotly_chart(create_h_stacked_bar(a2,'enabling-principle','count','alert-impact',True),use_container_width=True)
    r2c1.plotly_chart(create_h_stacked_bar(a3,'continent','count','alert-impact',False),use_container_width=True)
    r2c2.plotly_chart(create_h_stacked_bar(a4,'alert-country','count','alert-impact',False),use_container_width=True)

# TAB 2: Negative Events
with tab2:
    col1,col2,col3,col4,col5 = st.columns(5)
    actor_types = sorted(filtered_global['Actor of repression'].dropna().unique())
    subject_types = sorted(filtered_global['Subject of repression'].dropna().unique())
    mech_types = sorted(filtered_global['Mechanism of repression'].dropna().unique())
    event_types = sorted(filtered_global['Type of event'].dropna().unique())
    selected_actor_types = reactive_multiselect("Select Actor Type", actor_types, "selected_actor_types")
    selected_subject_types = reactive_multiselect("Select Subject Type", subject_types, "selected_subject_types")
    selected_mechanism_types = reactive_multiselect("Select Mechanism Type", mech_types, "selected_mechanism_types")
    selected_event_types = reactive_multiselect("Select Event Type", event_types, "selected_event_types")
    
    summary_data = filtered_global[
        (filtered_global['Actor of repression'].isin(selected_actor_types)) &
        (filtered_global['Subject of repression'].isin(selected_subject_types)) &
        (filtered_global['Mechanism of repression'].isin(selected_mechanism_types)) &
        (filtered_global['Type of event'].isin(selected_event_types)) &
        (filtered_global['alert-impact']=="Negative")
    ]
    render_summary_cards(summary_data)
    t1 = summary_data.groupby("Actor of repression").size().reset_index(name="count")
    t2 = summary_data.assign(**{"enabling-principle": summary_data["enabling-principle"].str.split(",")}).explode("enabling-principle")
    t2["enabling-principle"] = t2["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x))
    t2 = t2.groupby("enabling-principle").size().reset_index(name="count")
    t3 = summary_data.groupby("continent").size().reset_index(name="count")
    t4 = summary_data.groupby("alert-country").size().reset_index(name="count")
    t5 = summary_data.groupby("alert-type").size().reset_index(name="count")
    t6 = summary_data.groupby("month_name").size().reset_index(name="count")
    r1c1,r1c2,r1c3 = st.columns(3,gap="large")
    r2c1,r2c2,r2c3 = st.columns(3,gap="large")
    r3c1,r3c2 = st.columns(2,gap="large")
    r1c1.plotly_chart(create_bar_chart(t1,"Actor of repression","count",True),use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(t2,"enabling-principle","count",True),use_container_width=True)
    r1c3.plotly_chart(create_bar_chart(t3,"continent","count"),use_container_width=True)
    r2c1.plotly_chart(create_bar_chart(t4,"alert-country","count"),use_container_width=True)
    r2c2.plotly_chart(create_bar_chart(t5,"alert-type","count"),use_container_width=True)
    r2c3.plotly_chart(create_bar_chart(t6,"month_name","count"),use_container_width=True)

# TAB 3, 4, 5 remain similar, with charts rendered using the optimized functions
# TAB 5: Map
with tab5:
    df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
    if countries_gj:
        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]
        fig = px.choropleth_mapbox(df_map, geojson=countries_gj, locations="alert-country",
                                   featureidkey="properties.name", color="count",
                                   hover_name="alert-country", color_continuous_scale="Greens",
                                   mapbox_style="open-street-map", zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
        fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0}, height=500)
        st.plotly_chart(fig,use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
