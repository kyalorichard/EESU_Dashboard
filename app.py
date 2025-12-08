import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style='margin-top:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<p style='margin-top:0; color:gray; font-size:16px;'></p>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.css-1wa3eu0 .css-1d391kg {background-color:#660094 !important;color:white !important;}
.css-1wa3eu0 input {color:#660094 !important;}
.css-1gtu0r7 {background-color:#f2e6ff !important;color:#660094 !important;}
.css-1gtu0r7 div[role="option"]:hover {background-color:#b266ff !important;color:white !important;}
.css-1wa3eu0 {border-color:#660094 !important;}
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
.summary-card {background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%); color:white; padding:5px; border-radius:12px; text-align:center; margin:5px; box-shadow:0 4px 8px rgba(0,0,0,0.2);}
.summary-card h2 {font-size:22px; margin:5px 0;}
.summary-card p {font-size:12px; margin:0; opacity:0.9;}
.stTabs [role="tab"] button {font-size:20px; font-weight:bold;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600)
def load_data():
    parquet_file = Path.cwd() / "data" / "output_final.parquet"
    geojson_file = Path.cwd() / "data" / "countriess.geojson"
    meta_file = Path.cwd() / "data" / "countries_metadata.json"

    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame(), None, None
    df = pd.read_parquet(parquet_file)
    df = df[df['alert-country'].astype(str).str.strip() != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Dates
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')

    # Country metadata
    if meta_file.exists():
        with open(meta_file, encoding='utf-8') as f:
            country_meta = json.load(f)
        df['iso_alpha3'] = df['alert-country'].apply(lambda c: country_meta.get(c, {}).get("iso_alpha3"))
        df['continent'] = df['alert-country'].apply(lambda c: country_meta.get(c, {}).get("continent", "Unknown"))
    else:
        df['continent'] = "Unknown"

    # GeoJSON
    countries_gj = None
    if geojson_file.exists():
        try:
            with open(geojson_file) as f:
                countries_gj = json.load(f)
        except json.JSONDecodeError:
            st.error("Invalid countries.geojson format")

    return df, countries_gj, country_meta

data, countries_gj, country_meta = load_data()

# ---------------- HELPERS ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    return "<br>".join([" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)])

def reactive_multiselect(label, options, session_key):
    if session_key not in st.session_state:
        st.session_state[session_key] = options
    selected = st.multiselect(label, options, default=st.session_state[session_key])
    st.session_state[session_key] = selected if selected else options
    return st.session_state[session_key]

def filter_data(df, filters):
    filtered = df.copy()
    for col, sel in filters.items():
        if sel:
            filtered = filtered[filtered[col].isin(sel)]
    return filtered

def render_summary_cards(df):
    total_value = len(df)
    neg_alerts = len(df[df['alert-impact']=="Negative"])
    pos_alerts = len(df[df['alert-impact']=="Positive"])
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="summary-card"><p>Total Value</p><h1>{total_value}</h1></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="summary-card"><p>Negative Alerts</p><h1>{neg_alerts}</h1></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="summary-card"><p>Positive Alerts</p><h1>{pos_alerts}</h1></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="summary-card"><p>Max Value</p><h1>{total_value}</h1></div>', unsafe_allow_html=True)

def create_h_stacked_bar(df, y, x, color_col, horizontal=False, height=350):
    categories = sorted(df[color_col].unique())
    color_seq = ['#FFDB58', '#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat]
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            name=cat, orientation='h' if horizontal else 'v',
            marker_color=color_seq[i%len(color_seq)],
            text=df_cat[x] if not horizontal else df_cat[x],
            textposition='inside',
            textfont=dict(color='black' if color_seq[i]=='#FFDB58' else 'white', size=13)
        ))
    fig.update_layout(barmode='stack', height=height, margin=dict(l=120,r=20,t=20,b=20))
    return fig

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")
filters = {}
filters["continent"] = reactive_multiselect("Select Continent", sorted(data['continent'].dropna().unique()), "selected_continents")
country_options = sorted(data[data['continent'].isin(filters["continent"])]["alert-country"].dropna().unique())
filters["alert-country"] = reactive_multiselect("Select Country", country_options, "selected_countries")
filters["alert-type"] = reactive_multiselect("Select Alert Type", sorted(data['alert-type'].dropna().unique()), "selected_alert_types")
filters["enabling-principle"] = reactive_multiselect(
    "Select Enabling Principle",
    sorted(data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique()),
    "selected_enablinge_principle"
)
filters["alert-impact"] = reactive_multiselect("Select Alert Impact", sorted(data['alert-impact'].dropna().unique()), "selected_alert_impacts")
filters["month_name"] = reactive_multiselect("Select Month", sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B')), "selected_months")
filters["year"] = reactive_multiselect("Select Year", sorted(data['year'].dropna().unique()), "selected_years")
filtered_global = filter_data(data, filters)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Visualization map"])

# --------- TAB 1 OVERVIEW ---------
with tab1:
    render_summary_cards(filtered_global)
    df_clean = filtered_global.assign(**{"enabling-principle": filtered_global["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x))
    charts = [
        (filtered_global.groupby(["alert-type","alert-impact"]).size().reset_index(name='count'), "alert-type","alert-impact",True),
        (df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count'), "enabling-principle","alert-impact",True),
        (filtered_global.groupby(["continent","alert-impact"]).size().reset_index(name='count'), "continent","alert-impact",False),
        (filtered_global.groupby(["alert-country","alert-impact"]).size().reset_index(name='count'), "alert-country","alert-impact",False)
    ]
    cols = [*st.columns(2,gap="large"),*st.columns(2,gap="large")]
    for col, (dfc, y, color_col, horiz) in zip(cols, charts):
        col.plotly_chart(create_h_stacked_bar(dfc, y=y, x="count", color_col=color_col, horizontal=horiz), use_container_width=True)

# --------- TAB 2 NEGATIVE EVENTS ---------
with tab2:
    neg_data = filtered_global[filtered_global['alert-impact']=="Negative"]
    render_summary_cards(neg_data)
    col1, col2, col3, col4 = st.columns(4)
    filters_tab2 = {
        "Actor of repression": reactive_multiselect("Actor of Repression", sorted(neg_data['Actor of repression'].dropna().unique()), "selected_actor"),
        "Subject of repression": reactive_multiselect("Subject of Repression", sorted(neg_data['Subject of repression'].dropna().unique()), "selected_subject"),
        "Mechanism of repression": reactive_multiselect("Mechanism of Repression", sorted(neg_data['Mechanism of repression'].dropna().unique()), "selected_mechanism"),
        "Type of event": reactive_multiselect("Type of Event", sorted(neg_data['Type of event'].dropna().unique()), "selected_event_type")
    }
    filtered_tab2 = filter_data(neg_data, filters_tab2)
    charts_tab2 = [
        ("Actor of repression", filtered_tab2.groupby("Actor of repression").size().reset_index(name="count")),
        ("enabling-principle", filtered_tab2.assign(**{"enabling-principle": filtered_tab2["enabling-principle"].str.split(",")}).explode("enabling-principle").groupby("enabling-principle").size().reset_index(name="count")),
        ("continent", filtered_tab2.groupby("continent").size().reset_index(name="count")),
        ("alert-country", filtered_tab2.groupby("alert-country").size().reset_index(name="count")),
        ("alert-type", filtered_tab2.groupby("alert-type").size().reset_index(name="count")),
        ("month_name", filtered_tab2.groupby("month_name").size().reset_index(name="count"))
    ]
    cols = [*st.columns(3,gap="large"),*st.columns(3,gap="large")]
    for col, (x_col, dfc) in zip(cols, charts_tab2):
        col.plotly_chart(create_h_stacked_bar(dfc, y=x_col, x="count", color_col="count", horizontal=True), use_container_width=True)

# --------- TAB 5 MAP ---------
with tab5:
    df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
    geo_countries = [f['properties']['name'] for f in countries_gj['features']]
    df_map = df_map[df_map['alert-country'].isin(geo_countries)]
    fig = px.choropleth_mapbox(df_map, geojson=countries_gj, locations="alert-country", featureidkey="properties.name",
                               color="count", hover_name="alert-country", color_continuous_scale="Greens",
                               mapbox_style="open-street-map", zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
    fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0}, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align: center; color: gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
