import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path  

st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

# ---------------- LOAD MANUAL MAP ----------------
map_file = Path.cwd() / "data" / "manual_map.json"
if not map_file.exists():
    st.error(f"Manual map file not found: {map_file}")
    st.stop()

with open(map_file, "r", encoding="utf-8") as f:
    manual_map = json.load(f)

# ---- LOAD GeoJSON ----
data_dir = Path.cwd() / "data"
geojson_file = data_dir / "countries.geojson"

with open(geojson_file) as f:
    countries_gj = json.load(f)

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600)  # refresh cache every hour
def load_data():
    csv_file = Path.cwd() / "data" / "raw_data.csv"
    if not csv_file.exists():
        st.error(f"CSV file not found: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)

    if 'alert-country' not in df.columns:
        st.warning("No 'alert-country' column found in CSV.")
        return df

    # Clean country names
    df['alert-country'] = df['alert-country'].astype(str).str.strip()

    # Function to get ISO3 codes
    def get_iso3(country_name):
        if pd.isna(country_name) or country_name == "":
            return None
        country_name_clean = str(country_name).strip()
        if country_name_clean in manual_map:
            return manual_map[country_name_clean]
        try:
            return pycountry.countries.lookup(country_name_clean).alpha_3
        except:
            return None

    df['iso_alpha3'] = df['alert-country'].apply(get_iso3)

    # List missing countries once
    missing_countries = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing_countries) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing_countries)}")

    return df
    #return pd.read_csv(csv_file)

data = load_data()

# ---------------- REMOVE STREAMLIT DEFAULT TOP SPACING ----------------
st.markdown("""
<style>
    /* Remove Streamlit's default top padding */
    .css-18e3th9 {padding-top: 0rem;}
    /* Optional: reduce spacing around main container */
    .css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style='margin-bottom:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<p style='margin-top:0; color:gray; font-size:16px;'></p>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)  # tight separator

# ---------------- GLOBAL SIDEBAR FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)  # top of sidebar

st.sidebar.header("🌍 Global Filters")
# ---------------- SESSION STATE FILTERS ----------------
def initialize_session_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Multi-select with "Select All" functionality
def multiselect_with_all(label, options, key):
    selected = st.sidebar.multiselect(
        label,
        options=["Select All"] + list(options),
        default=initialize_session_state(key, ["Select All"])
    )
    if "Select All" in selected:
        st.session_state[key] = ["Select All"]
        return list(options)
    else:
        st.session_state[key] = selected
        return selected

country_options = data['alert-country'].dropna().unique()
selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")

alert_type_options = data['alert-type'].dropna().unique()
selected_alert_types = multiselect_with_all("Select Alert Type", alert_type_options, "selected_alert_types")

alert_impact_options = data['alert-impact'].dropna().unique()
selected_alert_impacts = multiselect_with_all("Select Alert Impact", alert_impact_options, "selected_alert_impacts")


# ---------------- FILTER DATA BASED ON SELECTION ----------------
filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['alert-impact'].isin(selected_alert_impacts))
]
# ---------------- CSS FOR SUMMARY CARDS & TABS ----------------
st.markdown("""
<style>
.summary-card {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 {
    font-size: 36px;
    margin: 5px 0;
}
.summary-card p {
    font-size: 16px;
    margin: 0;
    opacity: 0.9;
}
.summary-icon {
    font-size: 30px;
    margin-bottom: 5px;
}
/* Increase tabs name font size */
.stTabs [role="tab"] button {
    font-size: 22px;
    font-weight: bold;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTION TO RENDER SUMMARY CARDS ----------------
def render_summary_cards(data):
    total_value = data["alert-country"].count()
    avg_value = data["alert-type"].count()
    max_value = data["alert-impact"].count()
    min_value = data["alert-country"].count()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-icon">💰</div>
            <h2>{total_value}</h2>
            <p>Total Value</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="summary-card" style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)">
            <div class="summary-icon">📊</div>
            <h2>{avg_value:.2f}</h2>
            <p>Average Value</p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="summary-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%)">
            <div class="summary-icon">📈</div>
            <h2>{max_value}</h2>
            <p>Max Value</p>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="summary-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%)">
            <div class="summary-icon">📉</div>
            <h2>{min_value}</h2>
            <p>Min Value</p>
        </div>
        ''', unsafe_allow_html=True)

# ---------------- FUNCTION TO CREATE PLOTLY BAR CHART ----------------
def create_bar_chart(data, x, y, horizontal=False, height=400):
    fig = px.bar(
        data,
        x=x if not horizontal else y,
        y=y if not horizontal else x,
        orientation='h' if horizontal else 'v',
        color_discrete_sequence=['#660094'],
        text=y
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    return fig

# ---------------- FUNCTION TO GET DATA FOR SUMMARY CARDS ----------------
def get_summary_data(active_tab, tab2_country=[], tab2_alert_type=[], tab2_alert_impact=[]):
    data = filtered_global.copy()
    if active_tab == "Tab 2":
        data = data[
            (data["alert-country"].isin(tab2_country)) &
            (data["alert-type"].isin(tab2_alert_type)) &
            (data["alert-impact"].isin(tab2_alert_impact))
        ]
    return data

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others", "Visualization map"])

# ---------------- TAB 1 ----------------
with tab1:
    active_tab = "Tab 1"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📌 Overview")
    a1 = summary_data.groupby("alert-impact").size().reset_index(name="count")
    a2 = summary_data.groupby("alert-type").size().reset_index(name="count")
    a3 = summary_data.groupby("alert-country").size().reset_index(name="count")
    a4 = summary_data.groupby("alert-country").size().reset_index(name="count")

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(a1, x="alert-impact", y="count", horizontal=True), use_container_width=True, key="tab1_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(a2, x="alert-type", y="count", horizontal=True), use_container_width=True, key="tab1_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(a3, x="alert-country", y="count"), use_container_width=True, key="tab1_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(a4, x="alert-country", y="count"), use_container_width=True, key="tab1_chart4")

# ---------------- TAB 2 ----------------
with tab2:
    active_tab = "Tab 2"
    col1, col2, col3 = st.columns(3)
    with col1:
        tab2_country_filter = st.multiselect("alert-country", data["alert-country"].unique(),
                                              default=data["alert-country"].unique())
    with col2:
        tab2_type_filter = st.multiselect("alert-type (Tab 2)", data["alert-type"].unique(),
                                            default=data["alert-type"].unique())
    with col3:
        tab2_impact_filter = st.multiselect("alert-type (Tab 2)", data["alert-impact"].unique(),
                                             default=data["alert-impact"].unique())

    summary_data = get_summary_data(active_tab, tab2_country_filter, tab2_type_filter, tab2_impact_filter)
    render_summary_cards(summary_data)
    
    # Example: filter by 'alert_country' before counting
    filtered_summary2 = summary_data[summary_data['alert-impact'] == "Negative"]  

    st.header("📊 Negative Events Analysis")
    v1 = filtered_summary2.groupby("alert-country").size().reset_index(name="count")
    v2 = filtered_summary2.groupby("alert-type").size().reset_index(name="count")
    v3 = filtered_summary2.groupby("alert-country").size().reset_index(name="count")
    v4 = filtered_summary2.groupby("alert-country").size().reset_index(name="count")

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(v1, x="alert-country", y="count", horizontal=True), use_container_width=True, key="tab2_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(v2, x="alert-type", y="count", horizontal=True), use_container_width=True, key="tab2_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(v3, x="alert-country", y="count"), use_container_width=True, key="tab2_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(v4, x="alert-country", y="count"), use_container_width=True, key="tab2_chart4")

# ---------------- TAB 3 ----------------
with tab3:
    active_tab = "Tab 3"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📈 Positive Events Analysis")
    b1 = summary_data.groupby("alert-country").size().reset_index(name="count")
    b2 = summary_data.groupby("alert-type").size().reset_index(name="count")
    b3 = summary_data.groupby("alert-country").size().reset_index(name="count")
    b4 = summary_data.groupby("alert-country").size().reset_index(name="count")

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(b3, x="alert-country", y="count", horizontal=True), use_container_width=True, key="tab3_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(b4, x="alert-country", y="count", horizontal=True), use_container_width=True, key="tab3_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(b1, x="alert-country", y="count"), use_container_width=True, key="tab3_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(b2, x="alert-type", y="count"), use_container_width=True, key="tab3_chart4")

# ---------------- TAB 4 ----------------
with tab4:
    active_tab = "Tab 4"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📌 Others Analysis")
    d1 = summary_data.groupby("alert-country").size().reset_index(name="count")
    d2 = summary_data.groupby("alert-type").size().reset_index(name="count")
    d3 = summary_data.groupby("alert-country").size().reset_index(name="count")
    d4 = summary_data.groupby("alert-country").size().reset_index(name="count")

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(d1, x="alert-country", y="count", horizontal=True), use_container_width=True, key="tab4_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(d2, x="alert-type", y="count", horizontal=True), use_container_width=True, key="tab4_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(d3, x="alert-country", y="count"), use_container_width=True, key="tab4_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(d4, x="alert-country", y="count"), use_container_width=True, key="tab4_chart4")

# ---------------- TAB 5 ----------------
with tab5:
    active_tab = "Tab 5"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)
    
   # ---- THEME TOGGLE ----
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    bg_map = "rgba(10,10,30,1)"
    border = 0.6
    glow = 2
else:
    bg_map = "rgba(245,245,255,1)"
    border = 0.4
    glow = 0.8

# ---- PLOT CHOROPLETH MAP ----
df_map = summary_data.groupby("alert-country").size().reset_index(name="Count")

fig = px.choropleth(
    df_map,
    geojson=countries_gj,
    locations="alert-country",
    featureidkey="properties.name",  # match key based on your dataset
    color="Count",
    projection="natural earth",
    hover_name="alert-country"
)

# ---- MAP VISUAL ENHANCEMENT ----
fig.update_geos(
    showframe=False,
    showland=True,
    landcolor=None,
    showcountries=True,
    countrycolor=None,
    backgroundcolor=None,
    bgcolor=bg_map,
)

fig.update_traces(marker_line_width=border)

# Glow effect using border scale
fig.update_layout(
    geo=dict(
        showland=True,
        landcolor=None,
        countrywidth=border,
        lakecolor=None
    )
)

# ---- ADD COUNTRY COUNT LABELS ----
label_fig = px.scatter_geo(
    df_map,
    locations="alert-country",
    locationmode="country names",
    text="Count"
)
label_fig.update_traces(mode="text", textfont=dict(size=14))

for trace in label_fig.data:
    fig.add_trace(trace)

# ---- CLICK DRILL-DOWN INTERACTION ----
fig.update_layout(clickmode="event+select")

def country_click(select_event):
    if select_event and "points" in select_event:
        clicked_country = select_event["points"][0]["location"]
        st.session_state.alert_country = clicked_country
        st.sidebar.success(f"📌 Selected: {clicked_country}")

st.plotly_chart(fig, use_container_width=True, key="interactive_geojson_map", on_select=country_click)

# ---- COUNTRY DRILL-DOWN VIEW ----
if st.session_state.alert_country:
    st.subheader(f"📊 Details for {st.session_state.alert_country}")
    st.write(country_counts[st.session_state.alert_country], "total alerts")
    st.dataframe(filtered[filtered["alert-country"] == st.session_state.alert_country][[
        "alert-country","alert-type"
    ]])
# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align: center; color: gray;'>
    © 2025 EU SEE Dashboard. All rights reserved.
</div>
""", unsafe_allow_html=True)
