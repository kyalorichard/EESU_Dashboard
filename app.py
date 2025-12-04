import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path  
import plotly.graph_objects as go

st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

# ----------- LOAD MASTER COUNTRY ISO MAP -----------
with open(Path.cwd() / "data" / "countries_metadata.json", encoding="utf-8") as f:
    country_meta = json.load(f)
    
# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=0)  # refresh cache every hour
def load_data():
    csv_file = Path.cwd() / "data" / "raw_data.csv"
    if not csv_file.exists():
        st.error(f"CSV file not found: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    
    # Clean country names
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    # Remove unwanted placeholder countries
    df = df[df['alert-country'] != "Jose"]
    
    # Filter out rows with blank or missing alert-impact
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    if 'alert-country' not in df.columns:
        st.warning("No 'alert-country' column found in CSV.")
        return df

    # Clean country names
    df['alert-country'] = df['alert-country'].astype(str).str.strip()

    # ---- LOAD COUNTRIES METADATA JSON ----
    json_file = Path.cwd() / "data" / "countries_metadata.json"
    if not json_file.exists():
        st.error(f"Countries metadata JSON not found: {json_file}")
        return df

    with open(json_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    # Map ISO3
    def get_iso3(country_name):
        return country_meta.get(country_name, {}).get("iso_alpha3", None)

    # Map continent
    def get_continent(country_name):
        return country_meta.get(country_name, {}).get("continent", "Unknown")

    df['iso_alpha3'] = df['alert-country'].apply(get_iso3)
    df['continent'] = df['alert-country'].apply(get_continent)

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
        
# ---------------- CONTINENT AND COUNTRY FILTER ----------------
# Assume 'data' is your loaded dataframe with 'alert-country' and 'continent' columns

# Get unique continents
continent_options = sorted(data['continent'].dropna().unique())
selected_continents = multiselect_with_all("Select Continent", continent_options, "selected_continents")

# Filter countries based on selected continent(s)
if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(
        data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique()
    )

# Country selection
selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")

# ---------------- ALERT TYPE FILTER ----------------
alert_type_options = sorted(data['alert-type'].dropna().unique())
selected_alert_types = multiselect_with_all("Select Alert Type", alert_type_options, "selected_alert_types")

# ---------------- ALERT IMPACT FILTER ----------------
alert_impact_options = sorted(data['alert-impact'].dropna().unique())
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
    padding: 10px;
    border-radius: 15px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 {
    font-size: 30px;
    margin: 5px 0;
}
.summary-card p {
    font-size: 16px;
    margin: 0;
    opacity: 0.9;
}
.summary-icon {
    font-size: 20px;
    margin-bottom: 5px;
}
/* Increase tabs name font size */
.stTabs [role="tab"] button {
    font-size: 20px;
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
        <div style="display:flex; flex-direction:column; gap:6px;">
            <div class="summary-card" style="background:#fc4a1a; padding:10px;">
                <div class="summary-icon" style="font-size:24px;">📉</div>
                <h2 style="font-size:26px;">{data["alert-impact"].count()}</h2>
                <p style="font-size:14px;">Negative Alerts</p>
            </div>
            
            <div class="summary-card" style="background:#660094; padding:10px;">
                <div class="summary-icon" style="font-size:24px;">✅</div>
                <h2 style="font-size:26px;">{data["alert-country"].count()}</h2>
                <p style="font-size:14px;">Total Countries</p>
            </div>
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
    #fig.update_traces(textposition='outside')
     # Show labels inside the bars
    fig.update_traces(
        textposition='inside',
        insidetextanchor='end', # anchor at the end (top of bar segment)
        textfont=dict(size=13, color='white', family="Arial Black")  # bold & readable
    )
    # Bold axis line
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        bargap=0.2,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    return fig

# ---------------- FUNCTION TO CREATE HORIZONTAL STACKED BAR CHART WITH TOTALS ----------------
def create_h_stacked_bar(data, y, x, color_col, horizontal=False, height=400):
        # Prepare the stacked bar
    categories = sorted(data[color_col].unique())
    color_sequence = ['#FFDB58', '#660094']  # Map to categories

    fig = go.Figure()

    for i, cat in enumerate(categories):
        df_cat = data[data[color_col] == cat]
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            #y=df_cat[y],
            #x=df_cat[x],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=color_sequence[i % len(color_sequence)],
            text=df_cat[x] if not horizontal else df_cat[x],
            textposition='inside',
            insidetextanchor='end', # anchor at the end (top of bar segment)
            textfont=dict(color='black' if color_sequence[i] == '#FFDB58' else 'white', size=13, family="Arial Black"),
            hovertemplate=f"%{{y}}<br>{cat}: %{{x}}<extra></extra>"
        ))
        # Bold axis line
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        


    fig.update_layout(
        barmode='stack',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=120, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        bargap=0.2,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False)

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
    a2 = summary_data.groupby(["alert-type", "alert-impact"]).size().reset_index(name='count')
    a3 = summary_data.groupby(["continent", "alert-impact"]).size().reset_index(name='count')
    a4 = summary_data.groupby(["alert-country", "alert-impact"]).size().reset_index(name='count')


  
    
    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(a1, x="alert-impact", y="count", horizontal=True), use_container_width=True, key="tab1_chart1")
    with r1c2: st.plotly_chart(create_h_stacked_bar( a2, y="alert-type", x="count", color_col="alert-impact", horizontal=True), use_container_width=True, key="tab1_chart2")
    with r2c1: st.plotly_chart(create_h_stacked_bar( a3, y="continent", x="count", color_col="alert-impact", horizontal=False), use_container_width=True, key="tab1_chart3")
    with r2c2: st.plotly_chart(create_h_stacked_bar( a4, y="alert-country", x="count", color_col="alert-impact", horizontal=True), use_container_width=True, key="tab1_chart4")

# --- Create stacked bar chart data ---
   

    

    

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
    
# Count alerts per country
df_map = summary_data.groupby("alert-country").size().reset_index(name="Count")

# ---- USE MAPBOX SATELLITE STYLE ----
px.set_mapbox_access_token("YOUR_MAPBOX_TOKEN")  # optional if using Mapbox tiles

fig = px.choropleth_mapbox(
    df_map,
    geojson=None,  # can use locationmode="country names" instead
    locations="alert-country",
    color="Count",
    hover_name="alert-country",
    mapbox_style="satellite-streets",  # ✅ satellite look
    center={"lat": 10, "lon": 0},       # center map globally
    zoom=1,
    opacity=0.6,
)

# ---- Add country counts as labels ----
import plotly.graph_objects as go

labels = go.Scattermapbox(
    lat=[0]*len(df_map),  # placeholder lat/lon; or you can add proper coords
    lon=[0]*len(df_map),
    mode="text",
    text=df_map["Count"],
    hoverinfo="skip",
    textfont=dict(size=14)
)
fig.add_trace(labels)

# ---- Display Map ----
st.plotly_chart(fig, use_container_width=True, key="satellite_map")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align: center; color: gray;'>
    © 2025 EU SEE Dashboard. All rights reserved.
</div>
""", unsafe_allow_html=True)
