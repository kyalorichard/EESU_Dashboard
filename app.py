import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from streamlit_plotly_events import plotly_events

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<h1 style='margin-top:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<hr style='margin:5px 0'>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
/* Multi-select styling */
.css-1wa3eu0 .css-1d391kg {background-color: #660094 !important; color: white !important;}
.css-1wa3eu0 input {color: #660094 !important;}
.css-1gtu0r7 {background-color: #f2e6ff !important; color: #660094 !important;}
.css-1gtu0r7 div[role="option"]:hover {background-color: #b266ff !important; color: white !important;}
.css-1wa3eu0 {border-color: #660094 !important;}

/* Remove default spacing */
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}

/* Summary cards */
.summary-card {
   background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
   color: white; padding: 5px; border-radius: 12px; text-align: center; margin: 5px;
   box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 {font-size: 22px; margin: 5px 0;}
.summary-card p {font-size: 12px; margin: 0; opacity: 0.9;}
.stTabs [role="tab"] button {font-size: 20px; font-weight: bold;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600)
def load_data():
    parquet_file = Path("data/output_final.parquet")
    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_file)
    
    # Clean country names and remove placeholders
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'] != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Load metadata
    meta_file = Path("data/countries_metadata.json")
    if not meta_file.exists():
        st.error(f"Metadata JSON not found: {meta_file}")
        return df
    with open(meta_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    df['iso_alpha3'] = df['alert-country'].map(lambda c: country_meta.get(c, {}).get("iso_alpha3"))
    df['continent'] = df['alert-country'].map(lambda c: country_meta.get(c, {}).get("continent", "Unknown"))
    
    missing = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing)}")

    # Extract month & year
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    return df

data = load_data()

# ---------------- GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

def multiselect_with_all(label, options, key):
    selected = st.sidebar.multiselect(
        label, ["Select All"] + list(options), default=st.session_state.get(key, ["Select All"])
    )
    if "Select All" in selected:
        st.session_state[key] = ["Select All"]
        return list(options)
    st.session_state[key] = selected
    return selected

# Continent & Country filters
continent_options = sorted(data['continent'].dropna().unique())
selected_continents = multiselect_with_all("Select Continent", continent_options, "selected_continents")
if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(data[data['continent'].isin(selected_continents)]['alert-country'].unique())
selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")

# Alert Type, Enabling Principle, Alert Impact
alert_type_options = sorted(data['alert-type'].dropna().unique())
selected_alert_types = multiselect_with_all("Select Alert Type", alert_type_options, "selected_alert_types")

enabling_options = sorted(data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique())
selected_enabling = multiselect_with_all("Select Enabling Principle", enabling_options, "selected_enabling")

alert_impact_options = sorted(data['alert-impact'].dropna().unique())
selected_alert_impact = multiselect_with_all("Select Alert Impact", alert_impact_options, "selected_alert_impact")

# Month & Year
month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
selected_months = multiselect_with_all("Select Month", month_options, "selected_months")

year_options = sorted(data['year'].dropna().unique())
selected_years = multiselect_with_all("Select Year", year_options, "selected_years")

# Reset Filters
if st.sidebar.button("🔄 Reset Filters"):
    keys = ["selected_continents","selected_countries","selected_alert_types","selected_enabling",
            "selected_alert_impact","selected_months","selected_years"]
    for k in keys: st.session_state[k] = ["Select All"]

# ---------------- FILTER DATA ----------------
def contains_any(cell, selected):
    if pd.isna(cell): return False
    cell = str(cell)
    return any(sel in cell for sel in selected)

filtered = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling))) &
    (data['alert-impact'].isin(selected_alert_impact)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# ---------------- SUMMARY CARDS ----------------
def render_summary_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    total = df.shape[0]
    neg = df[df['alert-impact']=="Negative"].shape[0]
    pos = df[df['alert-impact']=="Positive"].shape[0]
    max_val = df['alert-impact'].count()
    
    col1.markdown(f'<div class="summary-card"><p>Total Alerts</p><h1>{total}</h1></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="summary-card"><p>Negative Alerts</p><h1>{neg}</h1></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="summary-card"><p>Positive Alerts</p><h1>{pos}</h1></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="summary-card"><p>Max Value</p><h1>{max_val}</h1></div>', unsafe_allow_html=True)

# ---------------- PLOTLY CHART FUNCTIONS ----------------
def create_bar_chart(df, x, y, horizontal=False, height=350):
    fig = px.bar(df, x=x if not horizontal else y, y=y if not horizontal else x,
                 orientation='h' if horizontal else 'v', color_discrete_sequence=['#660094'],
                 text=y)
    fig.update_traces(textposition='inside', insidetextanchor='end',
                      textfont=dict(size=13,color='white',family="Arial Black"))
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=20,b=20), uniformtext_minsize=12, uniformtext_mode='hide')
    fig.update_xaxes(showgrid=True, gridcolor='lightgray'); fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    return fig

def wrap_label(label, words_per_line=4):
    words = label.split()
    return "<br>".join([" ".join(words[i:i+words_per_line]) for i in range(0,len(words),words_per_line)])

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others", "Visualization map"])

# --------------- TAB 1 ----------------
with tab1:
    render_summary_cards(filtered)
    # Group data
    a1 = filtered.groupby(['alert-type','alert-impact']).size().reset_index(name='count')
    df_clean = filtered.assign(**{"enabling-principle": filtered["enabling-principle"].astype(str).str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().apply(wrap_label)
    a2 = df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count')
    a3 = filtered.groupby(["continent","alert-impact"]).size().reset_index(name='count')
    a4 = filtered.groupby(["alert-country","alert-impact"]).size().reset_index(name='count')

    r1c1,r1c2=st.columns(2); r2c1,r2c2=st.columns(2)
    r1c1.plotly_chart(create_bar_chart(a1,"alert-type","count",horizontal=True), use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(a2,"enabling-principle","count",horizontal=True), use_container_width=True)
    r2c1.plotly_chart(create_bar_chart(a3,"continent","count"), use_container_width=True)
    r2c2.plotly_chart(create_bar_chart(a4,"alert-country","count"), use_container_width=True)

# --------------- TAB 2 (Negative Events) ----------------
with tab2:
    st.subheader("Tab 2 Filters")
    col1,col2,col3,col4=st.columns(4)
    
    actor_opts = sorted(filtered['Actor of repression'].dropna().unique())
    subject_opts = sorted(filtered['Subject of repression'].dropna().unique())
    mech_opts = sorted(filtered['Mechanism of repression'].dropna().unique())
    event_opts = sorted(filtered['Type of event'].dropna().unique())
    
    selected_actor = col1.multiselect("Actor Type", ["Select All"]+list(actor_opts), default=["Select All"])
    if "Select All" in selected_actor: selected_actor=actor_opts
    
    selected_subject = col2.multiselect("Subject Type", ["Select All"]+list(subject_opts), default=["Select All"])
    if "Select All" in selected_subject: selected_subject=subject_opts
    
    selected_mech = col3.multiselect("Mechanism Type", ["Select All"]+list(mech_opts), default=["Select All"])
    if "Select All" in selected_mech: selected_mech=mech_opts
    
    selected_event = col4.multiselect("Event Type", ["Select All"]+list(event_opts), default=["Select All"])
    if "Select All" in selected_event: selected_event=event_opts
    
    # Filter data
    tab2_data = filtered[
        (filtered['Actor of repression'].isin(selected_actor)) &
        (filtered['Subject of repression'].isin(selected_subject)) &
        (filtered['Mechanism of repression'].isin(selected_mech)) &
        (filtered['Type of event'].isin(selected_event)) &
        (filtered['alert-impact']=="Negative")
    ]
    
    render_summary_cards(tab2_data)
    
    # Charts
    t1 = tab2_data.groupby("Actor of repression").size().reset_index(name="count")
    t2 = tab2_data.groupby("Subject of repression").size().reset_index(name="count")
    t3 = tab2_data.groupby("Mechanism of repression").size().reset_index(name="count")
    t4 = tab2_data.groupby("Type of event").size().reset_index(name="count")
    
    c1,c2=st.columns(2); c3,c4=st.columns(2)
    c1.plotly_chart(create_bar_chart(t1,"Actor of repression","count",horizontal=True), use_container_width=True)
    c2.plotly_chart(create_bar_chart(t2,"Subject of repression","count",horizontal=True), use_container_width=True)
    c3.plotly_chart(create_bar_chart(t3,"Mechanism of repression","count",horizontal=True), use_container_width=True)
    c4.plotly_chart(create_bar_chart(t4,"Type of event","count",horizontal=True), use_container_width=True)

# --------------- TAB 3 ----------------
with tab3:
    pos_data = filtered[filtered['alert-impact']=="Positive"]
    render_summary_cards(pos_data)

# --------------- TAB 4 ----------------
with tab4:
    render_summary_cards(filtered)

# --------------- TAB 5 (Map) ----------------
with tab5:
    df_map = filtered.groupby("alert-country").size().reset_index(name="count")
    geo_file = Path("data/countriess.geojson")
    if geo_file.exists():
        with open(geo_file) as f: countries_gj = json.load(f)
        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]
        fig = px.choropleth_mapbox(df_map, geojson=countries_gj, locations="alert-country",
                                   featureidkey="properties.name", color="count",
                                   hover_name="alert-country", hover_data={"count":True,"alert-country":False},
                                   color_continuous_scale="Greens",
                                   mapbox_style="open-street-map", zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
        fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0}, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align: center; color: gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
