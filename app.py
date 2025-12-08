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
<p style='margin-top:0; color:gray; font-size:16px;'></p>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)

# ---------------- MULTI-SELECT CUSTOM CSS ----------------
st.markdown("""
<style>
/* Multi-select customization */
.css-1wa3eu0 .css-1d391kg {background-color: #660094 !important; color: white !important;}
.css-1wa3eu0 input {color: #660094 !important;}
.css-1gtu0r7 {background-color: #f2e6ff !important; color: #660094 !important;}
.css-1gtu0r7 div[role="option"]:hover {background-color: #b266ff !important; color: white !important;}
.css-1wa3eu0 {border-color: #660094 !important;}
/* Remove Streamlit default spacing */
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
/* Summary cards */
.summary-card {background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%); color: white; padding: 5px; border-radius: 12px; text-align:center; margin:5px; box-shadow:0 4px 8px rgba(0,0,0,0.2);}
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
    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_file)
    
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'] != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Load countries metadata
    json_file = Path.cwd() / "data" / "countries_metadata.json"
    if not json_file.exists():
        st.error(f"Countries metadata JSON not found: {json_file}")
        return df
    with open(json_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    def get_iso3(country_name):
        return country_meta.get(country_name, {}).get("iso_alpha3", None)
    def get_continent(country_name):
        return country_meta.get(country_name, {}).get("continent", "Unknown")

    df['iso_alpha3'] = df['alert-country'].apply(get_iso3)
    df['continent'] = df['alert-country'].apply(get_continent)

    missing = df.loc[df['iso_alpha3'].isna(),'alert-country'].unique()
    if len(missing) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing)}")

    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    else:
        st.warning("No 'creation_date' column found.")

    return df

data = load_data()

# ---------------- GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

def multiselect_with_all(label, options, session_key):
    try:
        if session_key not in st.session_state:
            st.session_state[session_key] = ["Select All"]
        selected = st.sidebar.multiselect(label, ["Select All"]+list(options), default=st.session_state[session_key])
        if "Select All" in selected:
            st.session_state[session_key] = ["Select All"]
            return list(options)
        else:
            st.session_state[session_key] = selected
            return selected
    except Exception as e:
        st.error(f"Error in filter '{label}': {e}")
        return list(options)

# Global filters
continent_options = sorted(data['continent'].dropna().unique())
selected_continents = multiselect_with_all("Select Continent", continent_options, "selected_continents")

if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique())
selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")

alert_type_options = sorted(data['alert-type'].dropna().unique())
selected_alert_types = multiselect_with_all("Select Alert Type", alert_type_options, "selected_alert_types")

enabling_principle_options = sorted(data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique())
selected_enabling_principle = multiselect_with_all("Select Enabling Principle", enabling_principle_options, "selected_enabling_principle")

alert_impact_options = sorted(data['alert-impact'].dropna().unique())
selected_alert_impacts = multiselect_with_all("Select Alert Impact", alert_impact_options, "selected_alert_impacts")

month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m,format='%B').month)
selected_months = multiselect_with_all("Select Month", month_options, "selected_months")

year_options = sorted(data['year'].dropna().unique())
selected_years = multiselect_with_all("Select Year", year_options, "selected_years")

# Reset filters button
if st.sidebar.button("🔄 Reset Filters"):
    keys = ["selected_continents","selected_countries","selected_alert_types","selected_enabling_principle",
            "selected_alert_impacts","selected_months","selected_years"]
    for k in keys: st.session_state[k]=["Select All"]

# Filter data globally
def contains_any(cell, selected_values):
    if pd.isna(cell): return False
    return any(sel in str(cell) for sel in selected_values)

filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# ---------------- HELPER FUNCTIONS ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    return "<br>".join([" ".join(words[i:i+words_per_line]) for i in range(0,len(words),words_per_line)])

def render_summary_cards(data):
    total = data.shape[0]
    neg = data[data["alert-impact"]=="Negative"].shape[0]
    pos = data[data["alert-impact"]=="Positive"].shape[0]
    max_val = data["alert-impact"].count()
    min_val = data["alert-country"].count()

    cols = st.columns(4)
    for col, name, val in zip(cols, ["Total Value","Negative Alerts","Positive Alerts","Max Value"], [total,neg,pos,max_val]):
        col.markdown(f'<div class="summary-card"><p>{name}</p><h1>{val}</h1></div>', unsafe_allow_html=True)

def create_h_stacked_bar(df, y, x, color_col, horizontal=False):
    categories = sorted(df[color_col].unique())
    color_sequence = ['#FFDB58','#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat]
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=color_sequence[i % len(color_sequence)],
            text=df_cat[x] if not horizontal else df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if color_sequence[i]=='#FFDB58' else 'white', size=13, family="Arial Black"),
        ))
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.update_layout(barmode='stack', margin=dict(l=120,r=20,t=20,b=20), height=350)
    return fig

# ---------------- LOAD GEOJSON ----------------
geojson_file = Path.cwd() / "data" / "countriess.geojson"
countries_gj = None
if geojson_file.exists():
    try:
        with open(geojson_file) as f:
            countries_gj = json.load(f)
    except json.JSONDecodeError:
        st.error("Invalid countries.geojson format")
else:
    st.error("countries.geojson missing")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Visualization Map"])

# ---------------- TAB 1 ----------------
with tab1:
    render_summary_cards(filtered_global)
    a1 = filtered_global.groupby(["alert-type","alert-impact"]).size().reset_index(name="count")
    df_clean = filtered_global.assign(**{"enabling-principle": filtered_global["enabling-principle"].astype(str).str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x,4))
    a2 = df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name="count")
    a3 = filtered_global.groupby(["continent","alert-impact"]).size().reset_index(name="count")
    a4 = filtered_global.groupby(["alert-country","alert-impact"]).size().reset_index(name="count")

    r1c1,r1c2 = st.columns(2,gap="large")
    r2c1,r2c2 = st.columns(2,gap="large")
    r1c1.plotly_chart(create_h_stacked_bar(a1,"alert-type","count","alert-impact",horizontal=True), use_container_width=True)
    r1c2.plotly_chart(create_h_stacked_bar(a2,"enabling-principle","count","alert-impact",horizontal=True), use_container_width=True)
    r2c1.plotly_chart(create_h_stacked_bar(a3,"continent","count","alert-impact"), use_container_width=True)
    r2c2.plotly_chart(create_h_stacked_bar(a4,"alert-country","count","alert-impact"), use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.markdown("### Negative Events Filters")
    col1,col2,col3,col4 = st.columns(4)
    actor_options = sorted(filtered_global['Actor of repression'].dropna().unique())
    subject_options = sorted(filtered_global['Subject of repression'].dropna().unique())
    mechanism_options = sorted(filtered_global['Mechanism of repression'].dropna().unique())
    event_options = sorted(filtered_global['Type of event'].dropna().unique())

    selected_actor = col1.multiselect("Actor Type", ["Select All"]+list(actor_options), default=["Select All"])
    selected_subject = col2.multiselect("Subject Type", ["Select All"]+list(subject_options), default=["Select All"])
    selected_mechanism = col3.multiselect("Mechanism Type", ["Select All"]+list(mechanism_options), default=["Select All"])
    selected_event = col4.multiselect("Event Type", ["Select All"]+list(event_options), default=["Select All"])

    # Filter tab2 data
    tab2_data = filtered_global[filtered_global['alert-impact']=="Negative"]
    if "Select All" not in selected_actor:
        tab2_data = tab2_data[tab2_data['Actor of repression'].isin(selected_actor)]
    if "Select All" not in selected_subject:
        tab2_data = tab2_data[tab2_data['Subject of repression'].isin(selected_subject)]
    if "Select All" not in selected_mechanism:
        tab2_data = tab2_data[tab2_data['Mechanism of repression'].isin(selected_mechanism)]
    if "Select All" not in selected_event:
        tab2_data = tab2_data[tab2_data['Type of event'].isin(selected_event)]

    render_summary_cards(tab2_data)

    df_clean = tab2_data.assign(**{"enabling-principle": tab2_data["enabling-principle"].astype(str).str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x,4))

    t1 = tab2_data.groupby("Actor of repression").size().reset_index(name="count")
    t2 = df_clean.groupby("enabling-principle").size().reset_index(name="count")
    t3 = tab2_data.groupby("continent").size().reset_index(name="count")
    t4 = tab2_data.groupby("alert-country").size().reset_index(name="count")
    t5 = tab2_data.groupby("alert-type").size().reset_index(name="count")
    t6 = tab2_data.groupby("month_name").size().reset_index(name="count")

    r1c1,r1c2,r1c3= st.columns(3,gap="large")
    r2c1,r2c2,r2c3 = st.columns(3,gap="large")
    r3c1,r3c2 = st.columns(2,gap="large")

    r1c1.plotly_chart(create_h_stacked_bar(t1,"Actor of repression","count","alert-impact",horizontal=True),use_container_width=True)
    r1c2.plotly_chart(create_h_stacked_bar(t2,"enabling-principle","count","alert-impact",horizontal=True),use_container_width=True)
    r1c3.plotly_chart(create_h_stacked_bar(t3,"continent","count","alert-impact"),use_container_width=True)
    r2c1.plotly_chart(create_h_stacked_bar(t4,"alert-country","count","alert-impact"),use_container_width=True)
    r2c2.plotly_chart(create_h_stacked_bar(t5,"alert-type","count","alert-impact"),use_container_width=True)
    r2c3.plotly_chart(create_h_stacked_bar(t6,"month_name","count","alert-impact"),use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown("### Positive Events")
    tab3_data = filtered_global[filtered_global['alert-impact']=="Positive"]
    render_summary_cards(tab3_data)
    if not tab3_data.empty:
        df_clean = tab3_data.assign(**{"enabling-principle": tab3_data["enabling-principle"].astype(str).str.split(",")}).explode("enabling-principle")
        df_clean["enabling-principle"]=df_clean["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x,4))
        p1 = tab3_data.groupby("alert-type").size().reset_index(name="count")
        p2 = df_clean.groupby("enabling-principle").size().reset_index(name="count")
        p3 = tab3_data.groupby("continent").size().reset_index(name="count")
        p4 = tab3_data.groupby("alert-country").size().reset_index(name="count")
        r1c1,r1c2= st.columns(2,gap="large")
        r2c1,r2c2= st.columns(2,gap="large")
        r1c1.plotly_chart(create_h_stacked_bar(p1,"alert-type","count","alert-impact",horizontal=True),use_container_width=True)
        r1c2.plotly_chart(create_h_stacked_bar(p2,"enabling-principle","count","alert-impact",horizontal=True),use_container_width=True)
        r2c1.plotly_chart(create_h_stacked_bar(p3,"continent","count","alert-impact"),use_container_width=True)
        r2c2.plotly_chart(create_h_stacked_bar(p4,"alert-country","count","alert-impact"),use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    st.markdown("### Others")
    tab4_data = filtered_global[~filtered_global['alert-impact'].isin(["Positive","Negative"])]
    render_summary_cards(tab4_data)
    if not tab4_data.empty:
        o1 = tab4_data.groupby("alert-type").size().reset_index(name="count")
        o2 = tab4_data.groupby("continent").size().reset_index(name="count")
        o3 = tab4_data.groupby("alert-country").size().reset_index(name="count")
        r1c1,r1c2,r1c3 = st.columns(3,gap="large")
        r1c1.plotly_chart(create_h_stacked_bar(o1,"alert-type","count","alert-impact",horizontal=True),use_container_width=True)
        r1c2.plotly_chart(create_h_stacked_bar(o2,"continent","count","alert-impact"),use_container_width=True)
        r1c3.plotly_chart(create_h_stacked_bar(o3,"alert-country","count","alert-impact"),use_container_width=True)

# ---------------- TAB 5 ----------------
with tab5:
    st.markdown("### Map Visualization")
    tab5_data = filtered_global.dropna(subset=['iso_alpha3'])
    if not tab5_data.empty:
        map_fig = px.choropleth(
            tab5_data,
            locations="iso_alpha3",
            color="alert-impact",
            color_discrete_sequence=['#FFDB58','#660094'],
            hover_name="alert-country"
        )
        map_fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            margin=dict(l=0,r=0,t=0,b=0),
            height=600
        )
        map_fig.update_xaxes(visible=False)
        map_fig.update_yaxes(visible=False)
        st.plotly_chart(map_fig,use_container_width=True)


# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align: center; color: gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
