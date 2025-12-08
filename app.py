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

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Multi-select */
.css-1wa3eu0 .css-1d391kg { background-color: #660094 !important; color: white !important; }
.css-1wa3eu0 input { color: #660094 !important; }
.css-1gtu0r7 { background-color: #f2e6ff !important; color: #660094 !important; }
.css-1gtu0r7 div[role="option"]:hover { background-color: #b266ff !important; color: white !important; }
.css-1wa3eu0 { border-color: #660094 !important; }
/* Remove top spacing */
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
/* Summary cards */
.summary-card {
   background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
   color: white; padding: 5px; border-radius: 12px; text-align: center; margin: 5px;
   box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 { font-size: 22px; margin: 5px 0; }
.summary-card p { font-size: 12px; margin: 0; opacity: 0.9; }
.stTabs [role="tab"] button { font-size: 20px; font-weight: bold; }
footer { visibility: hidden; }
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

    # Load country metadata
    meta_file = Path.cwd() / "data" / "countries_metadata.json"
    if not meta_file.exists():
        st.error(f"Countries metadata JSON not found: {meta_file}")
        return df
    with open(meta_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    def get_iso3(country_name): return country_meta.get(country_name, {}).get("iso_alpha3", None)
    def get_continent(country_name): return country_meta.get(country_name, {}).get("continent", "Unknown")

    df['iso_alpha3'] = df['alert-country'].apply(get_iso3)
    df['continent'] = df['alert-country'].apply(get_continent)
    missing = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing)}")

    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    return df

data = load_data()

# ---------------- GLOBAL SIDEBAR FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

def safe_multiselect(label, options, session_key):
    """Multiselect with 'Select All' and error catching."""
    options = ["Select All"] + sorted(options)
    default = st.session_state.get(session_key, ["Select All"])
    try:
        selected = st.sidebar.multiselect(label, options, default=default, key=session_key)
    except Exception:
        selected = ["Select All"]
    if "Select All" in selected:
        st.session_state[session_key] = ["Select All"]
        return options[1:]
    st.session_state[session_key] = selected
    return selected

# Global filters
selected_continents = safe_multiselect("Select Continent", data['continent'].dropna().unique(), "selected_continents")
if "Select All" in selected_continents:
    country_options = data['alert-country'].dropna().unique()
else:
    country_options = data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique()
selected_countries = safe_multiselect("Select Country", country_options, "selected_countries")
selected_alert_types = safe_multiselect("Select Alert Type", data['alert-type'].dropna().unique(), "selected_alert_types")
selected_enabling_principle = safe_multiselect(
    "Select Enabling Principle",
    data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique(),
    "selected_enabling_principle"
)
selected_alert_impacts = safe_multiselect("Select Alert Impact", data['alert-impact'].dropna().unique(), "selected_alert_impacts")
month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
selected_months = safe_multiselect("Select Month", month_options, "selected_months")
year_options = sorted(data['year'].dropna().unique())
selected_years = safe_multiselect("Select Year", year_options, "selected_years")

# Reset button
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_continents","selected_countries","selected_alert_types",
                "selected_enabling_principle","selected_alert_impacts",
                "selected_months","selected_years"]:
        st.session_state[key] = ["Select All"]

# ---------------- FILTER DATA ----------------
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

# ---------------- CHART FUNCTIONS ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    return "<br>".join([" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)])

def create_h_stacked_bar(df, y, x="count", color_col="alert-impact", horizontal=False, height=350):
    if color_col not in df.columns:
        df[color_col] = "Unknown"
    categories = sorted(df[color_col].unique())
    color_sequence = ['#FFDB58', '#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat]
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=color_sequence[i % len(color_sequence)],
            text=df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if color_sequence[i] == '#FFDB58' else 'white', size=13, family="Arial Black"),
            hovertemplate=f"%{{y}}<br>{cat}: %{{x}}<extra></extra>"
        ))
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(barmode='stack', height=height, margin=dict(l=120,r=20,t=20,b=20),
                      uniformtext_minsize=12, uniformtext_mode='hide', bargap=0.2)
    return fig

def create_bar_chart(df, x, y="count", horizontal=False, height=350):
    fig = px.bar(df, x=x if not horizontal else y, y=y if not horizontal else x,
                 orientation='h' if horizontal else 'v', color_discrete_sequence=['#660094'],
                 text=y)
    fig.update_traces(textposition='inside', insidetextanchor='end',
                      textfont=dict(size=13, color='white', family="Arial Black"))
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=20,b=20),
                      uniformtext_minsize=12, uniformtext_mode='hide', bargap=0.2)
    return fig

def render_summary_cards(df):
    total = df.shape[0]
    neg = df[df['alert-impact']=="Negative"].shape[0]
    pos = df[df['alert-impact']=="Positive"].shape[0]
    col1,col2,col3,col4 = st.columns(4)
    with col1: st.markdown(f'<div class="summary-card"><p>Total Alerts</p><h1>{total}</h1></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="summary-card"><p>Negative Alerts</p><h1>{neg}</h1></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="summary-card"><p>Positive Alerts</p><h1>{pos}</h1></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="summary-card"><p>Max Value</p><h1>{total}</h1></div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Visualization map"])

# ---------------- TAB 1 ----------------
with tab1:
    summary_data = filtered_global.copy()
    render_summary_cards(summary_data)
    # Example stacked bar
    df1 = summary_data.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
    st.plotly_chart(create_h_stacked_bar(df1,"alert-type","count","alert-impact",horizontal=True), use_container_width=True, key="tab1_chart1")

# ---------------- TAB 2 ----------------
with tab2:
    summary_data = filtered_global[filtered_global['alert-impact']=="Negative"].copy()
    render_summary_cards(summary_data)
    
    # Tab 2 filters in one row
    col1,col2,col3,col4 = st.columns(4)
    actor_options = summary_data['Actor of repression'].dropna().unique()
    subject_options = summary_data['Subject of repression'].dropna().unique()
    mech_options = summary_data['Mechanism of repression'].dropna().unique()
    event_options = summary_data['Type of event'].dropna().unique()

    selected_actor = col1.multiselect("Actor Type", ["Select All"] + list(actor_options), default=["Select All"], key="tab2_actor")
    selected_subject = col2.multiselect("Subject Type", ["Select All"] + list(subject_options), default=["Select All"], key="tab2_subject")
    selected_mech = col3.multiselect("Mechanism Type", ["Select All"] + list(mech_options), default=["Select All"], key="tab2_mech")
    selected_event = col4.multiselect("Event Type", ["Select All"] + list(event_options), default=["Select All"], key="tab2_event")

    # Apply select all
    def apply_select_all(selected, options): return options if "Select All" in selected else selected
    selected_actor = apply_select_all(selected_actor, actor_options)
    selected_subject = apply_select_all(selected_subject, subject_options)
    selected_mech = apply_select_all(selected_mech, mech_options)
    selected_event = apply_select_all(selected_event, event_options)

    # Filter
    tab2_filtered = summary_data[
        (summary_data['Actor of repression'].isin(selected_actor)) &
        (summary_data['Subject of repression'].isin(selected_subject)) &
        (summary_data['Mechanism of repression'].isin(selected_mech)) &
        (summary_data['Type of event'].isin(selected_event))
    ]

    df2 = tab2_filtered.groupby("Actor of repression").size().reset_index(name='count')
    st.plotly_chart(create_bar_chart(df2,"Actor of repression","count",horizontal=True), use_container_width=True, key="tab2_chart1")

# ---------------- TAB 3 ----------------
with tab3:
    summary_data = filtered_global[filtered_global['alert-impact']=="Positive"].copy()
    render_summary_cards(summary_data)
    df3 = summary_data.groupby("alert-type").size().reset_index(name='count')
    st.plotly_chart(create_bar_chart(df3,"alert-type","count",horizontal=True), use_container_width=True, key="tab3_chart1")

# ---------------- TAB 4 ----------------
with tab4:
    summary_data = filtered_global.copy()
    render_summary_cards(summary_data)
    df4 = summary_data.groupby("alert-type").size().reset_index(name='count')
    st.plotly_chart(create_bar_chart(df4,"alert-type","count",horizontal=True), use_container_width=True, key="tab4_chart1")

# ---------------- TAB 5 ----------------
geojson_file = Path.cwd() / "data" / "countriess.geojson"
if geojson_file.exists():
    with open(geojson_file) as f:
        countries_gj = json.load(f)
else:
    countries_gj = None

with tab5:
    summary_data = filtered_global.copy()
    render_summary_cards(summary_data)
    if countries_gj:
        df_map = summary_data.groupby("alert-country").size().reset_index(name="count")
        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]
        fig = px.choropleth_mapbox(df_map, geojson=countries_gj,
                                   locations="alert-country", featureidkey="properties.name",
                                   color="count", hover_name="alert-country",
                                   color_continuous_scale="Greens", mapbox_style="open-street-map",
                                   zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
        fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0}, height=500,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align: center; color: gray;'>
    © 2025 EU SEE Dashboard. All rights reserved.
</div>
""", unsafe_allow_html=True)
