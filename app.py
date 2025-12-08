import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style='margin-top:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<p style='margin-top:0; color:gray; font-size:16px;'></p>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)

# ---------------- MULTI-SELECT WITH SELECT ALL ----------------
def multiselect_with_all(label, options, session_key):
    try:
        options = sorted(options)
        default = st.session_state.get(session_key, ["Select All"])
        selected = st.sidebar.multiselect(label, ["Select All"] + options, default=default)
        if "Select All" in selected or not selected:
            st.session_state[session_key] = ["Select All"]
            return options
        st.session_state[session_key] = selected
        return selected
    except Exception as e:
        st.warning(f"Filter '{label}' failed: {e}")
        return options

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600)
def load_data():
    parquet_file = Path("data/output_final.parquet")
    if not parquet_file.exists():
        st.error(f"Data file missing: {parquet_file}")
        return pd.DataFrame()

    df = pd.read_parquet(parquet_file)

    # Clean data
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'] != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Load metadata
    meta_file = Path("data/countries_metadata.json")
    if not meta_file.exists():
        st.error(f"Metadata missing: {meta_file}")
        return df
    with open(meta_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    df['iso_alpha3'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3", None))
    df['continent'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))

    missing_countries = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing_countries) > 0:
        st.warning(f"Missing ISO codes: {', '.join(missing_countries)}")

    # Extract month/year
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')

    return df

data = load_data()

# ---------------- SIDEBAR GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

selected_continents = multiselect_with_all("Select Continent", data['continent'].dropna().unique(), "selected_continents")
if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique())

selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")
selected_alert_types = multiselect_with_all("Select Alert Type", data['alert-type'].dropna().unique(), "selected_alert_types")
selected_enabling_principle = multiselect_with_all("Select Enabling Principle", data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique(), "selected_enabling_principle")
selected_alert_impacts = multiselect_with_all("Select Alert Impact", data['alert-impact'].dropna().unique(), "selected_alert_impacts")
month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
selected_months = multiselect_with_all("Select Month", month_options, "selected_months")
year_options = sorted(data['year'].dropna().unique())
selected_years = multiselect_with_all("Select Year", year_options, "selected_years")

# ---------------- FILTER DATA ----------------
def contains_any(cell, selected_values):
    if pd.isna(cell):
        return False
    return any(val in str(cell) for val in selected_values)

filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# ---------------- SUMMARY CARDS ----------------
def render_summary_cards(df):
    total = len(df)
    neg = len(df[df['alert-impact'] == "Negative"])
    pos = len(df[df['alert-impact'] == "Positive"])
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"<div class='summary-card'><p>Total</p><h1>{total}</h1></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='summary-card'><p>Negative</p><h1>{neg}</h1></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='summary-card'><p>Positive</p><h1>{pos}</h1></div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='summary-card'><p>Max</p><h1>{total}</h1></div>", unsafe_allow_html=True)

st.markdown("""
<style>
.summary-card {
   background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
   color: white; padding: 5px; border-radius: 12px; text-align:center; margin:5px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- PLOTLY FUNCTIONS ----------------
def create_h_stacked_bar(df, y, color_col, horizontal=False):
    categories = sorted(df[color_col].unique())
    color_sequence = ['#FFDB58', '#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        sub = df[df[color_col] == cat]
        fig.add_trace(go.Bar(
            x=sub[y] if not horizontal else sub[y],
            y=sub[y] if not horizontal else sub[y],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=color_sequence[i % len(color_sequence)],
            text=sub[y],
            textposition='inside',
            insidetextanchor='end'
        ))
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(barmode='stack', height=350, margin=dict(l=20,r=20,t=20,b=20))
    return fig

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Map"])

# ---------- TAB 1 ----------
with tab1:
    summary_data = filtered_global
    render_summary_cards(summary_data)
    st.plotly_chart(create_h_stacked_bar(summary_data, y="alert-type", color_col="alert-impact", horizontal=True))

# ---------- TAB 2 ----------
with tab2:
    col1,col2,col3,col4 = st.columns(4)
    with col1: selected_actor_types = multiselect_with_all("Actor", filtered_global['Actor of repression'].dropna().unique(), "selected_actor_types")
    with col2: selected_subject_types = multiselect_with_all("Subject", filtered_global['Subject of repression'].dropna().unique(), "selected_subject_types")
    with col3: selected_mechanism_types = multiselect_with_all("Mechanism", filtered_global['Mechanism of repression'].dropna().unique(), "selected_mechanism_types")
    with col4: selected_event_types = multiselect_with_all("Event", filtered_global['Type of event'].dropna().unique(), "selected_event_types")

    df_tab2 = filtered_global[
        (filtered_global['Actor of repression'].isin(selected_actor_types)) &
        (filtered_global['Subject of repression'].isin(selected_subject_types)) &
        (filtered_global['Mechanism of repression'].isin(selected_mechanism_types)) &
        (filtered_global['Type of event'].isin(selected_event_types)) &
        (filtered_global['alert-impact']=="Negative")
    ]
    render_summary_cards(df_tab2)
    st.plotly_chart(create_h_stacked_bar(df_tab2, y="Actor of repression", color_col="alert-impact", horizontal=True))

# ---------- TAB 3 ----------
with tab3:
    df_tab3 = filtered_global[filtered_global['alert-impact']=="Positive"]
    render_summary_cards(df_tab3)
    st.plotly_chart(create_h_stacked_bar(df_tab3, y="alert-country", color_col="alert-impact", horizontal=True))

# ---------- TAB 4 ----------
with tab4:
    render_summary_cards(filtered_global)
    st.plotly_chart(create_h_stacked_bar(filtered_global, y="alert-type", color_col="alert-impact", horizontal=True))

# ---------- TAB 5 (MAP) ----------
with tab5:
    df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
    geo_file = Path("data/countriess.geojson")
    if geo_file.exists():
        with open(geo_file) as f: countries_gj = json.load(f)
        geo_countries = [feat['properties']['name'] for feat in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]
        fig = px.choropleth_mapbox(df_map, geojson=countries_gj, locations="alert-country",
                                   featureidkey="properties.name", color="count",
                                   color_continuous_scale="Greens", mapbox_style="open-street-map",
                                   zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
        fig.update_layout(margin=dict(l=0,r=0,t=1,b=0), height=500, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("GeoJSON file missing for map.")

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
