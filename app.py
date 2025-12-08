import streamlit as st
import pandas as pd
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
<hr style='margin:5px 0'>
""", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Multi-select */
.css-1wa3eu0 .css-1d391kg {background-color: #660094 !important; color: white !important;}
.css-1wa3eu0 input {color: #660094 !important;}
.css-1gtu0r7 {background-color: #f2e6ff !important; color: #660094 !important;}
.css-1gtu0r7 div[role="option"]:hover {background-color: #b266ff !important; color: white !important;}
.css-1wa3eu0 {border-color: #660094 !important;}

/* Top spacing removal */
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}

/* Summary cards & tabs */
.summary-card {
   background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
   color: white; padding: 5px; border-radius: 12px; text-align: center; margin: 5px;
   box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 {font-size:22px;margin:5px 0;}
.summary-card p {font-size:12px;margin:0;opacity:0.9;}
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

    # Clean
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'] != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # Load country metadata
    meta_file = Path.cwd() / "data" / "countries_metadata.json"
    if not meta_file.exists():
        st.error("Countries metadata JSON not found")
        return df
    with open(meta_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    df['iso_alpha3'] = df['alert-country'].apply(lambda c: country_meta.get(c, {}).get("iso_alpha3"))
    df['continent'] = df['alert-country'].apply(lambda c: country_meta.get(c, {}).get("continent", "Unknown"))

    missing = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing)}")

    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')

    return df

data = load_data()

# ---------------- HELPER FUNCTIONS ----------------
def multiselect_with_all(label, options, session_key):
    selected = st.sidebar.multiselect(
        label, options=["Select All"] + list(options),
        default=st.session_state.get(session_key, ["Select All"])
    )
    if "Select All" in selected:
        st.session_state[session_key] = ["Select All"]
        return list(options)
    else:
        st.session_state[session_key] = selected
        return selected

def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    return "<br>".join([" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)])

def contains_any(cell_value, selected_values):
    if pd.isna(cell_value): return False
    return any(sel in str(cell_value) for sel in selected_values)

def render_summary_cards(df):
    total = df.shape[0]
    neg = df[df['alert-impact']=="Negative"].shape[0]
    pos = df[df['alert-impact']=="Positive"].shape[0]
    max_val = df['alert-impact'].count()
    min_val = df['alert-country'].count()
    cols = st.columns(4)
    for col, title, value in zip(cols, ["Total Value","Negative Alerts","Positive Alerts","Max Value"], [total,neg,pos,max_val]):
        col.markdown(f'<div class="summary-card"><p>{title}</p><h1>{value}</h1></div>', unsafe_allow_html=True)

def create_bar_chart(df, x, y, horizontal=False, height=350):
    fig = px.bar(df, x=x if not horizontal else y, y=y if not horizontal else x,
                 orientation='h' if horizontal else 'v',
                 color_discrete_sequence=['#660094'],
                 text=y)
    fig.update_traces(textposition='inside', insidetextanchor='end',
                      textfont=dict(size=13, color='white', family="Arial Black"))
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=20,b=20), xaxis_title=None, yaxis_title=None, uniformtext_minsize=12, uniformtext_mode='hide', bargap=0.2)
    return fig

# ---------------- SIDEBAR GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

# Reset all filters function
def reset_all_filters():
    keys = [
        "selected_continents", "selected_countries", "selected_alert_types",
        "selected_enabling_principle", "selected_alert_impacts",
        "selected_months", "selected_years",
        "selected_actor_types", "selected_subject_types",
        "selected_mechanism_types", "selected_event_types"
    ]
    for key in keys: st.session_state[key] = ["Select All"]

st.sidebar.button("🔄 Reset All Filters", on_click=reset_all_filters)

# Global multiselect filters
selected_continents = multiselect_with_all("Select Continent", sorted(data['continent'].dropna().unique()), "selected_continents")
if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique())
selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")
selected_alert_types = multiselect_with_all("Select Alert Type", sorted(data['alert-type'].dropna().unique()), "selected_alert_types")
enabling_principle_options = sorted(data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique())
selected_enabling_principle = multiselect_with_all("Select Enabling Principle", enabling_principle_options, "selected_enabling_principle")
selected_alert_impacts = multiselect_with_all("Select Alert Impact", sorted(data['alert-impact'].dropna().unique()), "selected_alert_impacts")
month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
selected_months = multiselect_with_all("Select Month", month_options, "selected_months")
year_options = sorted(data['year'].dropna().unique())
selected_years = multiselect_with_all("Select Year", year_options, "selected_years")

# ---------------- FILTER DATA ----------------
filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others", "Visualization map"])

with tab1:
    render_summary_cards(filtered_global)
    st.header("Distribution of Positive and Negative Events")
    df1 = filtered_global.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
    st.plotly_chart(create_bar_chart(df1, x="alert-type", y="count", horizontal=True), use_container_width=True)

with tab2:
    # Tab2 filters (actor, subject, mechanism, event)
    actor_types = multiselect_with_all("Select Actor Type", sorted(filtered_global['Actor of repression'].dropna().unique()), "selected_actor_types")
    subject_types = multiselect_with_all("Select Subject Type", sorted(filtered_global['Subject of repression'].dropna().unique()), "selected_subject_types")
    mechanism_types = multiselect_with_all("Select Mechanism Type", sorted(filtered_global['Mechanism of repression'].dropna().unique()), "selected_mechanism_types")
    event_types = multiselect_with_all("Select Event Type", sorted(filtered_global['Type of event'].dropna().unique()), "selected_event_types")

    tab2_data = filtered_global[
        (filtered_global['Actor of repression'].isin(actor_types)) &
        (filtered_global['Subject of repression'].isin(subject_types)) &
        (filtered_global['Mechanism of repression'].isin(mechanism_types)) &
        (filtered_global['Type of event'].isin(event_types)) &
        (filtered_global['alert-impact'] == "Negative")
    ]
    render_summary_cards(tab2_data)
    st.plotly_chart(create_bar_chart(tab2_data.groupby("Actor of repression").size().reset_index(name="count"), "Actor of repression", "count", horizontal=True), use_container_width=True)

with tab3:
    tab3_data = filtered_global[filtered_global['alert-impact']=="Positive"]
    render_summary_cards(tab3_data)
    st.plotly_chart(create_bar_chart(tab3_data.groupby("alert-country").size().reset_index(name="count"), "alert-country", "count", horizontal=True), use_container_width=True)

with tab4:
    render_summary_cards(filtered_global)
    st.plotly_chart(create_bar_chart(filtered_global.groupby("alert-type").size().reset_index(name="count"), "alert-type", "count", horizontal=True), use_container_width=True)

with tab5:
    geojson_file = Path.cwd() / "data" / "countriess.geojson"
    if geojson_file.exists():
        with open(geojson_file) as f:
            countries_gj = json.load(f)
        df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]
        fig = px.choropleth_mapbox(df_map, geojson=countries_gj, locations="alert-country",
                                   featureidkey="properties.name", color="count",
                                   hover_name="alert-country", mapbox_style="open-street-map",
                                   color_continuous_scale="Greens", zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
        fig.update_layout(height=500, margin={"r":0,"t":1,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
