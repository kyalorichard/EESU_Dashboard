import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path  
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="EU SEE Dashboard", layout="wide")


# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style='margin-top:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<p style='margin-top:0; color:gray; font-size:16px;'></p>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)  # tight separator

# ---------------- MULTI-SELECT CUSTOM CSS ----------------
st.markdown("""
<style>
/* Change multi-select background and text */
.css-1wa3eu0 .css-1d391kg {
    background-color: #660094 !important;  /* selected options */
    color: white !important;               /* selected text */
}

/* Placeholder text color */
.css-1wa3eu0 input {
    color: #660094 !important; 
}

/* Dropdown menu background */
.css-1gtu0r7 {
    background-color: #f2e6ff !important; 
    color: #660094 !important;
}

/* Hover effect on dropdown items */
.css-1gtu0r7 div[role="option"]:hover {
    background-color: #b266ff !important; 
    color: white !important;
}

/* Remove default border highlight */
.css-1wa3eu0 {
    border-color: #660094 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- REMOVE STREAMLIT DEFAULT TOP SPACING ----------------
st.markdown("""
<style>
    /* Remove Streamlit's default top padding */
    .css-18e3th9 {padding-top: 0rem;}
    /* Optional: reduce spacing around main container */
    .css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
</style>
""", unsafe_allow_html=True)


# ----------- LOAD MASTER COUNTRY ISO MAP -----------
with open(Path.cwd() / "data" / "countries_metadata.json", encoding="utf-8") as f:
    country_meta = json.load(f)

# ---------- ✅ Load Countries GeoJSON safely ----------
geojson_file = Path.cwd() / "data" / "countriess.geojson"

if not geojson_file.exists():
    st.error("❌ countries.geojson file missing inside /data folder")
    countries_gj = None
else:
    try:
        with open(geojson_file) as f:
            countries_gj = json.load(f)
    except json.JSONDecodeError:
        st.error("❌ Invalid countries.geojson format")
        countries_gj = None

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=0)  # refresh cache every hour
def load_data():
    parquet_file = Path.cwd() / "data" / "output_final.parquet"

    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_file)
    
    # Clean country names
    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    # Remove unwanted placeholder countries
    df = df[df['alert-country'] != "Jose"]
    
    # Filter out rows with blank or missing alert-impact
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    if 'alert-country' not in df.columns:
        st.warning("No 'alert-country' column found in the dataset.")
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

     # ---------------- EXTRACT MONTH AND YEAR ----------------
    if 'creation_date' in df.columns:
        # Convert to datetime
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        # Extract year and month
        df['year'] = df['creation_date'].dt.year
        #df['month'] = df['creation_date'].dt.month
        df['month_name'] = df['creation_date'].dt.strftime('%B')
        # Optional: create a combined string for easier dropdown
        #df['year_month'] = df['creation_date'].dt.to_period('M').astype(str)
    else:
        st.warning("No 'creation_date' column found in the dataset.")

    return df
    #return pd.read_csv(csv_file)

data = load_data()


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
        default=st.session_state.get(key, ["Select All"])
    )
    if "Select All" in selected:
        st.session_state[key] = ["Select All"]
        return list(options)
    else:
        st.session_state[key] = selected
        return selected
        
# ---------------- FILTER DATA BASED ON SELECTION (CONTAINS) --------------
def contains_any(cell_value, selected_values):
    if pd.isna(cell_value):
        return False
    cell_value = str(cell_value)
    return any(sel in cell_value for sel in selected_values)
        
# ---------------- CONTINENT AND COUNTRY FILTER ----------------
  
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

# ---------------- ENABLING PRINCIPLES FILTER ----------------
enabling_principle_options = sorted(data['enabling-principle'].dropna()
                            .str.split(",")
                            .explode()
                            .str.strip()
                            .unique()
                            .tolist())
selected_enablinge_principle = multiselect_with_all("Select Enabling Principle", enabling_principle_options, "selected_enablinge_principle")

# ---------------- ALERT IMPACT FILTER ----------------
alert_impact_options = sorted(data['alert-impact'].dropna().unique())
selected_alert_impacts = multiselect_with_all("Select Alert Impact", alert_impact_options, "selected_alert_impacts")

# ---------------- MONTH FILTER ----------------
# Get unique months in chronological order
month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
# Create a multi-select dropdown for months
selected_months = multiselect_with_all("Select Month", month_options, "selected_months")

# ---------------- YEAR FILTER ----------------
# Get unique years in ascending order
year_options = sorted(data['year'].dropna().unique())
# Create a multi-select dropdown for years
selected_years = multiselect_with_all("Select Year", year_options, "selected_years")



# Reset Filters button
if st.sidebar.button("🔄 Reset Filters") and not st.session_state.reset_triggered:
    st.session_state["selected_continents"] = ["Select All"]
    st.session_state["selected_countries"] = ["Select All"]
    st.session_state["selected_alert_types"] = ["Select All"]
    st.session_state["selected_enablinge_principle"] = ["Select All"]
    st.session_state["selected_alert_impacts"] = ["Select All"]
    st.session_state["selected_months"] = ["Select All"]
    st.session_state["selected_years"] = ["Select All"]
    
    # Mark that reset was triggered to avoid multiple reruns
    #st.session_state.reset_triggered = True
    #st.experimental_rerun()

# Clear the flag after rerun so the button works again
st.session_state.reset_triggered = False


# After rerun, clear the flag so next user click works
st.session_state.reset_triggered = False


# ---------------- FILTER DATA BASED ON SELECTION ----------------
filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply( lambda x: contains_any(x, selected_enablinge_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
    ]

# ---------------- CSS FOR SUMMARY CARDS & TABS ----------------
st.markdown("""
<style>
.summary-card {
   background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
    color: white;
    padding: 5px;
    border-radius: 12px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 {
    font-size: 22px;
    margin: 5px 0;
}
.summary-card p {
    font-size: 12px;
    margin: 0;
    opacity: 0.9;
}
.summary-icon {
    font-size: 14px;
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
    total_value = filtered_global["alert-country"].count()
    neg_alerts = filtered_global[filtered_global["alert-impact"] == "Negative"]["alert-impact"].count()
    pos_alerts = filtered_global[filtered_global["alert-impact"] == "Positive"]["alert-impact"].count()
    max_value = filtered_global["alert-impact"].count()
    min_value = filtered_global["alert-country"].count()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="summary-card">
            <p>Total Value</p>
            <h1>{total_value}</h1>
            
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="summary-card")">
             <p>Negative Alerts</p>
             <h1>{neg_alerts}</h1>
            
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="summary-card")">
            <p>Positive Alerts</p>
            <h1>{pos_alerts}</h1>
            
        </div>
        ''', unsafe_allow_html=True)

    with col4:
       st.markdown(f'''
        <div class="summary-card")">
            <p>Max Value</p>
            <h1>{max_value}</h1>
            
        </div>
        ''', unsafe_allow_html=True)
   
# ---------------- FUNCTION TO CREATE PLOTLY BAR CHART ----------------
def create_bar_chart(data, x, y, horizontal=False, height=350):
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
        #plot_bgcolor='white',
        #paper_bgcolor='lightgray',
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
def create_h_stacked_bar(data, y, x, color_col, horizontal=False, height=350):
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
        #plot_bgcolor='white',
        #paper_bgcolor='white',
        height=height,
        xaxis=dict(tickangle=90, automargin=True ),
        yaxis=dict(automargin=True ),
        margin=dict(l=120, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        bargap=0.2,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig
# ---------------- FUNCTION TO SHORTEN LONG SENTENCES ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)

# ---------------- FUNCTION TO GET DATA FOR SUMMARY CARDS ----------------
def get_summary_data(active_tab, selected_subject_filter=[], selected_mechanism_filter=[], selected_type_filter=[]):
    data = filtered_global.copy()
    if active_tab == "Tab 2":
        data = data[
            (data["actor of repression"].isin(selected_actor_filter)) &
            (data["Subject of repression"].isin(selected_subject_filter)) &
            (data["Mechanism of repression"].isin(selected_mechanism_filter)) &
            (data["Type of event"].isin(selected_type_filter))
        ]
        return data
        

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others", "Visualization map"])

# ---------------- TAB 1 ----------------
with tab1:
    active_tab = "Tab 1"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("Distribution of Positive and Negative Events")
    #a1 = summary_data.groupby("alert-impact").size().reset_index(name="count")
    a1 = summary_data.groupby(["alert-type", "alert-impact"]).size().reset_index(name='count')
    df_clean = (summary_data.assign(**{"enabling-principle": summary_data["enabling-principle"].astype(str).str.split(",")}).explode("enabling-principle"))
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip().apply(lambda x: wrap_label_by_words(x, words_per_line=4))

    a2 = df_clean.groupby(["enabling-principle", "alert-impact"]).size().reset_index(name='count')
    a3 = summary_data.groupby(["continent", "alert-impact"]).size().reset_index(name='count')
    a4 = summary_data.groupby(["alert-country", "alert-impact"]).size().reset_index(name='count')
   
    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")

    #with r1c1: st.plotly_chart(create_bar_chart(a1, x="alert-impact", y="count", horizontal=True), use_container_width=True, key="tab1_chart1")
    with r1c1: st.plotly_chart(create_h_stacked_bar( a1, y="alert-type", x="count", color_col="alert-impact", horizontal=True), use_container_width=True, key="tab1_chart1")
    with r1c2: st.plotly_chart(create_h_stacked_bar( a2, y="enabling-principle", x="count", color_col="alert-impact", horizontal=True), use_container_width=True, key="tab1_chart2")
    with r2c1: st.plotly_chart(create_h_stacked_bar( a3, y="continent", x="count", color_col="alert-impact", horizontal=False), use_container_width=True, key="tab1_chart3")
    with r2c2: st.plotly_chart(create_h_stacked_bar( a4, y="alert-country", x="count", color_col="alert-impact", horizontal=False), use_container_width=True, key="tab1_chart4")

# ---------------- TAB 2 ----------------
with tab2:
    active_tab = "Tab 2"
    col1, col2, col3, col4 = st.columns(4)
     
     
# ---------------- TAB 3 ----------------
with tab3:
    active_tab = "Tab 3"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    #st.header("📈 Positive Events Analysis")
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

    #st.header("📌 Others Analysis")
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
    
    df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
    # Only include countries that exist in the GeoJSON
    geo_countries = [feature['properties']['name'] for feature in countries_gj['features']]
    df_map = df_map[df_map['alert-country'].isin(geo_countries)]
    
       
    fig = px.choropleth_mapbox(
        df_map,
        geojson=countries_gj,
        locations="alert-country",                 # column in df_map
        featureidkey="properties.name",      # match your geojson property
        color="count",
        hover_name="alert-country",
        hover_data={"count": True,
                    "alert-country": False,},
        labels={
        "count": "Number of Alerts",
        },
        color_continuous_scale="Greens",
        mapbox_style="open-street-map",
        zoom=1,
        center={"lat": 10, "lon": 0},
        opacity=0.6
    )
    fig.update_layout(
        margin={"r":0,"t":1,"l":0,"b":0},
        height=500
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    
    
# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align: center; color: gray;'>
    © 2025 EU SEE Dashboard. All rights reserved.
</div>
""", unsafe_allow_html=True)
