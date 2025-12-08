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
.css-1wa3eu0 .css-1d391kg { background-color: #660094 !important; color: white !important; }
.css-1wa3eu0 input { color: #660094 !important; }
.css-1gtu0r7 { background-color: #f2e6ff !important; color: #660094 !important; }
.css-1gtu0r7 div[role="option"]:hover { background-color: #b266ff !important; color: white !important; }
.css-1wa3eu0 { border-color: #660094 !important; }
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
.summary-card { background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%); color: white; padding: 5px; border-radius: 12px; text-align: center; margin: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
.summary-card h2 { font-size: 22px; margin: 5px 0; }
.summary-card p { font-size: 12px; margin: 0; opacity: 0.9; }
.stTabs [role="tab"] button { font-size: 20px; font-weight: bold; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=0)
def load_data():
    parquet_file = Path.cwd() / "data" / "output_final.parquet"
    if not parquet_file.exists():
        st.error(f"Parquet file not found: {parquet_file}")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_file)

    df['alert-country'] = df['alert-country'].astype(str).str.strip()
    df = df[df['alert-country'] != "Jose"]
    df = df[df['alert-impact'].notna() & (df['alert-impact'].str.strip() != '')]

    # ------------------ ISO & Continent ------------------
    json_file = Path.cwd() / "data" / "countries_metadata.json"
    if not json_file.exists():
        st.error(f"Countries metadata JSON not found: {json_file}")
        return df

    with open(json_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    df['iso_alpha3'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3", None))
    df['continent'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))
    missing_countries = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing_countries) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing_countries)}")

    # ------------------ Month & Year ------------------
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    else:
        st.warning("No 'creation_date' column found in dataset.")

    return df

data = load_data()

# ---------------- GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

def multiselect_with_all(label, options, session_key):
    options = list(options)
    try:
        selected = st.sidebar.multiselect(label, ["Select All"] + options, default=st.session_state.get(session_key, ["Select All"]))
        if "Select All" in selected:
            st.session_state[session_key] = ["Select All"]
            return options
        else:
            st.session_state[session_key] = selected
            return selected
    except Exception:
        return options

selected_continents = multiselect_with_all("Select Continent", sorted(data['continent'].dropna().unique()), "selected_continents")
if "Select All" in selected_continents:
    country_options = sorted(data['alert-country'].dropna().unique())
else:
    country_options = sorted(data[data['continent'].isin(selected_continents)]['alert-country'].dropna().unique())
selected_countries = multiselect_with_all("Select Country", country_options, "selected_countries")
selected_alert_types = multiselect_with_all("Select Alert Type", sorted(data['alert-type'].dropna().unique()), "selected_alert_types")
selected_enablinge_principle = multiselect_with_all("Select Enabling Principle", sorted(data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique()), "selected_enablinge_principle")
selected_alert_impacts = multiselect_with_all("Select Alert Impact", sorted(data['alert-impact'].dropna().unique()), "selected_alert_impacts")
month_options = sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
selected_months = multiselect_with_all("Select Month", month_options, "selected_months")
year_options = sorted(data['year'].dropna().unique())
selected_years = multiselect_with_all("Select Year", year_options, "selected_years")

if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_continents","selected_countries","selected_alert_types","selected_enablinge_principle","selected_alert_impacts","selected_months","selected_years"]:
        st.session_state[key] = ["Select All"]
    st.experimental_rerun()

# ---------------- FILTER DATA ----------------
def contains_any(cell_value, selected_values):
    if pd.isna(cell_value): return False
    cell_value = str(cell_value)
    return any(sel in cell_value for sel in selected_values)

filtered_global = data[
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enablinge_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]

# ---------------- FUNCTIONS ----------------
def render_summary_cards(data):
    total_value = data.shape[0]
    neg_alerts = data[data['alert-impact']=="Negative"].shape[0]
    pos_alerts = data[data['alert-impact']=="Positive"].shape[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="summary-card"><p>Total Value</p><h1>{total_value}</h1></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="summary-card"><p>Negative Alerts</p><h1>{neg_alerts}</h1></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="summary-card"><p>Positive Alerts</p><h1>{pos_alerts}</h1></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="summary-card"><p>Max Value</p><h1>{total_value}</h1></div>', unsafe_allow_html=True)

def wrap_label_by_words(label, words_per_line=4):
    words = label.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)

def create_h_stacked_bar(df, y, x, color_col, horizontal=False, height=350):
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
            text=df_cat[x] if not horizontal else df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if color_sequence[i]=="#FFDB58" else 'white', size=13, family="Arial Black"),
            hovertemplate=f"%{{y}}<br>{cat}: %{{x}}<extra></extra>"
        ))
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(barmode='stack', height=height, margin=dict(l=120,r=20,t=20,b=20))
    return fig

# ---------------- LOAD GEOJSON ----------------
geojson_file = Path.cwd() / "data" / "countriess.geojson"
if geojson_file.exists():
    with open(geojson_file) as f:
        countries_gj = json.load(f)
else:
    st.error("❌ countries.geojson file missing inside /data folder")
    countries_gj = None

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others", "Visualization map"])

# ---------------- TAB 1 ----------------
with tab1:
    render_summary_cards(filtered_global)
    fig1 = create_h_stacked_bar(filtered_global.groupby("alert-country")["alert-impact"].count().reset_index(),
                                "alert-country","alert-impact","alert-impact",horizontal=True)
    st.plotly_chart(fig1, use_container_width=True)
    
# ---------------- TAB 2 ----------------
with tab2:
    # Filters in one row
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
    with fcol1:
        selected_continents_tab2 = st.multiselect("Select Continent", sorted(data['continent'].dropna().unique()), default=selected_continents)
    with fcol2:
        selected_countries_tab2 = st.multiselect("Select Country", sorted(data['alert-country'].dropna().unique()), default=selected_countries)
    with fcol3:
        selected_alert_types_tab2 = st.multiselect("Select Alert Type", sorted(data['alert-type'].dropna().unique()), default=selected_alert_types)
    with fcol4:
        selected_enablinge_principle_tab2 = st.multiselect("Select Enabling Principle", sorted(data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique()), default=selected_enablinge_principle)
    with fcol5:
        selected_alert_impacts_tab2 = st.multiselect("Select Alert Impact", sorted(data['alert-impact'].dropna().unique()), default=selected_alert_impacts)

    filtered_tab2 = filtered_global[
        (filtered_global['continent'].isin(selected_continents_tab2)) &
        (filtered_global['alert-country'].isin(selected_countries_tab2)) &
        (filtered_global['alert-type'].isin(selected_alert_types_tab2)) &
        (filtered_global['enabling-principle'].apply(lambda x: contains_any(x, selected_enablinge_principle_tab2))) &
        (filtered_global['alert-impact'].isin(selected_alert_impacts_tab2))
    ]
    render_summary_cards(filtered_tab2)
    fig_tab2 = create_h_stacked_bar(filtered_tab2.groupby("alert-country")["alert-impact"].count().reset_index(),
                                    "alert-country","alert-impact","alert-impact",horizontal=True)
    st.plotly_chart(fig_tab2, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    render_summary_cards(filtered_global)
    fig_tab3 = create_h_stacked_bar(filtered_global.groupby("alert-type")["alert-impact"].count().reset_index(),
                                    "alert-type","alert-impact","alert-impact")
    st.plotly_chart(fig_tab3, use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    render_summary_cards(filtered_global)
    fig_tab4 = create_h_stacked_bar(filtered_global.groupby("enabling-principle")["alert-impact"].count().reset_index(),
                                    "enabling-principle","alert-impact","alert-impact")
    st.plotly_chart(fig_tab4, use_container_width=True)

# ---------------- TAB 5 ----------------
with tab5:
    render_summary_cards(filtered_global)
    if countries_gj is not None:
        df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]
        fig = px.choropleth_mapbox(df_map, geojson=countries_gj, locations="alert-country",
                                   featureidkey="properties.name", color="count",
                                   hover_name="alert-country", hover_data={"count": True, "alert-country": False},
                                   color_continuous_scale="Greens", mapbox_style="open-street-map",
                                   zoom=1, center={"lat":10,"lon":0}, opacity=0.6)
        fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0}, height=500, xaxis_showgrid=False, yaxis_showgrid=False,
                          xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align:center; color: gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>
""", unsafe_allow_html=True)
