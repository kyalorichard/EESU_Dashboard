import streamlit as st
import pandas as pd
import numpy as np
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
<hr style='margin:5px 0'>
""", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Remove top spacing */
.css-18e3th9 {padding-top: 0rem;}
.css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}

/* Summary card styling */
.summary-card {
   background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
   color: white;
   padding: 5px;
   border-radius: 12px;
   text-align: center;
   margin: 5px;
   box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.stTabs [role="tab"] button {font-size: 20px; font-weight: bold;}
footer {visibility: hidden;}
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

    meta_file = Path.cwd() / "data" / "countries_metadata.json"
    country_meta = {}
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            country_meta = json.load(f)
    else:
        st.error(f"Countries metadata JSON not found: {meta_file}")

    # ISO codes & continent
    df['iso_alpha3'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3", None))
    df['continent'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))

    # Map continent to region
    def continent_to_region(continent):
        if continent == "Africa":
            return "Africa"
        elif continent in ["Asia", "Oceania"]:
            return "Asia and the Pacific"
        elif continent in ["Europe", "Middle East"]:
            return "The Middle East"
        elif continent in ["Americas", "North America", "South America", "Caribbean"]:
            return "Americas and the Caribbean"
        else:
            return "Unknown"

    df['region'] = df['continent'].apply(continent_to_region)

    # Warn about missing ISO codes
    missing_countries = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing_countries) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing_countries)}")

    # Process dates
    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    else:
        st.warning("No 'creation_date' column found in dataset.")

    return df

data = load_data()
    

# ---------------- MULTISELECT WITH SELECT ALL ----------------
def safe_multiselect(label, options, session_key, sidebar=True):
    options = sorted(list(options))
    options_with_all = ["Select All"] + options
    if session_key not in st.session_state:
        st.session_state[session_key] = ["Select All"]

    try:
        if sidebar:
            selected = st.sidebar.multiselect(label, options_with_all, default=st.session_state[session_key])
        else:
            selected = st.multiselect(label, options_with_all, default=st.session_state[session_key])
    except Exception:
        selected = ["Select All"]

    if "Select All" in selected or len(selected) == 0:
        st.session_state[session_key] = ["Select All"]
        return options
    else:
        st.session_state[session_key] = selected
        return selected
 

# ---------------- GLOBAL FILTERS (COMPACT SIDEBAR) ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

regions_labels = ["Africa", "The Middle East", "Asia and the Pacific", "Americas and the Caribbean"]
selected_regions = safe_multiselect("Select region", regions_labels, "selected_regions")
filtered_countries = data[data['region'].isin(selected_regions)] if "Select All" not in selected_regions else data
selected_countries = safe_multiselect("Select country", filtered_countries['alert-country'].dropna().unique(), "selected_countries")
selected_alert_impacts = safe_multiselect("Select Nature of event/alert", data['alert-impact'].dropna().unique(), "selected_alert_impacts")
selected_alert_types = safe_multiselect("Select Type of alert", data['alert-type'].dropna().unique(), "selected_alert_types")
selected_enabling_principle = safe_multiselect("Select enabling principle", 
                                               data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique(),
                                               "selected_enabling_principle")
selected_years = safe_multiselect("Select year", sorted(data['year'].dropna().unique()), "selected_years")

# Filter available months based on selected years
if "Select All" in selected_years:
    available_months = sorted(
        data['month_name'].dropna().unique(),
        key=lambda m: pd.to_datetime(m, format='%B').month
    )
else:
    available_months = sorted(
        data[data['year'].isin(selected_years)]['month_name'].dropna().unique(),
        key=lambda m: pd.to_datetime(m, format='%B').month
    )

# Month selection
selected_months = safe_multiselect(
    "Select Month", 
    available_months, 
    "selected_months"
)
#selected_months = safe_multiselect("Select month", sorted(data['month_name'].dropna().unique(), key=lambda m: pd.to_datetime(m, format='%B').month), "selected_months")

# Reset button
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_regions","selected_countries","selected_alert_types","selected_enabling_principle",
                "selected_alert_impacts","selected_months","selected_years"]:
        st.session_state[key] = ["Select All"]

# ---------------- FILTER DATA ----------------
def contains_any(cell_value, selected_values):
    if pd.isna(cell_value): return False
    return any(sel in str(cell_value) for sel in selected_values)

filtered_global = data[
    (data['region'].isin(selected_regions)) &
    (data['alert-country'].isin(selected_countries)) &
    (data['alert-type'].isin(selected_alert_types)) &
    (data['enabling-principle'].apply(lambda x: contains_any(x, selected_enabling_principle))) &
    (data['alert-impact'].isin(selected_alert_impacts)) &
    (data['month_name'].isin(selected_months)) &
    (data['year'].isin(selected_years))
]
# ---------------- LABEL WRAPPING ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = str(label).split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)


# ---------------- RESPONSIVE SUMMARY CARDS ----------------
def render_summary_cards(df, base_bar_height=25):
    """
    Render three summary cards with gradient background:
    1. Monitored Countries
    2. Total Alerts
    3. Alerts Breakdown (Negative vs Positive)
    
    Parameters:
        df (DataFrame): Filtered data
        base_bar_height (int): Base height of the horizontal bar
    """
    total_countries = df['alert-country'].nunique() if not df.empty else 0
    total_alerts = len(df) if not df.empty else 0
    negative = (df['alert-impact'] == "Negative").sum() if not df.empty else 0
    positive = (df['alert-impact'] == "Positive").sum() if not df.empty else 0
    total_np = negative + positive

    # Percentages
    neg_pct = round((negative / total_np) * 100, 1) if total_np else 0
    pos_pct = round((positive / total_np) * 100, 1) if total_np else 0

    # Adjust bar height and font size based on total alerts
    bar_height = max(base_bar_height, min(50, total_alerts // 10 + 20))
    font_size = max(10, min(16, 14 - int(total_alerts/100)))

    # Create columns
    col1, col2, col3 = st.columns(3)

    card_style = f"""
        background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
        color: white;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 5px;
    """

    # --- Monitored Countries ---
    with col1:
        st.markdown(f"""
<div style="{card_style}">
<p style="margin:0;font-size:20px;">Monitored Countries</p>
<h2 style="margin:6px 0;">{total_countries}</h2>
</div>
""", unsafe_allow_html=True)

    # --- Total Alerts ---
    with col2:
        st.markdown(f"""
<div style="{card_style}">
<p style="margin:0;font-size:20px;">Total Alerts</p>
<h2 style="margin:6px 0;">{total_alerts}</h2>
</div>
""", unsafe_allow_html=True)

    # --- Alerts Breakdown ---
    with col3:
        st.markdown(f"""
<div style="{card_style}">
<p style="margin:0;font-size:20px;">Alerts Breakdown</p>

<!-- Top numbers -->
<div style="display:flex; justify-content:space-between; font-size:16px; margin:6px 0;">
<span style="color:#FF4C4C;">Negative ● {negative}</span>
<span style="color:#00FFAA;">Positive ● {positive}</span>
</div>

<!-- Horizontal bar -->
<div style="display:flex; height:{bar_height}px; border-radius:8px; overflow:hidden;">
    <div style="width:{neg_pct}%; background:#FF4C4C; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:{font_size}px;">
        {neg_pct if neg_pct>5 else ''}%
    </div>
    <div style="width:{pos_pct}%; background:#00FFAA; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:{font_size}px;">
        {pos_pct if pos_pct>5 else ''}%
    </div>
</div>
</div>
""", unsafe_allow_html=True)
            
# ---------------- DYNAMIC BAR CHART ----------------
def create_bar_chart(df, x, y, horizontal=False):
    num_bars = df.shape[0]
    height = 350
    df = df.copy()
    df[x] = df[x].apply(lambda l: wrap_label_by_words(l, words_per_line=3))
    fig = px.bar(
        df,
        x=x if not horizontal else y,
        y=y if not horizontal else x,
        orientation='h' if horizontal else 'v',
        color_discrete_sequence=['#660094'],
        text=y
    )
    font_size = max(10, 20 - int(num_bars/5))
    fig.update_traces(
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(size=12, color='white', family="Arial Black")
    )
    # Bold axis line
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')           
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
       
    fig.update_xaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(height=height, margin=dict(l=120 if horizontal else 20, r=20, t=20, b=20))
    return fig

# ---------------- HORIZONTAL STACKED BAR ----------------
def create_h_stacked_bar(df, y, x="count", color_col="alert-impact", horizontal=False):
    categories = sorted(df[color_col].unique())
    color_sequence = ['#FFDB58', '#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat].copy()
        df_cat[y] = df_cat[y].apply(lambda l: wrap_label_by_words(l, words_per_line=4))
        fig.add_trace(go.Bar(
            x=df_cat[y] if not horizontal else df_cat[x],
            y=df_cat[x] if not horizontal else df_cat[y],
            name=cat,
            orientation='h' if horizontal else 'v',
            marker_color=color_sequence[i % len(color_sequence)],
            text=df_cat[x],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='black' if color_sequence[i]=="#FFDB58" else 'white', size=12, family="Arial Black"),
            hovertemplate=f"%{{y}}<br>{cat}: %{{x}}<extra></extra>"
        ))
    num_bars = df.shape[0]
    height = 350
    # Bold axis line
    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')        
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
              
    fig.update_layout(barmode='stack', height=height, margin=dict(l=120 if horizontal else 20, r=20, t=20, b=20))
    fig.update_xaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    return fig

# ---------------- TABS ----------------
#tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Visualization Map"])
tab1, tab2, tab3 = st.tabs(["Overview","Negative Events","Visualization Map"])

# ---------------- TAB 1 ----------------
with tab1:
    render_summary_cards(filtered_global)
    a1 = filtered_global.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
    df_clean = filtered_global.assign(**{"enabling-principle": filtered_global["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip()
    a2 = df_clean.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count')
    a3 = filtered_global.groupby(["region","alert-impact"]).size().reset_index(name='count')
    a4 = filtered_global.groupby(["alert-country","alert-impact"]).size().reset_index(name='count')
    r1c1,r1c2 = st.columns(2); r2c1,r2c2 = st.columns(2)
    r1c1.plotly_chart(create_h_stacked_bar(a1,y="alert-type",x="count",color_col="alert-impact",horizontal=True),use_container_width=True,  key="tab1_chart1")
    r1c2.plotly_chart(create_h_stacked_bar(a2,y="enabling-principle",x="count",color_col="alert-impact",horizontal=True),use_container_width=True,  key="tab1_chart2")
    r2c1.plotly_chart(create_h_stacked_bar(a3,y="region",x="count",color_col="alert-impact", horizontal=False),use_container_width=True,  key="tab1_chart3")
    r2c2.plotly_chart(create_h_stacked_bar(a4,y="alert-country",x="count",color_col="alert-impact", horizontal=False),use_container_width=True,  key="tab1_chart4")

# ---------------- TAB 2 (Negative Events + Cross-Analysis Heatmaps) ----------------
with tab2:
    st.markdown("## Filters & Overview")
    
    # --- Inline filters ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_actor_types = safe_multiselect(
            "Actor Type",
            reactive_df['Actor of repression'].dropna().unique(),
            "selected_actor_types",
            sidebar=False
        )
    with col2:
        selected_subject_types = safe_multiselect(
            "Subject Type",
            reactive_df['Subject of repression'].dropna().unique(),
            "selected_subject_types",
            sidebar=False
        )
    with col3:
        selected_mechanism_types = safe_multiselect(
            "Mechanism Type",
            reactive_df['Mechanism of repression'].dropna().unique(),
            "selected_mechanism_types",
            sidebar=False
        )
    with col4:
        selected_event_types = safe_multiselect(
            "Event Type",
            reactive_df['Type of event'].dropna().unique(),
            "selected_event_types",
            sidebar=False
        )

    # --- Filter data ---
    summary_data = reactive_df[
        reactive_df['Actor of repression'].isin(selected_actor_types) &
        reactive_df['Subject of repression'].isin(selected_subject_types) &
        reactive_df['Mechanism of repression'].isin(selected_mechanism_types) &
        reactive_df['Type of event'].isin(selected_event_types)
    ]

    # --- Summary cards ---
    render_summary_cards(summary_data)

    # --- Individual Indicator Bar Charts ---
    st.markdown("## Individual Indicators")
    r1c1,r1c2,r1c3 = st.columns(3)
    r2c1,r2c2,r2c3 = st.columns(3)

    t1 = summary_data.groupby("Actor of repression").size().reset_index(name="count")
    t2 = summary_data.groupby("Subject of repression").size().reset_index(name="count")
    t3 = summary_data.groupby("Mechanism of repression").size().reset_index(name="count")
    t4 = summary_data.groupby("Type of event").size().reset_index(name="count")
    t5 = summary_data.groupby("alert-type").size().reset_index(name="count")
    df_clean = summary_data.assign(**{"enabling-principle": summary_data["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip()
    t6 = df_clean.groupby("enabling-principle").size().reset_index(name="count")

    r1c1.plotly_chart(create_bar_chart(t1,"Actor of repression","count",horizontal=False), use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(t2,"Subject of repression","count",horizontal=False), use_container_width=True)
    r1c3.plotly_chart(create_bar_chart(t3,"Mechanism of repression","count",horizontal=False), use_container_width=True)

    r2c1.plotly_chart(create_bar_chart(t4,"Type of event","count",horizontal=True), use_container_width=True)
    r2c2.plotly_chart(create_bar_chart(t5,"alert-type","count",horizontal=True), use_container_width=True)
    r2c3.plotly_chart(create_bar_chart(t6,"enabling-principle","count",horizontal=True), use_container_width=True)

    # --- Top-N selection ---
    top_n_option = st.selectbox("Select Top N for visualizations", options=["Top 5", "Top 10", "All"], index=0)
    top_n_map = {"Top 5": 5, "Top 10": 10, "All": None}
    top_n = top_n_map[top_n_option]

    # --- Prepare Top-N nodes for Sankey ---
    top_actors = get_top_n_nodes(summary_data, 'Actor of repression', top_n)
    top_mechanisms = get_top_n_nodes(summary_data, 'Mechanism of repression', top_n)
    top_subjects = get_top_n_nodes(summary_data, 'Subject of repression', top_n)

    # --- Sankey Diagram ---
    nodes, links = build_sankey_links(summary_data, top_actors, top_mechanisms, top_subjects)
    sankey_legend()
    sankey_fig = create_sankey_hover(nodes, top_actors, top_mechanisms, top_subjects, links)
    st.plotly_chart(sankey_fig, use_container_width=True)

    # --- Heatmaps ---
    actor_mechanism_pivot = filter_top_n(summary_data, 'Actor of repression', 'Mechanism of repression', top_n)
    subject_mechanism_pivot = filter_top_n(summary_data, 'Subject of repression', 'Mechanism of repression', top_n)
    actor_subject_pivot = filter_top_n(summary_data, 'Actor of repression', 'Subject of repression', top_n)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_heatmap(actor_mechanism_pivot, summary_data, 'Actor of repression', 'Mechanism of repression', 
                                       "Actor → Mechanism (% of Actor Total)"), use_container_width=True)
    with col2:
        st.plotly_chart(create_heatmap(subject_mechanism_pivot, summary_data, 'Subject of repression', 'Mechanism of repression', 
                                       "Subject → Mechanism (% of Subject Total)"), use_container_width=True)
    with col3:
        st.plotly_chart(create_heatmap(actor_subject_pivot, summary_data, 'Actor of repression', 'Subject of repression', 
                                       "Actor → Subject (% of Actor Total)"), use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
