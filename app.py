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

# ---------------- TAB 2: Negative Events ----------------
# Filter negative alerts
reactive_df = filtered_global[filtered_global['alert-impact'] == "Negative"].copy()

# Ensure all required columns exist
required_columns = [
    'Actor of repression',
    'Subject of repression',
    'Mechanism of repression',
    'Type of event',
    'alert-type',
    'enabling-principle'
]

for col in required_columns:
    if col not in reactive_df.columns:
        reactive_df[col] = np.nan
        st.warning(f"Column '{col}' was missing and has been added as empty.")
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

# ---------------- HELPER FUNCTIONS ----------------
def filter_top_n(df, row_col, col_col, top_n=None):
    """
    Creates a pivot table for heatmaps, keeping only top-N rows if specified.
    """
    pivot_df = (
        df.groupby([row_col, col_col])
        .size()
        .reset_index(name='count')
    )

    if top_n is not None:
        top_rows = (
            pivot_df.groupby(row_col)['count']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        pivot_df = pivot_df[pivot_df[row_col].isin(top_rows)]

    heatmap_df = pivot_df.pivot(index=row_col, columns=col_col, values='count').fillna(0)
    return heatmap_df

def create_heatmap(pivot_df, title="Heatmap"):
    """
    Creates a Plotly heatmap from a pivot table.
    """
    if pivot_df.empty:
        return create_placeholder_chart("No data available")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            hoverongaps=False
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=80, r=20, t=40, b=80),
        height=350
    )
    return fig

def render_heatmaps(df, top_n):
    """
    Renders the three heatmaps for Negative Events tab.
    """
    actor_mechanism_pivot = filter_top_n(df, 'Actor of repression', 'Mechanism of repression', top_n)
    subject_mechanism_pivot = filter_top_n(df, 'Subject of repression', 'Mechanism of repression', top_n)
    actor_subject_pivot = filter_top_n(df, 'Actor of repression', 'Subject of repression', top_n)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_heatmap(actor_mechanism_pivot, "Actor → Mechanism (% of Actor Total)"), use_container_width=True)
    with col2:
        st.plotly_chart(create_heatmap(subject_mechanism_pivot, "Subject → Mechanism (% of Subject Total)"), use_container_width=True)
    with col3:
        st.plotly_chart(create_heatmap(actor_subject_pivot, "Actor → Subject (% of Actor Total)"), use_container_width=True)

# ---------------- SANKEY ----------------
def render_sankey(summary_df, top_n=None, width=900):
    if summary_df.empty:
        st.warning("No data available for Sankey")
        return go.Figure()
    def get_top_nodes(df, col, n):
        counts = df[col].value_counts()
        if n is not None: counts = counts.head(n)
        return counts.index.tolist()
    top_actors = get_top_nodes(summary_df, "Actor of repression", top_n)
    top_mechanisms = get_top_nodes(summary_df, "Mechanism of repression", top_n)
    top_subjects = get_top_nodes(summary_df, "Subject of repression", top_n)

    def wrap_label(label, max_words_per_line=2):
        words = str(label).split()
        lines = [" ".join(words[i:i+max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
        return "<br>".join(lines)

    actor_nodes = [wrap_label(f"Actor: {a}", 2) for a in top_actors]
    mechanism_nodes = [wrap_label(f"Mechanism: {m}", 2) for m in top_mechanisms]
    subject_nodes = [wrap_label(f"Subject: {s}", 2) for s in top_subjects]
    nodes = actor_nodes + mechanism_nodes + subject_nodes
    node_indices = {name: idx for idx, name in enumerate(nodes)}
    node_colors = ["#FF5733"]*len(actor_nodes) + ["#33C1FF"]*len(mechanism_nodes) + ["#33FF8A"]*len(subject_nodes)

    links = []
    df_am = summary_df[summary_df["Actor of repression"].isin(top_actors) & summary_df["Mechanism of repression"].isin(top_mechanisms)]
    for _, row in df_am.groupby(["Actor of repression","Mechanism of repression"]).size().reset_index(name="value").iterrows():
        links.append({"source": node_indices[wrap_label(f"Actor: {row['Actor of repression']}",2)],
                      "target": node_indices[wrap_label(f"Mechanism: {row['Mechanism of repression']}",2)],
                      "value": row["value"]})
    df_ms = summary_df[summary_df["Mechanism of repression"].isin(top_mechanisms) & summary_df["Subject of repression"].isin(top_subjects)]
    for _, row in df_ms.groupby(["Mechanism of repression","Subject of repression"]).size().reset_index(name="value").iterrows():
        links.append({"source": node_indices[wrap_label(f"Mechanism: {row['Mechanism of repression']}",2)],
                      "target": node_indices[wrap_label(f"Subject: {row['Subject of repression']}",2)],
                      "value": row["value"]})
    hover_text = [f"{nodes[s]} → {nodes[t]}: {v} alerts" for s,t,v in zip(
        [l['source'] for l in links],[l['target'] for l in links],[l['value'] for l in links])]
    fig_height = max(500, len(nodes)*40)

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(pad=30, thickness=25, line=dict(color="black", width=0.5), label=nodes, color=node_colors, font=dict(size=12,color='black'), hovertemplate='%{label}<extra></extra>'),
        link=dict(source=[l['source'] for l in links], target=[l['target'] for l in links], value=[l['value'] for l in links], hovertemplate=hover_text)
    ))

    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color="#FF5733"), name="Actor"),
        go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color="#33C1FF"), name="Mechanism"),
        go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color="#33FF8A"), name="Subject")
    ]
    for trace in legend_traces: fig.add_trace(trace)
    fig.update_layout(title_text="Flow of Negative Events", font_size=12, width=width, height=fig_height, margin=dict(l=50,r=50,t=50,b=50), showlegend=True)
    return fig
# ---------------- TAB 2: Negative Events ----------------
with tab2:
    st.markdown("## Filters & Overview")

    # Inline filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_actor_types = safe_multiselect("Actor Type",
                                               reactive_df['Actor of repression'].dropna().unique(),
                                               "selected_actor_types", sidebar=False)
    with col2:
        selected_subject_types = safe_multiselect("Subject Type",
                                                  reactive_df['Subject of repression'].dropna().unique(),
                                                  "selected_subject_types", sidebar=False)
    with col3:
        selected_mechanism_types = safe_multiselect("Mechanism Type",
                                                    reactive_df['Mechanism of repression'].dropna().unique(),
                                                    "selected_mechanism_types", sidebar=False)
    with col4:
        selected_event_types = safe_multiselect("Event Type",
                                                reactive_df['Type of event'].dropna().unique(),
                                                "selected_event_types", sidebar=False)

    # Filter data
    summary_data = reactive_df.copy()
    if "Select All" not in selected_actor_types:
        summary_data = summary_data[summary_data['Actor of repression'].isin(selected_actor_types)]
    if "Select All" not in selected_subject_types:
        summary_data = summary_data[summary_data['Subject of repression'].isin(selected_subject_types)]
    if "Select All" not in selected_mechanism_types:
        summary_data = summary_data[summary_data['Mechanism of repression'].isin(selected_mechanism_types)]
    if "Select All" not in selected_event_types:
        summary_data = summary_data[summary_data['Type of event'].isin(selected_event_types)]

    # Render summary cards
    render_summary_cards(summary_data)

    # Top-N selection
    if "top_n_option" not in st.session_state:
        st.session_state.top_n_option = "Top 5"
        st.session_state.top_n = 5

    def update_top_n():
        option = st.session_state.top_n_option
        st.session_state.top_n = {"Top 5":5, "Top 10":10, "All":None}[option]

    st.selectbox(
        "Select Top N for charts, heatmaps, and Sankey",
        options=["Top 5", "Top 10", "All"],
        index=["Top 5","Top 10","All"].index(st.session_state.top_n_option),
        key="top_n_option",
        on_change=update_top_n
    )

    # Render individual Top-N bar charts
    st.markdown("## Individual Indicators")
    r1c1,r1c2,r1c3 = st.columns(3)
    r2c1,r2c2,r2c3 = st.columns(3)

    def top_n_bar(df, col, n):
        grouped = df.groupby(col).size().reset_index(name="count").sort_values("count", ascending=False)
        if n is not None:
            grouped = grouped.head(n)
        return grouped

    t1 = top_n_bar(summary_data, "Actor of repression", st.session_state.top_n)
    t2 = top_n_bar(summary_data, "Subject of repression", st.session_state.top_n)
    t3 = top_n_bar(summary_data, "Mechanism of repression", st.session_state.top_n)
    t4 = top_n_bar(summary_data, "Type of event", st.session_state.top_n)
    t5 = top_n_bar(summary_data, "alert-type", st.session_state.top_n)

    df_clean = summary_data.assign(**{"enabling-principle": summary_data["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip()
    t6 = top_n_bar(df_clean, "enabling-principle", st.session_state.top_n)

    r1c1.plotly_chart(create_bar_chart(t1,"Actor of repression","count",horizontal=False), use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(t2,"Subject of repression","count",horizontal=False), use_container_width=True)
    r1c3.plotly_chart(create_bar_chart(t3,"Mechanism of repression","count",horizontal=False), use_container_width=True)
    r2c1.plotly_chart(create_bar_chart(t4,"Type of event","count",horizontal=True), use_container_width=True)
    r2c2.plotly_chart(create_bar_chart(t5,"alert-type","count",horizontal=True), use_container_width=True)
    r2c3.plotly_chart(create_bar_chart(t6,"enabling-principle","count",horizontal=True), use_container_width=True)

    # Render heatmaps
    render_heatmaps(summary_data, st.session_state.top_n)

    # Render Sankey diagram
    with st.expander("Show Flowchart (Sankey Diagram)"):
        st.plotly_chart(render_sankey(summary_data, st.session_state.top_n), use_container_width=True)

      
      # ---------------- TAB 3 (MAP) ----------------
with tab3:
    render_summary_cards(filtered_global)
    geo_file = Path.cwd() / "data" / "countriess.geojson"
    if geo_file.exists():
        with open(geo_file) as f: 
            countries_gj = json.load(f)
        
        # Base map data
        df_map = filtered_global.groupby("alert-country").size().reset_index(name="count")
        map_df = filtered_global.groupby(["alert-country","iso_alpha3"]).size().reset_index(name="count")

        geo_countries = [f['properties']['name'] for f in countries_gj['features']]
        df_map = df_map[df_map['alert-country'].isin(geo_countries)]

        # ----- Dynamic center & zoom -----
        if not df_map.empty:
            coords = []
            for feature in countries_gj['features']:
                if feature['properties']['name'] in df_map['alert-country'].values:
                    geometry = feature['geometry']
                    if geometry['type'] == "Polygon":
                        coords.extend(geometry['coordinates'][0])
                    elif geometry['type'] == "MultiPolygon":
                        for poly in geometry['coordinates']:
                            coords.extend(poly[0])

            if coords:
                lons, lats = zip(*coords)
                center = {"lat": np.mean(lats), "lon": np.mean(lons)}
                zoom = max(1, min(5, 2 / (max(lons)-min(lons) + 0.01)))
            else:
                center = {"lat":10,"lon":0}
                zoom = 2
        else:
            center = {"lat":10,"lon":0}
            zoom = 2

        # ----- Add advanced hover stats -----
        stats = (
            filtered_global
            .groupby("alert-country")
            .agg(
                total_alerts=("alert-impact", "size"),
                negative_alerts=("alert-impact", lambda x: (x == "Negative").sum()),
                positive_alerts=("alert-impact", lambda x: (x == "Positive").sum())
            )
            .reset_index()
        )

        df_map = df_map.merge(stats, on="alert-country", how="left")

        df_map["perc_negative"] = (
            (df_map["negative_alerts"] / df_map["total_alerts"]) * 100
        ).round(1)

        # ----- Main choropleth -----
        map_height = max(400, len(df_map)*20)

        fig = px.choropleth_mapbox(
            df_map,
            geojson=countries_gj,
            locations="alert-country",
            featureidkey="properties.name",
            color="count",
            hover_name="alert-country",
            hover_data={
                "count": False,
                "total_alerts": False,
                "negative_alerts": False,
                "positive_alerts": False,
                "perc_negative": False
            },
            color_continuous_scale="Greens",
            mapbox_style="open-street-map",
            zoom=zoom,
            center=center,
            opacity=0.8
        )

        # ----- Card-style hover tooltip -----
        fig.update_traces(
            hovertemplate=(
                "<b>%{location}</b><br>"
                "<span style='color:#FFD700'>●</span> Total Alerts: %{customdata[0]}<br>"
                "<span style='color:#FF4C4C'>●</span> Negative: %{customdata[1]}<br>"
                "<span style='color:#00FFAA'>●</span> Positive: %{customdata[2]}<br>"
                "% Negative: %{customdata[3]}%<extra></extra>"
            ),
            customdata=df_map[["total_alerts","negative_alerts","positive_alerts","perc_negative"]].values,
            hoverlabel=dict(
                bgcolor="#2D0055",
                font_size=13,
                font_family="Arial",
                font_color="white",
                bordercolor="#ffffff"
            ),
            marker_line_width=1,
            marker_line_color="black"
        )

        # ----- Bubble density overlay -----
        

        # ----- Final layout -----
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            height=map_height
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("GeoJSON file not found for map visualization.")

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
