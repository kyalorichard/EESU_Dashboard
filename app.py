import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import streamlit.components.v1 as components
import base64

st.set_page_config(page_title="EU SEE Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

EXEC_BRIEF_PATH = BASE_DIR / "docs" / "EU_SEE_Dashboard_Quick_Start_Executive.pdf"
USER_MANUAL_PATH = BASE_DIR / "docs" / "EU SEE Dashboard user manual.pdf"

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
    # Capitalize first letter of each option
    options = [str(opt).capitalize() for opt in sorted(options)]

    # Dropdown shows "Select All" first
    options_with_all = ["Select All"] + options

    # Initialize session state if not already set
    if session_key not in st.session_state:
        # Internally select all options
        st.session_state[session_key] = options.copy()
        # But display nothing by default so placeholder shows
        display_default = []
    else:
        # Only show options that exist in options_with_all
        display_default = [opt for opt in st.session_state[session_key] if opt in options_with_all]

    try:
        if sidebar:
            selected = st.sidebar.multiselect(
                label,
                options_with_all,
                default=display_default,
                key=session_key,
                help="Select options or leave empty to use all"
            )
        else:
            selected = st.multiselect(
                label,
                options_with_all,
                default=display_default,
                key=session_key,
                help="Select options or leave empty to use all"
            )
    except Exception:
        selected = options.copy()

    # If nothing selected or "Select All" chosen, return all options
    if "Select All" in selected or len(selected) == 0:
        st.session_state[session_key] = options.copy()
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
    font_size = max(12, min(16, 14 - int(total_alerts/100)))

    # Create columns
    col1, col2, col3 = st.columns(3)

    card_style = f"""
        background: linear-gradient(135deg, #660094 0%, #8a2be2 50%, #b266ff 100%);
        color: white;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 2px;
    """

    # --- Monitored Countries ---
    with col1:
        st.markdown(f"""
<div style="{card_style}">
<h1 style="margin:0;font-size:30px;font-weight:bold;">Monitored Countries</h1>
<h2 style="margin:0;font-size:30px;font-weight:bold;">{total_countries}</h2>
</div>
""", unsafe_allow_html=True)

    # --- Total Alerts ---
    with col2:
        st.markdown(f"""
<div style="{card_style}">
<h1 style="margin:0;font-size:30px;font-weight:bold;">Total Alerts</h2>
<h2 style="margin:0 0;font-size:30px;font-weight:bold;">{total_alerts}</h2>
</div>
""", unsafe_allow_html=True)

    # --- Alerts Breakdown ---
    with col3:
        st.markdown(f"""
<div style="{card_style}">
<h1 style="margin:0;font-size:30px;font-weight:bold;">Alerts Breakdown</h1>

<!-- Top numbers -->
<div style="display:flex; justify-content:space-between; font-size:14px; margin:2px 0;">
<span style="color:#FF4C4C;font-weight:bold;">Negative ● {negative}</span>
<span style="color:#00FFAA;font-weight:bold;">Positive ● {positive}</span>
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

def normalize_label(label: str) -> str:
    """
    Capitalize first character only, lowercase remaining characters.
    Safe for None/NaN.
    """
    if pd.isna(label):
        return ""
    label = str(label).strip()
    if len(label) == 0:
        return ""
    return label[0].upper() + label[1:].lower()
            
# ---------------- DYNAMIC BAR CHART ----------------
def create_bar_chart(df, x, y,title=None,horizontal=False):
    num_bars = df.shape[0]
    height = 350
    df = df.copy()
    df[x] = df[x].apply(lambda l: wrap_label_by_words(normalize_label(l), words_per_line=3))
   
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
    fig.update_layout(title=title,title_x=0.5, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ---------------- HORIZONTAL STACKED BAR ----------------
def create_h_stacked_bar(df, y, x="count", color_col="alert-impact",title=None, horizontal=False):
    categories = sorted(df[color_col].unique())
    color_sequence = ['#FFDB58', '#660094']
    fig = go.Figure()
    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat].copy()
        df_cat[y] = df_cat[y].apply(lambda l: wrap_label_by_words(normalize_label(l), words_per_line=4))
        
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
    fig.update_layout(title=title,title_x=0.5, margin=dict(l=10, r=10, t=40, b=10))
    return fig
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

# ---------------- FORMATTED HEATMAP ----------------
def create_heatmap(pivot_df, title="Heatmap"):
    """
    Creates a Plotly heatmap from a pivot table with formatted labels and hover info.
    """
    if pivot_df.empty:
        # Placeholder chart if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig
        
    # Wrap labels for better readability
    pivot_df.index = [wrap_label_by_words(normalize_label(str(i)), words_per_line=3) for i in pivot_df.index]
    pivot_df.columns = [wrap_label_by_words(normalize_label(str(i)), words_per_line=3) for i in pivot_df.columns]
  
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count", tickfont=dict(size=12))
        )
    )

    fig.update_layout(
        title=title,
        title_font=dict(size=18, color="#660094"),
        xaxis_title="",
        yaxis_title="",
        xaxis_tickangle=-45,
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=80, r=20, t=50, b=120),
        height=max(350, len(pivot_df)*35)
    )

    return fig
# ---------------- HELPER: Get Top-N Items ----------------
def get_top_n_items(df, col, top_n):
    """
    Returns a list of top-N items in a column based on frequency.
    If top_n is None, returns all items.
    """
    counts = df[col].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return counts.index.tolist()

# ---------------- UPDATED HEATMAP RENDER FUNCTION ----------------
def render_heatmaps(df, top_n=5):
    """
    Renders three heatmaps for Negative Events tab, handling multi-valued fields:
    - Actor → Mechanism
    - Subject → Mechanism
    - Actor → Subject

    Parameters:
        df (DataFrame): Filtered data (Negative Events)
        top_n (int or None): Number of top items to show per axis. Use None for all.
    """
    if df.empty:
        st.warning("No data available for heatmaps.")
        return

    # Explode multi-valued columns
    df_exploded = explode_multi_valued_columns(df, [
        "Actor of repression",
        "Subject of repression",
        "Mechanism of repression"
    ])

    # Determine Top-N items
    top_actors = get_top_n_items(df_exploded, "Actor of repression", top_n)
    top_subjects = get_top_n_items(df_exploded, "Subject of repression", top_n)
    top_mechanisms = get_top_n_items(df_exploded, "Mechanism of repression", top_n)

    # Filter to Top-N items
    df_top = df_exploded[
        df_exploded['Actor of repression'].isin(top_actors) &
        df_exploded['Subject of repression'].isin(top_subjects) &
        df_exploded['Mechanism of repression'].isin(top_mechanisms)
    ].copy()

    # Create pivot tables
    actor_mechanism_pivot = filter_top_n(df_top, 'Actor of repression', 'Mechanism of repression', top_n)
    subject_mechanism_pivot = filter_top_n(df_top, 'Subject of repression', 'Mechanism of repression', top_n)
    actor_subject_pivot = filter_top_n(df_top, 'Actor of repression', 'Subject of repression', top_n)

    # Consistent color scale
    all_values = pd.concat([
        actor_mechanism_pivot.stack(),
        subject_mechanism_pivot.stack(),
        actor_subject_pivot.stack()
    ])
    zmax = all_values.max() if not all_values.empty else 1

    # Render heatmaps in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = create_heatmap(actor_mechanism_pivot, title="Actor → Mechanism (# of Actor)")
        fig1.update_traces(zmin=0, zmax=zmax)
        st.plotly_chart(fig1, use_container_width=True, key="heatmap_actor_mechanism")

    with col2:
        fig2 = create_heatmap(subject_mechanism_pivot, title="Subject → Mechanism (# of Subject)")
        fig2.update_traces(zmin=0, zmax=zmax)
        st.plotly_chart(fig2, use_container_width=True, key="heatmap_subject_mechanism")

    with col3:
        fig3 = create_heatmap(actor_subject_pivot, title="Actor → Subject (# of Actor)")
        fig3.update_traces(zmin=0, zmax=zmax)
        st.plotly_chart(fig3, use_container_width=True, key="heatmap_actor_subject")

# ---------------- UPDATED SANKEY FUNCTION ----------------
def render_sankey(df, top_n=None, width=900):
    """
    Render a Sankey diagram for Negative Events:
    Actor → Mechanism → Subject
    """
    if df.empty:
        st.warning("No data available for Sankey")
        return go.Figure()

    # Helper: truncate long labels
    def truncate_label(label, max_chars=25):
        label = str(label)
        return label if len(label) <= max_chars else label[:max_chars-3] + "..."

    # Get top-N nodes
    def get_top_nodes(col):
        counts = df[col].value_counts()
        if top_n is not None:
            counts = counts.head(top_n)
        return counts.index.tolist()

    top_actors = get_top_nodes("Actor of repression")
    top_mechanisms = get_top_nodes("Mechanism of repression")
    top_subjects = get_top_nodes("Subject of repression")

    # Build node labels
    actor_nodes = [truncate_label(f"Actor: {a}") for a in top_actors]
    mechanism_nodes = [truncate_label(f"Mechanism: {m}") for m in top_mechanisms]
    subject_nodes = [truncate_label(f"Subject: {s}") for s in top_subjects]

    nodes = actor_nodes + mechanism_nodes + subject_nodes
    node_index = {name: i for i, name in enumerate(nodes)}

    node_colors = (
        ["#FF5733"] * len(actor_nodes) +
        ["#33C1FF"] * len(mechanism_nodes) +
        ["#33FF8A"] * len(subject_nodes)
    )
    
    links = []

    # Actor → Mechanism
    df_am = df[df["Actor of repression"].isin(top_actors) &
               df["Mechanism of repression"].isin(top_mechanisms)]
    for _, r in df_am.groupby(["Actor of repression", "Mechanism of repression"]).size().reset_index(name="value").iterrows():
        links.append(dict(
            source=node_index[truncate_label(f"Actor: {r['Actor of repression']}")],
            target=node_index[truncate_label(f"Mechanism: {r['Mechanism of repression']}")],
            value=r["value"]
        ))

    # Mechanism → Subject
    df_ms = df[df["Mechanism of repression"].isin(top_mechanisms) &
               df["Subject of repression"].isin(top_subjects)]
    for _, r in df_ms.groupby(["Mechanism of repression", "Subject of repression"]).size().reset_index(name="value").iterrows():
        links.append(dict(
            source=node_index[truncate_label(f"Mechanism: {r['Mechanism of repression']}")],
            target=node_index[truncate_label(f"Subject: {r['Subject of repression']}")],
            value=r["value"]
        ))

    # Figure height scales with number of nodes
    fig_height = max(500, len(nodes) * 40)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=40,             # spacing between nodes
            thickness=35,       # node thickness
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>"
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            hovertemplate="%{value} alerts<extra></extra>"
        )
    ))

    # Optional legend as scatter
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="#FF5733"),
        name="Actor of repression"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="#33C1FF"),
        name="Mechanism of repression"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="#33FF8A"),
        name="Subject of repression"
    ))

    fig.update_layout(
        title="Flow of Negative Events",
        font=dict(size=12, color="black"),
        height=fig_height,
        width=width,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )

    return fig

# ---------------- TOP-N BAR HELPER ----------------
def top_n_bar(df, col, top_n=None):
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=[col, "count"])
    
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    
    if top_n is not None:
        counts = counts.head(top_n)    
    return counts
# ---------------- EXPLODE MULTI-VALUED COLUMNS ----------------
def explode_multi_valued_columns(df, cols):
    """
    Explodes comma-separated values in specified columns.
    Each comma-separated value becomes a separate row.
    """
    df_exploded = df.copy()
    for col in cols:
        if col in df_exploded.columns:
            df_exploded[col] = df_exploded[col].fillna("").astype(str).str.split(",")
            df_exploded = df_exploded.explode(col)
            df_exploded[col] = df_exploded[col].str.strip()
    return df_exploded
    
 
# ---------------- TABS ----------------
#tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Visualization Map"])
tab1, tab2, tab3, tab4 = st.tabs(["Overview","Negative Events","Visualization Map","User Manual"])

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
    
    r1c1.plotly_chart(create_h_stacked_bar(a1,y="alert-type",x="count",color_col="alert-impact",title="Alert type distribution", horizontal=True),use_container_width=True,  key="tab1_chart1")
    r1c2.plotly_chart(create_h_stacked_bar(a2,y="enabling-principle",x="count",color_col="alert-impact",title="Enabling principles distribution", horizontal=True),use_container_width=True,  key="tab1_chart2")
    r2c1.plotly_chart(create_h_stacked_bar(a3,y="region",x="count",color_col="alert-impact",title="Region distribution", horizontal=False),use_container_width=True,  key="tab1_chart3")
    r2c2.plotly_chart(create_h_stacked_bar(a4,y="alert-country",x="count",color_col="alert-impact",title="Countries distribution", horizontal=False),use_container_width=True,  key="tab1_chart4")

# ---------------- TAB 2: Negative Events ----------------
with tab2:
    # Filter negative events
    reactive_df = filtered_global[filtered_global['alert-impact'] == "Negative"].copy()

    if reactive_df.empty:
        st.warning("No negative events available for the selected filters.")
    else:
        # ---------------- SUMMARY CARDS ----------------
        # Show totals BEFORE exploding multi-valued columns
        
        # ---------------- EXPLODE MULTI-VALUED COLUMNS ----------------
        cols_to_explode = ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]
        df_exploded = explode_multi_valued_columns(reactive_df, cols_to_explode)
        
        df_exploded = reactive_df.copy()
        for col in cols_to_explode:
            df_exploded[col] = df_exploded[col].str.split(",")
            df_exploded = df_exploded.explode(col)
            df_exploded[col] = df_exploded[col].str.strip()
        
        # ---------------- INLINE FILTERS ----------------
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_actor_types = safe_multiselect(
                "Actor Type",
                df_exploded['Actor of repression'].dropna().unique(),
                "selected_actor_types", sidebar=False
            )
        with col2:
            selected_subject_types = safe_multiselect(
                "Subject Type",
                df_exploded['Subject of repression'].dropna().unique(),
                "selected_subject_types", sidebar=False
            )
        with col3:
            selected_mechanism_types = safe_multiselect(
                "Mechanism Type",
                df_exploded['Mechanism of repression'].dropna().unique(),
                "selected_mechanism_types", sidebar=False
            )
        with col4:
            selected_event_types = safe_multiselect(
                "Event Type",
                df_exploded['Type of event'].dropna().unique(),
                "selected_event_types", sidebar=False
            )
       ##### -------- Tab 2 Summary card totals--------------------------
        reactive_df_updated= reactive_df[(reactive_df['Actor of repression'].apply(lambda x: contains_any(x, selected_actor_types))) &
            (reactive_df['Subject of repression'].apply(lambda x: contains_any(x, selected_subject_types))) &
            (reactive_df['Mechanism of repression'].apply(lambda x: contains_any(x, selected_mechanism_types))) &
            (reactive_df['Type of event'].apply(lambda x: contains_any(x, selected_event_types)))
        ]
        render_summary_cards(reactive_df_updated)

      
        filtered_df = df_exploded.copy()
        #filtered_df = reactive_df_updated.copy()
        
        tab2_actor = reactive_df_updated.assign(**{"Actor of repression": reactive_df_updated["Actor of repression"].str.split(",")}).explode("Actor of repression")
        tab2_actor["Actor of repression"] = tab2_actor["Actor of repression"].str.strip()
        m1 = tab2_actor.groupby(["Actor of repression","alert-impact"]).size().reset_index(name='count')

        tab2_subj = reactive_df_updated.assign(**{"Subject of repression": reactive_df_updated["Subject of repression"].str.split(",")}).explode("Subject of repression")
        tab2_subj["Subject of repression"] = tab2_subj["Subject of repression"].str.strip()
        m2 = tab2_subj.groupby(["Subject of repression","alert-impact"]).size().reset_index(name='count')

        tab2_mech = reactive_df_updated.assign(**{"Mechanism of repression": reactive_df_updated["Mechanism of repression"].str.split(",")}).explode("Mechanism of repression")
        tab2_mech["Mechanism of repression"] = tab2_mech["Mechanism of repression"].str.strip()
        m3 = tab2_mech.groupby(["Mechanism of repression","alert-impact"]).size().reset_index(name='count')

        tab2_type = reactive_df_updated.assign(**{"Type of event": reactive_df_updated["Type of event"].str.split(",")}).explode("Type of event")
        tab2_type["Type of event"] = tab2_type["Type of event"].str.strip()
        m4 = tab2_type.groupby(["Type of event","alert-impact"]).size().reset_index(name='count')

        tab2_alert = reactive_df_updated.assign(**{"alert-type": reactive_df_updated["alert-type"].str.split(",")}).explode("alert-type")
        tab2_alert["alert-type"] = tab2_alert["alert-type"].str.strip()
        m5 = tab2_alert.groupby(["alert-type","alert-impact"]).size().reset_index(name='count')
        
        tab2_enabling_principle = reactive_df_updated.assign(**{"enabling-principle": reactive_df_updated["enabling-principle"].str.split(",")}).explode("enabling-principle")
        tab2_enabling_principle["enabling-principle"] = tab2_enabling_principle["enabling-principle"].str.strip()
        m6 = tab2_enabling_principle.groupby(["enabling-principle","alert-impact"]).size().reset_index(name='count')
        
        # ---------------- BAR CHARTS ----------------
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)

        r1c1.plotly_chart(create_bar_chart(m1, "Actor of repression", "count",title="Negative actor of repression distribution"), use_container_width=True, key="tab2_chart1")
        r1c2.plotly_chart(create_bar_chart(m2, "Subject of repression", "count",title="Negative subject of repression distribution"), use_container_width=True, key="tab2_chart2")
        r1c3.plotly_chart(create_bar_chart(m3, "Mechanism of repression", "count",title="Negative mechanism of repression distribution"), use_container_width=True, key="tab2_chart3")
        r2c1.plotly_chart(create_bar_chart(m4, "Type of event", "count",title="Negative type of event distribution", horizontal=True), use_container_width=True, key="tab2_chart4")
        r2c2.plotly_chart(create_bar_chart(m5, "alert-type", "count",title="Negative alert type distribution", horizontal=True), use_container_width=True, key="tab2_chart5")
        r2c3.plotly_chart(create_bar_chart(m6, "enabling-principle", "count",title="Negative enabling principle distribution", horizontal=True), use_container_width=True, key="tab2_chart6")

        # ---------------- TOP-N CONFIG ----------------
        if "top_n_option" not in st.session_state:
            st.session_state.top_n_option = "Top 5"
            st.session_state.top_n = 5

        def update_top_n():
            st.session_state.top_n = {
                "Top 2": 2, "Top 3": 3, "Top 4": 4, "Top 5": 5, "All": None
            }[st.session_state.top_n_option]

        st.selectbox(
            "Select Top N for Heatmaps and Sankey Diagram",
            options=["Top 2", "Top 3", "Top 4", "Top 5", "All"],
            index=["Top 2", "Top 3", "Top 4", "Top 5", "All"].index(st.session_state.top_n_option),
            key="top_n_option",
            on_change=update_top_n
        )
        top_n = st.session_state.top_n
        # ---------------- HEATMAPS ----------------
        with st.expander("Show Heatmaps"):
            render_heatmaps(filtered_df, top_n=top_n)
        
        # ---------------- SANKEY DIAGRAM ----------------
        with st.expander("Show Flowchart (Sankey Diagram)"):
            st.plotly_chart(render_sankey(filtered_df, top_n=top_n), use_container_width=True)
            
        # ---------------- Tab two data preview ----------------
        with st.expander("Summary Data preview"):
            st.write(reactive_df_updated)     
      
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

        # Ensure numeric
        df_map["negative_alerts"] = pd.to_numeric(df_map["negative_alerts"], errors="coerce")
        df_map["total_alerts"] = pd.to_numeric(df_map["total_alerts"], errors="coerce")
        
        # Compute percentage safely
        df_map["perc_negative"] = ((df_map["negative_alerts"] / df_map["total_alerts"]) * 100).round(1)
        
        # Optional: handle NaN values if total_alerts was 0
        df_map["perc_negative"] = df_map["perc_negative"].fillna(0)
              
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

# -------------------------------USER MANUAL TAB------------------------------------       
     
with tab4:
    st.header("EU SEE Dashboard – Quick Start")

    st.markdown("""
    This section provides concise, decision-ready documentation for executives,
    donors, and policy stakeholders.
    """)

    # ---------------- EXECUTIVE BRIEF ----------------
    st.subheader("Executive Brief (1 Page)")
    st.caption("For senior leadership, donors, and policy reporting")

    if EXEC_BRIEF_PATH.exists():
        pdf_bytes = EXEC_BRIEF_PATH.read_bytes()

        st.download_button(
            "Download Executive Brief (PDF)",
            pdf_bytes,
            file_name="EU_SEE_Dashboard_Quick_Start_Executive.pdf",
            mime="application/pdf"
        )
        st.subheader("Executive Brief")
              
        
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f"""
            <iframe
                src="data:application/pdf;base64,{pdf_base64}"
                width="100%"
                height="550px"
                style="border:none;"
            ></iframe>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Executive Brief PDF not found.")

    st.divider()

    # ---------------- FULL USER MANUAL ----------------
    st.subheader("Full User Manual")
    st.caption("Detailed guidance for analysts and advanced users")

    if USER_MANUAL_PATH.exists():
        pdf_bytes = USER_MANUAL_PATH.read_bytes()

        st.download_button(
            "Download Full User Manual (PDF)",
            pdf_bytes,
            file_name="EU SEE Dashboard user manual.pdf",
            mime="application/pdf"
        )

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f"""
            <iframe
                src="data:application/pdf;base64,{pdf_base64}"
                width="100%"
                height="700px"
                style="border:none;"
            ></iframe>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("User Manual PDF not found.")

# ---------------- FOOTER ----------------
st.markdown("<hr><div style='text-align:center;color:gray;'>© 2025 EU SEE Dashboard. All rights reserved.</div>", unsafe_allow_html=True)
