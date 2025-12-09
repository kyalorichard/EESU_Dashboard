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
    if not meta_file.exists():
        st.error(f"Countries metadata JSON not found: {meta_file}")
        return df
    with open(meta_file, encoding="utf-8") as f:
        country_meta = json.load(f)

    df['iso_alpha3'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("iso_alpha3", None))
    df['continent'] = df['alert-country'].apply(lambda x: country_meta.get(x, {}).get("continent", "Unknown"))

    missing_countries = df.loc[df['iso_alpha3'].isna(), 'alert-country'].unique()
    if len(missing_countries) > 0:
        st.warning(f"Countries missing ISO codes: {', '.join(missing_countries)}")

    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        df['year'] = df['creation_date'].dt.year
        df['month_name'] = df['creation_date'].dt.strftime('%B')
    else:
        st.warning("No 'creation_date' column found in dataset.")

    return df

data = load_data()

# ---------------- HELPER FUNCTIONS ----------------
def wrap_label_by_words(label, words_per_line=4):
    words = str(label).split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)

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

def groupby_count(df, group_cols):
    return df.groupby(group_cols).size().reset_index(name='count')

def add_download_buttons(df, prefix="filtered"):
    if not df.empty:
        csv_name = f"{prefix}.csv"
        excel_name = f"{prefix}.xlsx"
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=csv_name,
            mime="text/csv"
        )
        st.download_button(
            label="📥 Download Excel",
            data=df.to_excel(index=False, engine='openpyxl'),
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No data to export.")

# ---------------- GLOBAL FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)
st.sidebar.header("🌍 Global Filters")

selected_continents = safe_multiselect("Select Continent", data['continent'].dropna().unique(), "selected_continents")
filtered_countries = data[data['continent'].isin(selected_continents)] if "Select All" not in selected_continents else data
selected_countries = safe_multiselect("Select Country", filtered_countries['alert-country'].dropna().unique(), "selected_countries")
selected_alert_types = safe_multiselect("Select Alert Type", data['alert-type'].dropna().unique(), "selected_alert_types")
selected_enabling_principle = safe_multiselect("Select Enabling Principle", 
                                               data['enabling-principle'].dropna().str.split(",").explode().str.strip().unique(),
                                               "selected_enabling_principle")
selected_alert_impacts = safe_multiselect("Select Alert Impact", data['alert-impact'].dropna().unique(), "selected_alert_impacts")
selected_years = safe_multiselect("Select Year", sorted(data['year'].dropna().unique()), "selected_years")

available_months = sorted(
    data[data['year'].isin(selected_years)]['month_name'].dropna().unique() if "Select All" not in selected_years else data['month_name'].dropna().unique(),
    key=lambda m: pd.to_datetime(m, format='%B').month
)
selected_months = safe_multiselect("Select Month", available_months, "selected_months")

# Reset button
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["selected_continents","selected_countries","selected_alert_types","selected_enabling_principle",
                "selected_alert_impacts","selected_months","selected_years"]:
        st.session_state[key] = ["Select All"]

# ---------------- FILTER DATA ----------------
def contains_any(cell_value, selected_values):
    if pd.isna(cell_value): return False
    return any(sel in str(cell_value) for sel in selected_values)

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
    cards = [
        {"label": "Total Alerts", "value": df.shape[0]},
        {"label": "Negative Alerts", "value": df[df['alert-impact']=="Negative"].shape[0]},
        {"label": "Positive Alerts", "value": df[df['alert-impact']=="Positive"].shape[0]},
        {"label": "Max Alerts", "value": df.shape[0]}
    ]
    col_count = min(len(cards), 4)
    cols = st.columns(col_count)
    for i, card in enumerate(cards):
        col = cols[i % col_count]
        col.markdown(
            f"""
            <div class="summary-card" style="padding:10px;">
                <p style="font-size:14px; margin:0;">{card['label']}</p>
                <h2 style="font-size:24px; margin:5px 0;">{card['value']}</h2>
            </div>
            """, unsafe_allow_html=True
        )

# ---------------- PLOTLY HOVER STYLE ----------------
hoverlabel_style = dict(
    bgcolor="#660094",
    font_size=12,
    font_family="Arial",
    font_color="white",
    bordercolor="black"
)

# ---------------- BAR CHART ----------------
def create_bar_chart(df, x, y, horizontal=False):
    df = df.copy()
    df[x] = df[x].apply(lambda l: wrap_label_by_words(l, words_per_line=3))
    hover_template = f"<b>%{{x if not horizontal else y}}</b><br>{y}: %{{y if not horizontal else x}}<extra></extra>"

    fig = px.bar(
        df,
        x=x if not horizontal else y,
        y=y if not horizontal else x,
        orientation='h' if horizontal else 'v',
        color_discrete_sequence=['#660094'],
        text=y
    )

    fig.update_traces(
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(size=12, color='white', family="Arial Black"),
        hovertemplate=hover_template,
        hoverlabel=hoverlabel_style
    )

    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')

    fig.update_xaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(height=350, margin=dict(l=120 if horizontal else 20, r=20, t=20, b=20))
    return fig

# ---------------- HORIZONTAL STACKED BAR ----------------
def create_h_stacked_bar(df, y, x="count", color_col="alert-impact", horizontal=False):
    categories = sorted(df[color_col].unique())
    color_sequence = ['#FFDB58', '#660094']
    fig = go.Figure()

    for i, cat in enumerate(categories):
        df_cat = df[df[color_col]==cat].copy()
        df_cat[y] = df_cat[y].apply(lambda l: wrap_label_by_words(l, words_per_line=4))
        hover_template = f"<b>%{{y if not horizontal else x}}</b><br>{cat}: %{{x if not horizontal else y}}<extra></extra>"

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
            hovertemplate=hover_template,
            hoverlabel=hoverlabel_style
        ))

    if horizontal:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    else:
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')

    fig.update_layout(barmode='stack', height=350, margin=dict(l=120 if horizontal else 20, r=20, t=20, b=20))
    fig.update_xaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title=None, showgrid=True, gridwidth=1, gridcolor='lightgray')
    return fig

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Negative Events","Positive Events","Others","Visualization Map"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Overview")
    add_download_buttons(filtered_global, "Overview_filtered")

    render_summary_cards(filtered_global)
    a1 = groupby_count(filtered_global, ["alert-type","alert-impact"])
    df_clean = filtered_global.assign(**{"enabling-principle": filtered_global["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip()
    a2 = groupby_count(df_clean, ["enabling-principle","alert-impact"])
    a3 = groupby_count(filtered_global, ["continent","alert-impact"])
    a4 = groupby_count(filtered_global, ["alert-country","alert-impact"])
    r1c1,r1c2 = st.columns(2); r2c1,r2c2 = st.columns(2)
    r1c1.plotly_chart(create_h_stacked_bar(a1,y="alert-type",x="count",color_col="alert-impact",horizontal=True),use_container_width=True)
    r1c2.plotly_chart(create_h_stacked_bar(a2,y="enabling-principle",x="count",color_col="alert-impact",horizontal=True),use_container_width=True)
    r2c1.plotly_chart(create_h_stacked_bar(a3,y="continent",x="count",color_col="alert-impact", horizontal=False),use_container_width=True)
    r2c2.plotly_chart(create_h_stacked_bar(a4,y="alert-country",x="count",color_col="alert-impact", horizontal=False),use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Negative Events")
    reactive_df = filtered_global[filtered_global['alert-impact']=="Negative"]
    add_download_buttons(reactive_df, "NegativeEvents_filtered")

    col1,col2,col3,col4 = st.columns(4)
    selected_actor_types = safe_multiselect("Actor Type", reactive_df['Actor of repression'].dropna().unique(), "selected_actor_types", sidebar=False)
    selected_subject_types = safe_multiselect("Subject Type", reactive_df['Subject of repression'].dropna().unique(), "selected_subject_types", sidebar=False)
    selected_mechanism_types = safe_multiselect("Mechanism Type", reactive_df['Mechanism of repression'].dropna().unique(), "selected_mechanism_types", sidebar=False)
    selected_event_types = safe_multiselect("Event Type", reactive_df['Type of event'].dropna().unique(), "selected_event_types", sidebar=False)

    summary_data = reactive_df[
        (reactive_df['Actor of repression'].isin(selected_actor_types)) &
        (reactive_df['Subject of repression'].isin(selected_subject_types)) &
        (reactive_df['Mechanism of repression'].isin(selected_mechanism_types)) &
        (reactive_df['Type of event'].isin(selected_event_types))
    ]

    render_summary_cards(summary_data)
    t1 = groupby_count(summary_data, ["Actor of repression"])
    t2 = groupby_count(summary_data, ["Subject of repression"])
    t3 = groupby_count(summary_data, ["Mechanism of repression"])
    t4 = groupby_count(summary_data, ["Type of event"])
    t5 = groupby_count(summary_data, ["alert-type"])
    df_clean = summary_data.assign(**{"enabling-principle": summary_data["enabling-principle"].str.split(",")}).explode("enabling-principle")
    df_clean["enabling-principle"] = df_clean["enabling-principle"].str.strip()
    t6 = groupby_count(df_clean, ["enabling-principle"])

    r1c1,r1c2,r1c3 = st.columns(3); r2c1,r2c2,r2c3 = st.columns(3)
    r1c1.plotly_chart(create_bar_chart(t1,"Actor of repression","count",horizontal=False),use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(t2,"Subject of repression","count",horizontal=False),use_container_width=True)
    r1c3.plotly_chart(create_bar_chart(t3,"Mechanism of repression","count",horizontal=False),use_container_width=True)
    r2c1.plotly_chart(create_bar_chart(t4,"Type of event","count",horizontal=True),use_container_width=True)
    r2c2.plotly_chart(create_bar_chart(t5,"alert-type","count",horizontal=True),use_container_width=True)
    r2c3.plotly_chart(create_bar_chart(t6,"enabling-principle","count",horizontal=True),use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Positive Events")
    positive_df = filtered_global[filtered_global['alert-impact']=="Positive"]
    add_download_buttons(positive_df, "PositiveEvents_filtered")
    render_summary_cards(positive_df)
    b1 = groupby_count(positive_df, ["alert-country"])
    b2 = groupby_count(positive_df, ["alert-type"])
    r1c1,r1c2 = st.columns(2)
    r2c1,r2c2 = st.columns(2)
    r1c1.plotly_chart(create_bar_chart(b1,"alert-country","count",horizontal=True), use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(b2,"alert-type","count",horizontal=True), use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Other Alerts")
    add_download_buttons(filtered_global, "OtherAlerts_filtered")
    render_summary_cards(filtered_global)
    d1 = groupby_count(filtered_global, ["alert-country"])
    d2 = groupby_count(filtered_global, ["alert-type"])
    r1c1,r1c2 = st.columns(2)
    r2c1,r2c2 = st.columns(2)
    r1c1.plotly_chart(create_bar_chart(d1,"alert-country","count",horizontal=True), use_container_width=True)
    r1c2.plotly_chart(create_bar_chart(d2,"alert-type","count",horizontal=True), use_container_width=True)

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("Visualization Map")
    map_df = filtered_global.groupby(["alert-country","iso_alpha3"]).size().reset_index(name="count")
    fig = px.choropleth(
        map_df,
        locations="iso_alpha3",
        color="count",
        hover_name="alert-country",
        color_continuous_scale="Purples"
    )
    fig.update_traces(
        hoverlabel=hoverlabel_style
    )
    st.plotly_chart(fig, use_container_width=True)
