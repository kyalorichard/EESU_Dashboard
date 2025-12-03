import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
st.set_page_config(page_title="EU SEE Dashboard", layout="wide")


# ---------------- LOAD DATA FROM CSV ----------------
@st.cache_data
def load_data():
    file_path = "data/raw_data.csv"
    try:
        df = pd.read_csv(file_path)

        # ✅ FIX: Normalize column names
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace("-", "_")
              .str.replace(" ", "_")
        )

        return df

    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# ---------------- SAMPLE DATA ----------------
@st.cache_data
def load_data():
    np.random.seed(42)
    return pd.DataFrame({
        "Country": np.random.choice(["Kenya", "Ethiopia", "Germany", "USA", "Brazil"], 200),
        "Region": np.random.choice(["East", "West", "North", "South"], 200),
        "Category": np.random.choice(["Agriculture", "Tech", "Health", "Finance"], 200),
        "Value": np.random.randint(20, 200, 200)
    })

df = load_data()

# ---------------- REMOVE STREAMLIT DEFAULT TOP SPACING ----------------
st.markdown("""
<style>
    /* Remove Streamlit's default top padding */
    .css-18e3th9 {padding-top: 0rem;}
    /* Optional: reduce spacing around main container */
    .css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<h1 style='margin-bottom:2px; line-height:1.1; color:#660094; font-size:52px;'>
    EU SEE Dashboard
</h1>
<p style='margin-top:0; color:gray; font-size:16px;'></p>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)  # tight separator

# ---------------- GLOBAL SIDEBAR FILTERS ----------------
st.sidebar.image("assets/eu-see-logo-rgb-wide.svg", width=500)  # top of sidebar
st.sidebar.header("🌍 Global Filters")

# ✅ FIX: Auto-detect the correct country column (even if user CSV used different naming)
country_col = next((c for c in df.columns if "country" in c), None)

if country_col:
    options = df[country_col].unique().tolist()
else:
    st.error("No country-related column found in your CSV!")
    options = []

selected_country = st.sidebar.selectbox("Country", ["all"] + options)



country_filter = st.sidebar.multiselect("Country", df["Country"].unique(), default=df["Country"].unique())
region_filter = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
filtered_global = df[(df["Country"].isin(country_filter)) & (df["Region"].isin(region_filter))]

# ---------------- CSS FOR SUMMARY CARDS & TABS ----------------
st.markdown("""
<style>
.summary-card {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.summary-card h2 {
    font-size: 36px;
    margin: 5px 0;
}
.summary-card p {
    font-size: 16px;
    margin: 0;
    opacity: 0.9;
}
.summary-icon {
    font-size: 30px;
    margin-bottom: 5px;
}
/* Increase tabs name font size */
.stTabs [role="tab"] button {
    font-size: 22px;
    font-weight: bold;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTION TO RENDER SUMMARY CARDS ----------------
def render_summary_cards(data):
    total_value = data["Value"].sum()
    avg_value = data["Value"].mean()
    max_value = data["Value"].max()
    min_value = data["Value"].min()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-icon">💰</div>
            <h2>{total_value}</h2>
            <p>Total Value</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="summary-card" style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)">
            <div class="summary-icon">📊</div>
            <h2>{avg_value:.2f}</h2>
            <p>Average Value</p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="summary-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%)">
            <div class="summary-icon">📈</div>
            <h2>{max_value}</h2>
            <p>Max Value</p>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="summary-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%)">
            <div class="summary-icon">📉</div>
            <h2>{min_value}</h2>
            <p>Min Value</p>
        </div>
        ''', unsafe_allow_html=True)

# ---------------- FUNCTION TO CREATE PLOTLY BAR CHART ----------------
def create_bar_chart(data, x, y, horizontal=False, height=400):
    fig = px.bar(
        data,
        x=x if not horizontal else y,
        y=y if not horizontal else x,
        orientation='h' if horizontal else 'v',
        color_discrete_sequence=['#660094'],
        text=y
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    return fig

# ---------------- FUNCTION TO GET DATA FOR SUMMARY CARDS ----------------
def get_summary_data(active_tab, tab2_category=[], tab2_region=[], tab2_country=[]):
    data = filtered_global.copy()
    if active_tab == "Tab 2":
        data = data[
            (data["Category"].isin(tab2_category)) &
            (data["Region"].isin(tab2_region)) &
            (data["Country"].isin(tab2_country))
        ]
    return data

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Negative Events", "Positive Events", "Others"])

# ---------------- TAB 1 ----------------
with tab1:
    active_tab = "Tab 1"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📌 Overview")
    a1 = summary_data.groupby("Country")["Value"].sum().reset_index()
    a2 = summary_data.groupby("Region")["Value"].sum().reset_index()
    a3 = summary_data.groupby("Country")["Value"].mean().reset_index()
    a4 = summary_data.groupby("Region")["Value"].mean().reset_index()

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(a1, x="Country", y="Value", horizontal=True), use_container_width=True, key="tab1_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(a2, x="Region", y="Value", horizontal=True), use_container_width=True, key="tab1_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(a3, x="Country", y="Value"), use_container_width=True, key="tab1_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(a4, x="Region", y="Value"), use_container_width=True, key="tab1_chart4")

# ---------------- TAB 2 ----------------
with tab2:
    active_tab = "Tab 2"
    col1, col2, col3 = st.columns(3)
    with col1:
        tab2_category_filter = st.multiselect("Category", df["Category"].unique(),
                                              default=df["Category"].unique())
    with col2:
        tab2_region_filter = st.multiselect("Region (Tab 2)", df["Region"].unique(),
                                            default=df["Region"].unique())
    with col3:
        tab2_country_filter = st.multiselect("Country (Tab 2)", df["Country"].unique(),
                                             default=df["Country"].unique())

    summary_data = get_summary_data(active_tab, tab2_category_filter, tab2_region_filter, tab2_country_filter)
    render_summary_cards(summary_data)

    st.header("📊 Negative Events Analysis")
    v1 = summary_data.groupby("Country")["Value"].sum().reset_index()
    v2 = summary_data.groupby("Region")["Value"].sum().reset_index()
    v3 = summary_data.groupby("Category")["Value"].mean().reset_index()
    v4 = summary_data.groupby("Country")["Value"].mean().reset_index()

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(v1, x="Country", y="Value", horizontal=True), use_container_width=True, key="tab2_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(v2, x="Region", y="Value", horizontal=True), use_container_width=True, key="tab2_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(v3, x="Category", y="Value"), use_container_width=True, key="tab2_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(v4, x="Country", y="Value"), use_container_width=True, key="tab2_chart4")

# ---------------- TAB 3 ----------------
with tab3:
    active_tab = "Tab 3"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📈 Positive Events Analysis")
    b1 = summary_data.groupby("Country")["Value"].mean().reset_index()
    b2 = summary_data.groupby("Region")["Value"].mean().reset_index()
    b3 = summary_data.groupby("Region")["Value"].sum().reset_index()
    b4 = summary_data.groupby("Country")["Value"].sum().reset_index()

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(b3, x="Region", y="Value", horizontal=True), use_container_width=True, key="tab3_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(b4, x="Country", y="Value", horizontal=True), use_container_width=True, key="tab3_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(b1, x="Country", y="Value"), use_container_width=True, key="tab3_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(b2, x="Region", y="Value"), use_container_width=True, key="tab3_chart4")

# ---------------- TAB 4 ----------------
with tab4:
    active_tab = "Tab 4"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📌 Others Analysis")
    d1 = summary_data.groupby("Country")["Value"].sum().reset_index()
    d2 = summary_data.groupby("Region")["Value"].sum().reset_index()
    d3 = summary_data.groupby("Category")["Value"].mean().reset_index()
    d4 = summary_data.groupby("Category")["Value"].sum().reset_index()

    r1c1, r1c2 = st.columns(2, gap="large")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r1c1: st.plotly_chart(create_bar_chart(d1, x="Country", y="Value", horizontal=True), use_container_width=True, key="tab4_chart1")
    with r1c2: st.plotly_chart(create_bar_chart(d2, x="Region", y="Value", horizontal=True), use_container_width=True, key="tab4_chart2")
    with r2c1: st.plotly_chart(create_bar_chart(d3, x="Category", y="Value"), use_container_width=True, key="tab4_chart3")
    with r2c2: st.plotly_chart(create_bar_chart(d4, x="Category", y="Value"), use_container_width=True, key="tab4_chart4")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align: center; color: gray;'>
    © 2025 EU SEE Dashboard. All rights reserved.
</div>
""", unsafe_allow_html=True)
