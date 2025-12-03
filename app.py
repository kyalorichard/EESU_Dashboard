import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Advanced Dashboard", layout="wide")

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

# ---------------- GLOBAL SIDEBAR FILTERS ----------------
st.sidebar.header("🌍 Global Filters")
country_filter = st.sidebar.multiselect("Country", df["Country"].unique(), default=df["Country"].unique())
region_filter = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
filtered_global = df[(df["Country"].isin(country_filter)) & (df["Region"].isin(region_filter))]

# ---------------- CSS FOR SUMMARY CARDS ----------------
st.markdown("""
<style>
.summary-card {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    padding: 20px;
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
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tab 2", "Tab 3", "Tab 4"])

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

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1: st.bar_chart(a1.set_index("Country"), horizontal=True)
    with r1c2: st.bar_chart(a2.set_index("Region"), horizontal=True)
    with r2c1: st.bar_chart(a3.set_index("Country"))
    with r2c2: st.bar_chart(a4.set_index("Region"))

# ---------------- TAB 2 ----------------
with tab2:
    active_tab = "Tab 2"

    # ---------------- TAB 2 FILTERS IN A SINGLE ROW ----------------
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

    # ---------------- FILTERED DATA ----------------
    summary_data = get_summary_data(active_tab,
                                    tab2_category_filter,
                                    tab2_region_filter,
                                    tab2_country_filter)

    # ---------------- RENDER UNIVERSAL SUMMARY CARDS ----------------
    render_summary_cards(summary_data)

    # ---------------- TAB 2 CHARTS ----------------
    st.header("📊 Tab 2 Analysis")
    v1 = summary_data.groupby("Country")["Value"].sum().reset_index()
    v2 = summary_data.groupby("Region")["Value"].sum().reset_index()
    v3 = summary_data.groupby("Category")["Value"].mean().reset_index()
    v4 = summary_data.groupby("Country")["Value"].mean().reset_index()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1: st.bar_chart(v1.set_index("Country"), horizontal=True)
    with r1c2: st.bar_chart(v2.set_index("Region"), horizontal=True)
    with r2c1: st.bar_chart(v3.set_index("Category"))
    with r2c2: st.bar_chart(v4.set_index("Country"))

# ---------------- TAB 3 ----------------
with tab3:
    active_tab = "Tab 3"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📈 Tab 3 Insights")
    b1 = summary_data.groupby("Country")["Value"].mean().reset_index()
    b2 = summary_data.groupby("Region")["Value"].mean().reset_index()
    b3 = summary_data.groupby("Region")["Value"].sum().reset_index()
    b4 = summary_data.groupby("Country")["Value"].sum().reset_index()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1: st.bar_chart(b3.set_index("Region"), horizontal=True)
    with r1c2: st.bar_chart(b4.set_index("Country"), horizontal=True)
    with r2c1: st.bar_chart(b1.set_index("Country"))
    with r2c2: st.bar_chart(b2.set_index("Region"))

# ---------------- TAB 4 ----------------
with tab4:
    active_tab = "Tab 4"
    summary_data = get_summary_data(active_tab)
    render_summary_cards(summary_data)

    st.header("📌 Tab 4 Summary")
    d1 = summary_data.groupby("Country")["Value"].sum().reset_index()
    d2 = summary_data.groupby("Region")["Value"].sum().reset_index()
    d3 = summary_data.groupby("Category")["Value"].mean().reset_index()
    d4 = summary_data.groupby("Category")["Value"].sum().reset_index()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1: st.bar_chart(d1.set_index("Country"), horizontal=True)
    with r1c2: st.bar_chart(d2.set_index("Region"), horizontal=True)
    with r2c1: st.bar_chart(d3.set_index("Category"))
    with r2c2: st.bar_chart(d4.set_index("Category"))
