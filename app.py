import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Advanced Dashboard", layout="wide")

# -------- SAMPLE DATA --------
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

# -------- GLOBAL SIDEBAR FILTERS --------
st.sidebar.header("🌍 Global Filters")
country_filter = st.sidebar.multiselect("Country", df["Country"].unique(), default=df["Country"].unique())
region_filter = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())

# Apply global filters
filtered_global = df[
    (df["Country"].isin(country_filter)) &
    (df["Region"].isin(region_filter))
]

# -------- TABS LAYOUT --------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tab 2", "Tab 3", "Tab 4"])
active_tab = st.session_state.get("active_tab", "Overview")

# Detect which tab is active
def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name
    return tab_name

# Tab-specific category filter (inside Tab 2)
tab2_category_filter = []

# -------- SUMMARY METRIC FUNCTION --------
def render_summary(data):
    total_value = data["Value"].sum()
    avg_value = data["Value"].mean()
    max_value = data["Value"].max()
    min_value = data["Value"].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"{total_value}")
    col2.metric("Average Value", f"{avg_value:.2f}")
    col3.metric("Max Value", f"{max_value}")
    col4.metric("Min Value", f"{min_value}")

# -------- DETERMINE DATA FOR SUMMARY CARDS --------
# For Tab 2, summary reacts to global + tab2 filter
# For other tabs, summary reacts only to global filters
with st.container():
    # Placeholder to detect tab and apply Tab2 filter if needed
    if "tab2_active" not in st.session_state:
        st.session_state.tab2_active = False

# -------- TAB 1 --------
with tab1:
    set_active_tab("Overview")
    filtered1 = filtered_global

    # Render summary cards based on filtered data
    render_summary(filtered1)

    st.header("📌 Overview")
    a1 = filtered1.groupby("Country")["Value"].sum().reset_index()
    a2 = filtered1.groupby("Region")["Value"].sum().reset_index()
    a3 = filtered1.groupby("Country")["Value"].mean().reset_index()
    a4 = filtered1.groupby("Region")["Value"].mean().reset_index()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        st.bar_chart(a1.set_index("Country"), horizontal=True)
    with r1c2:
        st.bar_chart(a2.set_index("Region"), horizontal=True)
    with r2c1:
        st.bar_chart(a3.set_index("Country"))
    with r2c2:
        st.bar_chart(a4.set_index("Region"))

# -------- TAB 2 --------
with tab2:
    set_active_tab("Tab 2")
    st.header("📊 Tab 2 Analysis")

    # Tab2-specific filter inside the tab
    tab2_category_filter = st.multiselect(
        "Select Category (Tab 2)", df["Category"].unique(), default=df["Category"].unique()
    )

    # Apply both global + Tab2 filter for summary
    filtered2 = filtered_global[filtered_global["Category"].isin(tab2_category_filter)]

    # Render summary cards based on global + tab2 filter
    render_summary(filtered2)

    # Prepare 4 charts 2x2
    v1 = filtered2.groupby("Country")["Value"].sum().reset_index()
    v2 = filtered2.groupby("Region")["Value"].sum().reset_index()
    v3 = filtered2.groupby("Category")["Value"].mean().reset_index()
    v4 = filtered2.groupby("Country")["Value"].mean().reset_index()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        st.bar_chart(v1.set_index("Country"), horizontal=True)
    with r1c2:
        st.bar_chart(v2.set_index("Region"), horizontal=True)
    with r2c1:
        st.bar_chart(v3.set_index("Category"))
    with r2c2:
        st.bar_chart(v4.set_index("Country"))

# -------- TAB 3 --------
with tab3:
    set_active_tab("Tab 3")
    filtered3 = filtered_global
    render_summary(filtered3)

    st.header("📈 Tab 3 Insights")
    b1 = filtered3.groupby("Country")["Value"].mean().reset_index()
    b2 = filtered3.groupby("Region")["Value"].mean().reset_index()
    b3 = filtered3.groupby("Region")["Value"].sum().reset_index()
    b4 = filtered3.groupby("Country")["Value"].sum().reset_index()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        st.bar_chart(b3.set_index("Region"), horizontal=True)
    with r1c2:
        st.bar_chart(b4.set_index("Country"), horizontal=True)
    with r2c1:
        st.bar_chart(b1.set_index("Country"))
    with r2c2:
        st.bar_chart(b2.set_index("Region"))

# -------- TAB 4 --------
with tab4:
    set_active_tab("Tab 4")
