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

filtered = df[
    (df["Country"].isin(country_filter)) &
    (df["Region"].isin(region_filter))
]

# -------- TAB-SPECIFIC SIDEBAR FOR TAB 2 --------
tab2_category_filter = None
if "tab_selection" not in st.session_state:
    st.session_state.tab_selection = "Overview"

def tab2_sidebar():
    st.sidebar.header("📌 Tab 2 Filters")
    return st.sidebar.multiselect("Category", df["Category"].unique(), default=df["Category"].unique())

# -------- TABS LAYOUT --------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tab 2", "Tab 3", "Tab 4"])

# Enable Tab-2 sidebar filters only when Tab 2 is active
with tab2:
    tab2_category_filter = tab2_sidebar()
    filtered_tab2 = filtered[filtered["Category"].isin(tab2_category_filter)]
    
    st.header("📊 Tab 2 Analysis")
    
    # ---- 4 Charts: 2 horizontal + 2 vertical ----
    v1 = filtered_tab2.groupby("Country")["Value"].sum().reset_index()
    v2 = filtered_tab2.groupby("Region")["Value"].sum().reset_index()
    v3 = filtered_tab2.groupby("Category")["Value"].mean().reset_index()
    v4 = filtered_tab2.groupby("Category")["Value"].sum().reset_index()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.bar_chart(v1.set_index("Country"), horizontal=True)
    with col2:
        st.bar_chart(v2.set_index("Region"), horizontal=True)
    with col3:
        st.bar_chart(v3.set_index("Category"))
    with col4:
        st.bar_chart(v4.set_index("Category"))

# -------- TAB 1 --------
with tab1:
    st.header("📌 Overview")
    
    a1 = filtered.groupby("Country")["Value"].mean().reset_index()
    a2 = filtered.groupby("Region")["Value"].mean().reset_index()
    a3 = filtered.groupby("Region")["Value"].sum().reset_index()
    a4 = filtered.groupby("Country")["Value"].sum().reset_index()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.bar_chart(a4.set_index("Country"), horizontal=True)
    with c2:
        st.bar_chart(a3.set_index("Region"), horizontal=True)
    with c3:
        st.bar_chart(a1.set_index("Country"))
    with c4:
        st.bar_chart(a2.set_index("Region"))

# -------- TAB 3 --------
with tab3:
    st.header("📈 Tab 3 Insights")

    b1 = filtered.groupby("Region")["Value"].sum().reset_index()
    b2 = filtered.groupby("Country")["Value"].mean().reset_index()
    b3 = filtered.groupby("Region")["Value"].mean().reset_index()
    b4 = filtered.groupby("Country")["Value"].sum().reset_index()

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.bar_chart(b1.set_index("Region"), horizontal=True)
    with p2:
        st.bar_chart(b4.set_index("Country"), horizontal=True)
    with p3:
        st.bar_chart(b2.set_index("Country"))
    with p4:
        st.bar_chart(b3.set_index("Region"))

# -------- TAB 4 --------
with tab4:
    st.header("📌 Tab 4 Summary")

    d1 = filtered.groupby("Category")["Value"].sum().reset_index()
    d2 = filtered.groupby("Category")["Value"].mean().reset_index()
    d3 = filtered.groupby("Country")["Value"].sum().reset_index()
    d4 = filtered.groupby("Region")["Value"].sum().reset_index()

    x1, x2, x3, x4 = st.columns(4)
    with x1:
        st.bar_chart(d3.set_index("Country"), horizontal=True)
    with x2:
        st.bar_chart(d4.set_index("Region"), horizontal=True)
    with x3:
        st.bar_chart(d1.set_index("Category"))
    with x4:
        st.bar_chart(d2.set_index("Category"))
