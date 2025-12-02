import streamlit as st
import pandas as pd
import numpy as np

# --------------------------
# Sample Data
# --------------------------
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=10)
categories = ["A", "B", "C"]
tags = ["X", "Y", "Z"]

def create_df():
    return pd.DataFrame({
        "Date": np.random.choice(dates, 10),
        "Category": np.random.choice(categories, 10),
        "Tag": np.random.choice(tags, 10),
        "Value1": np.random.randint(0, 100, 10),
        "Value2": np.random.randint(0, 100, 10)
    })

df1 = create_df()
df2 = create_df()
df3 = create_df()
df4 = create_df()

# --------------------------
# Sidebar: Global Filters
# --------------------------
st.sidebar.header("Global Filters")

# Dropdown
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + categories)

# Multi-select
selected_tags = st.sidebar.multiselect("Select Tags", options=tags, default=tags)

# Date range
start_date, end_date = st.sidebar.date_input("Select Date Range", [df1["Date"].min(), df1["Date"].max()])

# Numeric slider
min_value, max_value = st.sidebar.slider("Select Value1 Range", 0, 100, (0, 100))

# --------------------------
# Filter Function
# --------------------------
def filter_df(df):
    filtered = df.copy()
    if selected_category != "All":
        filtered = filtered[filtered["Category"] == selected_category]
    filtered = filtered[filtered["Tag"].isin(selected_tags)]
    filtered = filtered[(filtered["Date"] >= pd.to_datetime(start_date)) & (filtered["Date"] <= pd.to_datetime(end_date))]
    filtered = filtered[(filtered["Value1"] >= min_value) & (filtered["Value1"] <= max_value)]
    return filtered

df1_filtered = filter_df(df1)
df2_filtered = filter_df(df2)
df3_filtered = filter_df(df3)
df4_filtered = filter_df(df4)

# --------------------------
# Main Page: Tabs and Tab-Specific Filters
# --------------------------
st.title("Dashboard with Global and Tab-Specific Filters")

tab1, tab2, tab3, tab4 = st.tabs(["Table 1", "Table 2", "Table 3", "Table 4"])

# --- Tab 1 ---
with tab1:
    st.subheader("Table 1")
    # Tab-specific filter example
    val1_filter = st.number_input("Filter Value2 > ", min_value=0, max_value=100, value=0)
    df_tab1 = df1_filtered[df1_filtered["Value2"] > val1_filter]
    st.dataframe(df_tab1)

# --- Tab 2 ---
with tab2:
    st.subheader("Table 2")
    val2_filter = st.number_input("Filter Value2 < ", min_value=0, max_value=100, value=100)
    df_tab2 = df2_filtered[df2_filtered["Value2"] < val2_filter]
    st.dataframe(df_tab2)

# --- Tab 3 ---
with tab3:
    st.subheader("Table 3")
    # Example multi-select filter specific to Tab 3
    tag_tab3 = st.multiselect("Filter by Tag (Tab 3)", options=tags, default=tags)
    df_tab3 = df3_filtered[df3_filtered["Tag"].isin(tag_tab3)]
    st.dataframe(df_tab3)

# --- Tab 4 ---
with tab4:
    st.subheader("Table 4")
    # Example date filter specific to Tab 4
    start_tab4, end_tab4 = st.date_input("Tab 4 Date Range", [df4_filtered["Date"].min(), df4_filtered["Date"].max()])
    df_tab4 = df4_filtered[(df4_filtered["Date"] >= pd.to_datetime(start_tab4)) & (df4_filtered["Date"] <= pd.to_datetime(end_tab4))]
    st.dataframe(df_tab4)
