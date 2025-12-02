import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# --------------------------
# Remove Streamlit padding/margin for full width
# --------------------------
st.markdown("""
<style>
    .block-container {
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }
    .stColumn > div {
        padding-left: 0rem;
        padding-right: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Sample Data
# --------------------------
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=10)
categories = ["A", "B", "C"]
tags = ["X", "Y", "Z"]
countries = ["Kenya", "Ethiopia", "Uganda", "Tanzania"]

def create_df():
    return pd.DataFrame({
        "Date": np.random.choice(dates, 10),
        "Category": np.random.choice(categories, 10),
        "Tag": np.random.choice(tags, 10),
        "Country": np.random.choice(countries, 10),
        "Value1": np.random.randint(0, 100, 10),
        "Value2": np.random.randint(0, 100, 10)
    })

df1 = create_df()
df2 = create_df()
df3 = create_df()
df_map = pd.concat([df1, df2, df3])
df_line = create_df()  # for line charts

# --------------------------
# Session State
# --------------------------
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "All"

# --------------------------
# Sidebar Filters
# --------------------------
st.sidebar.header("Global Filters")
selected_category = st.sidebar.selectbox("Category", ["All"] + categories)
selected_tags = st.sidebar.multiselect("Tags", tags, default=tags)
start_date, end_date = st.sidebar.date_input("Date Range", [df_map['Date'].min(), df_map['Date'].max()])
min_value, max_value = st.sidebar.slider("Value1 Range", 0, 100, (0,100))
if st.sidebar.button("Reset Filters"):
    st.session_state.selected_country = "All"

# --------------------------
# Filter Function
# --------------------------
def filter_data(df, country_filter):
    df_filtered = df.copy()
    if selected_category != "All":
        df_filtered = df_filtered[df_filtered["Category"]==selected_category]
    df_filtered = df_filtered[df_filtered["Tag"].isin(selected_tags)]
    df_filtered = df_filtered[(df_filtered["Date"] >= pd.to_datetime(start_date)) & (df_filtered["Date"] <= pd.to_datetime(end_date))]
    df_filtered = df_filtered[(df_filtered["Value1"] >= min_value) & (df_filtered["Value1"] <= max_value)]
    if country_filter != "All":
        df_filtered = df_filtered[df_filtered["Country"]==country_filter]
    return df_filtered

df1_f = filter_data(df1, st.session_state.selected_country)
df2_f = filter_data(df2, st.session_state.selected_country)
df3_f = filter_data(df3, st.session_state.selected_country)
df_line_f = filter_data(df_line, st.session_state.selected_country)
df_map_f = filter_data(df_map, st.session_state.selected_country)

# --------------------------
# Top Summary: Single card with all metrics horizontally
# --------------------------
summary_values = [
    ("Total Value1", df1_f['Value1'].sum()),
    ("Avg Value1", df1_f['Value1'].mean()),
    ("Total Value2", df2_f['Value2'].sum()),
    ("Avg Value2", df2_f['Value2'].mean()),
    ("Count Records", len(df_map_f))
]

st.markdown("""
<style>
.summary-card-horizontal {
    display: flex;
    flex-direction: row;  /* horizontal layout */
    justify-content: space-around; /* space evenly between metrics */
    align-items: center;
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 20px;
    font-family: Arial;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 20px;
}
.summary-card-horizonta
