import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

# Reset filters button
if st.sidebar.button("Reset Filters"):
    st.session_state['selected_category'] = "All"
    st.session_state['selected_tags'] = tags
    st.session_state['start_date'] = df1["Date"].min()
    st.session_state['end_date'] = df1["Date"].max()
    st.session_state['min_value'] = 0
    st.session_state['max_value'] = 100

# Initialize session state if not set
if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = "All"
if 'selected_tags' not in st.session_state:
    st.session_state['selected_tags'] = tags
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = df1["Date"].min()
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = df1["Date"].max()
if 'min_value' not in st.session_state:
    st.session_state['min_value'] = 0
if 'max_value' not in st.session_state:
    st.session_state['max_value'] = 100

# Global filters
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + categories, index=["All"] + categories.index(st.session_state['selected_category']) if st.session_state['selected_category'] != "All" else 0)
selected_tags = st.sidebar.multiselect("Select Tags", options=tags, default=st.session_state['selected_tags'])
start_date, end_date = st.sidebar.date_input("Select Date Range", [st.session_state['start_date'], st.session_state['end_date']])
min_value, max_value = st.sidebar.slider("Select Value1 Range", 0, 100, (st.session_state['min_value'], st.session_state['max_value']))

# Update session state
st.session_state['selected_category'] = selected_category
st.session_state['selected_tags'] = selected_tags
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date
st.session_state['min_value'] = min_value
st.session_state['max_value'] = max_value

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
# Main Page: Tabs with Bar Plots
# --------------------------
st.title("Dashboard with Global Filters and Bar Plots")

tab1, tab2, tab3, tab4 = st.tabs(["Plot 1", "Plot 2", "Plot 3", "Plot 4"])

def create_bar_plot(df, y_value="Value1", color_value="Category"):
    if df.empty:
        st.warning("No data for the selected filters.")
        return
    chart = alt.Chart(df).mark_bar().encode(
        x='Date:T',
        y=alt.Y(f'{y_value}:Q', title=y_value),
        color=color_value,
        tooltip=['Date', 'Category', 'Tag', 'Value1', 'Value2']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

with tab1:
    st.subheader("Plot 1")
    create_bar_plot(df1_filtered)

with tab2:
    st.subheader("Plot 2")
    create_bar_plot(df2_filtered)

with tab3:
    st.subheader("Plot 3")
    create_bar_plot(df3_filtered)

with tab4:
    st.subheader("Plot 4")
    create_bar_plot(df4_filtered)
