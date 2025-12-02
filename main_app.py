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
        padding-left: 20px;
        padding-right: 20px;
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
# Top Summary: Equal size, spacing, one row at the top
# --------------------------
summary_values = [
    ("Total Value1", df_filtered[value1_col].sum()),
    ("Avg Value1", df_filtered[value1_col].mean()),
    ("Total Value2", df_filtered[value2_col].sum()),
    ("Avg Value2", df_filtered[value2_col].mean()),
    ("Count Records", len(df_filtered))
]

st.markdown("""
<style>
    .summary-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        width: 100%;
        margin-top: -10px;
    }
    .summary-card {
        flex: 1;
        background: linear-gradient(145deg, #e0e0e0, #ffffff);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.15), -3px -3px 8px rgba(255,255,255,0.7);
        transition: all 0.2s ease-in-out;
        height: 110px;
        min-width: 120px;
    }
    .summary-card:hover {
        transform: translateY(-4px);
        box-shadow: 5px 5px 15px rgba(0,0,0,0.25), -5px -5px 15px rgba(255,255,255,0.9);
    }
    .summary-title {
        font-size: 14px;
        margin-bottom: 5px;
        color: #444;
    }
    .summary-value {
        font-size: 22px;
        font-weight: bold;
        color: #2F4F4F;
    }
</style>
<div class="summary-row">
""", unsafe_allow_html=True)

for title, value in summary_values:
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">{title}</div>
        <div class="summary-value">{value:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Bar Plot Function with 3D/light background
# --------------------------
def create_bar_plot(df, title):
    if df.empty:
        st.warning(f"No data for {title}")
        return
    # Wrap chart in a div with rounded corners
    st.markdown("<div style='border-radius:30px; overflow:hidden; padding:20px; background-color:#f0f0f3;'>", unsafe_allow_html=True)
    chart = alt.Chart(df).mark_bar().encode(
        x='Date:T',
        y='Value1:Q',
        color='Category:N',
        tooltip=['Date','Category','Tag','Country','Value1','Value2']
    ).properties(
        height=600,
        background='#f0f0f3'
    ).configure_axis(
        grid=True,
        gridColor='#dcdcdc'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
# --------------------------

# Line Chart Function with 3D/light background
# --------------------------
def create_line_chart(df, title):
    if df.empty:
        st.warning("No data")
        return
    st.markdown("<div style='border-radius:30px; overflow:hidden; padding:20px; background-color:#f0f0f3;'>", unsafe_allow_html=True)
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="Date:T",
        y="Value1:Q",
        color="Category:N",
        tooltip=['Date','Category','Value1']
    ).properties(
        height=400,
        background='#f0f0f3'
    ).configure_axis(
        grid=True,
        gridColor='#dcdcdc'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Three Bar Plots
# --------------------------
st.subheader("Bar Plots")
plots = [("Plot 1", df1_f), ("Plot 2", df2_f), ("Plot 3", df3_f)]
with st.container():
    cols = st.columns(len(plots), gap="small")
    for idx, (title, df_tab) in enumerate(plots):
        with cols[idx]:
            st.subheader(title)
            create_bar_plot(df_tab, title)

# --------------------------
# Map + Line Chart on the same row (only countries with data)
# --------------------------
st.subheader("Map & Line Chart")
with st.container():
    col_map, col_line = st.columns([1,1], gap="medium")  # equal width

    # Map
    with col_map:
        agg_df = df_map_f.groupby("Country").agg({"Value1":"sum","Value2":"sum"}).reset_index()
        # Only keep countries with data
        agg_df = agg_df[(agg_df["Value1"] > 0) | (agg_df["Value2"] > 0)]
        agg_df["iso_alpha"] = agg_df["Country"].map({"Kenya":"KEN","Ethiopia":"ETH","Uganda":"UGA","Tanzania":"TZA"})

        fig_map = px.choropleth(
            agg_df,
            locations="iso_alpha",
            color="Value1",
            hover_name="Country",
            hover_data={"Value1":True,"Value2":True,"iso_alpha":False},
            color_continuous_scale="Viridis",
            scope="africa"
        )
        fig_map.update_layout(
            paper_bgcolor='#f0f0f3',
            plot_bgcolor='#f0f0f3'
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Line Chart
    with col_line:
        create_line_chart(df_line_f, "Line Chart")
