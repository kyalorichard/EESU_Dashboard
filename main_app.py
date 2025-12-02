import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# --------------------------
# Sample Data
# --------------------------
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=10)
categories = ["A", "B", "C"]
tags = ["X", "Y", "Z"]
countries = ["Kenya", "Ethiopia", "Uganda", "Tanzania"]
country_iso = {"Kenya":"KEN","Ethiopia":"ETH","Uganda":"UGA","Tanzania":"TZA"}

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
df4 = create_df()

# --------------------------
# Session State for Map Click
# --------------------------
if "map_click_country" not in st.session_state:
    st.session_state.map_click_country = "All"

# --------------------------
# Sidebar: Global Filters
# --------------------------
st.sidebar.header("Global Filters")
if st.sidebar.button("Reset Filters"):
    st.session_state.update({
        'selected_category': 'All',
        'selected_tags': tags,
        'selected_country': 'All',
        'start_date': df1['Date'].min(),
        'end_date': df1['Date'].max(),
        'min_value': 0,
        'max_value': 100,
        'map_click_country': 'All'
    })

for key, default in [
    ('selected_category', 'All'),
    ('selected_tags', tags),
    ('selected_country', 'All'),
    ('start_date', df1['Date'].min()),
    ('end_date', df1['Date'].max()),
    ('min_value', 0),
    ('max_value', 100)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Country filter will prioritize map click if any
selected_country = st.sidebar.selectbox(
    "Country",
    ["All"] + countries,
    index=0 if st.session_state.map_click_country == "All" else countries.index(st.session_state.map_click_country)+1
)

selected_category = st.sidebar.selectbox("Category", ["All"] + categories)
selected_tags = st.sidebar.multiselect("Tags", tags, default=st.session_state['selected_tags'])
start_date, end_date = st.sidebar.date_input("Date Range", [st.session_state['start_date'], st.session_state['end_date']])
min_value, max_value = st.sidebar.slider("Value1 Range", 0, 100, (st.session_state['min_value'], st.session_state['max_value']))

st.session_state.update({
    'selected_category': selected_category,
    'selected_tags': selected_tags,
    'selected_country': selected_country,
    'start_date': start_date,
    'end_date': end_date,
    'min_value': min_value,
    'max_value': max_value
})

# --------------------------
# Filter Function
# --------------------------
def filter_df(df):
    df_filtered = df.copy()
    if selected_category != "All":
        df_filtered = df_filtered[df_filtered["Category"] == selected_category]
    df_filtered = df_filtered[df_filtered["Tag"].isin(selected_tags)]
    if selected_country != "All":
        df_filtered = df_filtered[df_filtered["Country"] == selected_country]
    df_filtered = df_filtered[(df_filtered["Date"] >= pd.to_datetime(start_date)) & (df_filtered["Date"] <= pd.to_datetime(end_date))]
    df_filtered = df_filtered[(df_filtered["Value1"] >= min_value) & (df_filtered["Value1"] <= max_value)]
    return df_filtered

df1_f = filter_df(df1)
df2_f = filter_df(df2)
df3_f = filter_df(df3)
df4_f = filter_df(df4)

# --------------------------
# Styled Top Summary
# --------------------------
total_val = sum(df['Value1'].sum() for df in [df1_f, df2_f, df3_f, df4_f])
avg_val = np.mean([df['Value1'].mean() for df in [df1_f, df2_f, df3_f, df4_f]])

st.markdown(f"""
<div style='
    background-color: #4CAF50;
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-family: Arial;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
'>
    <h2 style='margin-bottom: 10px;'>Global Summary</h2>
    <h3>Total Value1: {total_val}</h3>
    <h3>Average Value1: {avg_val:.2f}</h3>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Functions
# --------------------------
def create_bar_plot(df, y="Value1", color="Category"):
    if df.empty:
        st.warning("No data for selected filters")
        return
    chart = alt.Chart(df).mark_bar().encode(
        x='Date:T',
        y=alt.Y(f'{y}:Q', title=y),
        color=color,
        tooltip=['Date','Category','Tag','Country','Value1','Value2']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def create_kpis(df):
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Value1 Sum", df["Value1"].sum())
    col2.metric("Value2 Sum", df["Value2"].sum())
    col3.metric("Avg Value1", round(df["Value1"].mean() if not df.empty else 0,2))
    col4.metric("Avg Value2", round(df["Value2"].mean() if not df.empty else 0,2))

# --------------------------
# Plotly Choropleth Map
# --------------------------
st.subheader("Interactive Country Map")

map_df = pd.concat([df1_f, df2_f, df3_f, df4_f])
if not map_df.empty:
    agg_df = map_df.groupby("Country").agg({
        "Value1": "sum",
        "Value2": "sum"
    }).reset_index()
    agg_df["iso_alpha"] = agg_df["Country"].map(country_iso)

    fig = px.choropleth(
        agg_df,
        locations="iso_alpha",
        color="Value1",
        hover_name="Country",
        hover_data={"Value1": True, "Value2": True, "iso_alpha": False},
        color_continuous_scale="Viridis",
        labels={"Value1": "Total Value1"},
        scope="africa"
    )

    # Capture click on map
    map_click = st.plotly_chart(fig, use_container_width=True)
    
    # Streamlit cannot directly capture Plotly click events,
    # but we can use a workaround using Plotly's selected data
    # in st.plotly_chart with 'click' events (requires Dash or custom JS).
    # For demonstration, we'll emulate selection via country dropdown update
else:
    st.warning("No data for selected filters to display on map.")

# --------------------------
# Responsive Side-by-Side Plots
# --------------------------
plots = [("Plot 1", df1_f), ("Plot 2", df2_f), ("Plot 3", df3_f), ("Plot 4", df4_f)]
cols = st.columns(4, gap="large")

for idx, (title, df_tab) in enumerate(plots):
    with cols[idx]:
        st.subheader(title)
        if title == "Plot 1":
            val2 = st.number_input("Value2 >", min_value=0, max_value=100, value=0, key='plot1_val2')
            df_filtered = df_tab[df_tab["Value2"] > val2]
        elif title == "Plot 2":
            val2 = st.number_input("Value2 <", min_value=0, max_value=100, value=100, key='plot2_val2')
            df_filtered = df_tab[df_tab["Value2"] < val2]
        elif title == "Plot 3":
            tags_sel = st.multiselect("Tags", options=tags, default=tags, key='plot3_tags')
            df_filtered = df_tab[df_tab["Tag"].isin(tags_sel)]
        else:
            start_tab, end_tab = st.date_input("Date Range", [df_tab["Date"].min(), df_tab["Date"].max()], key='plot4_date')
            df_filtered = df_tab[(df_tab["Date"] >= pd.to_datetime(start_tab)) & (df_tab["Date"] <= pd.to_datetime(end_tab))]

        create_kpis(df_filtered)
        create_bar_plot(df_filtered)
