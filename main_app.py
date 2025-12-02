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
all_data = pd.concat([df1, df2, df3, df4])

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
start_date, end_date = st.sidebar.date_input("Date Range", [all_data['Date'].min(), all_data['Date'].max()])
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

# --------------------------
# Top Summary Card
# --------------------------
def show_summary(df_list):
    total_val = sum(df['Value1'].sum() for df in df_list)
    avg_val = np.mean([df['Value1'].mean() for df in df_list])
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
        <h2>Global Summary</h2>
        <h3>Total Value1: {total_val}</h3>
        <h3>Average Value1: {avg_val:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Interactive Country Map
# --------------------------
st.subheader("Interactive Country Map")

map_df = filter_data(all_data, st.session_state.selected_country)
agg_df = map_df.groupby("Country").agg({"Value1":"sum","Value2":"sum"}).reset_index()
agg_df["iso_alpha"] = agg_df["Country"].map(country_iso)

fig = px.choropleth(
    agg_df,
    locations="iso_alpha",
    color="Value1",
    hover_name="Country",
    hover_data={"Value1":True, "Value2":True, "iso_alpha":False},
    color_continuous_scale="Viridis",
    scope="africa"
)

map_click = st.plotly_chart(fig, use_container_width=True)

# Capture click event
if map_click is not None:
    click_data = st.session_state.get("click_data")
    if click_data is None:
        st.session_state.click_data = {}
    # Check for clicked country
    if map_click.json:
        try:
            clicked_country = map_click.json["props"]["figure"]["layout"]["clickData"]["points"][0]["location"]
            for name, iso in country_iso.items():
                if iso == clicked_country:
                    st.session_state.selected_country = name
        except:
            pass

# --------------------------
# Responsive Bar Plots
# --------------------------
df1_f = filter_data(df1, st.session_state.selected_country)
df2_f = filter_data(df2, st.session_state.selected_country)
df3_f = filter_data(df3, st.session_state.selected_country)
df4_f = filter_data(df4, st.session_state.selected_country)

show_summary([df1_f, df2_f, df3_f, df4_f])

def create_bar_plot(df, title):
    if df.empty:
        st.warning(f"No data for {title}")
        return
    chart = alt.Chart(df).mark_bar().encode(
        x='Date:T',
        y='Value1:Q',
        color='Category:N',
        tooltip=['Date','Category','Tag','Country','Value1','Value2']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

cols = st.columns(4, gap="large")
plots = [("Plot 1", df1_f), ("Plot 2", df2_f), ("Plot 3", df3_f), ("Plot 4", df4_f)]
for idx, (title, df_tab) in enumerate(plots):
    with cols[idx]:
        st.subheader(title)
        create_bar_plot(df_tab, title)
