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
    flex-direction: col;  /* horizontal layout */
    justify-content: space-around; /* space evenly between metrics */
    align-items: center;
    width: 20%;
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 20px;
    font-family: Arial;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 20px;
}
.summary-card-horizontal:hover {
    transform: scale(1.02);
    box-shadow: 4px 4px 20px rgba(0,0,0,0.3);
}
.summary-item-horizontal {
    text-align: center;
    flex: 1; /* equal space for each metric */
}
@media (max-width: 768px) {
    .summary-card-horizontal {
        flex-direction: column; /* stack metrics vertically on small screens */
    }
    .summary-item-horizontal {
        margin: 5px 0;
    }
}
</style>
<div class="summary-card-horizontal">
""", unsafe_allow_html=True)

for title, value in summary_values:
    st.markdown(f"""
    <div class="summary-item-horizontal">
        <h4>{title}</h4>
        <h3>{value:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Three Bar Plots
# --------------------------
st.subheader("Bar Plots")
plots = [("Plot 1", df1_f), ("Plot 2", df2_f), ("Plot 3", df3_f)]

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

with st.container():
    cols = st.columns(len(plots), gap="small")
    for idx, (title, df_tab) in enumerate(plots):
        with cols[idx]:
            st.subheader(title)
            create_bar_plot(df_tab, title)

# --------------------------
# Map + Line Chart on the same row
# --------------------------
st.subheader("Map & Line Chart")
with st.container():
    col_map, col_line = st.columns([1,1], gap="medium")  # equal width
    # Map
    with col_map:
        agg_df = df_map_f.groupby("Country").agg({"Value1":"sum","Value2":"sum"}).reset_index()
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
        st.plotly_chart(fig_map, use_container_width=True)

    # Line Chart
    with col_line:
        if df_line_f.empty:
            st.warning("No data")
        else:
            line_chart = alt.Chart(df_line_f).mark_line(point=True).encode(
                x="Date:T",
                y="Value1:Q",
                color="Category:N",
                tooltip=['Date','Category','Value1']
            ).interactive()
            st.altair_chart(line_chart, use_container_width=True)
