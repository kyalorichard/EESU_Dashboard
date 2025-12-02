import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import time as t
# --------------------------
# CUSTOM CSS: Sidebar / Menu Font Sizes
# --------------------------
st.markdown("""
<style>
/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] {
    font-size: 30px !important;  /* adjust sidebar menu size */
}

/* Selectbox & multiselect font size */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    font-size: 18px !important;
}

/* Slider label font */
.stSlider label {
    font-size: 16px !important;
}

/* Sidebar section header */
.sidebar-header {
    font-size: 30px !important;
    font-weight: bold;
}

/* Top toolbar menu */
header div[data-testid="stToolbar"] {
    font-size: 30px !important;
}
</style>
""", unsafe_allow_html=True)

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
</style>
""", unsafe_allow_html=True)

#------------------------------
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
# Top Summary: Single card with all metrics horizontally
# --------------------------

summary_values = [
    ("Total Value1", df1_f['Value1'].sum(), "📊"),
    ("Avg Value1", df1_f['Value1'].mean(), "📈"),
    ("Total Value2", df2_f['Value2'].sum(), "💰"),
    ("Avg Value2", df2_f['Value2'].mean(), "📉"),
    ("Count Records", len(df_map_f), "🧾")
]


# sticky summary row CSS
st.markdown("""
<style>
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.metric-number {
  font-size: 26px;
  font-weight: bold;
}
.metric-label {
  font-size: 14px;
  color: #333;
  margin-top: 4px;
}
.metric-icon {
  font-size: 20px;
  margin-bottom: 4px;
}
.summary-row {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    justify-content: space-between;
    gap: 14px;
    width: 120px;
    padding: 12px 0 18px 0;
    background: #f5f7fa;
}
.metric-card {
    flex: 1;
    border-radius: 16px;
    padding: 12px;
    text-align: center;
    background: linear-gradient(145deg, #ffffff, #e6ebf1);
    box-shadow: 4px 4px 10px rgba(0,0,0,0.08), -4px -4px 10px rgba(255,255,255,0.8);
    animation: fadeInUp 0.4s ease-out forwards;
    height: 120px;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 6px 6px 18px rgba(0,0,0,0.14), -6px -6px 18px rgba(255,255,255,0.9);
}
</style>

<div class="summary-row">
""", unsafe_allow_html=True)

# number animation using placeholder increments
for i, (label, val, icon) in enumerate(summary_values):
    num_placeholder = st.empty()
    with num_placeholder:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-number metric-number-{i} metric-number">0.00</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Animate number count-up
    for v in np.linspace(0, val, 12):
        num_placeholder.markdown(
            f"""
            <script>
                document.querySelector('.metric-number-{i}').innerText = '{v:.2f}';
            </script>
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-number-{i} metric-number">{v:.2f}</div>
            """,
            unsafe_allow_html=True
        )
        t.sleep(0.05)

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
