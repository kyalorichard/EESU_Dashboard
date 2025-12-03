import streamlit as st
import pandas as pd
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Background for dashboard */
body, .css-18e3th9 {
    background: linear-gradient(145deg, #e0f0ff, #ffffff);
}

/* Summary cards */
.summary-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 10px;
}

.summary-number {
    font-size: 28px;
    font-weight: bold;
    color: #1f77b4;
}

.summary-label {
    font-size: 16px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
# Replace this with your actual data
@st.cache_data
def load_data():
    np.random.seed(0)
    data = pd.DataFrame({
        'Country': np.random.choice(['USA', 'UK', 'Germany', 'Kenya'], 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Value': np.random.randint(10, 100, 100)
    })
    return data

df = load_data()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect("Country", options=df['Country'].unique(), default=df['Country'].unique())
category_filter = st.sidebar.multiselect("Category", options=df['Category'].unique(), default=df['Category'].unique())

# Apply filters
filtered_data = df[(df['Country'].isin(country_filter)) & (df['Category'].isin(category_filter))]

# ---------- Summary Cards ----------
total_value = filtered_data['Value'].sum()
avg_value = filtered_data['Value'].mean()
max_value = filtered_data['Value'].max()
min_value = filtered_data['Value'].min()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="summary-card"><div class="summary-number">{total_value}</div><div class="summary-label">Total Value</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="summary-card"><div class="summary-number">{avg_value:.2f}</div><div class="summary-label">Average Value</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="summary-card"><div class="summary-number">{max_value}</div><div class="summary-label">Max Value</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="summary-card"><div class="summary-number">{min_value}</div><div class="summary-label">Min Value</div></div>', unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tab 2", "Tab 3", "Tab 4"])

with tab1:
    st.header("Overview")
    st.write("This tab can show main charts and metrics.")
    # Example placeholder chart
    st.bar_chart(filtered_data.groupby('Country')['Value'].sum())

with tab2:
    st.header("Tab 2")
    st.write("Content for Tab 2")
    st.bar_chart(filtered_data.groupby('Category')['Value'].sum())

with tab3:
    st.header("Tab 3")
    st.write("Content for Tab 3")
    st.line_chart(filtered_data.groupby('Country')['Value'].mean())

with tab4:
    st.header("Tab 4")
    st.write("Content for Tab 4")
    st.area_chart(filtered_data.groupby('Category')['Value'].sum())
