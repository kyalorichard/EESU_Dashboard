import streamlit as st
import pandas as pd
import numpy as np

# --- Load CSV ---
@st.cache_data
def load_csv(path="data.csv"):
    return pd.read_csv(path)

df = load_csv()

# --- Auto-Detect & Normalize Column Names ---
def detect_columns(df):
    col_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key == "country":
            col_map["country"] = col
        elif key == "region":
            col_map["region"] = col
        elif key == "category":
            col_map["category"] = col
        elif key == "value":
            col_map["value"] = col
    return col_map

cols = detect_columns(df)

# Ensure all expected columns exist
if len(cols) < 4:
    missing = {"country","region","category","value"} - set(cols.keys())
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# Rename dataframe columns to normalized names for internal use
df = df.rename(columns={
    cols["country"]: "Country",
    cols["region"]: "Region",
    cols["category"]: "Category",
    cols["value"]: "Value"
})

# --- Sidebar Filter Persistence ---
for key in ["country","region","category"]:
    if key not in st.session_state:
        st.session_state[key] = []

# --- Sidebar UI ---
st.sidebar.header("🌍 Filters (Auto-detected CSV)")

country_filter = st.sidebar.multiselect(
    "Country", df["Country"].dropna().unique(), default=st.session_state.country
)

region_filter = st.sidebar.multiselect(
    "Region", df["Region"].dropna().unique(), default=st.session_state.region
)

category_filter = st.sidebar.multiselect(
    "Category", df["Category"].dropna().unique(), default=st.session_state.category
)

# Save / Reset buttons
b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("💾 Save"):
        st.session_state.country = country_filter
        st.session_state.region = region_filter
        st.session_state.category = category_filter
        st.success("Saved!")

with b2:
    if st.button("🔁 Reset"):
        st.session_state.country = []
        st.session_state.region = []
        st.session_state.category = []
        st.rerun()

# --- Apply Filters ---
filtered_df = df.copy()

if country_filter:
    filtered_df = filtered_df[filtered_df["Country"].isin(country_filter)]
if region_filter:
    filtered_df = filtered_df[filtered_df["Region"].isin(region_filter)]
if category_filter:
    filtered_df = filtered_df[filtered_df["Category"].isin(category_filter)]

# --- Use filtered_df safely in your tabs ---
st.write("Filtered rows:", len(filtered_df))
st.dataframe(filtered_df.head(20), use_container_width=True)
