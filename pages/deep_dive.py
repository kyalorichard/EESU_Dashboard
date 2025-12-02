import streamlit as st
import pandas as pd

st.title("🔍 Deep Dive Tab")

df = pd.read_csv("app/data/alerts.csv")

# Tab-specific filters
value_min, value_max = st.slider("Value Range", float(df["Value"].min()), float(df["Value"].max()), (float(df["Value"].min()), float(df["Value"].max())))
region_focus = st.selectbox("Focus Region", ["All"] + list(df["Region"].unique()))

deep_df = df[(df["Value"] >= value_min) & (df["Value"] <= value_max)]
if region_focus != "All":
    deep_df = deep_df[deep_df["Region"] == region_focus]

st.dataframe(deep_df.head(20))
