import streamlit as st
import pandas as pd

st.title("📊 Analytics Tab")
df = pd.read_csv("app/data/alerts.csv")

st.subheader("Alerts by Country")
st.bar_chart(df["Country"].value_counts())
