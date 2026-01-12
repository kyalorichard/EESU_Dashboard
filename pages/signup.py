import streamlit as st
from auth.auth_db import create_user
from auth.security import hash_password

st.set_page_config(page_title="Request Access", layout="centered")

st.title("Request Dashboard Access")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Submit Request"):
    try:
        create_user(email, hash_password(password))
        st.success("Account created. Await admin approval.")
    except:
        st.error("Account already exists.")
