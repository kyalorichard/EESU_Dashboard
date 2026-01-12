import streamlit as st
from auth.db import create_user
from auth.security import hash_password


st.set_page_config(page_title="Signup", layout="centered")


st.title("Request Access")


email = st.text_input("Email")
password = st.text_input("Password", type="password")


if st.button("Submit"):
try:
create_user(email, hash_password(password))
st.success("Account created. Verify email and await admin approval.")
except Exception:
st.error("Account already exists")
