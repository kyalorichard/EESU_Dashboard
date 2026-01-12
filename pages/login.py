import streamlit as st
from auth.auth_db import get_user
from auth.security import verify_password

st.set_page_config(page_title="Login", layout="centered")

st.title("EU SEE Dashboard – Login")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):
    user = get_user(email)
    if user and verify_password(password, user[2]) and user[4] == 1:
        st.session_state.authenticated = True
        st.session_state.user_email = user[1]
        st.session_state.user_role = user[3]
        st.rerun()
    else:
        st.error("Invalid credentials or inactive account")
