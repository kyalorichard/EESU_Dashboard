import streamlit as st
from auth.db import get_user_by_email, log_event
from auth.security import verify_password

st.set_page_config(page_title="Login", layout="centered")

st.title("EU SEE Dashboard – Login")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):
user = get_user_by_email(email)
if not user:
st.error("Invalid credentials")
elif not user[5] or not user[6]:
st.error("Account inactive or email not verified")
elif verify_password(password, user[2]):
st.session_state.update({
"authenticated": True,
"user_id": user[0],
"user_email": user[1],
"user_role": user[3],
"org_id": user[4],
})
log_event(user[0], "LOGIN_SUCCESS")
st.switch_page("app.py")
else:
log_event(user[0], "LOGIN_FAILED")
st.error("Invalid credentials")
