import time
import streamlit as st


SESSION_TIMEOUT = 30 * 60

def enforce_session():
now = time.time()
last = st.session_state.get("last_activity", now)


if now - last > SESSION_TIMEOUT:
st.session_state.clear()
st.switch_page("pages/login.py")


st.session_state.last_activity = now
