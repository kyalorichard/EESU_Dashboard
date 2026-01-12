import streamlit as st
from auth.auth_db import list_users, update_user

st.title("User Management")

if st.session_state.user_role != "admin":
    st.error("Admin access only")
    st.stop()

users = list_users()

for uid, email, role, active in users:
    col1, col2, col3 = st.columns([3,2,2])
    col1.write(email)
    new_role = col2.selectbox("Role", ["admin","analyst","viewer"], index=["admin","analyst","viewer"].index(role))
    new_active = col3.checkbox("Active", value=bool(active))
    if st.button("Update", key=f"u{uid}"):
        update_user(uid, new_role, int(new_active))
        st.rerun()
