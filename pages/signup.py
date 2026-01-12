import streamlit as st
from auth.db import list_users, update_user


st.set_page_config(page_title="User Management", layout="wide")


if st.session_state.get("user_role") != "admin":
st.error("Admin access only")
st.stop()


st.title("User Management")


for uid, email, role, org_id, active in list_users():
c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
c1.write(email)
role_new = c2.selectbox("Role", ["admin", "analyst", "viewer"], index=["admin", "analyst", "viewer"].index(role), key=f"r{uid}")
org_new = c3.number_input("Org ID", value=org_id or 0, key=f"o{uid}")
active_new = c4.checkbox("Active", value=bool(active), key=f"a{uid}")


if st.button("Update", key=f"u{uid}"):
update_user(uid, role_new, org_new, int(active_new))
st.experimental_rerun()
