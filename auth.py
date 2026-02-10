# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials

# ---------------------------
# Firebase Admin Init
# ---------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# ---------------------------
# Pyrebase Init
# ---------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    firebase = pyrebase.initialize_app(firebase_cfg)
    firebase_auth = firebase.auth()

# ---------------------------
# Config
# ---------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))

# ---------------------------
# Helpers
# ---------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def init_state():
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("email", "")
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("user_role", "")
    st.session_state.setdefault("login_error", "")

# ---------------------------
# Logout
# ---------------------------
def logout_user():
    for k in ["user", "email", "name", "user_role"]:
        st.session_state.pop(k, None)
    # Streamlit reruns automatically on button click

# ---------------------------
# Sidebar Auth UI
# ---------------------------
def auth_ui():
    init_state()

    sidebar = st.sidebar

    # Logged in view
    if st.session_state.user:
        sidebar.success(f"👋 {st.session_state.name}")
        sidebar.button("Logout", on_click=logout_user)
        return  # Stop rendering login inputs

    # Logged out view
    email_input = sidebar.text_input("Email", key="email_input")
    password_input = sidebar.text_input("Password", type="password", key="password_input")

    if sidebar.button("Sign in"):
        email = email_input
        password = password_input

        # Domain check first
        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.session_state.login_error = "Access denied: your email domain is not allowed."
        elif not firebase_auth:
            st.session_state.login_error = "Firebase not initialized. Please contact admin."
        else:
            try:
                # Firebase login attempt
                firebase_auth.sign_in_with_email_and_password(email, password)
                # Login successful
                st.session_state.user = "email"
                st.session_state.email = email
                st.session_state.name = email.split("@")[0].title()
                st.session_state.user_role = "privileged"
                st.session_state.login_error = ""
            except Exception as e:
                err_str = str(e)
                # Friendly differentiation
                if "INVALID_PASSWORD" in err_str or "EMAIL_NOT_FOUND" in err_str or "INVALID_LOGIN_CREDENTIALS" in err_str:
                    st.session_state.login_error = "Invalid email or password. Please try again."
                else:
                    st.session_state.login_error = f"Login failed: {e}"

    if st.session_state.login_error:
        sidebar.error(st.session_state.login_error)

# ---------------------------
# Helper for privileged content
# ---------------------------
def is_privileged():
    return st.session_state.get("user_role") == "privileged"
