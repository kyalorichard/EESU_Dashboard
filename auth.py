# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import json

# -------------------------------------------------
# Firebase Admin Initialization
# -------------------------------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# -------------------------------------------------
# Pyrebase Initialization
# -------------------------------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))

if firebase_cfg:
    firebase = pyrebase.initialize_app(firebase_cfg)
    firebase_auth = firebase.auth()

# -------------------------------------------------
# Configuration
# -------------------------------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower().strip()

def parse_firebase_error(e):
    try:
        error_json = json.loads(e.args[1])
        return error_json["error"]["message"]
    except Exception:
        return str(e)

def init_state():
    defaults = {
        "user": None,
        "email": "",
        "name": "",
        "user_role": None,
        "email_verified": False,
        "idToken": None,
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# -------------------------------------------------
# Logout
# -------------------------------------------------
def logout_user():
    for key in [
        "user",
        "email",
        "name",
        "user_role",
        "email_verified",
        "idToken",
    ]:
        st.session_state.pop(key, None)

    st.rerun()

# -------------------------------------------------
# Privilege Check
# -------------------------------------------------
def is_privileged():
    return (
        st.session_state.get("user_role") == "privileged"
        and st.session_state.get("email_verified") is True
    )

# -------------------------------------------------
# Sidebar UI
# -------------------------------------------------
def auth_ui():
    init_state()
    sidebar = st.sidebar
    sidebar.markdown("## Account")

    # -----------------------------
    # Logged-in View
    # -----------------------------
    if st.session_state.user:

        if st.session_state.email_verified:
            sidebar.success(f"👋 {st.session_state.name} ✅ Verified")
        else:
            sidebar.warning(f"👋 {st.session_state.name} ⚠️ Email not verified")

            if sidebar.button("Resend Verification Email"):
                try:
                    refreshed = firebase_auth.refresh(
                        st.session_state.idToken
                    )
                    st.session_state.idToken = refreshed["idToken"]

                    firebase_auth.send_email_verification(
                        st.session_state.idToken
                    )

                    sidebar.success("Verification email resent.")
                except Exception:
                    sidebar.error("Unable to resend verification email.")

        sidebar.button("Logout", on_click=logout_user)
        return

    # -----------------------------
    # Login / Register Tabs
    # -----------------------------
    tab1, tab2 = sidebar.tabs(["Login", "Register"])

    # -----------------------------
    # LOGIN FORM
    # -----------------------------
    with tab1:
        with sidebar.form("login_form", clear_on_submit=False):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")

            if submitted:

                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                    return

                try:
                    user = firebase_auth.sign_in_with_email_and_password(
                        email, password
                    )

                    id_token = user["idToken"]
                    user_info = firebase_auth.get_account_info(id_token)

                    email_verified = user_info["users"][0].get(
                        "emailVerified", False
                    )

                    domain = get_email_domain(email)

                    if domain not in PRIVILEGED_DOMAINS:
                        st.error("Access restricted: unauthorized domain.")
                        return

                    st.session_state.user = user
                    st.session_state.email = email
                    st.session_state.name = email.split("@")[0].title()
                    st.session_state.email_verified = email_verified
                    st.session_state.idToken = id_token

                    if not email_verified:
                        st.warning(
                            "Please verify your email before accessing the dashboard."
                        )
                        st.session_state.user_role = "unverified"
                        return

                    st.session_state.user_role = "privileged"
                    st.rerun()

                except Exception as e:
                    error_code = parse_firebase_error(e)

                    if error_code in [
                        "INVALID_PASSWORD",
                        "EMAIL_NOT_FOUND",
                        "INVALID_LOGIN_CREDENTIALS",
                    ]:
                        st.error("Incorrect email or password.")
                    else:
                        st.error("Login failed. Please try again.")

    # -----------------------------
    # REGISTER FORM
    # -----------------------------
    with tab2:
        with sidebar.form("register_form", clear_on_submit=False):
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Register")

            if submitted:

                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                    return

                domain = get_email_domain(email)

                if domain not in PRIVILEGED_DOMAINS:
                    st.error("Registration restricted to approved domains.")
                    return

                try:
                    user = firebase_auth.create_user_with_email_and_password(
                        email, password
                    )

                    firebase_auth.send_email_verification(user["idToken"])

                    st.success(
                        "Registration successful. Check your email to verify your account."
                    )

                except Exception as e:
                    error_code = parse_firebase_error(e)

                    if error_code == "EMAIL_EXISTS":
                        st.error("This email is already registered.")
                    elif error_code == "WEAK_PASSWORD":
                        st.error("Password must be at least 6 characters.")
                    else:
                        st.error("Registration failed. Please try again.")
