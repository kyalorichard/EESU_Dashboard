# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import json

# -------------------------------------------------
# Firebase Admin Initialization
# -------------------------------------------------
try:
    if "firebase_admin" in st.secrets and not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"⚠️ Firebase Admin initialization failed.\n{str(e)}")
    st.stop()

# -------------------------------------------------
# Pyrebase Initialization
# -------------------------------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))

if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.warning(f"⚠️ Firebase authentication service unavailable.\n{str(e)}")

# -------------------------------------------------
# Configuration
# -------------------------------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)
if not PRIVILEGED_DOMAINS:
    st.warning("⚠️ No privileged domains configured. Access will be blocked.")

ERROR_MAP = {
    "EMAIL_EXISTS": "This email is already registered.",
    "INVALID_PASSWORD": "Incorrect email or password.",
    "EMAIL_NOT_FOUND": "Email not registered.",
    "WEAK_PASSWORD": "Password must be at least 6 characters.",
    "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password."
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_email_domain(email: str) -> str:
    return email.strip().split("@")[-1].lower()

def parse_firebase_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        error_json = json.loads(payload)
        return error_json.get("error", {}).get("message", str(e))
    except Exception:
        return str(e)

def init_state():
    defaults = {
        "user": None,
        "email": "",
        "name": "",
        "user_role": None,
        "email_verified": False,
        "idToken": None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def logout_user():
    for key in ["user", "email", "name", "user_role", "email_verified", "idToken"]:
        st.session_state.pop(key, None)
    st.rerun()

def is_privileged() -> bool:
    return (
        st.session_state.get("user_role") == "privileged"
        and st.session_state.get("email_verified") is True
    )

def refresh_id_token():
    """Refresh Firebase ID token if expired"""
    try:
        if st.session_state.get("idToken"):
            refreshed = firebase_auth.refresh(st.session_state.idToken)
            st.session_state.idToken = refreshed["idToken"]
    except Exception:
        pass

# -------------------------------------------------
# Authentication UI
# -------------------------------------------------
def auth_ui():
    init_state()
    sidebar = st.sidebar
    sidebar.markdown("## Account")

    refresh_id_token()  # Refresh token on each UI load

    # -----------------------------
    # Logged-in View
    # -----------------------------
    if st.session_state.user:
        if st.session_state.email_verified:
            sidebar.success(f"👋 {st.session_state.name} ✅ Verified")
        else:
            sidebar.warning(f"👋 {st.session_state.name} ⚠️ Email not verified")
            sidebar.markdown("You must verify your email to access the dashboard.")

            if sidebar.button("Resend Verification Email"):
                try:
                    refreshed = firebase_auth.refresh(st.session_state.idToken)
                    st.session_state.idToken = refreshed["idToken"]
                    firebase_auth.send_email_verification(st.session_state.idToken)
                    sidebar.success("Verification email resent successfully.")
                except Exception:
                    sidebar.error("Unable to resend verification email. Try again later.")

            if sidebar.button("Forgot Password"):
                try:
                    firebase_auth.send_password_reset_email(st.session_state.email)
                    sidebar.success(f"Password reset email sent to {st.session_state.email}.")
                except Exception as e:
                    sidebar.error(f"Failed to send reset email: {parse_firebase_error(e)}")

        sidebar.button("Logout", on_click=logout_user)
        return

    # -----------------------------
    # Login / Register Tabs
    # -----------------------------
    tab_login, tab_register = sidebar.tabs(["Login", "Register"])

    # -----------------------------
    # LOGIN FORM
    # -----------------------------
    with tab_login:
        with sidebar.form("login_form", clear_on_submit=True):
            email = st.text_input("Email", key="login_email").strip()
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Sign in")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                    return

                domain = get_email_domain(email)
                if domain not in PRIVILEGED_DOMAINS:
                    st.error(f"Access restricted: {domain} is not an authorized domain.")
                    return

                try:
                    user = firebase_auth.sign_in_with_email_and_password(email, password)
                    id_token = user["idToken"]
                    user_info = firebase_auth.get_account_info(id_token)
                    email_verified = user_info["users"][0].get("emailVerified", False)

                    st.session_state.user = user
                    st.session_state.email = email
                    st.session_state.name = email.split("@")[0].title()
                    st.session_state.email_verified = email_verified
                    st.session_state.idToken = id_token
                    st.session_state.user_role = "privileged" if email_verified else "unverified"

                    if not email_verified:
                        st.warning("Please verify your email before accessing the dashboard.")
                        return

                    st.rerun()

                except Exception as e:
                    error_code = parse_firebase_error(e)
                    st.error(ERROR_MAP.get(error_code, f"Login failed: {error_code}"))

        if st.button("Forgot Password?"):
            if not email:
                st.warning("Please enter your email above to reset your password.")
            else:
                try:
                    firebase_auth.send_password_reset_email(email)
                    st.success(f"Password reset email sent to {email}.")
                except Exception as e:
                    st.error(f"Failed to send reset email: {parse_firebase_error(e)}")

    # -----------------------------
    # REGISTER FORM
    # -----------------------------
    with tab_register:
        with sidebar.form("register_form", clear_on_submit=True):
            email = st.text_input("Email", key="reg_email").strip()
            password = st.text_input("Password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Register")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                    return

                domain = get_email_domain(email)
                if domain not in PRIVILEGED_DOMAINS:
                    st.error(f"Registration restricted: {domain} is not an approved domain.")
                    return

                try:
                    user = firebase_auth.create_user_with_email_and_password(email, password)
                    firebase_auth.send_email_verification(user["idToken"])

                    st.success("Registration successful. Check your email to verify your account.")

                    # Auto-login (unverified)
                    st.session_state.user = user
                    st.session_state.email = email
                    st.session_state.name = email.split("@")[0].title()
                    st.session_state.email_verified = False
                    st.session_state.idToken = user["idToken"]
                    st.session_state.user_role = "unverified"

                except Exception as e:
                    error_code = parse_firebase_error(e)
                    st.error(ERROR_MAP.get(error_code, f"Registration failed: {error_code}"))
