# auth.py (persistent login with refreshToken)

import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import json

# ---------------------------
# Firebase Admin Initialization
# ---------------------------
try:
    if "firebase_admin" in st.secrets and not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"⚠️ Firebase Admin initialization failed: {repr(e)}")
    st.stop()

# ---------------------------
# Pyrebase Initialization
# ---------------------------
firebase_cfg = dict(st.secrets.get("firebase", {}))
firebase_auth = None
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.warning(f"⚠️ Firebase authentication unavailable: {repr(e)}")

# ---------------------------
# Configuration
# ---------------------------
PRIVILEGED_DOMAINS = set(
    d.lower().lstrip("www.") for d in st.secrets.get("access", {}).get("privileged_domains", [])
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

# ---------------------------
# Helpers
# ---------------------------
def get_email_domain(email: str) -> str:
    return email.strip().split("@")[-1].lower().lstrip("www.")

def parse_firebase_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        if isinstance(payload, str):
            error_json = json.loads(payload)
            return error_json.get("error", {}).get("message", str(e))
        return str(e)
    except Exception:
        return str(e)

def format_name(email: str) -> str:
    return email.split("@")[0].replace(".", " ").title()

def init_state():
    defaults = {
        "email": "",
        "name": "",
        "user_role": None,
        "email_verified": False,
        "idToken": None,
        "refreshToken": None,
        "auth_tab": "Login",
        "forgot_email_sent": False
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def logout_user():
    for key in ["email", "name", "user_role", "email_verified", "idToken", "refreshToken"]:
        st.session_state.pop(key, None)
    st.rerun()

def is_privileged() -> bool:
    return st.session_state.get("user_role") == "privileged" and st.session_state.get("email_verified")

# ---------------------------
# Session Restore / Token Refresh
# ---------------------------
def restore_session():
    """Restore session on page reload using refreshToken"""
    if st.session_state.get("refreshToken") and firebase_auth:
        try:
            refreshed = firebase_auth.refresh(st.session_state.refreshToken)
            st.session_state.idToken = refreshed["idToken"]
            st.session_state.refreshToken = refreshed["refreshToken"]
        except Exception:
            logout_user()

# ---------------------------
# Authentication UI
# ---------------------------
def auth_ui():
    init_state()
    restore_session()
    sidebar = st.sidebar

    # -----------------------------
    # Logged-in View
    # -----------------------------
    if st.session_state.get("idToken"):
        sidebar.success(
            f"👋 {st.session_state.name} {'✅ Verified' if st.session_state.email_verified else '⚠️ Not verified'}"
        )

        if not st.session_state.email_verified:
            sidebar.markdown("Please verify your email to access the dashboard.")

            if sidebar.button("Resend Verification Email"):
                try:
                    firebase_auth.send_email_verification(st.session_state.idToken)
                    sidebar.success("Verification email resent successfully.")
                except Exception:
                    sidebar.error("Unable to resend verification email.")

            if sidebar.button("Forgot Password"):
                try:
                    firebase_auth.send_password_reset_email(st.session_state.email)
                    sidebar.success(f"Password reset email sent to {st.session_state.email}.")
                    st.session_state.forgot_email_sent = True
                except Exception as e:
                    sidebar.error(f"Failed to send reset email: {parse_firebase_error(e)}")

        sidebar.button("Logout", on_click=logout_user)
        return

    # -----------------------------
    # Tabs: Login / Register
    # -----------------------------
    tab_choice = sidebar.radio(
        "Select Action",
        ["Login", "Register"],
        index=0 if st.session_state.auth_tab == "Login" else 1,
        key="auth_tab_radio"
    )
    st.session_state.auth_tab = tab_choice

    # -----------------------------
    # LOGIN FORM
    # -----------------------------
    if tab_choice == "Login":
        with sidebar.form("login_form", clear_on_submit=True):
            email = st.text_input("Email", key="login_email").strip()
            password = st.text_input("Password", type="password", key="login_pass")

            submitted = st.form_submit_button("Sign in")
            forgot_pass = st.form_submit_button("Forgot Password?")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                elif get_email_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access restricted to privileged domains.")
                else:
                    try:
                        user = firebase_auth.sign_in_with_email_and_password(email, password)
                        id_token = user["idToken"]
                        refresh_token = user["refreshToken"]
                        info = firebase_auth.get_account_info(id_token)
                        verified = info["users"][0].get("emailVerified", False)

                        # Save minimal session info
                        st.session_state.idToken = id_token
                        st.session_state.refreshToken = refresh_token
                        st.session_state.email = email
                        st.session_state.name = format_name(email)
                        st.session_state.email_verified = verified
                        st.session_state.user_role = "privileged" if verified else "unverified"

                        if not verified:
                            st.warning("Please verify your email before accessing the dashboard.")
                        st.rerun()
                    except Exception as e:
                        st.error(ERROR_MAP.get(parse_firebase_error(e), f"Login failed: {parse_firebase_error(e)}"))

            if forgot_pass:
                if not email:
                    st.warning("Enter your email to reset your password.")
                else:
                    try:
                        firebase_auth.send_password_reset_email(email)
                        st.success(f"Password reset email sent to {email}.")
                        st.session_state.forgot_email_sent = True
                    except Exception as e:
                        st.error(f"Failed to send reset email: {parse_firebase_error(e)}")

        if st.session_state.forgot_email_sent:
            if st.button("Back to Login"):
                st.session_state.forgot_email_sent = False
                st.session_state.login_email = ""
                st.session_state.login_pass = ""
                st.rerun()

    # -----------------------------
    # REGISTER FORM
    # -----------------------------
    if tab_choice == "Register":
        with sidebar.form("register_form", clear_on_submit=True):
            email = st.text_input("Email", key="reg_email").strip()
            password = st.text_input("Password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Register")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                elif get_email_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Registration restricted to privileged domains.")
                else:
                    try:
                        user = firebase_auth.create_user_with_email_and_password(email, password)
                        firebase_auth.send_email_verification(user["idToken"])

                        # Save minimal session info
                        st.session_state.idToken = user["idToken"]
                        st.session_state.refreshToken = user["refreshToken"]
                        st.session_state.email = email
                        st.session_state.name = format_name(email)
                        st.session_state.email_verified = False
                        st.session_state.user_role = "unverified"

                        st.success("Registration successful. Verify your email to login.")
                    except Exception as e:
                        st.error(ERROR_MAP.get(parse_firebase_error(e), f"Registration failed: {parse_firebase_error(e)}"))
