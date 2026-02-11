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
        "login_error": "",
        "register_error": "",
        "register_success": "",
        "email_verified": False,
        "idToken": None,
        "resend_success": "",
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# -------------------------------------------------
# Logout
# -------------------------------------------------
def logout_user():
    keys = [
        "user",
        "email",
        "name",
        "user_role",
        "email_verified",
        "idToken",
        "resend_success",
    ]
    for k in keys:
        st.session_state.pop(k, None)

    st.rerun()

# -------------------------------------------------
# Authorization Check
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

    # --------------------------------------------
    # Logged-in View
    # --------------------------------------------
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

                    st.session_state.resend_success = (
                        "Verification email resent. Check your inbox."
                    )
                except Exception:
                    st.session_state.resend_success = (
                        "Unable to resend verification email."
                    )

            if st.session_state.resend_success:
                sidebar.info(st.session_state.resend_success)

        sidebar.button("Logout", on_click=logout_user)
        return

    # --------------------------------------------
    # Login / Register Tabs
    # --------------------------------------------
    action = sidebar.radio("Select action:", ["Login", "Register"])

    if action == "Login":
        login_ui(sidebar)
    else:
        register_ui(sidebar)

# -------------------------------------------------
# Login
# -------------------------------------------------
def login_ui(sidebar):
    email = sidebar.text_input("Email", key="login_email")
    password = sidebar.text_input(
        "Password", type="password", key="login_password"
    )

    if sidebar.button("Sign in"):

        st.session_state.login_error = ""

        if not firebase_auth:
            st.session_state.login_error = "Authentication service unavailable."
            return

        try:
            user = firebase_auth.sign_in_with_email_and_password(
                email, password
            )

            st.session_state.idToken = user["idToken"]

            user_info = firebase_auth.get_account_info(
                user["idToken"]
            )

            email_verified = user_info["users"][0].get(
                "emailVerified", False
            )

            domain = get_email_domain(email)

            # Domain restriction
            if domain not in PRIVILEGED_DOMAINS:
                st.session_state.login_error = (
                    "Access restricted: your email domain is not authorized."
                )
                return

            # Set session user
            st.session_state.user = user
            st.session_state.email = email
            st.session_state.name = email.split("@")[0].title()
            st.session_state.email_verified = email_verified

            if not email_verified:
                st.session_state.user_role = "unverified"
                st.session_state.login_error = (
                    "Please verify your email before accessing the dashboard."
                )
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
                st.session_state.login_error = "Incorrect email or password."
            else:
                st.session_state.login_error = "Login failed. Please try again."

    if st.session_state.login_error:
        sidebar.error(st.session_state.login_error)

# -------------------------------------------------
# Register
# -------------------------------------------------
def register_ui(sidebar):
    email = sidebar.text_input("Email", key="register_email")
    password = sidebar.text_input(
        "Password", type="password", key="register_password"
    )

    if sidebar.button("Register"):

        st.session_state.register_error = ""
        st.session_state.register_success = ""

        if not firebase_auth:
            st.session_state.register_error = "Authentication service unavailable."
            return

        domain = get_email_domain(email)

        if domain not in PRIVILEGED_DOMAINS:
            st.session_state.register_error = (
                "Registration restricted to approved domains."
            )
            return

        try:
            user = firebase_auth.create_user_with_email_and_password(
                email, password
            )

            firebase_auth.send_email_verification(user["idToken"])

            st.session_state.register_success = (
                "Registration successful. Check your email to verify your account."
            )

        except Exception as e:
            error_code = parse_firebase_error(e)

            if error_code == "EMAIL_EXISTS":
                st.session_state.register_error = (
                    "This email is already registered. Please log in."
                )
            elif error_code == "WEAK_PASSWORD":
                st.session_state.register_error = (
                    "Password must be at least 6 characters."
                )
            else:
                st.session_state.register_error = (
                    "Registration failed. Please try again."
                )

    if st.session_state.register_error:
        sidebar.error(st.session_state.register_error)

    if st.session_state.register_success:
        sidebar.success(st.session_state.register_success)
