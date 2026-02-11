# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials

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
    st.secrets.get("access", {}).get("privileged_domains", [])
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

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

# -------------------------------------------------
# Authorization Check
# -------------------------------------------------
def is_privileged():
    """
    Only verified users with allowed domain
    get privileged access.
    """
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

            # Resend verification
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

        # Domain restriction
        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.session_state.login_error = (
                "Access denied: your email domain is not allowed."
            )
            return

        if not firebase_auth:
            st.session_state.login_error = (
                "Authentication service unavailable."
            )
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

            # Set session user
            st.session_state.user = "email"
            st.session_state.email = email
            st.session_state.name = email.split("@")[0].title()
            st.session_state.email_verified = email_verified

            if email_verified:
                st.session_state.user_role = "privileged"
                st.session_state.login_error = ""
            else:
                st.session_state.user_role = "unverified"
                st.session_state.login_error = (
                    "Please verify your email to unlock full access."
                )

        except Exception as e:
            err_str = str(e)

            if (
                "INVALID_PASSWORD" in err_str
                or "EMAIL_NOT_FOUND" in err_str
                or "INVALID_LOGIN_CREDENTIALS" in err_str
            ):
                st.session_state.login_error = (
                    "Invalid email or password."
                )
            else:
                st.session_state.login_error = (
                    "Login failed. Please try again."
                )

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

        # Domain restriction
        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.session_state.register_error = (
                "Registration denied: email domain not permitted."
            )
            return

        if not firebase_auth:
            st.session_state.register_error = (
                "Authentication service unavailable."
            )
            return

        try:
            user = firebase_auth.create_user_with_email_and_password(
                email, password
            )

            firebase_auth.send_email_verification(user["idToken"])

            st.session_state.register_success = (
                "Registration successful. "
                "Check your email to verify your account."
            )
            st.session_state.register_error = ""

        except Exception as e:
            err_str = str(e)

            if "EMAIL_EXISTS" in err_str:
                st.session_state.register_error = (
                    "Email already registered. Please log in."
                )
            else:
                st.session_state.register_error = (
                    "Registration failed. Please try again."
                )

    if st.session_state.register_error:
        sidebar.error(st.session_state.register_error)

    if st.session_state.register_success:
        sidebar.success(st.session_state.register_success)
