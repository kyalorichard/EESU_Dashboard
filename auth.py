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
    st.session_state.setdefault("register_error", "")
    st.session_state.setdefault("register_success", "")
    st.session_state.setdefault("email_unverified", False)
    st.session_state.setdefault("resend_success", "")
    st.session_state.setdefault("idToken", None)
    st.session_state.setdefault("email_verified", False)

# ---------------------------
# Logout
# ---------------------------
def logout_user():
    for k in ["user", "email", "name", "user_role", "email_unverified", "resend_success", "idToken", "email_verified"]:
        st.session_state.pop(k, None)

# ---------------------------
# Check if user is privileged
# ---------------------------
def is_privileged():
    return st.session_state.get("user_role") == "privileged"

# ---------------------------
# Sidebar Auth UI
# ---------------------------
def auth_ui():
    init_state()
    sidebar = st.sidebar
    sidebar.markdown("## Account")

    # Logged in view
    if st.session_state.user:
        # Show name with verification badge
        if st.session_state.email_verified:
            sidebar.success(f"👋 {st.session_state.name} ✅")
        else:
            sidebar.warning(f"👋 {st.session_state.name} ⚠️ Email not verified")

        # Resend verification button if not verified
        if not st.session_state.email_verified:
            if sidebar.button("Resend Verification Email"):
                try:
                    firebase_auth.send_email_verification(st.session_state.idToken)
                    st.session_state.resend_success = "Verification email resent! Check your inbox."
                except Exception as e:
                    st.session_state.resend_success = f"Failed to resend: {e}"
            if st.session_state.resend_success:
                sidebar.info(st.session_state.resend_success)

        sidebar.button("Logout", on_click=logout_user)
        return

    # Tabs for Login / Register
    tab = sidebar.radio("Select action:", ["Login", "Register"])
    if tab == "Login":
        login_ui(sidebar)
    else:
        register_ui(sidebar)

# ---------------------------
# Login UI
# ---------------------------
def login_ui(sidebar):
    email_input = sidebar.text_input("Email", key="login_email")
    password_input = sidebar.text_input("Password", type="password", key="login_password")

    if sidebar.button("Sign in"):
        email = email_input
        password = password_input

        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.session_state.login_error = "Access denied: your email domain is not allowed."
            return

        if not firebase_auth:
            st.session_state.login_error = "Firebase not initialized. Contact admin."
            return

        try:
            user = firebase_auth.sign_in_with_email_and_password(email, password)
            st.session_state.idToken = user['idToken']

            # Check email verification
            user_info = firebase_auth.get_account_info(user['idToken'])
            email_verified = user_info['users'][0].get('emailVerified', False)
            st.session_state.email_verified = email_verified

            if not email_verified:
                st.session_state.login_error = "Please verify your email before logging in."
                st.session_state.email_unverified = True
                return

            # Successful login
            st.session_state.user = "email"
            st.session_state.email = email
            st.session_state.name = email.split("@")[0].title()
            st.session_state.user_role = "privileged"
            st.session_state.login_error = ""
            st.session_state.email_unverified = False

        except Exception as e:
            err_str = str(e)
            if "INVALID_PASSWORD" in err_str or "EMAIL_NOT_FOUND" in err_str or "INVALID_LOGIN_CREDENTIALS" in err_str:
                st.session_state.login_error = "Invalid email or password. Please try again."
            else:
                st.session_state.login_error = f"Login failed: {e}"

    if st.session_state.login_error:
        sidebar.error(st.session_state.login_error)

# ---------------------------
# Register UI
# ---------------------------
def register_ui(sidebar):
    email_input = sidebar.text_input("Email", key="register_email")
    password_input = sidebar.text_input("Password", type="password", key="register_password")

    if sidebar.button("Register"):
        email = email_input
        password = password_input

        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.session_state.register_error = "Registration denied: your email domain is not allowed."
            st.session_state.register_success = ""
            return

        if not firebase_auth:
            st.session_state.register_error = "Firebase not initialized. Contact admin."
            return

        try:
            user = firebase_auth.create_user_with_email_and_password(email, password)
            firebase_auth.send_email_verification(user['idToken'])
            st.session_state.register_success = "Registration successful! Check your email to verify before logging in."
            st.session_state.register_error = ""
        except Exception as e:
            err_str = str(e)
            if "EMAIL_EXISTS" in err_str:
                st.session_state.register_error = "Email already registered. Please log in."
            else:
                st.session_state.register_error = f"Registration failed: {e}"
            st.session_state.register_success = ""

    if st.session_state.register_error:
        sidebar.error(st.session_state.register_error)
    if st.session_state.register_success:
        sidebar.success(st.session_state.register_success)
