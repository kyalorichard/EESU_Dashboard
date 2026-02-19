# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import json
from streamlit_cookies_manager import EncryptedCookieManager

# ------------------------------
# Firebase Admin Initialization
# ------------------------------
try:
    if "firebase_admin" in st.secrets and not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"⚠️ Firebase Admin init failed: {str(e)}")
    st.stop()

# ------------------------------
# Pyrebase Initialization
# ------------------------------
firebase_cfg = dict(st.secrets.get("firebase", {}))
firebase_auth = None
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.warning(f"⚠️ Firebase auth unavailable: {str(e)}")

# ------------------------------
# Privileged domains
# ------------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)
if not PRIVILEGED_DOMAINS:
    st.warning("⚠️ No privileged domains configured. Access may be blocked.")

# ------------------------------
# Error messages mapping
# ------------------------------
ERROR_MAP = {
    "EMAIL_EXISTS": "This email is already registered.",
    "INVALID_PASSWORD": "Incorrect email or password.",
    "EMAIL_NOT_FOUND": "Email not registered.",
    "WEAK_PASSWORD": "Password must be at least 6 characters.",
    "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password."
}

# ------------------------------
# Helpers
# ------------------------------
def get_email_domain(email: str) -> str:
    return email.strip().split("@")[-1].lower()

def parse_firebase_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        error_json = json.loads(payload)
        return error_json.get("error", {}).get("message", str(e))
    except Exception:
        return str(e)

# ------------------------------
# Initialize EncryptedCookieManager
# ------------------------------
def get_cookies_manager():
    if "cookies_manager" not in st.session_state:
        password = st.secrets.get("cookie_password")
        if not password:
            st.error("Cookie password not set in secrets.toml")
            st.stop()
        st.session_state["cookies_manager"] = EncryptedCookieManager(
            prefix="myapp", password=password
        )
    return st.session_state["cookies_manager"]

cookies = get_cookies_manager()

if not cookies.ready():
    st.info("Loading session…")
    st.stop()

# ------------------------------
# Session Initialization
# ------------------------------
def init_state():
    defaults = {
        "user": None,
        "email": "",
        "name": "",
        "user_role": None,
        "email_verified": False,
        "idToken": None,
        "auth_tab": "Login"
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # Restore from cookies if available
    if "user" in cookies:
        st.session_state["user"] = cookies.get("user")
        st.session_state["email"] = cookies.get("email")
        st.session_state["name"] = cookies.get("name")
        st.session_state["user_role"] = cookies.get("user_role")
        st.session_state["email_verified"] = cookies.get("email_verified", False)

# ------------------------------
# Logout
# ------------------------------
def logout_user():
    keys = ["user", "email", "name", "user_role", "email_verified", "idToken"]
    for key in keys:
        st.session_state.pop(key, None)
        if key in cookies:
            cookies.pop(key)
    cookies.save()
    st.rerun()

# ------------------------------
# Check if user is privileged
# ------------------------------
def is_privileged() -> bool:
    return (
        st.session_state.get("user_role") == "privileged"
        and st.session_state.get("email_verified") is True
    )

# ------------------------------
# Refresh ID token
# ------------------------------
def refresh_id_token():
    try:
        if st.session_state.get("idToken") and firebase_auth:
            refreshed = firebase_auth.refresh(st.session_state.idToken)
            st.session_state.idToken = refreshed["idToken"]
    except Exception:
        pass

# ------------------------------
# Authentication UI
# ------------------------------
def auth_ui():
    init_state()
    sidebar = st.sidebar
    refresh_id_token()  # Refresh token if needed

    # ----------------- Logged-in view -----------------
    if st.session_state.user:
        if st.session_state.email_verified:
            sidebar.success(f"👋 {st.session_state.name} ✅ Verified")
        else:
            sidebar.warning(f"👋 {st.session_state.name} ⚠️ Email not verified")
            sidebar.markdown("Verify your email to access the dashboard.")
            if sidebar.button("Resend Verification Email"):
                try:
                    firebase_auth.send_email_verification(st.session_state.idToken)
                    sidebar.success("Verification email resent successfully.")
                except Exception:
                    sidebar.error("Unable to resend verification email.")

        sidebar.button("Logout", on_click=logout_user)
        return

    # ----------------- Tabs: Login / Register -----------------
    tab_choice = sidebar.radio(
        "Select Action",
        ["Login", "Register"],
        index=0 if st.session_state.auth_tab == "Login" else 1,
        key="auth_tab_radio"
    )
    st.session_state.auth_tab = tab_choice

    # ----------------- LOGIN FORM -----------------
    if tab_choice == "Login":
        with sidebar.form("login_form", clear_on_submit=True):
            email = st.text_input("Email", key="login_email").strip()
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Sign in")
            forgot_pass = st.form_submit_button("Forgot Password?")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                else:
                    domain = get_email_domain(email)
                    if domain not in PRIVILEGED_DOMAINS:
                        st.error(f"Access restricted: {domain} is not authorized.")
                    else:
                        try:
                            user = firebase_auth.sign_in_with_email_and_password(email, password)
                            id_token = user["idToken"]
                            user_info = firebase_auth.get_account_info(id_token)
                            email_verified = user_info["users"][0].get("emailVerified", False)

                            # Store in session
                            st.session_state.user = user
                            st.session_state.email = email
                            st.session_state.name = email.split("@")[0].title()
                            st.session_state.email_verified = email_verified
                            st.session_state.idToken = id_token
                            st.session_state.user_role = "privileged" if email_verified else "unverified"

                            # Save in cookies
                            cookies["user"] = user
                            cookies["email"] = email
                            cookies["name"] = st.session_state.name
                            cookies["user_role"] = st.session_state.user_role
                            cookies["email_verified"] = email_verified
                            cookies.save()

                            if not email_verified:
                                st.warning("Please verify your email before accessing the dashboard.")
                                return

                            st.rerun()
                        except Exception as e:
                            error_code = parse_firebase_error(e)
                            st.error(ERROR_MAP.get(error_code, f"Login failed: {error_code}"))

            if forgot_pass:
                if not email:
                    st.warning("Enter your email to reset password.")
                else:
                    try:
                        firebase_auth.send_password_reset_email(email)
                        st.success(f"Password reset email sent to {email}.")
                    except Exception as e:
                        st.error(f"Failed to send reset email: {parse_firebase_error(e)}")

    # ----------------- REGISTER FORM -----------------
    if tab_choice == "Register":
        with sidebar.form("register_form", clear_on_submit=True):
            email = st.text_input("Email", key="reg_email").strip()
            password = st.text_input("Password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Register")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                else:
                    domain = get_email_domain(email)
                    if domain not in PRIVILEGED_DOMAINS:
                        st.error(f"Registration restricted: {domain} is not allowed.")
                    else:
                        try:
                            user = firebase_auth.create_user_with_email_and_password(email, password)
                            firebase_auth.send_email_verification(user["idToken"])

                            st.success("Registration successful. Verify your email before accessing the dashboard.")

                            # Auto-login (unverified)
                            st.session_state.user = user
                            st.session_state.email = email
                            st.session_state.name = email.split("@")[0].title()
                            st.session_state.email_verified = False
                            st.session_state.idToken = user["idToken"]
                            st.session_state.user_role = "unverified"

                            # Save in cookies
                            cookies["user"] = user
                            cookies["email"] = email
                            cookies["name"] = st.session_state.name
                            cookies["user_role"] = st.session_state.user_role
                            cookies["email_verified"] = False
                            cookies.save()
                        except Exception as e:
                            error_code = parse_firebase_error(e)
                            st.error(ERROR_MAP.get(error_code, f"Registration failed: {error_code}"))
