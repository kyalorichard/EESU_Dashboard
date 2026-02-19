# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import json
from streamlit_cookies_manager import EncryptedCookieManager

# -------------------------------
# Firebase Admin Initialization
# -------------------------------
try:
    if "firebase_admin" in st.secrets and not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"⚠️ Firebase Admin initialization failed.\n{str(e)}")
    st.stop()

# -------------------------------
# Pyrebase Initialization
# -------------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.warning(f"⚠️ Firebase authentication service unavailable.\n{str(e)}")

<<<<<<< HEAD
# -------------------------------------------------
# Privileged Domains
# -------------------------------------------------
=======
# -------------------------------
# Configuration
# -------------------------------
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)

# -------------------------------------------------
# Error Messages
# -------------------------------------------------
ERROR_MAP = {
    "EMAIL_EXISTS": "This email is already registered.",
    "INVALID_PASSWORD": "Incorrect email or password.",
    "EMAIL_NOT_FOUND": "Email not registered.",
    "WEAK_PASSWORD": "Password must be at least 6 characters.",
    "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password."
}

<<<<<<< HEAD
# -------------------------------------------------
# Cookies Manager
# -------------------------------------------------
def get_cookies_manager() -> EncryptedCookieManager:
    """
    Initialize the cookies manager.
    """
    cookies = EncryptedCookieManager(
        prefix="myapp",  # unique prefix for this app
        password=st.secrets.get("cookies_password", "fallback-secret"),
    )
    cookies.load()  # async load
    return cookies

# -------------------------------------------------
=======
# -------------------------------
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f
# Helpers
# -------------------------------
def get_email_domain(email: str) -> str:
    return email.strip().split("@")[-1].lower()

def parse_firebase_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        error_json = json.loads(payload)
        return error_json.get("error", {}).get("message", str(e))
    except Exception:
        return str(e)

<<<<<<< HEAD
def logout_user():
    # Clear session state
    for key in ["user", "email", "name", "user_role", "email_verified", "idToken"]:
        st.session_state.pop(key, None)

    # Clear cookies
    cookies = get_cookies_manager()
    if cookies.ready():
        for key in ["user", "email", "name", "user_role", "email_verified"]:
            if key in cookies:
                cookies[key] = ""
        cookies.save()
    st.experimental_rerun()
=======
# -------------------------------
# Cookies Manager (lazy init)
# -------------------------------
def get_cookies_manager():
    if "cookies_manager" not in st.session_state:
        st.session_state["cookies_manager"] = EncryptedCookieManager(prefix="myapp")
    return st.session_state["cookies_manager"]

# -------------------------------
# Session State Initialization
# -------------------------------
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

# -------------------------------
# Logout
# -------------------------------
def logout_user():
    keys = ["user", "email", "name", "user_role", "email_verified", "idToken"]
    for key in keys:
        st.session_state.pop(key, None)
    cookies = get_cookies_manager()
    cookies.clear()
    cookies.save()
    st.rerun()
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f

# -------------------------------
# Check Privileged
# -------------------------------
def is_privileged() -> bool:
    return (
        st.session_state.get("user_role") == "privileged"
        and st.session_state.get("email_verified") is True
    )

<<<<<<< HEAD
def refresh_id_token():
    """Refresh Firebase ID token if expired"""
    try:
        if st.session_state.get("idToken") and firebase_auth:
            refreshed = firebase_auth.refresh(st.session_state.idToken)
            st.session_state.idToken = refreshed["idToken"]
    except Exception:
        pass

def init_state_from_cookies():
    """
    Load session from cookies if available and cookies are ready.
    """
    cookies = get_cookies_manager()
    if not cookies.ready():
        return False  # stop until ready

    # Restore session from cookies
    if "user" in cookies:
        st.session_state.user = cookies.get("user")
        st.session_state.email_verified = cookies.get("email_verified", False)
        st.session_state.user_role = cookies.get("user_role", None)
        st.session_state.email = cookies.get("email", "")
        st.session_state.name = cookies.get("name", "")
    return True

# -------------------------------------------------
=======
# -------------------------------
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f
# Authentication UI
# -------------------------------
def auth_ui():
<<<<<<< HEAD
    refresh_id_token()
    cookies = get_cookies_manager()

    # Wait for cookies to load
    if not cookies.ready():
        st.info("Loading session...")
        return

    # Restore session from cookies
    init_state_from_cookies()
=======
    init_state()
    cookies = get_cookies_manager()

    # Wait until cookies are ready
    if not cookies.ready():
        cookies.save()
        st.stop()

    # Restore session from cookies
    if "user" in cookies and not st.session_state.get("user"):
        st.session_state.user = cookies.get("user")
        st.session_state.email_verified = cookies.get("email_verified", False)
        st.session_state.user_role = cookies.get("user_role", None)
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f

    sidebar = st.sidebar

    # -----------------------------
    # Logged-in View
    # -----------------------------
    if st.session_state.get("user"):
        if st.session_state.email_verified:
            sidebar.success(f"👋 {st.session_state.name} ✅ Verified")
        else:
            sidebar.warning(f"👋 {st.session_state.name} ⚠️ Email not verified")

<<<<<<< HEAD
            if sidebar.button("Resend Verification Email"):
                try:
                    refreshed = firebase_auth.refresh(st.session_state.idToken)
                    st.session_state.idToken = refreshed["idToken"]
                    firebase_auth.send_email_verification(st.session_state.idToken)
                    sidebar.success("Verification email resent successfully.")
                except Exception:
                    sidebar.error("Unable to resend verification email. Try again later.")

        sidebar.button("Logout", on_click=logout_user)
=======
        if sidebar.button("Logout"):
            logout_user()
            return
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f
        return

    # -----------------------------
    # Tabs: Login / Register
    # -----------------------------
    tab_choice = sidebar.radio(
        "Select Action",
        ["Login", "Register"],
        index=0,
        key="auth_tab_radio"
    )

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

                            # Save in session
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
                            # Save to cookies
                            cookies["user"] = user
                            cookies["email_verified"] = email_verified
                            cookies["user_role"] = st.session_state.user_role
                            cookies.save()

                            if not email_verified:
                                st.warning("Please verify your email before accessing the dashboard.")
                                return

                            st.experimental_rerun()
                        except Exception as e:
                            error_code = parse_firebase_error(e)
                            st.error(ERROR_MAP.get(error_code, f"Login failed: {error_code}"))

            if forgot_pass:
                if not email:
                    st.warning("Enter your email above to reset your password.")
                else:
                    try:
                        firebase_auth.send_password_reset_email(email)
                        st.success(f"Password reset email sent to {email}.")
                    except Exception as e:
                        st.error(f"Failed to send reset email: {parse_firebase_error(e)}")

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
                else:
                    domain = get_email_domain(email)
                    if domain not in PRIVILEGED_DOMAINS:
                        st.error(f"Registration restricted: {domain} is not approved.")
                    else:
                        try:
                            user = firebase_auth.create_user_with_email_and_password(email, password)
                            firebase_auth.send_email_verification(user["idToken"])

                            st.success("Registration successful. Check your email to verify account.")

                            st.session_state.user = user
                            st.session_state.email = email
                            st.session_state.name = email.split("@")[0].title()
                            st.session_state.email_verified = False
                            st.session_state.idToken = user["idToken"]
                            st.session_state.user_role = "unverified"

<<<<<<< HEAD
                            # Save in cookies
                            cookies["user"] = user
                            cookies["email"] = email
                            cookies["name"] = st.
=======
                            # Save to cookies
                            cookies["user"] = user
                            cookies["email_verified"] = False
                            cookies["user_role"] = "unverified"
                            cookies.save()

                        except Exception as e:
                            error_code = parse_firebase_error(e)
                            st.error(ERROR_MAP.get(error_code, f"Registration failed: {error_code}"))
>>>>>>> 29beb2b1f42d4398be7a27330a039700c82b258f
