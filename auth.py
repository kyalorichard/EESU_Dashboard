# auth_safe_full_v2.py — full robust auth with safe cookies
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
from streamlit_cookies_manager import EncryptedCookieManager, CookiesNotReady
import json
import time

DEBUG = True  # Set False in production

# ---------------------- Firebase Admin ----------------------
firebase_admin_available = False
if not firebase_admin._apps:
    try:
        if "firebase_admin" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
            firebase_admin.initialize_app(cred)
            firebase_admin_available = True
        else:
            if DEBUG:
                st.warning("⚠️ firebase_admin missing in secrets.toml")
    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase Admin init failed: {e}")

# ---------------------- Firebase Client ----------------------
firebase_available = False
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
        firebase_available = True
    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase init failed: {e}")
else:
    if DEBUG:
        st.warning("⚠️ firebase config missing in secrets.toml")

# ---------------------- Privileged Domains ----------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)

# ---------------------- Cookie Manager ----------------------
def get_cookies():
    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            if DEBUG:
                st.warning("Cookie password missing in secrets.toml")
            return None
        st.session_state.cookies = EncryptedCookieManager(prefix="myapp", password=password)

    cookies = st.session_state.cookies

    # Wait until cookies are ready (max 1 second)
    start = time.time()
    while True:
        try:
            if cookies.ready():
                break
        except CookiesNotReady:
            if time.time() - start > 1.0:
                if DEBUG:
                    st.sidebar.warning("Cookies not ready after waiting 1 second.")
                return None
            time.sleep(0.05)

    # Try to sync/load safely
    try:
        if hasattr(cookies, "sync"):
            cookies.sync()
        elif hasattr(cookies, "load"):
            cookies.load()
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Could not sync/load cookies: {e}")

    return cookies

# ---------------------- Session ----------------------
def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "restored": False,
        "password": None,  # optional, for resending verification
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def restore_session():
    cookies = get_cookies()
    if not cookies:
        st.session_state.restored = True
        return

    if not st.session_state.restored:
        try:
            if "email" in cookies:
                st.session_state.user = True
                st.session_state.email = cookies.get("email")
                st.session_state.name = cookies.get("name")
                st.session_state.role = cookies.get("role")
                st.session_state.email_verified = cookies.get("email_verified", False)
        except CookiesNotReady:
            if DEBUG:
                st.sidebar.warning("Tried to access cookies before ready.")
        except Exception as e:
            if DEBUG:
                st.sidebar.warning(f"Error restoring session: {e}")
        st.session_state.restored = True

def logout():
    cookies = get_cookies()
    if cookies:
        try:
            for key in ["email", "name", "role", "email_verified"]:
                if key in cookies:
                    del cookies[key]
            cookies.save()
        except CookiesNotReady:
            if DEBUG:
                st.sidebar.warning("Tried to delete cookies before ready.")
        except Exception:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------------------- Helpers ----------------------
def parse_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        data = json.loads(payload)
        return data.get("error", {}).get("message", str(e))
    except Exception:
        return str(e)

def get_domain(email):
    return email.strip().split("@")[-1].lower()

def is_privileged():
    return (
        st.session_state.get("user", False)
        and st.session_state.get("email_verified", False)
        and st.session_state.get("role") == "privileged"
    )

# ---------------------- Auth UI ----------------------
def auth_ui():
    init_session()
    restore_session()
    sidebar = st.sidebar

    # ----- Logged-In View -----
    if st.session_state.user:
        sidebar.success(f"👋 {st.session_state.name}")

        # Unverified users
        if not st.session_state.email_verified:
            sidebar.warning("Email not verified.")
            sidebar.info("Please verify your email before accessing the dashboard.")

            # Resend Verification Email
            if firebase_available:
                if sidebar.button("Resend Verification Email"):
                    try:
                        if st.session_state.password:
                            user = firebase_auth.sign_in_with_email_and_password(
                                st.session_state.email,
                                st.session_state.password
                            )
                            firebase_auth.send_email_verification(user["idToken"])
                            sidebar.success(f"Verification email resent to {st.session_state.email}.")
                        else:
                            sidebar.warning("Password required to resend verification email. Please log in again.")
                    except Exception as e:
                        sidebar.error(f"Failed to resend verification email: {parse_error(e)}")
            else:
                sidebar.warning("Firebase not available. Cannot resend verification email.")

            if sidebar.button("Logout"):
                logout()
            return

        if sidebar.button("Logout"):
            logout()
        return

    # ----- Tabs -----
    action = sidebar.radio("Select Action", ["Login", "Register"])

    # ----- LOGIN -----
    if action == "Login":
        with sidebar.form("login_form"):
            email = st.text_input("Email", key="login_email").strip()
            password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Sign in")

            if login_submit:
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access restricted to approved domains.")
                elif not firebase_available:
                    st.warning("Firebase not available. Cannot log in.")
                else:
                    try:
                        user = firebase_auth.sign_in_with_email_and_password(email, password)
                        info = firebase_auth.get_account_info(user["idToken"])
                        verified = info["users"][0]["emailVerified"]
                        role = "privileged" if verified else "restricted"

                        st.session_state.user = True
                        st.session_state.email = email
                        st.session_state.name = email.split("@")[0].title()
                        st.session_state.email_verified = verified
                        st.session_state.role = role
                        st.session_state.password = password

                        cookies = get_cookies()
                        if cookies:
                            try:
                                cookies["email"] = email
                                cookies["name"] = st.session_state.name
                                cookies["email_verified"] = verified
                                cookies["role"] = role
                                cookies.save()
                            except CookiesNotReady:
                                if DEBUG:
                                    st.sidebar.warning("Cookies not ready when saving session.")

                        st.rerun()
                    except Exception as e:
                        st.error(parse_error(e))

        # Forgot Password form
        with sidebar.form("forgot_password_form"):
            st.markdown("---")
            st.write("Forgot your password? Enter your email below to reset it.")
            forgot_email = st.text_input("Email for password reset", key="forgot_email_input")
            reset_submit = st.form_submit_button("Send Reset Email")

            if reset_submit:
                if not forgot_email:
                    st.warning("Please enter your email.")
                elif get_domain(forgot_email) not in PRIVILEGED_DOMAINS:
                    st.error("Password reset restricted to approved domains.")
                elif not firebase_available:
                    st.warning("Firebase not available. Cannot reset password.")
                else:
                    try:
                        firebase_auth.send_password_reset_email(forgot_email)
                        st.success(f"Password reset email sent to {forgot_email}.")
                    except Exception as e:
                        st.error(f"Failed to send reset email: {parse_error(e)}")

    # ----- REGISTER -----
    if action == "Register":
        with sidebar.form("register_form"):
            email = st.text_input("Email").strip()
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register")

            if submit:
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Registration restricted to approved domains.")
                elif not firebase_available:
                    st.warning("Firebase not available. Cannot register.")
                else:
                    try:
                        user = firebase_auth.create_user_with_email_and_password(email, password)
                        firebase_auth.send_email_verification(user["idToken"])
                        st.success("Registration successful. Check your email to verify.")
                    except Exception as e:
                        st.error(parse_error(e))