# auth.py (with DEBUG sidebar)
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
from streamlit_cookies_manager import EncryptedCookieManager
import json

# ----------------- DEBUG FLAG -----------------
DEBUG = True  # Set False in production

# -------------------------------------------------
# Firebase Admin Initialization
# -------------------------------------------------
if not firebase_admin._apps:
    if "firebase_admin" not in st.secrets:
        st.error("Missing firebase_admin in secrets.toml")
        st.stop()

    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# -------------------------------------------------
# Firebase Client (Pyrebase)
# -------------------------------------------------
firebase_cfg = dict(st.secrets.get("firebase", {}))
if not firebase_cfg:
    st.error("Firebase config missing in secrets.toml")
    st.stop()

firebase = pyrebase.initialize_app(firebase_cfg)
firebase_auth = firebase.auth()

# -------------------------------------------------
# Privileged Domains
# -------------------------------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)

# -------------------------------------------------
# Cookie Manager
# -------------------------------------------------
def get_cookies():
    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            st.error("Cookie password missing in secrets.toml")
            return None

        st.session_state.cookies = EncryptedCookieManager(prefix="myapp", password=password)

    cookies = st.session_state.cookies

    # Wait until cookies are ready (max 1 second)
    try:
        if not cookies.ready():
            start = time.time()
            while not cookies.ready() and time.time() - start < 1.0:
                time.sleep(0.05)  # small delay
            if not cookies.ready():
                if DEBUG:
                    st.sidebar.warning("Cookies not ready after waiting.")
                return None
    except CookiesNotReady:
        if DEBUG:
            st.sidebar.warning("Caught CookiesNotReady exception.")
        return None
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Cookie load error: {e}")
        return None

    # Try to sync/load without raising
    try:
        if hasattr(cookies, "sync"):
            cookies.sync()
        elif hasattr(cookies, "load"):
            cookies.load()
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Could not sync/load cookies: {e}")

    return cookies
# -------------------------------------------------
# Session Initialization
# -------------------------------------------------
def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "restored": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# -------------------------------------------------
# Restore Session From Cookies
# -------------------------------------------------
def restore_session():
    cookies = get_cookies()
    if not cookies:
        st.session_state.restored = True
        return

    if not st.session_state.restored:
        if "email" in cookies:
            st.session_state.user = True
            st.session_state.email = cookies.get("email")
            st.session_state.name = cookies.get("name")
            st.session_state.role = cookies.get("role")
            st.session_state.email_verified = cookies.get("email_verified", False)
        st.session_state.restored = True

# -------------------------------------------------
# Logout
# -------------------------------------------------
def logout():
    cookies = get_cookies()
    if cookies:
        for key in ["email", "name", "role", "email_verified"]:
            if key in cookies:
                del cookies[key]
        try:
            cookies.save()
        except Exception:
            pass

    # Clear session safely
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
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

# -------------------------------------------------
# Authentication UI
# -------------------------------------------------
def auth_ui():
    init_session()
    restore_session()

    sidebar = st.sidebar

    # ---------------- Logged-In View ----------------
    if st.session_state.user:
        sidebar.success(f"👋 {st.session_state.name}")

        if not st.session_state.email_verified:
            sidebar.warning("Email not verified.")
            sidebar.info("Please verify your email before accessing the dashboard.")
            if sidebar.button("Logout"):
                logout()
            return

        if sidebar.button("Logout"):
            logout()
        return

    # ---------------- Tabs ----------------
    action = sidebar.radio("Select Action", ["Login", "Register"])

    # ================= LOGIN =================
    if action == "Login":
        with sidebar.form("login_form"):
            email = st.text_input("Email").strip()
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in")

            if submit:
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access restricted to approved domains.")
                    return

                try:
                    user = firebase_auth.sign_in_with_email_and_password(email, password)
                    info = firebase_auth.get_account_info(user["idToken"])
                    verified = info["users"][0]["emailVerified"]
                    role = "privileged" if verified else "restricted"

                    # Store session
                    st.session_state.user = True
                    st.session_state.email = email
                    st.session_state.name = email.split("@")[0].title()
                    st.session_state.email_verified = verified
                    st.session_state.role = role

                    # Store SAFE cookie data (no JWT)
                    cookies = get_cookies()
                    if cookies:
                        cookies["email"] = email
                        cookies["name"] = st.session_state.name
                        cookies["email_verified"] = verified
                        cookies["role"] = role
                        try:
                            cookies.save()
                        except Exception:
                            pass

                    st.rerun()

                except Exception as e:
                    st.error(parse_error(e))

    # ================= REGISTER =================
    if action == "Register":
        with sidebar.form("register_form"):
            email = st.text_input("Email").strip()
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register")

            if submit:
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Registration restricted to approved domains.")
                    return

                try:
                    user = firebase_auth.create_user_with_email_and_password(email, password)
                    firebase_auth.send_email_verification(user["idToken"])
                    st.success("Registration successful. Check your email to verify.")

                except Exception as e:
                    st.error(parse_error(e))

    # ---------------- DEBUG INFO ----------------
    if DEBUG:
        sidebar.markdown("---")
        sidebar.markdown("**DEBUG INFO**")
        sidebar.text(f"Session State: {st.session_state}")
        cookies = get_cookies()
        sidebar.text(f"Cookies: {dict(cookies) if cookies else 'None'}")