# auth_safe.py
import streamlit as st
import json
import time

DEBUG = True  # Set False in production

# -----------------------------
# Optional Imports
# -----------------------------
try:
    import pyrebase
    HAS_PYREBASE = True
except ImportError:
    HAS_PYREBASE = False

try:
    import firebase_admin
    from firebase_admin import credentials
    HAS_FIREBASE_ADMIN = True
except ImportError:
    HAS_FIREBASE_ADMIN = False

try:
    from streamlit_cookies_manager import EncryptedCookieManager
    HAS_COOKIES = True
except ImportError:
    HAS_COOKIES = False

# -----------------------------
# 1️⃣ Firebase Admin (Safe)
# -----------------------------
def init_firebase_admin():
    if not HAS_FIREBASE_ADMIN:
        if DEBUG:
            st.warning("firebase_admin not installed, skipping Admin init.")
        return None

    secrets_admin = st.secrets.get("firebase_admin")
    if not secrets_admin:
        if DEBUG:
            st.warning("Firebase Admin secrets missing in secrets.toml, skipping init.")
        return None

    try:
        private_key = secrets_admin["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": secrets_admin["project_id"],
            "private_key_id": secrets_admin["private_key_id"],
            "private_key": private_key,
            "client_email": secrets_admin["client_email"],
            "client_id": secrets_admin["client_id"],
            "auth_uri": secrets_admin["auth_uri"],
            "token_uri": secrets_admin["token_uri"],
            "auth_provider_x509_cert_url": secrets_admin["auth_provider_x509_cert_url"],
            "client_x509_cert_url": secrets_admin["client_x509_cert_url"],
            "universe_domain": secrets_admin.get("universe_domain", "googleapis.com")
        })
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        return firebase_admin
    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase Admin init failed: {e}")
        return None

# -----------------------------
# 2️⃣ Firebase Client (Pyrebase, Safe)
# -----------------------------
def init_firebase_client():
    cfg = st.secrets.get("firebase", {})
    if not cfg:
        if DEBUG:
            st.warning("Firebase client config missing in secrets.toml, skipping client init.")
        return None, None

    if not HAS_PYREBASE:
        if DEBUG:
            st.warning("pyrebase not installed, skipping client init.")
        return None, None

    try:
        firebase = pyrebase.initialize_app(cfg)
        auth = firebase.auth()
        return firebase, auth
    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase client init failed: {e}")
        return None, None

firebase_admin_app = init_firebase_admin()
firebase_client, firebase_auth = init_firebase_client()

# -----------------------------
# 3️⃣ Privileged Domains
# -----------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)

# -----------------------------
# 4️⃣ Cookie Manager (Safe)
# -----------------------------
def get_cookies():
    if not HAS_COOKIES:
        if DEBUG:
            st.warning("streamlit_cookies_manager not installed, cookies disabled.")
        return None

    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            if DEBUG:
                st.warning("Cookie password missing in secrets.toml, skipping cookies.")
            return None
        st.session_state.cookies = EncryptedCookieManager(prefix="myapp", password=password)

    cookies = st.session_state.cookies

    # Wait until cookies are ready (max 1 sec)
    try:
        start = time.time()
        while not cookies.ready() and time.time() - start < 1.0:
            time.sleep(0.05)
        if not cookies.ready():
            return None
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Cookie load error: {e}")
        return None

    try:
        if hasattr(cookies, "sync"):
            cookies.sync()
        elif hasattr(cookies, "load"):
            cookies.load()
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Could not sync/load cookies: {e}")

    return cookies

# -----------------------------
# 5️⃣ Session Helpers
# -----------------------------
def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "restored": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def restore_session():
    cookies = get_cookies()
    if not cookies:
        st.session_state.restored = True
        return
    if st.session_state.restored:
        return
    try:
        if cookies.ready() and "email" in cookies:
            st.session_state.user = True
            st.session_state.email = cookies.get("email")
            st.session_state.name = cookies.get("name")
            st.session_state.role = cookies.get("role")
            st.session_state.email_verified = cookies.get("email_verified", False)
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Error restoring session: {e}")
    st.session_state.restored = True

def logout():
    cookies = get_cookies()
    if cookies and cookies.ready():
        for key in ["email", "name", "role", "email_verified"]:
            if key in cookies:
                del cookies[key]
        try:
            cookies.save()
        except Exception:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

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

# -----------------------------
# 6️⃣ Authentication UI
# -----------------------------
def auth_ui():
    init_session()
    restore_session()
    sidebar = st.sidebar

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

    action = sidebar.radio("Select Action", ["Login", "Register"])

    # --- LOGIN ---
    if action == "Login":
        if not firebase_auth:
            sidebar.warning("Firebase auth not initialized, login disabled.")
            return
        with sidebar.form("login_form"):
            email = st.text_input("Email").strip()
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            submit = col1.form_submit_button("Sign in")
            forgot = col2.form_submit_button("Forgot Password")

            if submit:
                if not email or not password:
                    st.error("Enter email and password.")
                    return
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access restricted to approved domains.")
                    return
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

                    cookies = get_cookies()
                    if cookies and cookies.ready():
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

            if forgot:
                if not email:
                    st.warning("Enter email above for password reset.")
                    return
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Reset restricted to approved domains.")
                    return
                try:
                    firebase_auth.send_password_reset_email(email)
                    st.success("Password reset email sent.")
                except Exception as e:
                    st.error(parse_error(e))

    # --- REGISTER ---
    if action == "Register":
        if not firebase_auth:
            sidebar.warning("Firebase auth not initialized, registration disabled.")
            return
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