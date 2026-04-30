# auth.py
import json
import time
from pathlib import Path

import streamlit as st

DEBUG = False  # Set True only while debugging deployment/auth issues

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
    firebase_admin = None
    credentials = None
    HAS_FIREBASE_ADMIN = False

try:
    from streamlit_cookies_manager import EncryptedCookieManager
    HAS_COOKIES = True
except ImportError:
    EncryptedCookieManager = None
    HAS_COOKIES = False


# -----------------------------
# Firebase Admin
# -----------------------------
def init_firebase_admin():
    if not HAS_FIREBASE_ADMIN:
        if DEBUG:
            st.warning("firebase_admin not installed; skipping Admin init.")
        return None

    secrets_admin = st.secrets.get("firebase_admin", {})
    if not secrets_admin:
        if DEBUG:
            st.warning("Firebase Admin secrets missing in secrets.toml; skipping Admin init.")
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
            "universe_domain": secrets_admin.get("universe_domain", "googleapis.com"),
        })
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        return firebase_admin
    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase Admin init failed: {e}")
        return None


# -----------------------------
# Firebase Client
# -----------------------------
def init_firebase_client():
    cfg = st.secrets.get("firebase", {})
    if not cfg:
        if DEBUG:
            st.warning("Firebase client config missing in secrets.toml; login disabled.")
        return None, None

    if not HAS_PYREBASE:
        if DEBUG:
            st.warning("pyrebase not installed; login disabled.")
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
# Access Control
# -----------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)


def get_domain(email: str) -> str:
    return str(email or "").strip().split("@")[-1].lower()


def is_privileged():
    return (
        st.session_state.get("user", False)
        and st.session_state.get("email_verified", False)
        and st.session_state.get("role") == "privileged"
    )


# -----------------------------
# Cookies
# -----------------------------
def get_cookies():
    if not HAS_COOKIES:
        if DEBUG:
            st.warning("streamlit_cookies_manager not installed; cookies disabled.")
        return None

    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            if DEBUG:
                st.warning("Cookie password missing in secrets.toml; cookies disabled.")
            return None
        st.session_state.cookies = EncryptedCookieManager(prefix="eusee", password=password)

    cookies = st.session_state.cookies
    try:
        start = time.time()
        while not cookies.ready() and time.time() - start < 1.0:
            time.sleep(0.05)
        if not cookies.ready():
            return None
        if hasattr(cookies, "sync"):
            cookies.sync()
        elif hasattr(cookies, "load"):
            cookies.load()
    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Cookie load error: {e}")
        return None

    return cookies


# -----------------------------
# Session Helpers
# -----------------------------
def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "restored": False,
        "auth_mode": "Login",
        "auth_remember": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def restore_session():
    if st.session_state.get("restored"):
        return

    cookies = get_cookies()
    if cookies and cookies.ready():
        try:
            if "email" in cookies:
                st.session_state.user = True
                st.session_state.email = cookies.get("email")
                st.session_state.name = cookies.get("name")
                st.session_state.role = cookies.get("role")
                st.session_state.email_verified = str(cookies.get("email_verified", "False")) == "True"
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

    for key in ["user", "email", "name", "role", "email_verified", "restored", "auth_mode", "auth_remember"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def parse_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        data = json.loads(payload)
        msg = data.get("error", {}).get("message", str(e))
    except Exception:
        msg = str(e)

    friendly = {
        "EMAIL_NOT_FOUND": "No account was found for this email address.",
        "INVALID_PASSWORD": "The password is incorrect.",
        "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
        "EMAIL_EXISTS": "An account already exists for this email address.",
        "WEAK_PASSWORD": "Password is too weak. Use at least six characters.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
    }
    return friendly.get(msg, msg)


# -----------------------------
# Modal Styling
# -----------------------------
def _auth_modal_css():
    st.markdown(
        """
        <style>
        /* Hide sidebar auth remnants while modal is active */
        section[data-testid="stSidebar"] { filter: brightness(0.92); }

        /* Dim dashboard background behind the modal */
        .eusee-auth-backdrop {
            position: fixed;
            inset: 0;
            z-index: 9990;
            background: linear-gradient(135deg, rgba(12,15,42,0.76), rgba(65,28,94,0.68));
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }

        /* Streamlit dialog polish */
        div[data-testid="stDialog"] > div {
            border-radius: 22px !important;
            padding: 0 !important;
            max-width: 760px !important;
            box-shadow: 0 26px 80px rgba(10,15,40,0.38) !important;
            border: 1px solid rgba(255,255,255,0.55) !important;
            overflow: hidden !important;
        }
        div[data-testid="stDialog"] div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0 !important;
        }
        div[data-testid="stDialog"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem !important;
        }

        .auth-left-panel {
            min-height: 420px;
            background: linear-gradient(160deg, #ffffff 0%, #f9f5ff 58%, #f2eaff 100%);
            border-right: 1px solid #eee7f7;
            padding: 34px 30px 24px 30px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .auth-logo-title {
            font-family: Arial Black, Arial, sans-serif;
            color: #660094;
            font-size: 42px;
            line-height: 0.92;
            letter-spacing: -1px;
            margin: 0 0 8px 0;
        }
        .auth-logo-subtitle {
            color: #660094;
            font-family: Arial, sans-serif;
            font-size: 11px;
            font-weight: 900;
            line-height: 1.35;
            text-transform: uppercase;
        }
        .auth-shield {
            width: 145px;
            height: 145px;
            margin: 34px auto 0 auto;
            border-radius: 50%;
            background: radial-gradient(circle, #ffffff 0%, #f1e4ff 62%, #ead7ff 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 0 1px rgba(102,0,148,0.10), 0 14px 36px rgba(102,0,148,0.14);
            color: #660094;
            font-size: 56px;
        }
        .auth-right-title {
            font-family: Arial Black, Arial, sans-serif;
            color: #231942;
            text-align: center;
            font-size: 22px;
            margin: 14px 0 4px 0;
        }
        .auth-right-subtitle {
            font-family: Arial, sans-serif;
            color: #6b6475;
            text-align: center;
            font-size: 12px;
            margin-bottom: 14px;
        }
        .auth-caption {
            font-family: Arial, sans-serif;
            color: #6b6475;
            text-align: center;
            font-size: 11px;
            margin-top: 10px;
            line-height: 1.35;
        }
        div[data-testid="stDialog"] label p {
            font-size: 12px !important;
            font-weight: 800 !important;
            color: #332045 !important;
        }
        div[data-testid="stDialog"] input {
            border-radius: 10px !important;
            min-height: 42px !important;
            font-size: 12px !important;
        }
        div[data-testid="stDialog"] div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 22px 18px 22px !important;
        }
        div[data-testid="stDialog"] button[kind="primaryFormSubmit"],
        div[data-testid="stDialog"] button[kind="formSubmit"] {
            border-radius: 10px !important;
            min-height: 42px !important;
            font-weight: 900 !important;
        }
        </style>
        <div class="eusee-auth-backdrop"></div>
        """,
        unsafe_allow_html=True,
    )


def _save_cookie_session(email, name, verified, role, remember=False):
    if not remember:
        return
    cookies = get_cookies()
    if cookies and cookies.ready():
        cookies["email"] = email
        cookies["name"] = name
        cookies["email_verified"] = str(bool(verified))
        cookies["role"] = role
        try:
            cookies.save()
        except Exception:
            pass


def _render_auth_panel():
    _auth_modal_css()

    c1, c2 = st.columns([0.92, 1.08], gap="large")

    with c1:
        st.markdown(
            """
            <div class="auth-left-panel">
                <div>
                    <div class="auth-logo-title">EU SEE</div>
                    <div class="auth-logo-subtitle">Supporting<br>An Enabling Environment<br>For Civil Society</div>
                </div>
                <div class="auth-shield">🔐</div>
                <div class="auth-caption">Secure dashboard access for approved EU SEE users.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="auth-right-title">Welcome Back</div>
            <div class="auth-right-subtitle">Sign in to continue to the EU SEE Dashboard</div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("auth_mode", "Login") == "Login":
            with st.form("eusee_login_modal_form"):
                email = st.text_input("Email", placeholder="Enter your email").strip()
                password = st.text_input("Password", placeholder="Enter your password", type="password")
                remember = st.checkbox("Remember me", value=st.session_state.get("auth_remember", False))
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if not firebase_auth:
                        st.error("Firebase authentication is not initialized.")
                        return False
                    if not email or not password:
                        st.error("Enter email and password.")
                        return False
                    if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
                        st.error("Access is restricted to approved domains.")
                        return False
                    try:
                        user = firebase_auth.sign_in_with_email_and_password(email, password)
                        info = firebase_auth.get_account_info(user["idToken"])
                        verified = bool(info["users"][0].get("emailVerified", False))
                        role = "privileged" if verified else "restricted"

                        st.session_state.user = True
                        st.session_state.email = email
                        st.session_state.name = email.split("@")[0].replace(".", " ").title()
                        st.session_state.email_verified = verified
                        st.session_state.role = role
                        st.session_state.auth_remember = remember
                        _save_cookie_session(email, st.session_state.name, verified, role, remember)
                        st.rerun()
                    except Exception as e:
                        st.error(parse_error(e))
                        return False

            col_a, col_b, col_c = st.columns([1, 0.55, 1])
            with col_b:
                st.caption("or")
            if st.button("Login with SSO", use_container_width=True, disabled=True):
                st.info("SSO can be connected later if enabled by your identity provider.")
            st.markdown('<div class="auth-caption">Need access? Contact the EU SEE dashboard administrator.</div>', unsafe_allow_html=True)
            if st.button("Create an account", use_container_width=True):
                st.session_state.auth_mode = "Register"
                st.rerun()

            with st.expander("Forgot password?"):
                reset_email = st.text_input("Reset email", placeholder="Enter your email", key="reset_email_modal").strip()
                if st.button("Send password reset", use_container_width=True):
                    if not firebase_auth:
                        st.error("Firebase authentication is not initialized.")
                    elif not reset_email:
                        st.warning("Enter your email first.")
                    elif PRIVILEGED_DOMAINS and get_domain(reset_email) not in PRIVILEGED_DOMAINS:
                        st.error("Password reset is restricted to approved domains.")
                    else:
                        try:
                            firebase_auth.send_password_reset_email(reset_email)
                            st.success("Password reset email sent.")
                        except Exception as e:
                            st.error(parse_error(e))

        else:
            with st.form("eusee_register_modal_form"):
                email = st.text_input("Email", placeholder="Enter your email").strip()
                password = st.text_input("Password", placeholder="Create a password", type="password")
                submitted = st.form_submit_button("Register", use_container_width=True)
                if submitted:
                    if not firebase_auth:
                        st.error("Firebase authentication is not initialized.")
                        return False
                    if not email or not password:
                        st.error("Enter email and password.")
                        return False
                    if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
                        st.error("Registration is restricted to approved domains.")
                        return False
                    try:
                        user = firebase_auth.create_user_with_email_and_password(email, password)
                        firebase_auth.send_email_verification(user["idToken"])
                        st.success("Registration successful. Check your email to verify your account.")
                    except Exception as e:
                        st.error(parse_error(e))
                        return False
            if st.button("Back to sign in", use_container_width=True):
                st.session_state.auth_mode = "Login"
                st.rerun()

    return False


# -----------------------------
# Authentication UI Entrypoint
# -----------------------------
def auth_ui():
    """Non-blocking floating-left authentication launcher."""
    init_session()
    restore_session()

    if st.session_state.get("user"):
        with st.sidebar:
            st.markdown("""
            <style>
            .auth-floating-card {
                background: linear-gradient(135deg, rgba(102,0,148,0.18), rgba(255,255,255,0.06));
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 14px;
                padding: 10px 12px;
                margin: 8px 0 10px 0;
                box-shadow: 0 8px 22px rgba(0,0,0,0.10);
                font-family: Arial, sans-serif;
            }
            .auth-floating-title {font-size:12px;font-weight:800;color:#ffffff;margin-bottom:3px;}
            .auth-floating-subtitle {font-size:10.5px;color:rgba(255,255,255,0.78);line-height:1.35;}
            </style>
            """, unsafe_allow_html=True)
            st.markdown(f'''
            <div class="auth-floating-card">
                <div class="auth-floating-title">👋 {st.session_state.get('name') or 'Signed in'}</div>
                <div class="auth-floating-subtitle">Authenticated EU SEE dashboard session</div>
            </div>
            ''', unsafe_allow_html=True)
            if not st.session_state.get("email_verified"):
                st.warning("Email not verified. Please verify your email before accessing privileged dashboard features.")
            if st.button("Logout", use_container_width=True, key="auth_sidebar_logout"):
                logout()
        return

    with st.sidebar:
        st.markdown("""
        <style>
        div[data-testid="stPopover"] > button {
            width: 100%;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            background: linear-gradient(135deg, #660094 0%, #3b075c 100%) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 24px rgba(102,0,148,0.28) !important;
            min-height: 42px !important;
        }
        div[data-testid="stPopover"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(102,0,148,0.34) !important;
        }
        .auth-left-note {
            margin: 6px 0 10px 0;
            font-family: Arial, sans-serif;
            font-size: 10.5px;
            color: rgba(255,255,255,0.72);
            line-height: 1.35;
        }
        </style>
        <div class="auth-left-note">Sign in only when you need privileged access. The dashboard remains visible.</div>
        """, unsafe_allow_html=True)
        if hasattr(st, "popover"):
            with st.popover("🔐 Sign in / Access", use_container_width=True):
                _render_auth_panel()
        else:
            with st.expander("🔐 Sign in / Access", expanded=False):
                _render_auth_panel()
    return
