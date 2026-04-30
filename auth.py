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
# Floating Authentication Panel Styling
# -----------------------------
def _auth_modal_css():
    """Compact non-blocking styling for the left floating/popover auth panel."""
    st.markdown(
        """
        <style>
        /* IMPORTANT: no full-screen backdrop, no blur, no dashboard dimming. */
        .eusee-auth-panel {
            width: 100%;
            max-width: 360px;
            background: #ffffff;
            border-radius: 18px;
            border: 1px solid rgba(102, 0, 148, 0.14);
            box-shadow: 0 16px 44px rgba(34, 12, 56, 0.18);
            overflow: hidden;
            font-family: Arial, sans-serif;
        }
        .eusee-auth-header {
            padding: 16px 16px 12px 16px;
            background: linear-gradient(135deg, #fbf7ff 0%, #f1e6fb 100%);
            border-bottom: 1px solid rgba(102, 0, 148, 0.10);
        }
        .eusee-auth-brand {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .eusee-auth-logo {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 20px;
            background: linear-gradient(135deg, #660094, #008CAA);
            box-shadow: 0 10px 22px rgba(102, 0, 148, 0.22);
        }
        .eusee-auth-title {
            font-size: 15px;
            font-weight: 900;
            color: #241037;
            line-height: 1.15;
        }
        .eusee-auth-subtitle {
            font-size: 11px;
            color: #6c6275;
            line-height: 1.35;
            margin-top: 3px;
        }
        .eusee-auth-body {
            padding: 14px 16px 16px 16px;
        }
        .eusee-auth-mode {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 9px;
            border-radius: 999px;
            background: #f7effc;
            color: #660094;
            font-size: 10.5px;
            font-weight: 900;
            margin-bottom: 8px;
        }
        .eusee-auth-note {
            font-size: 10.5px;
            color: #6c6275;
            line-height: 1.35;
            margin: 8px 0 2px 0;
        }
        .eusee-auth-footer {
            border-top: 1px solid #f0e7f7;
            padding: 10px 16px 14px 16px;
            background: #fffaff;
            font-size: 10.5px;
            color: #6c6275;
            line-height: 1.35;
        }
        div[data-testid="stPopoverBody"] {
            padding: 0 !important;
            border-radius: 18px !important;
            overflow: hidden !important;
        }
        div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem !important;
        }
        div[data-testid="stPopoverBody"] label p,
        div[data-testid="stExpander"] label p {
            font-size: 11.5px !important;
            font-weight: 800 !important;
            color: #332045 !important;
        }
        div[data-testid="stPopoverBody"] input,
        div[data-testid="stExpander"] input {
            border-radius: 10px !important;
            min-height: 38px !important;
            font-size: 12px !important;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stForm"],
        div[data-testid="stExpander"] div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stPopoverBody"] button[kind="primaryFormSubmit"],
        div[data-testid="stPopoverBody"] button[kind="formSubmit"],
        div[data-testid="stExpander"] button[kind="primaryFormSubmit"],
        div[data-testid="stExpander"] button[kind="formSubmit"] {
            border-radius: 10px !important;
            min-height: 39px !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #660094 0%, #008CAA 100%) !important;
            border: 0 !important;
        }
        </style>
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
    """Compact floating auth panel rendered inside st.popover/st.expander. Non-blocking."""
    _auth_modal_css()

    st.markdown(
        """
        <div class="eusee-auth-panel">
            <div class="eusee-auth-header">
                <div class="eusee-auth-brand">
                    <div class="eusee-auth-logo">🔐</div>
                    <div>
                        <div class="eusee-auth-title">EU SEE Access</div>
                        <div class="eusee-auth-subtitle">Sign in for privileged dashboard features.</div>
                    </div>
                </div>
            </div>
            <div class="eusee-auth-body">
        """,
        unsafe_allow_html=True,
    )

    mode = st.session_state.get("auth_mode", "Login")
    mode_label = "Welcome back" if mode == "Login" else "Create account"
    st.markdown(f'<div class="eusee-auth-mode">{mode_label}</div>', unsafe_allow_html=True)

    if mode == "Login":
        with st.form("eusee_login_float_form"):
            email = st.text_input("Email", placeholder="Enter your email", key="float_login_email").strip()
            password = st.text_input("Password", placeholder="Enter your password", type="password", key="float_login_password")
            remember = st.checkbox("Remember me", value=st.session_state.get("auth_remember", False), key="float_login_remember")
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

        st.markdown('<div class="eusee-auth-note">Need an account or forgot your password?</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create account", use_container_width=True, key="float_create_account_btn"):
                st.session_state.auth_mode = "Register"
                st.rerun()
        with c2:
            if st.button("Reset password", use_container_width=True, key="float_reset_toggle_btn"):
                st.session_state.auth_reset_open = not st.session_state.get("auth_reset_open", False)
                st.rerun()

        if st.session_state.get("auth_reset_open", False):
            with st.form("eusee_reset_float_form"):
                reset_email = st.text_input("Reset email", placeholder="Enter your email", key="float_reset_email").strip()
                reset_submit = st.form_submit_button("Send reset email", use_container_width=True)
                if reset_submit:
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
        with st.form("eusee_register_float_form"):
            email = st.text_input("Email", placeholder="Enter your email", key="float_register_email").strip()
            password = st.text_input("Password", placeholder="Create a password", type="password", key="float_register_password")
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

        if st.button("Back to sign in", use_container_width=True, key="float_back_login_btn"):
            st.session_state.auth_mode = "Login"
            st.rerun()

    st.markdown(
        """
            </div>
            <div class="eusee-auth-footer">
                The panel is non-blocking. You can keep using the dashboard while this access panel is open.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False


# -----------------------------
# Authentication UI Entrypoint
# -----------------------------
def auth_ui():
    """Strictly non-blocking sidebar authentication."""
    init_session()
    restore_session()

    # Safety reset: keep the dashboard clickable; do not use dialog/popover/expander.
    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"], div[data-testid="stApp"], section.main, .main {
            filter: none !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "auth_sidebar_open" not in st.session_state:
        st.session_state.auth_sidebar_open = False

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
        .auth-inline-launch-note {
            margin: 8px 0 8px 0;
            font-family: Arial, sans-serif;
            font-size: 10.5px;
            color: rgba(255,255,255,0.72);
            line-height: 1.35;
        }
        div[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            border-radius: 14px !important;
            min-height: 42px !important;
            font-weight: 850 !important;
        }
        </style>
        <div class="auth-inline-launch-note">
            Sign in only when you need privileged access. This panel does not blur or block the dashboard.
        </div>
        """, unsafe_allow_html=True)

        label = "▾ Hide sign in" if st.session_state.auth_sidebar_open else "🔐 Sign in / Access"
        if st.button(label, use_container_width=True, key="auth_inline_toggle_btn"):
            st.session_state.auth_sidebar_open = not st.session_state.auth_sidebar_open
            st.rerun()

        if st.session_state.auth_sidebar_open:
            _render_auth_panel()
    return
