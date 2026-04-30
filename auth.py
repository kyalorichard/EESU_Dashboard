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


def is_authenticated():
    """Return True when the user has a verified authenticated session."""
    init_session()
    restore_session()
    return (
        st.session_state.get("user", False)
        and bool(st.session_state.get("email_verified", False))
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
                    st.session_state.auth_view = False
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
    """Premium non-blocking sign-in view shown after Sign in / Access is clicked."""
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        st.rerun()

    mode = st.session_state.get("auth_mode", "Login")
    mode_title = "Welcome back" if mode == "Login" else "Create account"
    mode_copy = (
        "Sign in to unlock privileged dashboard features and return with an active session."
        if mode == "Login"
        else "Register with an approved organisational email, then verify your email before access is enabled."
    )

    st.markdown(f"""
    <style>
    .eusee-auth-wrap {{ width:100%; margin:8px 0 20px 0; font-family:Arial, sans-serif; }}
    .eusee-auth-shell {{
        width:min(1120px, 98vw); margin:0 auto; display:grid; grid-template-columns:0.95fr 1.05fr;
        background:#fff; border:1px solid rgba(102,0,148,.12); border-radius:30px; overflow:hidden;
        box-shadow:0 22px 60px rgba(45,0,85,.16);
    }}
    .eusee-auth-left {{
        min-height:440px; padding:42px 42px 34px; color:#fff; position:relative;
        background: radial-gradient(circle at 86% 18%, rgba(255,219,88,.34), transparent 22%),
                    radial-gradient(circle at 18% 88%, rgba(0,140,170,.25), transparent 25%),
                    linear-gradient(145deg,#2d0055 0%,#660094 55%,#008CAA 130%);
    }}
    .eusee-auth-left::after {{ content:""; position:absolute; inset:18px; border-radius:24px; border:1px solid rgba(255,255,255,.14); pointer-events:none; }}
    .eusee-auth-brand {{ display:flex; align-items:center; gap:12px; position:relative; z-index:1; margin-bottom:34px; }}
    .eusee-auth-mark {{ width:50px; height:50px; border-radius:17px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.22); font-size:24px; box-shadow:0 16px 34px rgba(0,0,0,.18); }}
    .eusee-auth-name {{ font-size:28px; font-weight:950; line-height:1; letter-spacing:-.6px; }}
    .eusee-auth-subbrand {{ font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:rgba(255,255,255,.72); margin-top:5px; }}
    .eusee-auth-hero {{ position:relative; z-index:1; max-width:410px; }}
    .eusee-auth-hero h2 {{ margin:0 0 12px 0; font-size:31px; line-height:1.12; font-weight:950; color:#fff; }}
    .eusee-auth-hero p {{ margin:0; font-size:13px; line-height:1.62; color:rgba(255,255,255,.84); }}
    .eusee-auth-benefits {{ margin-top:28px; display:grid; gap:10px; position:relative; z-index:1; }}
    .eusee-auth-benefit {{ width:fit-content; padding:9px 12px; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.15); font-size:12px; font-weight:800; }}
    .eusee-auth-right {{ padding:34px 42px; background:linear-gradient(180deg,rgba(249,247,252,.96),#fff 34%); }}
    .eusee-auth-top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:20px; }}
    .eusee-auth-pill {{ padding:8px 12px; border-radius:999px; background:rgba(102,0,148,.08); color:#660094; font-size:12px; font-weight:950; }}
    .eusee-auth-mode {{ padding:7px 10px; border-radius:999px; background:#fff9dc; color:#6f5200; border:1px solid rgba(255,219,88,.65); font-size:11px; font-weight:900; }}
    .eusee-auth-heading {{ margin:0 0 6px 0; font-size:30px; font-weight:950; color:#231331; letter-spacing:-.4px; }}
    .eusee-auth-copy {{ margin:0 0 18px 0; font-size:13px; color:#6b6174; line-height:1.55; max-width:560px; }}
    .eusee-auth-card {{ background:rgba(255,255,255,.96); border:1px solid rgba(102,0,148,.10); border-radius:22px; padding:22px; box-shadow:0 16px 38px rgba(45,0,85,.09); }}
    .eusee-auth-card label p {{ color:#332045!important; font-weight:900!important; font-size:12px!important; }}
    .eusee-auth-card input {{ border-radius:14px!important; min-height:46px!important; border:1px solid rgba(102,0,148,.16)!important; font-size:13px!important; }}
    .eusee-auth-card div[data-testid="stForm"] {{ border:0!important; padding:0!important; }}
    .eusee-auth-card button[kind="primaryFormSubmit"], .eusee-auth-card button[kind="formSubmit"] {{ min-height:46px!important; border-radius:14px!important; font-weight:950!important; background:linear-gradient(135deg,#660094 0%,#008CAA 100%)!important; border:0!important; box-shadow:0 10px 22px rgba(102,0,148,.18)!important; }}
    .eusee-auth-card .stButton>button {{ border-radius:14px!important; min-height:40px!important; font-weight:900!important; }}
    .eusee-auth-notes {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:14px; }}
    .eusee-auth-note {{ background:#fbf8fe; border:1px solid #efe4f7; border-radius:15px; padding:10px 11px; color:#4d405d; font-size:11px; line-height:1.35; }}
    .eusee-auth-note b {{ color:#660094; }}
    .eusee-auth-footer {{ margin-top:14px; color:#70667a; font-size:11.5px; line-height:1.45; }}
    @media(max-width:900px) {{ .eusee-auth-shell{{grid-template-columns:1fr}} .eusee-auth-left{{min-height:320px;padding:34px}} .eusee-auth-right{{padding:28px 22px}} .eusee-auth-notes{{grid-template-columns:1fr}} }}
    </style>
    <div class="eusee-auth-wrap"><div class="eusee-auth-shell">
      <section class="eusee-auth-left">
        <div class="eusee-auth-brand"><div class="eusee-auth-mark">🔐</div><div><div class="eusee-auth-name">EU SEE</div><div class="eusee-auth-subbrand">Dashboard Access</div></div></div>
        <div class="eusee-auth-hero"><h2>Secure access to deeper dashboard intelligence.</h2><p>Sign in to access protected analytical features, country-level views, exports, and privileged indicators while keeping the public dashboard available.</p></div>
        <div class="eusee-auth-benefits"><div class="eusee-auth-benefit">✓ Approved-domain access control</div><div class="eusee-auth-benefit">✓ Active session after login</div><div class="eusee-auth-benefit">✓ No pop-up overlay or dashboard blocking</div></div>
      </section>
      <section class="eusee-auth-right">
        <div class="eusee-auth-top"><div class="eusee-auth-pill">Secure sign-in workspace</div><div class="eusee-auth-mode">{mode}</div></div>
        <h2 class="eusee-auth-heading">{mode_title}</h2><p class="eusee-auth-copy">{mode_copy}</p>
        <div class="eusee-auth-card">
    """, unsafe_allow_html=True)

    if mode == "Login":
        with st.form("eusee_login_route_form"):
            email = st.text_input("Email address", placeholder="name@organisation.org", key="route_login_email").strip()
            password = st.text_input("Password", placeholder="Enter your password", type="password", key="route_login_password")
            remember = st.checkbox("Keep me signed in on this device", value=st.session_state.get("auth_remember", True), key="route_login_remember")
            submitted = st.form_submit_button("Sign in and return to dashboard", use_container_width=True)
            if submitted:
                if not firebase_auth:
                    st.error("Firebase authentication is not initialized.")
                elif not email or not password:
                    st.error("Enter email and password.")
                elif PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access is restricted to approved domains.")
                else:
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
                        st.session_state.auth_view = False
                        _save_cookie_session(email, st.session_state.name, verified, role, remember)
                        st.rerun()
                    except Exception as e:
                        st.error(parse_error(e))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create account", use_container_width=True, key="route_create_account_btn"):
                st.session_state.auth_mode = "Register"
                st.session_state.auth_reset_open = False
                st.rerun()
        with c2:
            if st.button("Reset password", use_container_width=True, key="route_reset_toggle_btn"):
                st.session_state.auth_reset_open = not st.session_state.get("auth_reset_open", False)
                st.rerun()
        if st.session_state.get("auth_reset_open", False):
            st.markdown("---")
            with st.form("eusee_reset_route_form"):
                reset_email = st.text_input("Password reset email", placeholder="name@organisation.org", key="route_reset_email").strip()
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
        with st.form("eusee_register_route_form"):
            email = st.text_input("Email address", placeholder="name@organisation.org", key="route_register_email").strip()
            password = st.text_input("Password", placeholder="Create a strong password", type="password", key="route_register_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                if not firebase_auth:
                    st.error("Firebase authentication is not initialized.")
                elif not email or not password:
                    st.error("Enter email and password.")
                elif PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Registration is restricted to approved domains.")
                else:
                    try:
                        user = firebase_auth.create_user_with_email_and_password(email, password)
                        firebase_auth.send_email_verification(user["idToken"])
                        st.success("Registration successful. Check your email to verify your account, then sign in.")
                    except Exception as e:
                        st.error(parse_error(e))
        if st.button("Back to sign in", use_container_width=True, key="route_back_login_btn"):
            st.session_state.auth_mode = "Login"
            st.rerun()

    st.markdown("""
        </div>
        <div class="eusee-auth-notes"><div class="eusee-auth-note"><b>Access</b><br>Approved organisational domains only.</div><div class="eusee-auth-note"><b>Security</b><br>Email verification is required.</div><div class="eusee-auth-note"><b>Session</b><br>Remembered users return faster.</div></div>
        <div class="eusee-auth-footer">Use the button below to return to the public dashboard without signing in.</div>
      </section>
    </div></div>
    """, unsafe_allow_html=True)

    if st.button("← Back to dashboard", use_container_width=False, key="route_back_dashboard_btn"):
        st.session_state.auth_view = False
        st.session_state.auth_mode = "Login"
        st.session_state.auth_reset_open = False
        st.rerun()
