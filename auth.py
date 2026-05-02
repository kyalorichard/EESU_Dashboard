# auth.py
import json
import time
import streamlit as st

DEBUG = False

# ============================================================
# Optional imports
# ============================================================
try:
    import pyrebase
    HAS_PYREBASE = True
except ImportError:
    pyrebase = None
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


# ============================================================
# Firebase setup
# ============================================================
def init_firebase_admin():
    if not HAS_FIREBASE_ADMIN:
        return None

    secrets_admin = st.secrets.get("firebase_admin", {})
    if not secrets_admin:
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


def init_firebase_client():
    cfg = st.secrets.get("firebase", {})
    if not cfg or not HAS_PYREBASE:
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

PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)


# ============================================================
# Session helpers
# ============================================================
def get_domain(email: str) -> str:
    return str(email or "").strip().split("@")[-1].lower()


def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "restored": False,
        "auth_mode": "Login",
        "auth_remember": True,
        "auth_view": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def is_authenticated():
    init_session()
    restore_session()
    return bool(st.session_state.get("user") and st.session_state.get("email_verified"))


def is_privileged():
    init_session()
    restore_session()
    return (
        st.session_state.get("user", False)
        and st.session_state.get("email_verified", False)
        and st.session_state.get("role") == "privileged"
    )


# ============================================================
# Cookies
# ============================================================
def get_cookies():
    if not HAS_COOKIES:
        return None

    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            return None

        st.session_state.cookies = EncryptedCookieManager(
            prefix="eusee",
            password=password
        )

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

    except Exception:
        return None

    return cookies


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
                st.session_state.email_verified = str(
                    cookies.get("email_verified", "False")
                ) == "True"
        except Exception:
            pass

    st.session_state.restored = True


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

    for key in [
        "user", "email", "name", "role", "email_verified", "restored",
        "auth_mode", "auth_remember", "auth_view"
    ]:
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


# ============================================================
# CSS
# ============================================================
def _auth_page_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        section[data-testid="stSidebar"],
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        footer {
            display: none !important;
        }

        html, body, .stApp {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            min-height: 100vh !important;
            font-family: Inter, Arial, sans-serif !important;
            background:
                radial-gradient(circle at 28% 40%, rgba(37, 99, 235, 0.18), transparent 26%),
                radial-gradient(circle at 82% 12%, rgba(124, 58, 237, 0.15), transparent 28%),
                linear-gradient(135deg, #020617 0%, #071426 42%, #0b1b3a 100%) !important;
            color: #ffffff !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            margin: 0;
        }

        .eusee-topbar {
            min-height: 74px;
            padding: 0 44px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(2, 6, 23, 0.62);
            backdrop-filter: blur(18px);
            display: flex;
            align-items: center;
        }

        .brand-line {
            display: flex;
            align-items: center;
            gap: 13px;
            height: 74px;
        }

        .brand-stars {
            color: #facc15 !important;
            font-size: 24px;
            font-weight: 900;
            letter-spacing: -5px;
        }

        .brand-main {
            color: #ffffff !important;
            font-size: 25px;
            font-weight: 900;
            letter-spacing: -0.045em;
        }

        .brand-sub {
            color: rgba(226, 232, 240, 0.76) !important;
            font-size: 14px;
            letter-spacing: 0.055em;
            font-weight: 700;
        }

        .top-nav {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 22px;
            height: 74px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            padding: 9px 15px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: white !important;
            font-size: 13px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
            white-space: nowrap;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.85);
            display: inline-block;
        }

        .top-link {
            color: rgba(255,255,255,0.86) !important;
            font-size: 13px;
            font-weight: 650;
            white-space: nowrap;
        }

        .page-pad {
            padding: 34px 56px 18px 56px;
        }

        .left-panel {
            min-height: 720px;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: hidden;
            box-sizing: border-box;
        }

        .left-panel::after {
            content: "";
            position: absolute;
            right: -54px;
            top: 130px;
            width: 550px;
            height: 550px;
            opacity: 0.72;
            background:
                radial-gradient(circle, rgba(37,99,235,0.36) 0 2px, transparent 2px);
            background-size: 9px 9px;
            mask-image: radial-gradient(circle, black 0%, transparent 72%);
            -webkit-mask-image: radial-gradient(circle, black 0%, transparent 72%);
            pointer-events: none;
        }

        .left-panel::before {
            content: "";
            position: absolute;
            right: -10px;
            top: 250px;
            width: 530px;
            height: 330px;
            border: 1px solid rgba(59,130,246,0.34);
            border-left: none;
            border-bottom: none;
            border-radius: 50%;
            transform: rotate(-17deg);
            opacity: 0.62;
            pointer-events: none;
        }

        .left-inner {
            position: relative;
            z-index: 2;
            max-width: 600px;
        }

        .left-title {
            color: #ffffff !important;
            font-size: 52px;
            line-height: 1.05;
            font-weight: 950;
            letter-spacing: -0.05em;
            margin: 0 0 4px 0;
        }

        .left-gradient {
            display: block;
            font-size: 44px;
            line-height: 1.08;
            font-weight: 950;
            letter-spacing: -0.05em;
            margin: 0 0 20px 0;
            background: linear-gradient(90deg, #a855f7 0%, #38bdf8 76%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .left-lead {
            max-width: 520px;
            margin-bottom: 36px;
            color: rgba(226, 232, 240, 0.88) !important;
            font-size: 18px;
            line-height: 1.58;
            font-weight: 500;
        }

        .feature-icon {
            width: 58px;
            height: 58px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, rgba(124,58,237,0.54), rgba(30,64,175,0.48));
            color: #ddd6fe !important;
            font-size: 27px;
            box-shadow: 0 16px 34px rgba(0,0,0,0.18);
        }

        .feature-title {
            color: #ffffff !important;
            font-size: 19px;
            font-weight: 850;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
        }

        .feature-desc {
            color: rgba(226, 232, 240, 0.78) !important;
            font-size: 15px;
            line-height: 1.38;
            margin: 0;
        }

        .left-rule {
            width: 88%;
            height: 1px;
            background: rgba(148, 163, 184, 0.22);
            margin: 26px 0 24px 0;
        }

        .security-icon {
            width: 68px;
            height: 68px;
            border-radius: 23px;
            display: grid;
            place-items: center;
            color: #22d3ee !important;
            border: 1px solid rgba(34, 211, 238, 0.72);
            font-size: 29px;
        }

        .security-title {
            margin: 0 0 8px 0;
            color: #22d3ee !important;
            font-size: 17px;
            font-weight: 850;
        }

        .security-desc {
            margin: 0;
            color: rgba(226, 232, 240, 0.80) !important;
            line-height: 1.56;
            font-size: 15px;
        }

        .copyright {
            position: absolute;
            left: 0;
            bottom: 2px;
            z-index: 2;
            color: rgba(226, 232, 240, 0.62) !important;
            font-size: 12.5px;
        }

        .auth-card div[data-testid="stVerticalBlockBorderWrapper"] {
            width: min(735px, 100%) !important;
            min-height: 720px !important;
            margin: 0 auto !important;
            color: #0f172a !important;
            border-radius: 18px !important;
            border: 1px solid rgba(255,255,255,0.82) !important;
            background:
                radial-gradient(circle at 88% 6%, rgba(124, 58, 237, 0.07), transparent 24%),
                #ffffff !important;
            box-shadow: 0 32px 95px rgba(0,0,0,0.42) !important;
        }

        .auth-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 30px 46px !important;
        }

        .auth-card * {
            color: #0f172a;
        }

        .auth-card .stButton > button {
            border-radius: 10px !important;
            border: 1px solid #d7dce7 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-weight: 750 !important;
            min-height: 42px !important;
            transition: all 0.15s ease !important;
        }

        .auth-card .stButton > button:hover {
            border-color: #7c3aed !important;
            color: #5b21b6 !important;
            box-shadow: 0 8px 18px rgba(124,58,237,0.12) !important;
        }

        .back-btn .stButton > button {
            width: auto !important;
            min-height: 42px !important;
            padding: 0.45rem 1rem !important;
            border: 1px solid rgba(124, 58, 237, 0.45) !important;
            color: #5b21b6 !important;
            background: #ffffff !important;
            border-radius: 10px !important;
            font-weight: 850 !important;
            box-shadow: 0 6px 16px rgba(124,58,237,0.09) !important;
        }

        .login-lock {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #f1e9ff;
            color: #6d28d9 !important;
            font-size: 31px;
            font-weight: 900;
        }

        .login-heading h1 {
            margin: 0 0 9px 0 !important;
            color: #0b102f !important;
            font-size: 32px !important;
            line-height: 1.12 !important;
            font-weight: 950 !important;
            letter-spacing: -0.04em !important;
        }

        .login-heading p {
            margin: 0 !important;
            color: #475569 !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        .or-divider {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 18px;
            align-items: center;
            color: #64748b !important;
            font-weight: 750;
            margin: 30px 0 24px 0;
            font-size: 14px;
        }

        .or-divider::before,
        .or-divider::after {
            content: "";
            height: 1px;
            background: #d5dbe8;
        }

        .auth-card label p {
            color: #0f172a !important;
            font-size: 13px !important;
            font-weight: 850 !important;
            margin-bottom: 5px !important;
        }

        .auth-card input {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        .auth-card div[data-testid="stTextInput"] {
            margin-bottom: 13px !important;
        }

        .auth-card div[data-testid="stTextInput"] input {
            height: 54px !important;
            border-radius: 10px !important;
            border: 1px solid #d5dbe8 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-size: 15px !important;
            box-shadow: none !important;
        }

        .auth-card div[data-testid="stTextInput"] input:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
        }

        .auth-card div[data-testid="stCheckbox"] {
            margin-top: 4px !important;
            margin-bottom: 14px !important;
        }

        .auth-card div[data-testid="stCheckbox"] label,
        .auth-card div[data-testid="stCheckbox"] label span {
            color: #0f172a !important;
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        .auth-card button[kind="primaryFormSubmit"],
        .auth-card button[kind="formSubmit"] {
            width: 100% !important;
            min-height: 58px !important;
            height: 58px !important;
            border-radius: 10px !important;
            border: none !important;
            color: #ffffff !important;
            font-size: 18px !important;
            font-weight: 900 !important;
            background: linear-gradient(90deg, #6d28d9, #2563eb) !important;
            box-shadow: 0 10px 25px rgba(37,99,235,0.30) !important;
            transition: all 0.15s ease !important;
        }

        .auth-card button[kind="primaryFormSubmit"]:hover,
        .auth-card button[kind="formSubmit"]:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        .switch-text {
            text-align: center;
            margin: 20px 0 14px 0;
            color: #64748b !important;
            font-size: 14.5px;
            font-weight: 500;
        }

        .switch-text strong {
            color: #4f46e5 !important;
            font-weight: 850;
            margin-left: 12px;
        }

        .notice {
            margin-top: 22px;
            border: 1px solid rgba(245, 158, 11, 0.48);
            background: linear-gradient(180deg, #fffbeb 0%, #fff7ed 100%);
            border-radius: 12px;
            padding: 18px 20px;
            color: #1f2937 !important;
        }

        .notice h3 {
            margin: 0 0 8px 0 !important;
            color: #3b2f0b !important;
            font-size: 15.5px !important;
            font-weight: 850 !important;
        }

        .notice p {
            margin: 5px 0 !important;
            color: #111827 !important;
            font-size: 13.5px !important;
        }

        .notice-check {
            color: #16a34a !important;
            font-weight: 900;
            margin-right: 8px;
        }

        .bottom-footer {
            text-align: center;
            color: rgba(226,232,240,0.68) !important;
            font-size: 13px;
            margin: 12px 0 18px 0;
            font-weight: 500;
        }

        .auth-card div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
        }

        .auth-card div[data-testid="stForm"] > div {
            gap: 0.35rem !important;
        }

        @media (max-width: 980px) {
            .top-nav {
                display: none !important;
            }

            .page-pad {
                padding: 24px;
            }

            .left-panel {
                min-height: auto;
                padding: 32px 0;
            }

            .copyright {
                position: static;
                margin-top: 34px;
            }

            .auth-card div[data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto !important;
                margin-top: 24px !important;
            }
        }

        @media (max-width: 640px) {
            .eusee-topbar {
                padding: 0 18px;
            }

            .brand-sub {
                display: none !important;
            }

            .page-pad {
                padding: 18px;
            }

            .left-title {
                font-size: 38px;
            }

            .left-gradient {
                font-size: 32px;
            }

            .auth-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 24px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Render UI
# ============================================================
def _render_topbar():
    st.markdown('<div class="eusee-topbar">', unsafe_allow_html=True)
    left, right = st.columns([0.60, 0.40], vertical_alignment="center")

    with left:
        st.markdown(
            """
            <div class="brand-line">
                <span class="brand-stars">✦✦✦</span>
                <span class="brand-main">EU SEE</span>
                <span class="brand-sub">INTELLIGENCE PLATFORM</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="top-nav">
                <span class="status-pill"><span class="status-dot"></span>System Status: <strong>Operational</strong></span>
                <span class="top-link">ⓘ Help</span>
                <span class="top-link">▤ Docs</span>
                <span class="top-link">◎ English⌄</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_feature(icon, title, desc):
    icon_col, text_col = st.columns([0.14, 0.86], vertical_alignment="center")
    with icon_col:
        st.markdown(f'<div class="feature-icon">{icon}</div>', unsafe_allow_html=True)
    with text_col:
        st.markdown(f'<p class="feature-title">{title}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="feature-desc">{desc}</p>', unsafe_allow_html=True)


def _render_left_panel():
    st.markdown('<div class="left-panel"><div class="left-inner">', unsafe_allow_html=True)

    st.markdown('<div class="left-title">EU SEE</div>', unsafe_allow_html=True)
    st.markdown('<div class="left-gradient">Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="left-lead">
            Secure access to real-time geopolitical analytics,
            risk signals, and cross-country monitoring across
            South East Europe and beyond.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_feature("◎", "86 Countries Monitored", "Comprehensive coverage and real-time updates")
    _render_feature("▮", "Real-time Signal Processing", "AI-powered detection and analytics engine")
    _render_feature("◇", "AI-driven Risk Classification", "Advanced models for early risk identification")

    st.markdown('<div class="left-rule"></div>', unsafe_allow_html=True)

    sec_icon, sec_text = st.columns([0.16, 0.84], vertical_alignment="center")
    with sec_icon:
        st.markdown('<div class="security-icon">🔐</div>', unsafe_allow_html=True)
    with sec_text:
        st.markdown('<p class="security-title">Enterprise-grade security</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="security-desc">Your data is protected with end-to-end encryption and strict access controls.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('</div><div class="copyright">© 2024 EU SEE Intelligence Platform. All rights reserved.</div></div>', unsafe_allow_html=True)


def _render_login_header(mode_title, mode_subtitle):
    lock_col, text_col = st.columns([0.13, 0.87], vertical_alignment="center")
    with lock_col:
        st.markdown('<div class="login-lock">▣</div>', unsafe_allow_html=True)
    with text_col:
        st.markdown(
            f"""
            <div class="login-heading">
                <h1>{mode_title}</h1>
                <p>{mode_subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)


def _render_notice():
    st.markdown(
        """
        <div class="notice">
            <h3>🛡️ Secure Access Notice</h3>
            <p><span class="notice-check">✓</span>Access restricted to verified institutional domains</p>
            <p><span class="notice-check">✓</span>All activity is logged and monitored</p>
            <p><span class="notice-check">✓</span>Session protected with enterprise-grade encryption</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Form actions
# ============================================================
def _set_auth_mode(mode: str):
    st.session_state.auth_mode = mode
    st.rerun()


def _back_to_dashboard():
    st.session_state.auth_view = False
    st.rerun()


def _login_form():
    with st.form("eusee_login_form", clear_on_submit=False):
        email = st.text_input(
            "Email address",
            placeholder="✉   name@organization.org",
            key="login_email",
        ).strip()

        password = st.text_input(
            "Password",
            placeholder="🔒   Enter your password",
            type="password",
            key="login_password",
        )

        remember = st.checkbox(
            "Keep me signed in on this device",
            value=st.session_state.get("auth_remember", True),
            key="login_remember",
        )

        submitted = st.form_submit_button(
            "🔒  Access Intelligence Dashboard",
            use_container_width=True
        )

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized.")
            return

        if not email or not password:
            st.error("Enter email and password.")
            return

        if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Access is restricted to approved institutional domains.")
            return

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

            st.success("Signed in successfully. Redirecting to dashboard...")
            st.rerun()

        except Exception as e:
            st.error(parse_error(e))

    st.markdown(
        '<div class="switch-text">Don’t have an account? <strong>Create account</strong></div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create account", use_container_width=True, key="switch_register_btn"):
            _set_auth_mode("Register")
    with c2:
        if st.button("Forgot password?", use_container_width=True, key="switch_reset_btn"):
            _set_auth_mode("Reset")


def _register_form():
    with st.form("eusee_register_form", clear_on_submit=False):
        email = st.text_input(
            "Email address",
            placeholder="✉   name@organization.org",
            key="register_email",
        ).strip()

        password = st.text_input(
            "Password",
            placeholder="🔒   Create a secure password",
            type="password",
            key="register_password",
        )

        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized.")
            return

        if not email or not password:
            st.error("Enter email and password.")
            return

        if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Registration is restricted to approved institutional domains.")
            return

        try:
            user = firebase_auth.create_user_with_email_and_password(email, password)
            firebase_auth.send_email_verification(user["idToken"])
            st.success("Registration successful. Check your email to verify your account, then sign in.")
        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to sign in", use_container_width=True, key="register_back_btn"):
        _set_auth_mode("Login")


def _reset_form():
    with st.form("eusee_reset_form", clear_on_submit=False):
        reset_email = st.text_input(
            "Email address",
            placeholder="✉   name@organization.org",
            key="reset_email",
        ).strip()

        submitted = st.form_submit_button(
            "Send password reset link",
            use_container_width=True
        )

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized.")
            return

        if not reset_email:
            st.warning("Enter your email address first.")
            return

        if PRIVILEGED_DOMAINS and get_domain(reset_email) not in PRIVILEGED_DOMAINS:
            st.error("Password reset is restricted to approved institutional domains.")
            return

        try:
            firebase_auth.send_password_reset_email(reset_email)
            st.success("Password reset email sent.")
        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to sign in", use_container_width=True, key="reset_back_btn"):
        _set_auth_mode("Login")


# ============================================================
# Main auth page
# ============================================================
def _render_auth_page():
    _auth_page_css()
    _render_topbar()

    mode = st.session_state.get("auth_mode", "Login")

    mode_title = {
        "Login": "Sign in to your workspace",
        "Register": "Create your account",
        "Reset": "Reset password",
    }.get(mode, "Sign in to your workspace")

    mode_subtitle = {
        "Login": "Access your authorized EU SEE dashboard and analytics.",
        "Register": "Request access using an approved institutional email address.",
        "Reset": "Enter your approved email address to receive a password reset link.",
    }.get(mode, "Access your authorized EU SEE dashboard and analytics.")

    st.markdown('<div class="page-pad">', unsafe_allow_html=True)

    left_col, right_col = st.columns([0.445, 0.555], gap="large", vertical_alignment="center")

    with left_col:
        _render_left_panel()

    with right_col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        with st.container(border=True):
            back_left, back_right = st.columns([0.68, 0.32])
            with back_right:
                st.markdown('<div class="back-btn">', unsafe_allow_html=True)
                if st.button("←  Back to dashboard", key="back_to_dashboard_auth_btn"):
                    _back_to_dashboard()
                st.markdown('</div>', unsafe_allow_html=True)

            _render_login_header(mode_title, mode_subtitle)

            if mode == "Login":
                _login_form()
            elif mode == "Register":
                _register_form()
            else:
                _reset_form()

            _render_notice()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="bottom-footer">🔒 Secure authentication &nbsp; • &nbsp; Protected access &nbsp; • &nbsp; Compliance ready</div>',
        unsafe_allow_html=True,
    )


def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_auth_page()
