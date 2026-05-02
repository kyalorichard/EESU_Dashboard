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
        "auth_remember": False,
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

        :root {
            --bg0: #020617;
            --bg1: #071126;
            --ink: #0b102f;
            --muted: #475569;
            --violet: #6d28d9;
            --violet2: #7c3aed;
            --cyan: #0891b2;
            --blue: #2563eb;
        }

        section[data-testid="stSidebar"],
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] {
            display: none !important;
        }

        html, body, .stApp {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            min-height: 100vh !important;
            background:
                radial-gradient(circle at 34% 40%, rgba(37, 99, 235, 0.20), transparent 25%),
                radial-gradient(circle at 82% 10%, rgba(124, 58, 237, 0.14), transparent 25%),
                linear-gradient(135deg, #020617 0%, #061126 48%, #020617 100%) !important;
            color: #ffffff !important;
            font-family: Inter, Arial, sans-serif !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            margin: 0;
        }

        .auth-topbar {
            height: 76px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 42px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(2, 6, 23, 0.50);
            backdrop-filter: blur(18px);
            box-sizing: border-box;
        }

        .brand-head {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .brand-badge {
            width: 46px;
            height: 46px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            color: #facc15;
            font-size: 26px;
            letter-spacing: -6px;
            transform: rotate(-8deg);
        }

        .brand-title {
            display: flex;
            align-items: baseline;
            gap: 16px;
        }

        .brand-title strong {
            color: white;
            font-size: 27px;
            font-weight: 900;
            letter-spacing: -0.045em;
        }

        .brand-title span {
            color: rgba(226, 232, 240, 0.78);
            font-size: 15px;
            letter-spacing: 0.045em;
            font-weight: 500;
        }

        .top-actions {
            display: flex;
            align-items: center;
            gap: 28px;
            color: rgba(255,255,255,0.92);
            font-size: 16px;
            font-weight: 500;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 17px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.12);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 18px rgba(16, 185, 129, 0.80);
        }

        .auth-page {
            padding: 28px 52px 18px 52px;
            box-sizing: border-box;
        }

        /* Main Streamlit column row should vertically align like the target image */
        .auth-page div[data-testid="stHorizontalBlock"] {
            align-items: center !important;
        }

        .left-card {
            position: relative;
            min-height: 742px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }

        .left-card::after {
            content: "";
            position: absolute;
            right: -58px;
            top: 132px;
            width: 575px;
            height: 575px;
            opacity: 0.78;
            background:
                radial-gradient(circle, rgba(37,99,235,0.36) 0 2px, transparent 2px);
            background-size: 9px 9px;
            mask-image: radial-gradient(circle, black 0%, transparent 73%);
            -webkit-mask-image: radial-gradient(circle, black 0%, transparent 73%);
            pointer-events: none;
        }

        .left-card::before {
            content: "";
            position: absolute;
            right: -20px;
            top: 250px;
            width: 540px;
            height: 330px;
            border: 1px solid rgba(59,130,246,0.35);
            border-left: none;
            border-bottom: none;
            border-radius: 50%;
            transform: rotate(-18deg);
            opacity: 0.65;
            pointer-events: none;
        }

        .left-inner {
            position: relative;
            z-index: 2;
            max-width: 610px;
        }

        .left-main-title {
            margin: 0 0 18px 0;
            font-size: 52px;
            line-height: 1.08;
            font-weight: 950;
            letter-spacing: -0.05em;
            color: #ffffff;
        }

        .left-main-title .gradient {
            display: block;
            background: linear-gradient(90deg, #a855f7 0%, #38bdf8 76%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .left-desc {
            max-width: 525px;
            margin-bottom: 34px;
            color: rgba(226, 232, 240, 0.88);
            font-size: 18px;
            line-height: 1.56;
            font-weight: 500;
        }

        .features {
            display: flex;
            flex-direction: column;
            gap: 22px;
            margin-bottom: 34px;
        }

        .feature {
            display: grid;
            grid-template-columns: 62px 1fr;
            gap: 18px;
            align-items: center;
        }

        .feature-ico {
            width: 58px;
            height: 58px;
            border-radius: 13px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, rgba(124,58,237,0.48), rgba(30,64,175,0.46));
            color: #ddd6fe;
            font-size: 27px;
            box-shadow: 0 16px 34px rgba(0,0,0,0.16);
        }

        .feature h3 {
            margin: 0 0 6px 0;
            color: #ffffff;
            font-size: 20px;
            font-weight: 850;
            letter-spacing: -0.02em;
        }

        .feature p {
            margin: 0;
            color: rgba(226, 232, 240, 0.78);
            font-size: 15.5px;
            line-height: 1.35;
        }

        .left-line {
            width: 86%;
            height: 1px;
            background: rgba(148, 163, 184, 0.22);
            margin: 20px 0 28px 0;
        }

        .security {
            display: grid;
            grid-template-columns: 76px 1fr;
            gap: 18px;
            align-items: center;
            max-width: 550px;
        }

        .security-ico {
            width: 72px;
            height: 72px;
            border-radius: 24px;
            display: grid;
            place-items: center;
            color: #22d3ee;
            border: 1px solid rgba(34, 211, 238, 0.72);
            font-size: 31px;
        }

        .security h3 {
            margin: 0 0 8px 0;
            color: #22d3ee;
            font-size: 18px;
            font-weight: 850;
        }

        .security p {
            margin: 0;
            color: rgba(226, 232, 240, 0.80);
            line-height: 1.58;
            font-size: 15.5px;
        }

        .copyright {
            position: absolute;
            left: 0;
            bottom: 0;
            z-index: 2;
            color: rgba(226, 232, 240, 0.64);
            font-size: 13px;
        }

        /* Login card: Streamlit bordered container */
        .auth-login-card div[data-testid="stVerticalBlockBorderWrapper"] {
            width: min(760px, 100%) !important;
            min-height: 750px !important;
            margin: 0 auto !important;
            color: #0f172a !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.82) !important;
            background:
                radial-gradient(circle at 90% 8%, rgba(124, 58, 237, 0.075), transparent 22%),
                #ffffff !important;
            box-shadow: 0 32px 90px rgba(0,0,0,0.40) !important;
        }

        .auth-login-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 28px 48px 30px 48px !important;
        }

        .back-holder {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 18px;
        }

        .back-holder + div {
            margin-top: 0 !important;
        }

        .back-holder .fake-space {
            height: 0;
        }

        .auth-login-card .stButton > button {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-weight: 750 !important;
            min-height: 42px !important;
        }

        .auth-login-card div[data-testid="stHorizontalBlock"] .stButton > button {
            width: 100% !important;
        }

        .auth-back-btn .stButton > button {
            width: auto !important;
            min-height: 42px !important;
            padding: 0.45rem 1rem !important;
            border: 1px solid rgba(124, 58, 237, 0.55) !important;
            color: #5b21b6 !important;
            background: #ffffff !important;
            border-radius: 8px !important;
            font-weight: 850 !important;
            box-shadow: 0 6px 16px rgba(124,58,237,0.10) !important;
        }

        .login-title-row {
            display: grid;
            grid-template-columns: 78px 1fr;
            gap: 22px;
            align-items: center;
            margin: 6px 0 34px 0;
        }

        .login-lock {
            width: 74px;
            height: 74px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #f1e9ff;
            color: #6d28d9;
            font-size: 32px;
        }

        .login-title-row h1 {
            margin: 0 0 9px 0;
            color: #0b102f;
            font-size: 34px;
            line-height: 1.10;
            font-weight: 950;
            letter-spacing: -0.04em;
        }

        .login-title-row p {
            margin: 0;
            color: #475569;
            font-size: 17px;
            font-weight: 500;
        }

        .or-row {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 18px;
            align-items: center;
            color: #64748b;
            font-weight: 750;
            margin: 16px 0 24px 0;
            font-size: 16px;
        }

        .or-row::before,
        .or-row::after {
            content: "";
            height: 1px;
            background: #cbd5e1;
        }

        .auth-login-card label p {
            color: #0f172a !important;
            font-size: 14px !important;
            font-weight: 850 !important;
            margin-bottom: 4px !important;
        }

        .auth-login-card div[data-testid="stTextInput"] {
            margin-bottom: 13px !important;
        }

        .auth-login-card div[data-testid="stTextInput"] input {
            height: 55px !important;
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-size: 16px !important;
            box-shadow: none !important;
        }

        .auth-login-card div[data-testid="stTextInput"] input:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
        }

        .auth-login-card div[data-testid="stCheckbox"] {
            margin-top: 2px !important;
            margin-bottom: 14px !important;
        }

        .auth-login-card div[data-testid="stCheckbox"] label {
            color: #0f172a !important;
            font-size: 15px !important;
            font-weight: 500 !important;
        }

        .auth-login-card button[kind="primaryFormSubmit"],
        .auth-login-card button[kind="formSubmit"] {
            width: 100% !important;
            min-height: 58px !important;
            border-radius: 8px !important;
            border: 0 !important;
            color: #ffffff !important;
            font-size: 20px !important;
            font-weight: 900 !important;
            background: linear-gradient(90deg, #6d28d9 0%, #2563eb 56%, #0891b2 100%) !important;
            box-shadow: 0 14px 26px rgba(37,99,235,0.28) !important;
        }

        .auth-login-card button[kind="primaryFormSubmit"]:hover,
        .auth-login-card button[kind="formSubmit"]:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
        }

        .below-form-link {
            text-align: center;
            margin: 20px 0 14px 0;
            color: #475569;
            font-size: 15.5px;
            font-weight: 500;
        }

        .below-form-link span {
            color: #4f46e5;
            font-weight: 850;
            margin-left: 14px;
        }

        .notice-box {
            margin-top: 22px;
            border: 1px solid rgba(245, 158, 11, 0.50);
            background: linear-gradient(180deg, #fffbeb 0%, #fff7ed 100%);
            border-radius: 10px;
            padding: 18px 22px;
            color: #1f2937;
            display: grid;
            grid-template-columns: 36px 1fr;
            gap: 14px;
        }

        .notice-icon {
            color: #f59e0b;
            font-size: 25px;
            margin-top: 2px;
        }

        .notice-box h3 {
            margin: 0 0 8px 0;
            color: #3b2f0b;
            font-size: 16px;
            font-weight: 850;
        }

        .notice-box p {
            margin: 5px 0;
            color: #111827;
            font-size: 14px;
        }

        .notice-box .check {
            color: #16a34a;
            font-weight: 900;
            margin-right: 8px;
        }

        .bottom-footer {
            text-align: center;
            color: rgba(226,232,240,0.70);
            font-size: 14px;
            margin: 14px 0 18px 0;
            font-weight: 500;
        }

        /* Hide Streamlit's extra blank spacing in this page */
        .auth-login-card div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
        }

        .auth-login-card div[data-testid="stForm"] > div {
            gap: 0.35rem !important;
        }

        @media (max-width: 1200px) {
            .auth-page {
                padding: 28px 32px 18px 32px;
            }

            .left-main-title {
                font-size: 46px;
            }

            .auth-login-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 28px 36px !important;
            }
        }

        @media (max-width: 980px) {
            .auth-page div[data-testid="stHorizontalBlock"] {
                display: block !important;
            }

            .left-card {
                min-height: auto;
                padding: 36px 0;
            }

            .copyright {
                position: static;
                margin-top: 38px;
            }

            .auth-login-card div[data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto !important;
                margin-top: 24px !important;
            }

            .top-actions {
                display: none;
            }
        }

        @media (max-width: 640px) {
            .auth-topbar {
                padding: 0 18px;
            }

            .brand-title span {
                display: none;
            }

            .auth-page {
                padding: 20px 18px;
            }

            .left-main-title {
                font-size: 38px;
            }

            .login-title-row {
                grid-template-columns: 1fr;
            }

            .auth-login-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 24px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Static HTML sections
# ============================================================
def _render_topbar():
    st.markdown(
        """
        <div class="auth-topbar">
            <div class="brand-head">
                <div class="brand-badge">✦✦✦</div>
                <div class="brand-title">
                    <strong>EU SEE</strong>
                    <span>INTELLIGENCE PLATFORM</span>
                </div>
            </div>

            <div class="top-actions">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>System Status: <strong>Operational</strong></span>
                </div>
                <span>ⓘ Help</span>
                <span>▤ Docs</span>
                <span>◎ English⌄</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_left_panel():
    st.markdown(
        """
        <div class="left-card">
            <div class="left-inner">
                <div class="left-main-title">
                    EU SEE
                    <span class="gradient">Intelligence Platform</span>
                </div>

                <div class="left-desc">
                    Secure access to real-time geopolitical analytics,
                    risk signals, and cross-country monitoring across
                    South East Europe and beyond.
                </div>

                <div class="features">
                    <div class="feature">
                        <div class="feature-ico">◎</div>
                        <div>
                            <h3>86 Countries Monitored</h3>
                            <p>Comprehensive coverage and real-time updates</p>
                        </div>
                    </div>

                    <div class="feature">
                        <div class="feature-ico">▮</div>
                        <div>
                            <h3>Real-time Signal Processing</h3>
                            <p>AI-powered detection and analytics engine</p>
                        </div>
                    </div>

                    <div class="feature">
                        <div class="feature-ico">◇</div>
                        <div>
                            <h3>AI-driven Risk Classification</h3>
                            <p>Advanced models for early risk identification</p>
                        </div>
                    </div>
                </div>

                <div class="left-line"></div>

                <div class="security">
                    <div class="security-ico">🔐</div>
                    <div>
                        <h3>Enterprise-grade security</h3>
                        <p>Your data is protected with end-to-end encryption and strict access controls.</p>
                    </div>
                </div>
            </div>

            <div class="copyright">
                © 2024 EU SEE Intelligence Platform. All rights reserved.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_login_header(mode_title, mode_subtitle):
    st.markdown(
        f"""
        <div class="login-title-row">
            <div class="login-lock">▣</div>
            <div>
                <h1>{mode_title}</h1>
                <p>{mode_subtitle}</p>
            </div>
        </div>

        <div class="or-row">OR</div>
        """,
        unsafe_allow_html=True,
    )


def _render_notice():
    st.markdown(
        """
        <div class="notice-box">
            <div class="notice-icon">🛡️</div>
            <div>
                <h3>Secure Access Notice</h3>
                <p><span class="check">✓</span>Access restricted to verified institutional domains</p>
                <p><span class="check">✓</span>All activity is logged and monitored</p>
                <p><span class="check">✓</span>Session protected with enterprise-grade encryption</p>
            </div>
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
            placeholder="✉   name@organization.org"
        ).strip()

        password = st.text_input(
            "Password",
            placeholder="🔒   Enter your password",
            type="password"
        )

        remember = st.checkbox(
            "Keep me signed in on this device",
            value=st.session_state.get("auth_remember", True)
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
        """
        <div class="below-form-link">
            Don’t have an account? <span>Create account</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create account", use_container_width=True, key="switch_register"):
            _set_auth_mode("Register")
    with c2:
        if st.button("Forgot password?", use_container_width=True, key="switch_reset"):
            _set_auth_mode("Reset")


def _register_form():
    with st.form("eusee_register_form", clear_on_submit=False):
        email = st.text_input(
            "Email address",
            placeholder="✉   name@organization.org"
        ).strip()

        password = st.text_input(
            "Password",
            placeholder="🔒   Create a secure password",
            type="password"
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
            st.success("Registration successful. Check your email to verify your account.")
        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to sign in", use_container_width=True, key="register_back"):
        _set_auth_mode("Login")


def _reset_form():
    with st.form("eusee_reset_form", clear_on_submit=False):
        reset_email = st.text_input(
            "Email address",
            placeholder="✉   name@organization.org"
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

    if st.button("Back to sign in", use_container_width=True, key="reset_back"):
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
        "Register": "Request dashboard access using an approved institutional email address.",
        "Reset": "Enter your approved email address to receive a password reset link.",
    }.get(mode, "Access your authorized EU SEE dashboard and analytics.")

    st.markdown('<div class="auth-page">', unsafe_allow_html=True)

    left_col, right_col = st.columns([0.445, 0.555], gap="large")

    with left_col:
        _render_left_panel()

    with right_col:
        st.markdown('<div class="auth-login-card">', unsafe_allow_html=True)

        with st.container(border=True):
            # Right-aligned button using columns, not open HTML around the button
            b1, b2 = st.columns([0.70, 0.30])
            with b2:
                st.markdown('<div class="auth-back-btn">', unsafe_allow_html=True)
                if st.button("←  Back to dashboard", key="back_to_dashboard_auth"):
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
        """
        <div class="bottom-footer">
            🔒 Secure authentication &nbsp; • &nbsp; Protected access &nbsp; • &nbsp; Compliance ready
        </div>
        """,
        unsafe_allow_html=True,
    )


def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_auth_page()
