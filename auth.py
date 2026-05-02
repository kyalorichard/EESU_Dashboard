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
# Full-page EU SEE themed login CSS
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
            color: #ffffff !important;
            background:
                radial-gradient(circle at 28% 45%, rgba(37, 99, 235, 0.22), transparent 28%),
                radial-gradient(circle at 75% 10%, rgba(124, 58, 237, 0.17), transparent 30%),
                linear-gradient(135deg, #020617 0%, #061426 50%, #020617 100%) !important;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                radial-gradient(circle at 48% 50%, rgba(255, 215, 120, 0.20) 0 1px, transparent 2px),
                radial-gradient(circle at 58% 45%, rgba(59, 130, 246, 0.45) 0 2px, transparent 3px),
                radial-gradient(circle at 44% 60%, rgba(59, 130, 246, 0.35) 0 2px, transparent 3px),
                linear-gradient(120deg, transparent 20%, rgba(37,99,235,0.10) 45%, transparent 70%);
            opacity: 0.55;
        }

        .stApp::after {
            content: "";
            position: fixed;
            left: 18%;
            top: 16%;
            width: 760px;
            height: 760px;
            pointer-events: none;
            background:
                radial-gradient(circle, rgba(37,99,235,0.40) 0 2px, transparent 2px);
            background-size: 12px 12px;
            mask-image: radial-gradient(circle, black 0%, transparent 68%);
            -webkit-mask-image: radial-gradient(circle, black 0%, transparent 68%);
            opacity: 0.34;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            position: relative;
            z-index: 2;
        }

        div[data-testid="stMarkdownContainer"] p {
            margin: 0;
        }

        .login-page {
            min-height: 100vh;
            padding: 52px 56px 28px 56px;
            box-sizing: border-box;
        }

        .logo-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 76px;
        }

        .logo-mark {
            width: 64px;
            height: 64px;
            border-radius: 18px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, #7c3aed, #2563eb 70%);
            color: white;
            font-size: 34px;
            font-weight: 950;
            box-shadow: 0 18px 40px rgba(37,99,235,0.35);
        }

        .logo-title {
            color: #ffffff !important;
            font-size: 34px;
            line-height: 1;
            font-weight: 950;
            letter-spacing: -0.05em;
        }

        .logo-subtitle {
            color: rgba(226,232,240,0.82) !important;
            font-size: 15px;
            letter-spacing: 0.055em;
            margin-top: 6px;
            font-weight: 650;
        }

        .left-copy h1 {
            color: #ffffff !important;
            font-size: 48px;
            line-height: 1.13;
            font-weight: 950;
            letter-spacing: -0.055em;
            margin: 0 0 20px 0;
        }

        .left-copy .accent {
            background: linear-gradient(90deg, #a855f7, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .accent-line {
            width: 66px;
            height: 3px;
            border-radius: 99px;
            background: linear-gradient(90deg, #a855f7, #38bdf8);
            margin: 24px 0 30px 0;
        }

        .left-copy p {
            max-width: 430px;
            color: rgba(226,232,240,0.86) !important;
            font-size: 17px;
            line-height: 1.62;
            font-weight: 500;
            margin-bottom: 64px;
        }

        .feature-row {
            margin-bottom: 24px;
        }

        .feature-icon {
            width: 58px;
            height: 58px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: rgba(124, 58, 237, 0.18);
            color: #a855f7 !important;
            font-size: 27px;
            box-shadow: inset 0 0 0 1px rgba(168,85,247,0.18);
        }

        .feature-title {
            color: #ffffff !important;
            font-size: 16px;
            font-weight: 850;
            margin: 0 0 6px 0;
        }

        .feature-desc {
            color: rgba(226,232,240,0.74) !important;
            font-size: 14.5px;
            line-height: 1.45;
            margin: 0;
        }

        .back-dashboard-wrap .stButton > button {
            max-width: 360px !important;
            height: 68px !important;
            border-radius: 13px !important;
            border: 1px solid rgba(168,85,247,0.75) !important;
            background: rgba(2, 6, 23, 0.48) !important;
            color: #ffffff !important;
            font-size: 20px !important;
            font-weight: 850 !important;
            box-shadow: 0 16px 38px rgba(0,0,0,0.22) !important;
        }

        .back-dashboard-wrap .stButton > button:hover {
            border-color: #c084fc !important;
            box-shadow: 0 0 0 4px rgba(168,85,247,0.14), 0 16px 38px rgba(0,0,0,0.22) !important;
            color: #ffffff !important;
        }

        .auth-panel div[data-testid="stVerticalBlockBorderWrapper"] {
            width: min(610px, 100%) !important;
            min-height: 660px !important;
            margin: 16px auto 0 auto !important;
            border-radius: 24px !important;
            border: 1px solid rgba(124, 58, 237, 0.45) !important;
            background:
                radial-gradient(circle at top center, rgba(124,58,237,0.14), transparent 35%),
                rgba(15, 23, 42, 0.82) !important;
            backdrop-filter: blur(24px) !important;
            box-shadow: 0 34px 90px rgba(0,0,0,0.48) !important;
        }

        .auth-panel div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 44px 48px 36px 48px !important;
        }

        .lock-icon {
            width: 92px;
            height: 92px;
            margin: 0 auto 26px auto;
            display: grid;
            place-items: center;
            border-radius: 50%;
            border: 1px solid rgba(168,85,247,0.45);
            color: #a855f7 !important;
            font-size: 42px;
            background: rgba(15,23,42,0.40);
            box-shadow: 0 0 34px rgba(168,85,247,0.18);
        }

        .panel-title {
            color: #ffffff !important;
            text-align: center;
            font-size: 32px;
            line-height: 1.1;
            font-weight: 900;
            letter-spacing: -0.045em;
            margin-bottom: 8px;
        }

        .panel-subtitle {
            color: rgba(226,232,240,0.78) !important;
            text-align: center;
            font-size: 16px;
            margin-bottom: 34px;
        }

        .auth-panel label p {
            color: #ffffff !important;
            font-size: 14px !important;
            font-weight: 800 !important;
            margin-bottom: 6px !important;
        }

        .auth-panel div[data-testid="stTextInput"] {
            margin-bottom: 16px !important;
        }

        .auth-panel div[data-testid="stTextInput"] input {
            height: 56px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            background: rgba(15,23,42,0.55) !important;
            color: #ffffff !important;
            font-size: 15px !important;
            box-shadow: none !important;
        }

        .auth-panel div[data-testid="stTextInput"] input::placeholder {
            color: rgba(226,232,240,0.50) !important;
        }

        .auth-panel div[data-testid="stTextInput"] input:focus {
            border-color: #a855f7 !important;
            box-shadow: 0 0 0 3px rgba(168,85,247,0.16) !important;
        }

        .auth-panel div[data-testid="stCheckbox"] label,
        .auth-panel div[data-testid="stCheckbox"] label span {
            color: rgba(255,255,255,0.88) !important;
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        .auth-panel button[kind="primaryFormSubmit"],
        .auth-panel button[kind="formSubmit"] {
            width: 100% !important;
            height: 58px !important;
            border: none !important;
            border-radius: 10px !important;
            background: linear-gradient(90deg, #9333ea, #2563eb) !important;
            color: #ffffff !important;
            font-size: 18px !important;
            font-weight: 900 !important;
            box-shadow: 0 18px 34px rgba(37,99,235,0.30) !important;
        }

        .auth-panel button[kind="primaryFormSubmit"]:hover,
        .auth-panel button[kind="formSubmit"]:hover {
            filter: brightness(1.06);
            transform: translateY(-1px);
        }

        .auth-panel .stButton > button {
            height: 48px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(148,163,184,0.30) !important;
            background: rgba(15,23,42,0.38) !important;
            color: rgba(255,255,255,0.90) !important;
            font-size: 14px !important;
            font-weight: 750 !important;
        }

        .auth-panel .stButton > button:hover {
            border-color: #a855f7 !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 3px rgba(168,85,247,0.12) !important;
        }

        .helper-text {
            text-align: center;
            color: rgba(226,232,240,0.70) !important;
            font-size: 14px;
            margin: 18px 0 12px 0;
        }

        .security-strip {
            margin-top: 34px;
            padding-top: 26px;
            border-top: 1px solid rgba(148,163,184,0.15);
        }

        .security-item {
            color: rgba(226,232,240,0.82) !important;
            font-size: 14px;
            line-height: 1.35;
        }

        .security-item strong {
            color: #ffffff !important;
            font-size: 15px;
        }

        @media (max-width: 1000px) {
            .login-page {
                padding: 32px 24px;
            }

            .logo-row {
                margin-bottom: 36px;
            }

            .left-copy h1 {
                font-size: 38px;
            }

            .auth-panel div[data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto !important;
                margin-top: 32px !important;
            }
        }

        @media (max-width: 640px) {
            .login-page {
                padding: 24px 18px;
            }

            .logo-title {
                font-size: 28px;
            }

            .logo-mark {
                width: 54px;
                height: 54px;
            }

            .auth-panel div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 34px 24px 28px 24px !important;
            }

            .back-dashboard-wrap .stButton > button {
                max-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UI helpers
# ============================================================
def _mode_text(mode):
    data = {
        "Login": {
            "title": "Welcome back",
            "subtitle": "Sign in to access your dashboard",
            "submit": "Sign in",
        },
        "Register": {
            "title": "Create account",
            "subtitle": "Register with your approved organizational email",
            "submit": "Create account",
        },
        "Reset": {
            "title": "Reset password",
            "subtitle": "Receive a secure password reset link",
            "submit": "Send reset link",
        },
    }
    return data.get(mode, data["Login"])


def _back_to_dashboard():
    st.session_state.auth_view = False
    st.rerun()


def _set_auth_mode(mode: str):
    st.session_state.auth_mode = mode
    st.rerun()


def _render_feature(icon, title, desc):
    c1, c2 = st.columns([0.16, 0.84], vertical_alignment="center")
    with c1:
        st.markdown(f'<div class="feature-icon">{icon}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<p class="feature-title">{title}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="feature-desc">{desc}</p>', unsafe_allow_html=True)


# ============================================================
# Forms
# ============================================================
def _login_form():
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email address", placeholder="name@organization.org").strip()
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        remember = st.checkbox(
            "Keep me signed in",
            value=st.session_state.get("auth_remember", True),
        )
        submitted = st.form_submit_button("Sign in", use_container_width=True)

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

    st.markdown('<div class="helper-text">Need access or forgot your password?</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create account", use_container_width=True):
            _set_auth_mode("Register")
    with c2:
        if st.button("Forgot password", use_container_width=True):
            _set_auth_mode("Reset")


def _register_form():
    with st.form("register_form", clear_on_submit=False):
        email = st.text_input("Email address", placeholder="name@organization.org").strip()
        password = st.text_input("Password", placeholder="Create a secure password", type="password")
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

    if st.button("Back to sign in", use_container_width=True):
        _set_auth_mode("Login")


def _reset_form():
    with st.form("reset_form", clear_on_submit=False):
        reset_email = st.text_input("Email address", placeholder="name@organization.org").strip()
        submitted = st.form_submit_button("Send password reset link", use_container_width=True)

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

    if st.button("Back to sign in", use_container_width=True):
        _set_auth_mode("Login")


# ============================================================
# Main auth page
# ============================================================
def _render_auth_page():
    _auth_page_css()
    mode = st.session_state.get("auth_mode", "Login")
    text = _mode_text(mode)

    st.markdown('<div class="login-page">', unsafe_allow_html=True)

    left_col, right_col = st.columns([0.46, 0.54], gap="large", vertical_alignment="center")

    with left_col:
        st.markdown(
            """
            <div class="logo-row">
                <div class="logo-mark">S</div>
                <div>
                    <div class="logo-title">EU SEE</div>
                    <div class="logo-subtitle">INTELLIGENCE PLATFORM</div>
                </div>
            </div>

            <div class="left-copy">
                <h1>Intelligence for<br>a <span class="accent">Safer Europe.</span></h1>
                <div class="accent-line"></div>
                <p>
                    Advanced analytics and real-time intelligence empowering
                    decision makers across South East Europe and beyond.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_feature("♢", "Secure & Trusted", "Enterprise-grade security and data protection")
        _render_feature("▥", "Real-time Intelligence", "Live data, smart dashboards and actionable insights")
        _render_feature("◎", "Regional Coverage", "Monitoring 86+ countries across South East Europe and beyond")

        st.markdown('<div style="height: 26px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="back-dashboard-wrap">', unsafe_allow_html=True)
        if st.button("←  Back to dashboard", use_container_width=True):
            _back_to_dashboard()
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="lock-icon">🔒</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-title">{text["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-subtitle">{text["subtitle"]}</div>', unsafe_allow_html=True)

            if mode == "Login":
                _login_form()
            elif mode == "Register":
                _register_form()
            else:
                _reset_form()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="security-strip">', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown('<div class="security-item"><strong>256-bit</strong><br>SSL Encryption</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="security-item"><strong>Secure</strong><br>Infrastructure</div>', unsafe_allow_html=True)
        with s3:
            st.markdown('<div class="security-item"><strong>GDPR</strong><br>Compliant</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_auth_page()
