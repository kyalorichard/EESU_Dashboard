# auth.py
import json
import time
import html
import streamlit as st

DEBUG = False

# -----------------------------
# Optional imports
# -----------------------------
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


# -----------------------------
# Firebase setup
# -----------------------------
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


# -----------------------------
# Session helpers
# -----------------------------
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


# -----------------------------
# Cookies
# -----------------------------
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


# -----------------------------
# Styling
# -----------------------------
def _auth_page_css():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            display: none !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 33% 42%, rgba(37, 99, 235, 0.22), transparent 25%),
                radial-gradient(circle at 80% 12%, rgba(124, 58, 237, 0.15), transparent 26%),
                linear-gradient(135deg, #020617 0%, #071126 48%, #020617 100%) !important;
            color: #ffffff;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
        }

        .auth-shell {
            min-height: 100vh;
            font-family: Inter, Arial, sans-serif;
        }

        .auth-topbar {
            height: 72px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 38px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(2, 6, 23, 0.58);
            backdrop-filter: blur(18px);
        }

        .brand-head {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-stars {
            color: #facc15;
            font-size: 24px;
            letter-spacing: -6px;
            width: 52px;
        }

        .brand-name {
            font-size: 26px;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: -0.04em;
        }

        .brand-platform {
            font-size: 14px;
            letter-spacing: 0.06em;
            color: rgba(226, 232, 240, 0.78);
            margin-left: 12px;
        }

        .top-actions {
            display: flex;
            align-items: center;
            gap: 26px;
            color: rgba(255,255,255,0.90);
            font-size: 15px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            padding: 10px 16px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.13);
            box-shadow: 0 8px 24px rgba(0,0,0,0.20);
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 18px rgba(16, 185, 129, 0.75);
        }

        .auth-body {
            min-height: calc(100vh - 72px);
            display: grid;
            grid-template-columns: 45% 55%;
            align-items: center;
            gap: 38px;
            padding: 32px 58px 26px 58px;
        }

        .left-panel {
            position: relative;
            min-height: 760px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: hidden;
        }

        .map-orb {
            position: absolute;
            right: -40px;
            top: 145px;
            width: 560px;
            height: 560px;
            opacity: 0.78;
            background:
                radial-gradient(circle, rgba(37,99,235,0.32) 0 2px, transparent 2px);
            background-size: 10px 10px;
            mask-image: radial-gradient(circle, black 0%, transparent 72%);
            -webkit-mask-image: radial-gradient(circle, black 0%, transparent 72%);
        }

        .left-content {
            position: relative;
            z-index: 2;
            max-width: 620px;
        }

        .left-title {
            font-size: 52px;
            line-height: 1.06;
            font-weight: 950;
            letter-spacing: -0.045em;
            margin-bottom: 20px;
            color: #ffffff;
        }

        .left-title .gradient {
            display: block;
            background: linear-gradient(90deg, #a855f7 0%, #22d3ee 78%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .left-description {
            max-width: 520px;
            color: rgba(226, 232, 240, 0.88);
            font-size: 19px;
            line-height: 1.52;
            margin-bottom: 34px;
        }

        .feature-list {
            display: flex;
            flex-direction: column;
            gap: 22px;
            margin-bottom: 36px;
        }

        .feature-item {
            display: grid;
            grid-template-columns: 58px 1fr;
            gap: 18px;
            align-items: center;
        }

        .feature-icon {
            width: 58px;
            height: 58px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, rgba(124,58,237,0.48), rgba(30,64,175,0.44));
            color: #ddd6fe;
            font-size: 26px;
        }

        .feature-item h3 {
            margin: 0 0 6px 0;
            font-size: 20px;
            color: white;
            font-weight: 850;
        }

        .feature-item p {
            margin: 0;
            font-size: 15px;
            color: rgba(226, 232, 240, 0.76);
        }

        .left-divider {
            width: 84%;
            height: 1px;
            background: rgba(148, 163, 184, 0.22);
            margin: 20px 0 28px 0;
        }

        .security-row {
            display: grid;
            grid-template-columns: 72px 1fr;
            gap: 18px;
            max-width: 520px;
            align-items: center;
        }

        .security-icon {
            width: 72px;
            height: 72px;
            border-radius: 24px;
            display: grid;
            place-items: center;
            color: #22d3ee;
            border: 1px solid rgba(34, 211, 238, 0.72);
            font-size: 30px;
        }

        .security-row h3 {
            margin: 0 0 8px 0;
            color: #22d3ee;
            font-size: 18px;
        }

        .security-row p {
            margin: 0;
            color: rgba(226, 232, 240, 0.78);
            line-height: 1.55;
            font-size: 15px;
        }

        .left-footer {
            position: absolute;
            left: 0;
            bottom: 8px;
            color: rgba(226, 232, 240, 0.62);
            font-size: 13px;
        }

        .login-card {
            width: min(760px, 100%);
            min-height: 720px;
            margin: 0 auto;
            background:
                radial-gradient(circle at 90% 8%, rgba(124, 58, 237, 0.08), transparent 22%),
                #ffffff;
            color: #0f172a;
            border-radius: 14px;
            padding: 28px 48px 28px 48px;
            box-shadow: 0 32px 90px rgba(0,0,0,0.38);
            border: 1px solid rgba(255,255,255,0.70);
        }

        .back-row {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 22px;
        }

        .auth-back-button .stButton > button {
            width: auto !important;
            border: 1px solid rgba(124, 58, 237, 0.55) !important;
            color: #5b21b6 !important;
            background: #ffffff !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
            padding: 0.55rem 1rem !important;
            box-shadow: 0 6px 16px rgba(124,58,237,0.10);
        }

        .login-heading {
            display: grid;
            grid-template-columns: 78px 1fr;
            gap: 22px;
            align-items: center;
            margin-bottom: 32px;
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

        .login-heading h1 {
            margin: 0 0 8px 0;
            color: #0b102f;
            font-size: 34px;
            line-height: 1.1;
            font-weight: 950;
            letter-spacing: -0.035em;
        }

        .login-heading p {
            margin: 0;
            color: #475569;
            font-size: 17px;
        }

        .or-row {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 18px;
            align-items: center;
            color: #64748b;
            font-weight: 700;
            margin: 12px 0 24px 0;
        }

        .or-row::before,
        .or-row::after {
            content: "";
            height: 1px;
            background: #cbd5e1;
        }

        label p {
            color: #0f172a !important;
            font-size: 14px !important;
            font-weight: 850 !important;
        }

        div[data-testid="stTextInput"] input {
            height: 54px !important;
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            font-size: 16px !important;
            color: #0f172a !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12) !important;
        }

        div[data-testid="stCheckbox"] label {
            color: #0f172a !important;
            font-size: 15px !important;
        }

        button[kind="primaryFormSubmit"],
        button[kind="formSubmit"] {
            min-height: 58px !important;
            border-radius: 8px !important;
            border: 0 !important;
            color: white !important;
            font-size: 20px !important;
            font-weight: 900 !important;
            background: linear-gradient(90deg, #6d28d9 0%, #0284c7 100%) !important;
            box-shadow: 0 14px 26px rgba(37, 99, 235, 0.28) !important;
        }

        .stButton > button {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-weight: 750 !important;
        }

        .form-link-row {
            text-align: center;
            margin: 22px 0 12px 0;
            color: #475569;
            font-size: 15px;
        }

        .form-link-row span {
            color: #4f46e5;
            font-weight: 800;
        }

        .secure-notice {
            margin-top: 22px;
            border: 1px solid rgba(245, 158, 11, 0.55);
            background: linear-gradient(180deg, #fffbeb 0%, #fff7ed 100%);
            border-radius: 10px;
            padding: 18px 22px;
            color: #1f2937;
        }

        .secure-notice h3 {
            margin: 0 0 9px 0;
            font-size: 16px;
            color: #3b2f0b;
        }

        .secure-notice p {
            margin: 5px 0;
            font-size: 14px;
        }

        .secure-notice .check {
            color: #16a34a;
            font-weight: 900;
            margin-right: 7px;
        }

        .bottom-footer {
            text-align: center;
            color: rgba(226,232,240,0.70);
            font-size: 14px;
            margin-top: 18px;
        }

        @media (max-width: 1100px) {
            .auth-body {
                grid-template-columns: 1fr;
                padding: 24px;
            }

            .left-panel {
                min-height: auto;
                padding: 40px 0;
            }

            .left-footer {
                position: static;
                margin-top: 40px;
            }

            .login-card {
                min-height: auto;
            }
        }

        @media (max-width: 720px) {
            .auth-topbar {
                padding: 0 18px;
            }

            .top-actions {
                display: none;
            }

            .brand-platform {
                display: none;
            }

            .left-title {
                font-size: 40px;
            }

            .login-card {
                padding: 24px;
            }

            .login-heading {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# HTML blocks
# -----------------------------
def _topbar_html():
    return """
    <div class="auth-shell">
        <div class="auth-topbar">
            <div class="brand-head">
                <div class="brand-stars">✦✦✦</div>
                <div>
                    <span class="brand-name">EU SEE</span>
                    <span class="brand-platform">INTELLIGENCE PLATFORM</span>
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
    """


def _left_panel_html():
    return """
        <div class="left-panel">
            <div class="map-orb"></div>
            <div class="left-content">
                <div class="left-title">
                    EU SEE
                    <span class="gradient">Intelligence Platform</span>
                </div>

                <div class="left-description">
                    Secure access to real-time geopolitical analytics,
                    risk signals, and cross-country monitoring across
                    South East Europe and beyond.
                </div>

                <div class="feature-list">
                    <div class="feature-item">
                        <div class="feature-icon">◎</div>
                        <div>
                            <h3>86 Countries Monitored</h3>
                            <p>Comprehensive coverage and real-time updates</p>
                        </div>
                    </div>

                    <div class="feature-item">
                        <div class="feature-icon">▮</div>
                        <div>
                            <h3>Real-time Signal Processing</h3>
                            <p>AI-powered detection and analytics engine</p>
                        </div>
                    </div>

                    <div class="feature-item">
                        <div class="feature-icon">◇</div>
                        <div>
                            <h3>AI-driven Risk Classification</h3>
                            <p>Advanced models for early risk identification</p>
                        </div>
                    </div>
                </div>

                <div class="left-divider"></div>

                <div class="security-row">
                    <div class="security-icon">🔐</div>
                    <div>
                        <h3>Enterprise-grade security</h3>
                        <p>Your data is protected with end-to-end encryption and strict access controls.</p>
                    </div>
                </div>
            </div>

            <div class="left-footer">
                © 2024 EU SEE Intelligence Platform. All rights reserved.
            </div>
        </div>
    """


def _login_header_html(mode_title, mode_subtitle):
    return f"""
        <div class="login-heading">
            <div class="login-lock">🔒</div>
            <div>
                <h1>{html.escape(mode_title)}</h1>
                <p>{html.escape(mode_subtitle)}</p>
            </div>
        </div>

        <div class="or-row">OR</div>
    """


def _secure_notice_html():
    return """
        <div class="secure-notice">
            <h3>🛡️ Secure Access Notice</h3>
            <p><span class="check">✓</span>Access restricted to verified institutional domains</p>
            <p><span class="check">✓</span>All activity is logged and monitored</p>
            <p><span class="check">✓</span>Session protected with enterprise-grade encryption</p>
        </div>
    """


# -----------------------------
# Form actions
# -----------------------------
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
            placeholder="name@organization.org"
        ).strip()

        password = st.text_input(
            "Password",
            placeholder="Enter your password",
            type="password"
        )

        remember = st.checkbox(
            "Keep me signed in on this device",
            value=st.session_state.get("auth_remember", False)
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
        <div class="form-link-row">
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
            placeholder="name@organization.org"
        ).strip()

        password = st.text_input(
            "Password",
            placeholder="Create a secure password",
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
            placeholder="name@organization.org"
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


# -----------------------------
# Main auth page
# -----------------------------
def _render_auth_page():
    _auth_page_css()

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

    st.markdown(_topbar_html(), unsafe_allow_html=True)

    body_left, body_right = st.columns([0.45, 0.55], gap="large")

    with body_left:
        st.markdown(_left_panel_html(), unsafe_allow_html=True)

    with body_right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        st.markdown('<div class="back-row"><div class="auth-back-button">', unsafe_allow_html=True)
        if st.button("←  Back to dashboard", key="back_to_dashboard_auth"):
            _back_to_dashboard()
        st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown(
            _login_header_html(mode_title, mode_subtitle),
            unsafe_allow_html=True
        )

        if mode == "Login":
            _login_form()
        elif mode == "Register":
            _register_form()
        else:
            _reset_form()

        st.markdown(_secure_notice_html(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="bottom-footer">
            🔒 Secure authentication &nbsp; • &nbsp; Protected access &nbsp; • &nbsp; Compliance ready
        </div>
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
