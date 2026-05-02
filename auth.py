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
# Professional centered UI
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
            min-height: 100vh !important;
            font-family: Inter, Arial, sans-serif !important;
            background:
                radial-gradient(circle at 18% 18%, rgba(124, 58, 237, 0.22), transparent 32%),
                radial-gradient(circle at 82% 82%, rgba(8, 145, 178, 0.20), transparent 36%),
                linear-gradient(135deg, #020617 0%, #071426 48%, #0f172a 100%) !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            margin: 0;
        }

        .login-shell {
            min-height: 100vh;
            padding: 46px 18px 28px 18px;
        }

        .top-brand {
            text-align: center;
            margin-bottom: 22px;
        }

        .top-brand .logo {
            width: 58px;
            height: 58px;
            border-radius: 18px;
            display: inline-grid;
            place-items: center;
            background: linear-gradient(135deg, #6d28d9, #2563eb);
            color: #ffffff;
            font-size: 26px;
            font-weight: 900;
            box-shadow: 0 16px 34px rgba(37,99,235,0.30);
            margin-bottom: 14px;
        }

        .top-brand h1 {
            margin: 0;
            color: #ffffff !important;
            font-size: 30px;
            line-height: 1.15;
            font-weight: 950;
            letter-spacing: -0.045em;
        }

        .top-brand p {
            margin: 8px 0 0 0;
            color: rgba(226,232,240,0.78) !important;
            font-size: 14.5px;
            line-height: 1.5;
        }

        /* Only style the main auth card */
        .auth-card div[data-testid="stVerticalBlockBorderWrapper"] {
            max-width: 460px !important;
            margin: 0 auto !important;
            border-radius: 24px !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)) !important;
            box-shadow: 0 30px 90px rgba(0,0,0,0.45) !important;
        }

        .auth-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 32px 32px 28px 32px !important;
        }

        .mode-pill {
            width: fit-content;
            margin: 0 auto 22px auto;
            padding: 8px 13px;
            border-radius: 999px;
            color: #5b21b6 !important;
            background: #f3e8ff;
            border: 1px solid #e9d5ff;
            font-size: 12px;
            font-weight: 850;
            letter-spacing: 0.03em;
        }

        .card-title {
            text-align: center;
            color: #0b102f !important;
            font-size: 24px;
            font-weight: 900;
            letter-spacing: -0.035em;
            margin-bottom: 7px;
        }

        .card-subtitle {
            text-align: center;
            color: #64748b !important;
            font-size: 13.5px;
            line-height: 1.48;
            margin-bottom: 24px;
        }

        .auth-card label p {
            color: #111827 !important;
            font-size: 13px !important;
            font-weight: 850 !important;
            margin-bottom: 5px !important;
        }

        .auth-card div[data-testid="stTextInput"] input {
            height: 50px !important;
            border-radius: 12px !important;
            border: 1px solid #d7dce7 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-size: 14.5px !important;
            box-shadow: none !important;
        }

        .auth-card div[data-testid="stTextInput"] input:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124,58,237,0.13) !important;
        }

        .auth-card div[data-testid="stCheckbox"] {
            margin-top: 2px !important;
            margin-bottom: 12px !important;
        }

        .auth-card div[data-testid="stCheckbox"] label,
        .auth-card div[data-testid="stCheckbox"] label span {
            color: #334155 !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
        }

        .auth-card button[kind="primaryFormSubmit"],
        .auth-card button[kind="formSubmit"] {
            width: 100% !important;
            min-height: 52px !important;
            border-radius: 12px !important;
            border: none !important;
            color: #ffffff !important;
            font-size: 15.5px !important;
            font-weight: 900 !important;
            background: linear-gradient(90deg, #6d28d9, #2563eb) !important;
            box-shadow: 0 12px 26px rgba(37,99,235,0.28) !important;
            transition: all .15s ease !important;
        }

        .auth-card button[kind="primaryFormSubmit"]:hover,
        .auth-card button[kind="formSubmit"]:hover {
            filter: brightness(1.04);
            transform: translateY(-1px);
        }

        .auth-card .stButton > button {
            min-height: 42px !important;
            border-radius: 12px !important;
            border: 1px solid #d7dce7 !important;
            background: #ffffff !important;
            color: #334155 !important;
            font-weight: 750 !important;
            font-size: 13.5px !important;
        }

        .auth-card .stButton > button:hover {
            border-color: #7c3aed !important;
            color: #5b21b6 !important;
            box-shadow: 0 8px 18px rgba(124,58,237,0.12) !important;
        }

        .helper-text {
            text-align: center;
            color: #64748b !important;
            font-size: 13px;
            margin: 16px 0 10px 0;
        }

        .notice {
            margin-top: 18px;
            padding: 13px 14px;
            border-radius: 14px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #4b3b14 !important;
            font-size: 12px;
            line-height: 1.45;
        }

        .notice strong {
            color: #3b2f0b !important;
        }

        .footer-note {
            text-align: center;
            color: rgba(226,232,240,0.68) !important;
            font-size: 12.5px;
            margin-top: 18px;
        }

        @media (max-width: 520px) {
            .login-shell {
                padding-top: 28px;
            }

            .auth-card div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 28px 22px 24px 22px !important;
            }

            .top-brand h1 {
                font-size: 26px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mode_text(mode):
    data = {
        "Login": {
            "pill": "SECURE ACCESS",
            "title": "Sign in to dashboard",
            "subtitle": "Use your approved organizational account to access EU SEE analytics.",
        },
        "Register": {
            "pill": "ACCOUNT REGISTRATION",
            "title": "Create your account",
            "subtitle": "Register with an approved institutional email address.",
        },
        "Reset": {
            "pill": "PASSWORD RECOVERY",
            "title": "Reset your password",
            "subtitle": "Enter your email address to receive a password reset link.",
        },
    }
    return data.get(mode, data["Login"])


def _back_to_dashboard():
    st.session_state.auth_view = False
    st.rerun()


def _set_auth_mode(mode: str):
    st.session_state.auth_mode = mode
    st.rerun()


def _login_form():
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email address", placeholder="name@organization.org").strip()
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        remember = st.checkbox(
            "Keep me signed in on this device",
            value=st.session_state.get("auth_remember", True),
        )
        submitted = st.form_submit_button("Access Dashboard", use_container_width=True)

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


def _render_auth_page():
    _auth_page_css()
    mode = st.session_state.get("auth_mode", "Login")
    text = _mode_text(mode)

    st.markdown('<div class="login-shell">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="top-brand">
            <div class="logo">✦</div>
            <h1>EU SEE Intelligence Platform</h1>
            <p>Secure access to dashboard analytics and protected intelligence features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centering is handled by Streamlit columns; no fragile open HTML around widgets.
    left, center, right = st.columns([1, 0.58, 1])

    with center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f'<div class="mode-pill">{text["pill"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">{text["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-subtitle">{text["subtitle"]}</div>', unsafe_allow_html=True)

            if mode == "Login":
                _login_form()
            elif mode == "Register":
                _register_form()
            else:
                _reset_form()

            st.markdown(
                """
                <div class="notice">
                    <strong>Secure Access Notice</strong><br>
                    Access is restricted to verified institutional users. Sessions are protected and activity may be monitored.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("← Back to dashboard", use_container_width=True):
                _back_to_dashboard()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">EU SEE Dashboard · Secure authentication · Protected access</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_auth_page()
