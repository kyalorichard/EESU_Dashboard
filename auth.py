# auth.py
import json
import time
import streamlit as st
import streamlit.components.v1 as components

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
        "auth_action": None,
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
        "auth_mode", "auth_remember", "auth_view", "auth_action"
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
# Page shell
# ============================================================
def _hide_streamlit_chrome():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        footer {
            display: none !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        .stApp {
            background: #020617 !important;
        }

        iframe {
            display: block !important;
            width: 100% !important;
            border: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _login_component_html(mode="Login", message="", message_type="info"):
    is_login = mode == "Login"
    is_register = mode == "Register"
    is_reset = mode == "Reset"

    title = {
        "Login": "Sign in to your workspace",
        "Register": "Create your account",
        "Reset": "Reset password",
    }.get(mode, "Sign in to your workspace")

    subtitle = {
        "Login": "Access your authorized EU SEE dashboard and analytics.",
        "Register": "Request access using an approved institutional email address.",
        "Reset": "Enter your approved email address to receive a password reset link.",
    }.get(mode, "Access your authorized EU SEE dashboard and analytics.")

    button_label = {
        "Login": "Access Intelligence Dashboard",
        "Register": "Create account",
        "Reset": "Send password reset link",
    }.get(mode, "Access Intelligence Dashboard")

    password_display = "block" if mode in ["Login", "Register"] else "none"
    remember_display = "flex" if mode == "Login" else "none"
    forgot_display = "inline-flex" if mode == "Login" else "none"

    msg_html = ""
    if message:
        color = "#166534" if message_type == "success" else "#991b1b" if message_type == "error" else "#1e40af"
        bg = "#ecfdf5" if message_type == "success" else "#fef2f2" if message_type == "error" else "#eff6ff"
        border = "#bbf7d0" if message_type == "success" else "#fecaca" if message_type == "error" else "#bfdbfe"
        msg_html = f'<div class="message" style="color:{color};background:{bg};border-color:{border};">{message}</div>'

    mode_hint = ""
    if is_login:
        mode_hint = 'Don’t have an account? <button name="action" value="switch_register" class="link-btn" form="auth_form">Create account</button>'
    elif is_register:
        mode_hint = '<button name="action" value="switch_login" class="link-btn" form="auth_form">Back to sign in</button>'
    elif is_reset:
        mode_hint = '<button name="action" value="switch_login" class="link-btn" form="auth_form">Back to sign in</button>'

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {{
    --bg: #020617;
    --panel: #ffffff;
    --ink: #0b102f;
    --muted: #475569;
    --border: #d7dce8;
    --violet: #6d28d9;
    --blue: #2563eb;
    --cyan: #0891b2;
}}

* {{
    box-sizing: border-box;
}}

html, body {{
    margin: 0;
    padding: 0;
    min-height: 100%;
    font-family: Inter, Arial, sans-serif;
    background:
        radial-gradient(circle at 30% 42%, rgba(37, 99, 235, 0.20), transparent 25%),
        radial-gradient(circle at 84% 12%, rgba(124, 58, 237, 0.16), transparent 28%),
        linear-gradient(135deg, #020617 0%, #071426 42%, #0b1b3a 100%);
    color: white;
}}

.page {{
    min-height: 100vh;
}}

.topbar {{
    height: 74px;
    padding: 0 42px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(2, 6, 23, 0.62);
    backdrop-filter: blur(18px);
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.eu-mark {{
    color: #facc15;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -5px;
    transform: rotate(-6deg);
}}

.brand-main {{
    font-size: 25px;
    font-weight: 900;
    letter-spacing: -0.045em;
}}

.brand-sub {{
    font-size: 14px;
    letter-spacing: 0.055em;
    color: rgba(226,232,240,0.78);
    font-weight: 650;
}}

.nav {{
    display: flex;
    align-items: center;
    gap: 26px;
    color: rgba(255,255,255,0.88);
    font-size: 15px;
    font-weight: 600;
}}

.status {{
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 15px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.13);
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}}

.dot {{
    width: 11px;
    height: 11px;
    border-radius: 999px;
    background: #10b981;
    box-shadow: 0 0 18px rgba(16,185,129,0.86);
}}

.main {{
    min-height: calc(100vh - 74px);
    display: grid;
    grid-template-columns: 44.5% 55.5%;
    align-items: center;
    gap: 42px;
    padding: 34px 56px 18px 56px;
}}

.left {{
    position: relative;
    min-height: 720px;
    display: flex;
    align-items: center;
    overflow: hidden;
}}

.left::after {{
    content: "";
    position: absolute;
    right: -54px;
    top: 130px;
    width: 550px;
    height: 550px;
    opacity: 0.72;
    background: radial-gradient(circle, rgba(37,99,235,0.36) 0 2px, transparent 2px);
    background-size: 9px 9px;
    mask-image: radial-gradient(circle, black 0%, transparent 72%);
    -webkit-mask-image: radial-gradient(circle, black 0%, transparent 72%);
}}

.left::before {{
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
}}

.left-content {{
    position: relative;
    z-index: 2;
    max-width: 600px;
}}

.left h1 {{
    margin: 0 0 4px 0;
    font-size: 52px;
    line-height: 1.05;
    font-weight: 950;
    letter-spacing: -0.05em;
}}

.gradient-title {{
    margin: 0 0 20px 0;
    font-size: 44px;
    line-height: 1.08;
    font-weight: 950;
    letter-spacing: -0.05em;
    background: linear-gradient(90deg, #a855f7 0%, #38bdf8 76%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.lead {{
    max-width: 520px;
    margin-bottom: 36px;
    color: rgba(226,232,240,0.88);
    font-size: 18px;
    line-height: 1.58;
    font-weight: 500;
}}

.feature {{
    display: grid;
    grid-template-columns: 64px 1fr;
    gap: 18px;
    align-items: center;
    margin-bottom: 22px;
}}

.feature-icon {{
    width: 58px;
    height: 58px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    color: #ddd6fe;
    background: linear-gradient(145deg, rgba(124,58,237,0.54), rgba(30,64,175,0.48));
    font-size: 27px;
    box-shadow: 0 16px 34px rgba(0,0,0,0.18);
}}

.feature h3 {{
    margin: 0 0 6px 0;
    color: white;
    font-size: 19px;
    font-weight: 850;
    letter-spacing: -0.02em;
}}

.feature p {{
    margin: 0;
    color: rgba(226,232,240,0.78);
    font-size: 15px;
    line-height: 1.38;
}}

.rule {{
    width: 88%;
    height: 1px;
    background: rgba(148,163,184,0.22);
    margin: 26px 0 24px 0;
}}

.security {{
    display: grid;
    grid-template-columns: 76px 1fr;
    gap: 18px;
    align-items: center;
}}

.security-icon {{
    width: 68px;
    height: 68px;
    border-radius: 23px;
    display: grid;
    place-items: center;
    color: #22d3ee;
    border: 1px solid rgba(34,211,238,0.72);
    font-size: 29px;
}}

.security h3 {{
    margin: 0 0 8px 0;
    color: #22d3ee;
    font-size: 17px;
    font-weight: 850;
}}

.security p {{
    margin: 0;
    max-width: 460px;
    color: rgba(226,232,240,0.80);
    line-height: 1.56;
    font-size: 15px;
}}

.copyright {{
    position: absolute;
    left: 0;
    bottom: 2px;
    color: rgba(226,232,240,0.62);
    font-size: 12.5px;
    z-index: 2;
}}

.right {{
    display: flex;
    justify-content: center;
    align-items: center;
}}

.card {{
    width: min(735px, 100%);
    min-height: 720px;
    padding: 30px 46px;
    border-radius: 18px;
    background:
        radial-gradient(circle at 88% 6%, rgba(124, 58, 237, 0.07), transparent 24%),
        white;
    color: var(--ink);
    border: 1px solid rgba(255,255,255,0.82);
    box-shadow: 0 32px 95px rgba(0,0,0,0.42);
}}

.card-top {{
    display: flex;
    justify-content: flex-end;
    margin-bottom: 26px;
}}

.back-btn {{
    height: 42px;
    padding: 0 16px;
    border-radius: 10px;
    border: 1px solid rgba(124,58,237,0.45);
    background: white;
    color: #5b21b6;
    font-size: 14px;
    font-weight: 850;
    cursor: pointer;
    box-shadow: 0 6px 16px rgba(124,58,237,0.09);
}}

.header {{
    display: grid;
    grid-template-columns: 78px 1fr;
    gap: 22px;
    align-items: center;
    margin-bottom: 30px;
}}

.lock {{
    width: 72px;
    height: 72px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: #f1e9ff;
    color: #6d28d9;
    font-size: 31px;
    font-weight: 900;
}}

.header h2 {{
    margin: 0 0 9px 0;
    color: #0b102f;
    font-size: 32px;
    line-height: 1.12;
    font-weight: 950;
    letter-spacing: -0.04em;
}}

.header p {{
    margin: 0;
    color: #475569;
    font-size: 16px;
    font-weight: 500;
}}

.divider {{
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 18px;
    align-items: center;
    color: #64748b;
    font-size: 14px;
    font-weight: 750;
    margin: 30px 0 24px 0;
}}

.divider::before,
.divider::after {{
    content: "";
    height: 1px;
    background: #d5dbe8;
}}

.message {{
    padding: 11px 13px;
    border: 1px solid;
    border-radius: 10px;
    margin-bottom: 14px;
    font-size: 13px;
    font-weight: 650;
}}

.form-group {{
    margin-bottom: 16px;
}}

.form-label {{
    display: block;
    color: #0f172a;
    font-size: 13px;
    font-weight: 850;
    margin-bottom: 7px;
}}

.input {{
    width: 100%;
    height: 54px;
    border-radius: 10px;
    border: 1px solid #d5dbe8;
    background: white;
    color: #0f172a;
    font-size: 15px;
    padding: 0 15px;
    outline: none;
}}

.input:focus {{
    border-color: #7c3aed;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12);
}}

.form-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 4px 0 16px 0;
}}

.remember {{
    display: {remember_display};
    align-items: center;
    gap: 9px;
    color: #0f172a;
    font-size: 14px;
    font-weight: 500;
}}

.remember input {{
    width: 18px;
    height: 18px;
    accent-color: #6d28d9;
}}

.forgot {{
    display: {forgot_display};
    border: 0;
    background: transparent;
    color: #4f46e5;
    font-size: 14px;
    font-weight: 750;
    cursor: pointer;
}}

.submit {{
    width: 100%;
    height: 58px;
    border: none;
    border-radius: 10px;
    background: linear-gradient(90deg, #6d28d9, #2563eb);
    color: white;
    font-size: 18px;
    font-weight: 900;
    cursor: pointer;
    box-shadow: 0 10px 25px rgba(37,99,235,0.30);
    transition: transform .15s ease, filter .15s ease;
}}

.submit:hover {{
    transform: translateY(-1px);
    filter: brightness(1.05);
}}

.mode-note {{
    text-align: center;
    margin: 20px 0 14px 0;
    color: #64748b;
    font-size: 14.5px;
    font-weight: 500;
}}

.link-btn {{
    border: 0;
    background: transparent;
    color: #4f46e5;
    font-size: 14.5px;
    font-weight: 850;
    margin-left: 10px;
    cursor: pointer;
}}

.notice {{
    margin-top: 22px;
    border: 1px solid rgba(245,158,11,0.48);
    background: linear-gradient(180deg, #fffbeb 0%, #fff7ed 100%);
    border-radius: 12px;
    padding: 18px 20px;
    color: #1f2937;
}}

.notice h3 {{
    margin: 0 0 8px 0;
    color: #3b2f0b;
    font-size: 15.5px;
    font-weight: 850;
}}

.notice p {{
    margin: 5px 0;
    color: #111827;
    font-size: 13.5px;
}}

.check {{
    color: #16a34a;
    font-weight: 900;
    margin-right: 8px;
}}

.footer {{
    text-align: center;
    color: rgba(226,232,240,0.68);
    font-size: 13px;
    margin: 12px 0 18px 0;
    font-weight: 500;
}}

@media (max-width: 980px) {{
    .nav {{
        display: none;
    }}
    .main {{
        grid-template-columns: 1fr;
        padding: 24px;
    }}
    .left {{
        min-height: auto;
        padding: 32px 0;
    }}
    .copyright {{
        position: static;
        margin-top: 34px;
    }}
    .card {{
        min-height: auto;
    }}
}}

@media (max-width: 640px) {{
    .topbar {{
        padding: 0 18px;
    }}
    .brand-sub {{
        display: none;
    }}
    .main {{
        padding: 18px;
    }}
    .left h1 {{
        font-size: 38px;
    }}
    .gradient-title {{
        font-size: 32px;
    }}
    .card {{
        padding: 24px;
    }}
    .header {{
        grid-template-columns: 1fr;
    }}
}}
</style>
</head>
<body>
<div class="page">
    <div class="topbar">
        <div class="brand">
            <span class="eu-mark">✦✦✦</span>
            <span class="brand-main">EU SEE</span>
            <span class="brand-sub">INTELLIGENCE PLATFORM</span>
        </div>
        <div class="nav">
            <span class="status"><span class="dot"></span>System Status: <strong>Operational</strong></span>
            <span>ⓘ Help</span>
            <span>▤ Docs</span>
            <span>◎ English⌄</span>
        </div>
    </div>

    <main class="main">
        <section class="left">
            <div class="left-content">
                <h1>EU SEE</h1>
                <div class="gradient-title">Intelligence Platform</div>
                <div class="lead">
                    Secure access to real-time geopolitical analytics,
                    risk signals, and cross-country monitoring across
                    South East Europe and beyond.
                </div>

                <div class="feature">
                    <div class="feature-icon">◎</div>
                    <div>
                        <h3>86 Countries Monitored</h3>
                        <p>Comprehensive coverage and real-time updates</p>
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">▮</div>
                    <div>
                        <h3>Real-time Signal Processing</h3>
                        <p>AI-powered detection and analytics engine</p>
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">◇</div>
                    <div>
                        <h3>AI-driven Risk Classification</h3>
                        <p>Advanced models for early risk identification</p>
                    </div>
                </div>

                <div class="rule"></div>

                <div class="security">
                    <div class="security-icon">🔐</div>
                    <div>
                        <h3>Enterprise-grade security</h3>
                        <p>Your data is protected with end-to-end encryption and strict access controls.</p>
                    </div>
                </div>
            </div>
            <div class="copyright">© 2024 EU SEE Intelligence Platform. All rights reserved.</div>
        </section>

        <section class="right">
            <div class="card">
                <form method="GET" id="auth_form">
                    <div class="card-top">
                        <button class="back-btn" name="action" value="back" type="submit">← Back to dashboard</button>
                    </div>

                    <div class="header">
                        <div class="lock">▣</div>
                        <div>
                            <h2>{title}</h2>
                            <p>{subtitle}</p>
                        </div>
                    </div>

                    <div class="divider">OR</div>

                    {msg_html}

                    <div class="form-group">
                        <label class="form-label" for="email">Email address</label>
                        <input class="input" id="email" name="email" type="email" placeholder="✉   name@organization.org" autocomplete="email">
                    </div>

                    <div class="form-group" style="display:{password_display};">
                        <label class="form-label" for="password">Password</label>
                        <input class="input" id="password" name="password" type="password" placeholder="🔒   Enter your password" autocomplete="current-password">
                    </div>

                    <div class="form-row">
                        <label class="remember">
                            <input type="checkbox" name="remember" value="1" checked>
                            Keep me signed in on this device
                        </label>
                        <button class="forgot" name="action" value="switch_reset" type="submit">Forgot password?</button>
                    </div>

                    <button class="submit" name="action" value="{mode.lower()}" type="submit">🔒 {button_label}</button>

                    <div class="mode-note">{mode_hint}</div>
                </form>

                <div class="notice">
                    <h3>🛡️ Secure Access Notice</h3>
                    <p><span class="check">✓</span>Access restricted to verified institutional domains</p>
                    <p><span class="check">✓</span>All activity is logged and monitored</p>
                    <p><span class="check">✓</span>Session protected with enterprise-grade encryption</p>
                </div>
            </div>
        </section>
    </main>

    <div class="footer">🔒 Secure authentication &nbsp; • &nbsp; Protected access &nbsp; • &nbsp; Compliance ready</div>
</div>
</body>
</html>
"""


# ============================================================
# Fallback functional form
# This is intentionally kept below the component for now.
# Use this to actually authenticate while the component gives the stable layout.
# ============================================================
def _functional_auth_controls():
    mode = st.session_state.get("auth_mode", "Login")

    with st.expander("Authentication controls", expanded=True):
        st.caption("Use this functional panel while the HTML layout is stabilized.")

        if mode == "Login":
            with st.form("functional_login"):
                email = st.text_input("Email address", placeholder="name@organization.org")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember = st.checkbox("Keep me signed in", value=True)
                submitted = st.form_submit_button("Sign in")

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

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create account", use_container_width=True):
                    st.session_state.auth_mode = "Register"
                    st.rerun()
            with c2:
                if st.button("Forgot password", use_container_width=True):
                    st.session_state.auth_mode = "Reset"
                    st.rerun()

        elif mode == "Register":
            with st.form("functional_register"):
                email = st.text_input("Email address", placeholder="name@organization.org")
                password = st.text_input("Password", type="password", placeholder="Create a secure password")
                submitted = st.form_submit_button("Create account")

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

            if st.button("Back to sign in", use_container_width=True):
                st.session_state.auth_mode = "Login"
                st.rerun()

        else:
            with st.form("functional_reset"):
                email = st.text_input("Email address", placeholder="name@organization.org")
                submitted = st.form_submit_button("Send password reset link")

            if submitted:
                if not firebase_auth:
                    st.error("Firebase authentication is not initialized.")
                    return

                if not email:
                    st.error("Enter your email address.")
                    return

                if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Password reset is restricted to approved institutional domains.")
                    return

                try:
                    firebase_auth.send_password_reset_email(email)
                    st.success("Password reset email sent.")
                except Exception as e:
                    st.error(parse_error(e))

            if st.button("Back to sign in", use_container_width=True):
                st.session_state.auth_mode = "Login"
                st.rerun()


# ============================================================
# Main auth page
# ============================================================
def _render_auth_page():
    _hide_streamlit_chrome()

    mode = st.session_state.get("auth_mode", "Login")

    # Pixel-stable visual layout
    components.html(
        _login_component_html(mode=mode),
        height=930,
        scrolling=False,
    )

    # Functional auth underneath. Once the visual design is approved,
    # replace this with a component callback or keep it as a compact fallback.
    _functional_auth_controls()


def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_auth_page()
