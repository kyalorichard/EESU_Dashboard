# auth.py
import json
import time

import streamlit as st

# Set True only while debugging deployment/auth issues.
# This version also shows critical Firebase init errors even when DEBUG=False,
# because login cannot work without Firebase client auth.
DEBUG = False


# -----------------------------
# Optional Imports
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
# Firebase Admin
# -----------------------------
def init_firebase_admin():
    if not HAS_FIREBASE_ADMIN:
        if DEBUG:
            st.warning("firebase-admin is not installed; skipping Firebase Admin init.")
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
def _firebase_required_keys():
    return ["apiKey", "authDomain", "projectId", "storageBucket", "messagingSenderId", "appId"]


def _firebase_config():
    """
    Return a validated Firebase client config.

    Kept separate from initialization so the dashboard can load normally in
    guest mode even if Firebase is not configured correctly.
    """
    if not HAS_PYREBASE:
        return None, "Firebase client package missing. Add `pyrebase4` to requirements.txt and redeploy."

    cfg_raw = st.secrets.get("firebase", {})
    if not cfg_raw:
        return None, "Firebase config missing. Add the `[firebase]` block to `.streamlit/secrets.toml`."

    cfg = dict(cfg_raw)
    missing = [k for k in _firebase_required_keys() if not cfg.get(k)]
    if missing:
        return None, "Firebase config is incomplete. Missing keys: " + ", ".join(missing)

    # Pyrebase expects this key even when Realtime Database is not used.
    cfg.setdefault("databaseURL", "")
    return cfg, None


@st.cache_resource(show_spinner=False)
def get_firebase_client_cached():
    """
    Stable cached Firebase client.

    Streamlit reruns the script frequently. Initializing Firebase globally on
    every rerun can make login unstable. This cached resource keeps the Pyrebase
    client stable for the running app process.
    """
    cfg, error = _firebase_config()
    if error:
        return None, None, error

    try:
        firebase = pyrebase.initialize_app(cfg)
        auth = firebase.auth()
        if not auth:
            return None, None, "Firebase auth object was not created."
        return firebase, auth, None
    except Exception as e:
        return None, None, f"Firebase initialization failed: {e}"


@st.cache_resource(show_spinner=False)
def get_firebase_admin_cached():
    """Stable cached Firebase Admin app, used only when available."""
    return init_firebase_admin()


def get_firebase_auth(show_error=False):
    """
    Get Firebase auth safely.

    This function prevents normal dashboard usage from being blocked by Firebase
    errors. Errors are shown only from the login/register/reset flows.
    """
    firebase, auth, error = get_firebase_client_cached()
    if error and show_error:
        st.error(f"❌ {error}")
    return auth


# Keep these names for compatibility with any other modules importing them.
firebase_admin_app = None
firebase_client = None
firebase_auth = None


# -----------------------------
# Access Control
# -----------------------------
PRIVILEGED_DOMAINS = set(
    str(d).lower().strip()
    for d in st.secrets.get("access", {}).get("privileged_domains", [])
    if str(d).strip()
)


def get_domain(email: str) -> str:
    return str(email or "").strip().split("@")[-1].lower()


def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "id_token": None,
        "refresh_token": None,
        "token_created_at": None,
        "restored": False,
        "auth_mode": "Login",
        "auth_remember": False,
        "auth_view": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


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
        if DEBUG:
            st.warning("streamlit-cookies-manager is not installed; cookies disabled.")
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


def restore_session():
    """
    Restore a lightweight remembered session.

    This restores dashboard role state only. It does not try to keep a Firebase
    ID token alive forever, because Pyrebase token refresh is not consistently
    reliable in all Streamlit deployments. Users can simply sign in again when
    the Firebase token expires.
    """
    if st.session_state.get("restored"):
        return

    cookies = get_cookies()
    if cookies and cookies.ready():
        try:
            if "email" in cookies:
                email = str(cookies.get("email") or "").lower().strip()
                verified = str(cookies.get("email_verified", "False")) == "True"
                role = str(cookies.get("role") or ("privileged" if verified else "restricted"))

                st.session_state.user = bool(email)
                st.session_state.email = email
                st.session_state.name = cookies.get("name") or email.split("@")[0].replace(".", " ").title()
                st.session_state.role = role
                st.session_state.email_verified = verified
        except Exception as e:
            if DEBUG:
                st.sidebar.warning(f"Error restoring session: {e}")

    st.session_state.restored = True


def _save_cookie_session(email, name, verified, role, remember=False):
    """
    Save only safe, non-sensitive dashboard session metadata.

    Do not store Firebase ID tokens in cookies. This avoids token expiry and
    security issues causing login to break after some time.
    """
    if not remember:
        return

    cookies = get_cookies()
    if cookies and cookies.ready():
        cookies["email"] = str(email or "").lower().strip()
        cookies["name"] = str(name or "")
        cookies["email_verified"] = str(bool(verified))
        cookies["role"] = str(role or "guest")
        try:
            cookies.save()
        except Exception:
            pass


def _clear_cookie_session():
    cookies = get_cookies()
    if cookies and cookies.ready():
        for key in [
            "email", "name", "role", "email_verified",
            "id_token", "refresh_token", "local_id"
        ]:
            if key in cookies:
                del cookies[key]
        try:
            cookies.save()
        except Exception:
            pass


def logout():
    _clear_cookie_session()

    for key in [
        "user", "email", "name", "role", "email_verified",
        "id_token", "refresh_token", "token_created_at",
        "restored", "auth_mode", "auth_remember", "auth_view"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


def _set_logged_in_session(email, verified, role, remember, user_payload=None):
    """Centralized session write after a successful Firebase sign-in."""
    name = email.split("@")[0].replace(".", " ").title()

    st.session_state.user = True
    st.session_state.email = email
    st.session_state.name = name
    st.session_state.email_verified = bool(verified)
    st.session_state.role = role
    st.session_state.auth_remember = bool(remember)
    st.session_state.auth_view = False
    st.session_state.restored = True

    # Store tokens only in Streamlit session_state, not persistent cookies.
    if user_payload:
        st.session_state.id_token = user_payload.get("idToken")
        st.session_state.refresh_token = user_payload.get("refreshToken")
        st.session_state.token_created_at = time.time()

    _save_cookie_session(email, name, verified, role, remember)


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
        "USER_DISABLED": "This user account has been disabled.",
    }
    return friendly.get(msg, msg)


# -----------------------------
# Premium Auth Styling
# -----------------------------
def _auth_page_css():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        header[data-testid="stHeader"] { background: transparent !important; }

        html, body, .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(102, 0, 148, 0.12), transparent 28%),
                radial-gradient(circle at 94% 18%, rgba(0, 140, 170, 0.14), transparent 28%),
                radial-gradient(circle at 50% 100%, rgba(255, 219, 88, 0.15), transparent 30%),
                linear-gradient(135deg, #FBF8FD 0%, #F7FBFD 45%, #FFFAF0 100%) !important;
        }

        .block-container {
            max-width: 1120px !important;
            padding-top: 1.7rem !important;
            padding-bottom: 1.7rem !important;
        }

        .auth-page-title {
            text-align: center;
            margin-bottom: 18px;
            font-family: Arial, sans-serif;
        }

        .auth-page-title h1 {
            margin: 0;
            color: #231942;
            font-size: 32px;
            font-family: Arial Black, Arial, sans-serif;
            letter-spacing: -0.6px;
        }

        .auth-page-title p {
            margin: 8px 0 0 0;
            color: #6f667a;
            font-size: 13px;
        }

        div[data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }

        div[data-testid="stHorizontalBlock"] > div {
            display: flex !important;
            flex-direction: column !important;
        }

        div[data-testid="stHorizontalBlock"] > div > div {
            flex: 1 1 auto !important;
        }

        .auth-brand-card {
            height: 100% !important;
            min-height: 560px;
            border-radius: 28px;
            padding: 38px 36px 30px 36px;
            color: white;
            background:
                radial-gradient(circle at 88% 16%, rgba(255,219,88,0.34), transparent 25%),
                radial-gradient(circle at 20% 88%, rgba(255,255,255,0.15), transparent 30%),
                linear-gradient(155deg, #660094 0%, #4b006f 55%, #008CAA 100%);
            box-shadow: 0 24px 70px rgba(35,25,66,0.22);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            overflow: hidden;
        }

        .brand-eyebrow {
            font-size: 11px;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            font-weight: 900;
            color: #FFDB58;
            margin-bottom: 12px;
            font-family: Arial, sans-serif;
        }

        .brand-title {
            font-family: Arial Black, Arial, sans-serif;
            font-size: 46px;
            line-height: 0.96;
            letter-spacing: -1.4px;
            margin-bottom: 16px;
        }

        .brand-text {
            font-family: Arial, sans-serif;
            font-size: 14px;
            line-height: 1.62;
            max-width: 440px;
            color: rgba(255,255,255,0.93);
        }

        .brand-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 24px;
        }

        .brand-badge {
            border: 1px solid rgba(255,255,255,0.26);
            background: rgba(255,255,255,0.13);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 800;
            font-family: Arial, sans-serif;
            color: white;
        }

        .auth-brand-footer {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 30px;
        }

        .mini-stat {
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.19);
            border-radius: 16px;
            padding: 12px;
        }

        .mini-stat-value {
            color: #FFDB58;
            font-size: 18px;
            font-weight: 900;
            font-family: Arial Black, Arial, sans-serif;
        }

        .mini-stat-label {
            color: rgba(255,255,255,0.84);
            font-size: 10px;
            font-family: Arial, sans-serif;
            margin-top: 3px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            height: 100% !important;
            min-height: 560px !important;
            border-radius: 28px !important;
            border: 1px solid rgba(102,0,148,0.12) !important;
            box-shadow: 0 24px 70px rgba(35,25,66,0.14) !important;
            background: rgba(255,255,255,0.94) !important;
            backdrop-filter: blur(14px) !important;
            display: flex !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            width: 100%;
            height: 100%;
            padding: 32px 36px 28px 36px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            box-sizing: border-box;
        }

        .auth-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: #f7f0fb;
            color: #660094;
            border: 1px solid #eadcf3;
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 11px;
            font-weight: 900;
            font-family: Arial, sans-serif;
        }

        .form-title {
            font-family: Arial Black, Arial, sans-serif;
            color: #231942;
            font-size: 28px;
            letter-spacing: -0.3px;
            margin: 18px 0 6px 0;
        }

        .form-subtitle {
            font-family: Arial, sans-serif;
            color: #6f667a;
            font-size: 13px;
            line-height: 1.55;
            margin-bottom: 18px;
        }

        .mode-card {
            display: grid;
            grid-template-columns: 1fr 1fr;
            background: #f8f5fb;
            border: 1px solid #eee6f6;
            border-radius: 14px;
            padding: 4px;
            margin-bottom: 16px;
            gap: 4px;
        }

        .mode-active, .mode-inactive {
            text-align: center;
            border-radius: 11px;
            padding: 9px 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: 900;
        }

        .mode-active {
            color: white;
            background: linear-gradient(135deg, #660094, #008CAA);
            box-shadow: 0 8px 20px rgba(102,0,148,0.20);
        }

        .mode-inactive {
            color: #6f667a;
        }

        .auth-note {
            background: #fffaf0;
            border: 1px solid rgba(255,219,88,0.55);
            border-left: 4px solid #FFDB58;
            border-radius: 14px;
            padding: 11px 13px;
            color: #4b3b14;
            font-size: 11.5px;
            line-height: 1.42;
            font-family: Arial, sans-serif;
            margin-top: 14px;
        }

        label p {
            font-size: 12px !important;
            font-weight: 900 !important;
            color: #332045 !important;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 12px !important;
            min-height: 44px !important;
            font-size: 13px !important;
            border: 1px solid #e7ddec !important;
            background: #fcfbfd !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #660094 !important;
            box-shadow: 0 0 0 2px rgba(102,0,148,0.10) !important;
        }

        div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
        }

        button[kind="primaryFormSubmit"], button[kind="formSubmit"] {
            border-radius: 12px !important;
            min-height: 45px !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #660094, #008CAA) !important;
            border: 0 !important;
        }

        button {
            border-radius: 12px !important;
            font-weight: 800 !important;
        }

        .small-footer {
            text-align: center;
            color: #91869b;
            font-size: 10.5px;
            font-family: Arial, sans-serif;
            margin-top: 16px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1.2rem !important;
            }

            .auth-brand-card {
                min-height: auto;
                padding: 28px;
            }

            .brand-title {
                font-size: 36px;
            }

            .auth-brand-footer {
                grid-template-columns: 1fr;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 24px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_auth_mode(mode: str):
    st.session_state.auth_mode = mode
    st.rerun()


def _back_to_dashboard():
    st.session_state.auth_view = False
    st.rerun()


def _login_form():
    with st.form("eusee_login_premium_form"):
        email = st.text_input("Email address", placeholder="name@organization.org").strip().lower()
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        remember = st.checkbox("Keep me signed in on this device", value=st.session_state.get("auth_remember", False))
        submitted = st.form_submit_button("Sign in to dashboard", use_container_width=True)

    if submitted:
        auth_obj = get_firebase_auth(show_error=True)

        if not auth_obj:
            st.warning("Login is temporarily unavailable. The dashboard remains available in guest mode.")
            return

        if not email or not password:
            st.error("Enter email and password.")
            return

        if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Access is restricted to approved domains.")
            return

        try:
            user = auth_obj.sign_in_with_email_and_password(email, password)
            info = auth_obj.get_account_info(user["idToken"])
            verified = bool(info["users"][0].get("emailVerified", False))
            role = "privileged" if verified else "restricted"

            if not verified:
                st.warning("Your account is signed in but email verification is still pending. Please verify your email to unlock privileged access.")

            _set_logged_in_session(
                email=email,
                verified=verified,
                role=role,
                remember=remember,
                user_payload=user,
            )

            st.success("Signed in successfully. Redirecting to dashboard...")
            st.rerun()

        except Exception as e:
            st.error(parse_error(e))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create account", use_container_width=True, key="switch_to_register"):
            _set_auth_mode("Register")
    with col2:
        if st.button("Forgot password", use_container_width=True, key="switch_to_reset"):
            _set_auth_mode("Reset")


def _register_form():
    with st.form("eusee_register_premium_form"):
        email = st.text_input("Email address", placeholder="name@organization.org").strip().lower()
        password = st.text_input("Password", placeholder="Create a secure password", type="password")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        auth_obj = get_firebase_auth(show_error=True)

        if not auth_obj:
            st.warning("Account registration is temporarily unavailable. The dashboard remains available in guest mode.")
            return

        if not email or not password:
            st.error("Enter email and password.")
            return

        if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Registration is restricted to approved domains.")
            return

        try:
            user = auth_obj.create_user_with_email_and_password(email, password)
            auth_obj.send_email_verification(user["idToken"])
            st.success("Registration successful. Check your email to verify your account, then sign in.")
        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to sign in", use_container_width=True, key="register_back_login"):
        _set_auth_mode("Login")


def _reset_form():
    with st.form("eusee_reset_premium_form"):
        reset_email = st.text_input("Email address", placeholder="name@organization.org").strip().lower()
        submitted = st.form_submit_button("Send password reset link", use_container_width=True)

    if submitted:
        auth_obj = get_firebase_auth(show_error=True)

        if not auth_obj:
            st.warning("Password reset is temporarily unavailable because login services are not configured.")
            return

        if not reset_email:
            st.warning("Enter your email first.")
            return

        if PRIVILEGED_DOMAINS and get_domain(reset_email) not in PRIVILEGED_DOMAINS:
            st.error("Password reset is restricted to approved domains.")
            return

        try:
            auth_obj.send_password_reset_email(reset_email)
            st.success("Password reset email sent.")
        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to sign in", use_container_width=True, key="reset_back_login"):
        _set_auth_mode("Login")


def _render_premium_auth_page():
    _auth_page_css()
    mode = st.session_state.get("auth_mode", "Login")
    mode_title = {
        "Login": "Welcome back",
        "Register": "Create your account",
        "Reset": "Reset password",
    }.get(mode, "Welcome back")
    mode_subtitle = {
        "Login": "Access privileged EU SEE dashboard features using your approved organizational account.",
        "Register": "Request dashboard access using an approved domain. Email verification is required before privileged access.",
        "Reset": "Enter your approved email address and we will send a password reset link.",
    }.get(mode, "Access the EU SEE Dashboard.")

    st.markdown(
        """
        <div class="auth-page-title">
            <h1>EU SEE Dashboard Access</h1>
            <p>Secure sign-in for authorized users. Return to the dashboard at any time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.0, 1.05], gap="large")

    with left:
        st.markdown(
            """
            <div class="auth-brand-card">
                <div>
                    <div class="brand-eyebrow">Secure access portal</div>
                    <div class="brand-title">EU SEE<br>Dashboard</div>
                    <div class="brand-text">
                        Sign in to unlock privileged analytical views, protected dashboard features,
                        and role-based access for approved EU SEE users.
                    </div>
                    <div class="brand-badges">
                        <span class="brand-badge">🔐 Verified access</span>
                        <span class="brand-badge">🌍 86-country monitoring</span>
                        <span class="brand-badge">📊 Protected insights</span>
                    </div>
                </div>
                <div class="auth-brand-footer">
                    <div class="mini-stat"><div class="mini-stat-value">86</div><div class="mini-stat-label">Countries</div></div>
                    <div class="mini-stat"><div class="mini-stat-value">EU SEE</div><div class="mini-stat-label">Network data</div></div>
                    <div class="mini-stat"><div class="mini-stat-value">Secure</div><div class="mini-stat-label">Role-based access</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            top_a, top_b = st.columns([1.25, 0.75])
            with top_a:
                st.markdown('<div class="auth-pill">🔐 Authorized users only</div>', unsafe_allow_html=True)
            with top_b:
                if st.button("← Dashboard", use_container_width=True, key="premium_back_dashboard"):
                    _back_to_dashboard()

            st.markdown(f'<div class="form-title">{mode_title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="form-subtitle">{mode_subtitle}</div>', unsafe_allow_html=True)

            if mode == "Login":
                st.markdown('<div class="mode-card"><div class="mode-active">Sign in</div><div class="mode-inactive">Register</div></div>', unsafe_allow_html=True)
                _login_form()
            elif mode == "Register":
                st.markdown('<div class="mode-card"><div class="mode-inactive">Sign in</div><div class="mode-active">Register</div></div>', unsafe_allow_html=True)
                _register_form()
            else:
                st.markdown('<div class="mode-card"><div class="mode-active">Password reset</div><div class="mode-inactive">Secure email link</div></div>', unsafe_allow_html=True)
                _reset_form()

            domain_note = "Approved organizational domains only."
            if PRIVILEGED_DOMAINS:
                domain_note = "Approved domains: " + ", ".join(sorted(PRIVILEGED_DOMAINS))
            st.markdown(
                f'<div class="auth-note">💡 {domain_note}<br>After successful sign-in, you will return automatically to the dashboard with an active session.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="small-footer">EU SEE Dashboard · Secure authentication · Protected access</div>', unsafe_allow_html=True)


# -----------------------------
# Authentication UI Entrypoint
# -----------------------------
def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_premium_auth_page()
