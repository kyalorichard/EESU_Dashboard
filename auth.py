# auth.py
import json
import time

import streamlit as st

DEBUG = False


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


def _firebase_required_keys():
    return ["apiKey", "authDomain", "projectId", "storageBucket", "messagingSenderId", "appId"]


def init_firebase_client():
    if not HAS_PYREBASE:
        st.error("❌ Firebase client package missing. Add `pyrebase4` to requirements.txt and redeploy.")
        return None, None

    cfg_raw = st.secrets.get("firebase", {})
    if not cfg_raw:
        st.error("❌ Firebase config missing. Add the `[firebase]` block to `.streamlit/secrets.toml`.")
        return None, None

    cfg = dict(cfg_raw)
    missing = [k for k in _firebase_required_keys() if not cfg.get(k)]

    if missing:
        st.error("❌ Firebase config is incomplete. Missing keys: " + ", ".join(missing))
        return None, None

    cfg.setdefault("databaseURL", "")

    try:
        firebase = pyrebase.initialize_app(cfg)
        auth = firebase.auth()

        if not auth:
            st.error("❌ Firebase auth object was not created.")
            return None, None

        return firebase, auth

    except Exception as e:
        st.error(f"❌ Firebase initialization failed: {e}")
        return None, None


firebase_admin_app = init_firebase_admin()
firebase_client, firebase_auth = init_firebase_client()


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


def get_cookies():
    if not HAS_COOKIES:
        return None

    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            return None

        st.session_state.cookies = EncryptedCookieManager(
            prefix="eusee",
            password=password,
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

    except Exception as e:
        if DEBUG:
            st.sidebar.warning(f"Cookie load error: {e}")
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
                st.session_state.email = str(cookies.get("email") or "").lower().strip()
                st.session_state.name = cookies.get("name")
                st.session_state.role = cookies.get("role")
                st.session_state.email_verified = str(cookies.get("email_verified", "False")) == "True"
        except Exception as e:
            if DEBUG:
                st.sidebar.warning(f"Error restoring session: {e}")

    st.session_state.restored = True


def _save_cookie_session(email, name, verified, role, remember=False):
    if not remember:
        return

    cookies = get_cookies()

    if cookies and cookies.ready():
        cookies["email"] = str(email or "").lower().strip()
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
        "user",
        "email",
        "name",
        "role",
        "email_verified",
        "restored",
        "auth_mode",
        "auth_remember",
        "auth_view",
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
        "USER_DISABLED": "This user account has been disabled.",
    }

    return friendly.get(msg, msg)


def _auth_page_css():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        html, body, .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(102, 0, 148, 0.10), transparent 28%),
                radial-gradient(circle at 94% 18%, rgba(0, 140, 170, 0.12), transparent 28%),
                radial-gradient(circle at 50% 100%, rgba(255, 219, 88, 0.12), transparent 30%),
                linear-gradient(135deg, #FBF8FD 0%, #F7FBFD 45%, #FFFAF0 100%) !important;
        }

        .block-container {
            max-width: 880px !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        .auth-page-title {
            text-align: center;
            margin-bottom: 22px;
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
            margin: 9px 0 0 0;
            color: #6f667a;
            font-size: 14px;
            line-height: 1.5;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 28px !important;
            border: 1px solid rgba(102,0,148,0.12) !important;
            box-shadow: 0 24px 70px rgba(35,25,66,0.14) !important;
            background: rgba(255,255,255,0.96) !important;
            backdrop-filter: blur(14px) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 34px 38px 30px 38px !important;
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
            margin: 20px 0 6px 0;
        }

        .form-subtitle {
            font-family: Arial, sans-serif;
            color: #6f667a;
            font-size: 13px;
            line-height: 1.55;
            margin-bottom: 18px;
        }

        .mode-card {
            display: flex;
            align-items: flex-end;
            gap: 24px;
            margin-bottom: 22px;
            border-bottom: 1px solid #e9e3f1;
            padding-bottom: 0;
        }

        .mode-active {
            position: relative;
            font-family: Arial, sans-serif;
            font-size: 13px;
            font-weight: 800;
            color: #231942;
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0 2px 12px 2px;
            box-shadow: none;
            cursor: default;
            pointer-events: none;
            user-select: none;
            letter-spacing: 0.1px;
        }

        .mode-active::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -1px;
            width: 100%;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(135deg, #660094, #008CAA);
        }

        .auth-note {
            background: #fffaf0;
            border: 1px solid rgba(255,219,88,0.55);
            border-left: 4px solid #FFDB58;
            border-radius: 14px;
            padding: 12px 14px;
            color: #4b3b14;
            font-size: 11.5px;
            line-height: 1.45;
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

        button[kind="primaryFormSubmit"],
        button[kind="formSubmit"] {
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
                max-width: 100% !important;
                padding: 1.2rem 1rem !important;
            }

            .auth-page-title h1 {
                font-size: 26px;
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
    with st.form("eusee_login_form"):
        email = st.text_input(
            "Email address",
            placeholder="name@organization.org",
        ).strip().lower()

        password = st.text_input(
            "Password",
            placeholder="Enter your password",
            type="password",
        )

        remember = st.checkbox(
            "Keep me signed in on this device",
            value=st.session_state.get("auth_remember", False),
        )

        submitted = st.form_submit_button(
            "Sign in to Dashboard",
            use_container_width=True,
        )

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized. See the Firebase error shown above.")
            return

        if not email or not password:
            st.error("Enter email and password.")
            return

        if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Access is restricted to approved EUSEE partner accounts.")
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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Create Account", use_container_width=True, key="switch_to_register"):
            _set_auth_mode("Register")

    with col2:
        if st.button("Forgot Password", use_container_width=True, key="switch_to_reset"):
            _set_auth_mode("Reset")


def _register_form():
    with st.form("eusee_register_form"):
        email = st.text_input(
            "Email address",
            placeholder="name@organization.org",
        ).strip().lower()

        password = st.text_input(
            "Password",
            placeholder="Create a secure password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Create Account",
            use_container_width=True,
        )

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized. See the Firebase error shown above.")
            return

        if not email or not password:
            st.error("Enter email and password.")
            return

        if PRIVILEGED_DOMAINS and get_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Registration is restricted to approved EUSEE partner accounts.")
            return

        try:
            user = firebase_auth.create_user_with_email_and_password(email, password)
            firebase_auth.send_email_verification(user["idToken"])

            st.success("Registration successful. Check your email to verify your account, then sign in.")

        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to Sign In", use_container_width=True, key="register_back_login"):
        _set_auth_mode("Login")


def _reset_form():
    with st.form("eusee_reset_form"):
        reset_email = st.text_input(
            "Email address",
            placeholder="name@organization.org",
        ).strip().lower()

        submitted = st.form_submit_button(
            "Send Password Reset Link",
            use_container_width=True,
        )

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized. See the Firebase error shown above.")
            return

        if not reset_email:
            st.warning("Enter your email first.")
            return

        if PRIVILEGED_DOMAINS and get_domain(reset_email) not in PRIVILEGED_DOMAINS:
            st.error("Password reset is restricted to approved EUSEE partner accounts.")
            return

        try:
            firebase_auth.send_password_reset_email(reset_email)
            st.success("Password reset email sent.")

        except Exception as e:
            st.error(parse_error(e))

    if st.button("Back to Sign In", use_container_width=True, key="reset_back_login"):
        _set_auth_mode("Login")


def _render_premium_auth_page():
    _auth_page_css()

    mode = st.session_state.get("auth_mode", "Login")

    mode_title = {
        "Login": "Sign in or create account",
        "Register": "Create account",
        "Reset": "Reset password",
    }.get(mode, "Sign in or create account")

    mode_subtitle = {
        "Login": "Use the form below to sign in or create an account.",
        "Register": "Create an account using your organizational email. Email verification is required before privileged access.",
        "Reset": "Enter your email address and we will send a password reset link.",
    }.get(mode, "Access the EUSEE Dashboard.")

    left_space, center, right_space = st.columns([0.18, 0.64, 0.18])

    with center:
        with st.container(border=True):
            top_a, top_b = st.columns([1.25, 0.75])

            with top_a:
                st.markdown(
                    '<div class="auth-pill">🔐 Authorized users only</div>',
                    unsafe_allow_html=True,
                )

            with top_b:
                if st.button("← Dashboard", use_container_width=True, key="premium_back_dashboard"):
                    _back_to_dashboard()

            st.markdown(f'<div class="form-title">{mode_title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="form-subtitle">{mode_subtitle}</div>', unsafe_allow_html=True)

            if mode == "Login":
                st.markdown(
                    """
                    <div class="mode-card">
                        <div class="mode-active">Sign in</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _login_form()

            elif mode == "Register":
                st.markdown(
                    """
                    <div class="mode-card">
                        <div class="mode-active">Create account</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _register_form()

            else:
                st.markdown(
                    """
                    <div class="mode-card">
                        <div class="mode-active">Password reset</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _reset_form()

            st.markdown(
                """
                <div class="auth-note">
                    Access is limited to approved EUSEE partner accounts.
                    After successful sign-in, you will return automatically to the dashboard.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="small-footer">EUSEE Dashboard · Secure authentication · Protected access</div>',
                unsafe_allow_html=True,
            )


def auth_ui():
    init_session()
    restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.auth_view = False
        return

    _render_premium_auth_page()
