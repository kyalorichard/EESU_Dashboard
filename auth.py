# auth.py
from __future__ import annotations

import base64
import json
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import pyrebase
    HAS_PYREBASE = True
except ImportError:
    pyrebase = None
    HAS_PYREBASE = False


COOKIE_NAME = "eusee_auth_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": "guest",
        "email_verified": False,
        "restored": False,
        "auth_mode": "Login",
        "auth_view": False,
        "id_token": None,
        "refresh_token": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def init_firebase_client():
    if not HAS_PYREBASE:
        st.error("❌ Add `pyrebase4` to requirements.txt.")
        return None, None

    cfg = dict(st.secrets.get("firebase", {}))
    cfg.setdefault("databaseURL", "")

    required = [
        "apiKey",
        "authDomain",
        "projectId",
        "storageBucket",
        "messagingSenderId",
        "appId",
    ]

    missing = [k for k in required if not cfg.get(k)]
    if missing:
        st.error("❌ Firebase config missing keys: " + ", ".join(missing))
        return None, None

    try:
        firebase = pyrebase.initialize_app(cfg)
        return firebase, firebase.auth()
    except Exception as e:
        st.error(f"❌ Firebase initialization failed: {e}")
        return None, None


firebase_client, firebase_auth = init_firebase_client()


PRIVILEGED_DOMAINS = set(
    str(d).lower().strip()
    for d in st.secrets.get("access", {}).get("privileged_domains", [])
    if str(d).strip()
)


def get_domain(email: str) -> str:
    return str(email or "").strip().split("@")[-1].lower()


def _encode_payload(payload: dict) -> str:
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _decode_payload(value: str) -> dict:
    try:
        raw = base64.urlsafe_b64decode(value.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _read_cookie() -> dict:
    try:
        raw = st.context.cookies.get(COOKIE_NAME)
    except Exception:
        raw = None

    if not raw:
        return {}

    return _decode_payload(raw)


def _set_cookie_and_reload(payload: dict):
    encoded = _encode_payload(payload)

    components.html(
        f"""
        <script>
        document.cookie = "{COOKIE_NAME}={encoded}; path=/; max-age={COOKIE_MAX_AGE}; SameSite=Lax";
        setTimeout(function() {{
            window.parent.location.reload();
        }}, 250);
        </script>
        """,
        height=0,
        width=0,
    )


def _clear_cookie_and_reload():
    components.html(
        f"""
        <script>
        document.cookie = "{COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax";
        setTimeout(function() {{
            window.parent.location.reload();
        }}, 250);
        </script>
        """,
        height=0,
        width=0,
    )


def refresh_firebase_token(refresh_token: str):
    api_key = st.secrets.get("firebase", {}).get("apiKey")

    if not api_key or not refresh_token:
        return None

    url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"

    try:
        response = requests.post(
            url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=15,
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


def _apply_state(email, name, role, id_token, refresh_token):
    st.session_state.user = True
    st.session_state.email = email
    st.session_state.name = name or email.split("@")[0].replace(".", " ").title()
    st.session_state.role = role or "privileged"
    st.session_state.email_verified = True
    st.session_state.id_token = id_token
    st.session_state.refresh_token = refresh_token
    st.session_state.auth_view = False
    st.session_state.restored = True


def restore_session():
    init_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.restored = True
        return True

    cookie_data = _read_cookie()

    if not cookie_data:
        st.session_state.restored = True
        return False

    email = str(cookie_data.get("email") or "").lower().strip()
    name = cookie_data.get("name") or ""
    role = cookie_data.get("role") or "privileged"
    refresh_token = cookie_data.get("refresh_token") or ""

    if not email or not refresh_token:
        st.session_state.restored = True
        return False

    refreshed = refresh_firebase_token(refresh_token)

    if not refreshed:
        st.session_state.restored = True
        return False

    id_token = refreshed.get("id_token")
    new_refresh_token = refreshed.get("refresh_token", refresh_token)

    _apply_state(
        email=email,
        name=name,
        role=role,
        id_token=id_token,
        refresh_token=new_refresh_token,
    )

    return True


def is_authenticated():
    restore_session()
    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
    )


def is_privileged():
    restore_session()
    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
        and st.session_state.get("role") == "privileged"
    )


def logout():
    for key in [
        "user",
        "email",
        "name",
        "role",
        "email_verified",
        "restored",
        "auth_mode",
        "auth_view",
        "id_token",
        "refresh_token",
    ]:
        st.session_state.pop(key, None)

    _clear_cookie_and_reload()


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


def _set_auth_mode(mode: str):
    st.session_state.auth_mode = mode
    st.rerun()


def _back_to_dashboard():
    st.session_state.auth_view = False
    st.rerun()


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
                radial-gradient(circle at 10% 10%, rgba(102, 0, 148, 0.14), transparent 30%),
                radial-gradient(circle at 90% 18%, rgba(0, 140, 170, 0.14), transparent 28%),
                radial-gradient(circle at 50% 100%, rgba(255, 219, 88, 0.16), transparent 34%),
                linear-gradient(135deg, #faf7fc 0%, #f7fbfd 50%, #fffaf0 100%) !important;
        }

        .block-container {
            max-width: 920px !important;
            padding-top: 2.4rem !important;
            padding-bottom: 2.4rem !important;
        }

        .auth-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, #f7f0fb, #f4fbfd);
            color: #4b006d;
            border: 1px solid rgba(102,0,148,0.16);
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 11.5px;
            font-weight: 900;
            font-family: Arial, sans-serif;
            letter-spacing: 0.15px;
            white-space: nowrap;
        }

        .mode-card {
            display: flex;
            align-items: flex-end;
            gap: 24px;
            margin: 0 0 24px 0;
            border-bottom: 1px solid #eee7f4;
        }

        .mode-active {
            position: relative;
            font-family: Arial, sans-serif;
            font-size: 15px;
            font-weight: 900;
            color: #231942;
            padding: 0 2px 13px 2px;
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
            background: linear-gradient(135deg, #fffaf0, #fffdf7);
            border: 1px solid rgba(255,219,88,0.58);
            border-left: 4px solid #FFDB58;
            border-radius: 16px;
            padding: 13px 15px;
            color: #4b3b14;
            font-size: 11.8px;
            line-height: 1.5;
            font-family: Arial, sans-serif;
            margin-top: 16px;
        }

        label p {
            font-size: 12px !important;
            font-weight: 900 !important;
            color: #332045 !important;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 14px !important;
            min-height: 46px !important;
            font-size: 13px !important;
            border: 1px solid #e5d9eb !important;
            background: #fcfbfd !important;
        }

        button[kind="primaryFormSubmit"],
        button[kind="formSubmit"] {
            border-radius: 14px !important;
            min-height: 47px !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #660094, #008CAA) !important;
            border: 0 !important;
            box-shadow: 0 10px 24px rgba(102,0,148,0.18) !important;
        }

        button {
            border-radius: 14px !important;
            font-weight: 850 !important;
            border-color: #e6ddec !important;
        }

        .small-footer {
            text-align: center;
            color: #91869b;
            font-size: 10.8px;
            font-family: Arial, sans-serif;
            margin-top: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

        submitted = st.form_submit_button(
            "Sign in to Dashboard",
            use_container_width=True,
        )

    if submitted:
        if not firebase_auth:
            st.error("Firebase authentication is not initialized.")
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

            if not verified:
                st.warning("Please verify your email before signing in.")
                return

            role = "privileged"
            name = email.split("@")[0].replace(".", " ").title()

            _apply_state(
                email=email,
                name=name,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
            )

            st.success("Signed in successfully. Redirecting to dashboard...")

            _set_cookie_and_reload(
                {
                    "email": email,
                    "name": name,
                    "role": role,
                    "refresh_token": user.get("refreshToken"),
                }
            )

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
            st.error("Firebase authentication is not initialized.")
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
            st.session_state.auth_mode = "Login"

        except Exception as e:
            st.error(parse_error(e))


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
            st.error("Firebase authentication is not initialized.")
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
            st.session_state.auth_mode = "Login"

        except Exception as e:
            st.error(parse_error(e))


def _render_premium_auth_page():
    _auth_page_css()

    mode = st.session_state.get("auth_mode", "Login")

    left_space, center, right_space = st.columns([0.16, 0.68, 0.16])

    with center:
        with st.container(border=True):
            top_a, top_b = st.columns([1.35, 0.65])

            with top_a:
                st.markdown(
                    '<div class="auth-pill">🔐 Authorized users only</div>',
                    unsafe_allow_html=True,
                )

            with top_b:
                if st.button("← Dashboard", use_container_width=True, key="premium_back_dashboard"):
                    _back_to_dashboard()

            if mode == "Login":
                st.markdown(
                    '<div class="mode-card"><div class="mode-active">Sign in</div></div>',
                    unsafe_allow_html=True,
                )
                _login_form()

            elif mode == "Register":
                st.markdown(
                    '<div class="mode-card"><div class="mode-active">Create account</div></div>',
                    unsafe_allow_html=True,
                )
                _register_form()

            else:
                st.markdown(
                    '<div class="mode-card"><div class="mode-active">Password reset</div></div>',
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