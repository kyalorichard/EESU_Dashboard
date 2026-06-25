# auth.py
import json
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
    import extra_streamlit_components as stx
    HAS_COOKIES = True
except ImportError:
    stx = None
    HAS_COOKIES = False


COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

COOKIE_KEYS = [
    "email",
    "name",
    "role",
    "email_verified",
    "id_token",
    "refresh_token",
]


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
    return [
        "apiKey",
        "authDomain",
        "projectId",
        "storageBucket",
        "messagingSenderId",
        "appId",
    ]


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
        "auth_remember": True,
        "auth_view": False,
        "id_token": None,
        "refresh_token": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_cookies():
    if not HAS_COOKIES:
        return None

    if "cookie_manager" not in st.session_state:
        st.session_state.cookie_manager = stx.CookieManager(
            key="eusee_cookie_manager"
        )

    return st.session_state.cookie_manager


def _set_authenticated_session(
    email,
    name,
    verified,
    role,
    id_token=None,
    refresh_token=None,
):
    st.session_state.user = True
    st.session_state.email = str(email or "").lower().strip()
    st.session_state.name = name
    st.session_state.email_verified = bool(verified)
    st.session_state.role = role
    st.session_state.id_token = id_token
    st.session_state.refresh_token = refresh_token


def _save_cookie_session(
    email,
    name,
    verified,
    role,
    id_token=None,
    refresh_token=None,
    remember=True,
):
    if not remember:
        return

    cookies = get_cookies()
    if not cookies:
        return

    cookies.set("email", str(email or "").lower().strip(), max_age=COOKIE_MAX_AGE)
    cookies.set("name", str(name or ""), max_age=COOKIE_MAX_AGE)
    cookies.set("email_verified", str(bool(verified)), max_age=COOKIE_MAX_AGE)
    cookies.set("role", str(role or ""), max_age=COOKIE_MAX_AGE)

    if id_token:
        cookies.set("id_token", str(id_token), max_age=COOKIE_MAX_AGE)

    if refresh_token:
        cookies.set("refresh_token", str(refresh_token), max_age=COOKIE_MAX_AGE)


def _clear_cookie_session():
    cookies = get_cookies()
    if not cookies:
        return

    for key in COOKIE_KEYS:
        try:
            cookies.delete(key)
        except Exception:
            pass


def restore_session():
    init_session()

    if st.session_state.get("restored"):
        return

    cookies = get_cookies()

    if not cookies:
        return

    try:
        refresh_token = cookies.get("refresh_token")
        email_cookie = str(cookies.get("email") or "").lower().strip()
        name_cookie = cookies.get("name")
        role_cookie = cookies.get("role")
        verified_cookie = str(cookies.get("email_verified", "False")) == "True"

        if not refresh_token and not email_cookie:
            return

        if refresh_token and firebase_auth:
            try:
                refreshed = firebase_auth.refresh(refresh_token)

                new_id_token = refreshed.get("idToken")
                new_refresh_token = refreshed.get("refreshToken", refresh_token)

                info = firebase_auth.get_account_info(new_id_token)
                user_info = info["users"][0]

                email = str(user_info.get("email") or email_cookie).lower().strip()
                verified = bool(user_info.get("emailVerified", verified_cookie))
                name = name_cookie or email.split("@")[0].replace(".", " ").title()
                role = role_cookie or ("privileged" if verified else "restricted")

                _set_authenticated_session(
                    email=email,
                    name=name,
                    verified=verified,
                    role=role,
                    id_token=new_id_token,
                    refresh_token=new_refresh_token,
                )

                _save_cookie_session(
                    email=email,
                    name=name,
                    verified=verified,
                    role=role,
                    id_token=new_id_token,
                    refresh_token=new_refresh_token,
                    remember=True,
                )

            except Exception as e:
                if DEBUG:
                    st.warning(f"Token refresh failed: {e}")
                _clear_cookie_session()

        elif email_cookie:
            _set_authenticated_session(
                email=email_cookie,
                name=name_cookie,
                verified=verified_cookie,
                role=role_cookie,
                id_token=cookies.get("id_token"),
                refresh_token=refresh_token,
            )

    except Exception as e:
        if DEBUG:
            st.warning(f"Session restore failed: {e}")

    st.session_state.restored = True


def is_authenticated():
    init_session()
    restore_session()

    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
    )


def is_privileged():
    init_session()
    restore_session()

    return (
        st.session_state.get("user", False)
        and st.session_state.get("email_verified", False)
        and st.session_state.get("role") == "privileged"
    )


def logout():
    _clear_cookie_session()

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
        "id_token",
        "refresh_token",
        "cookie_manager",
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
        "TOKEN_EXPIRED": "Your login session expired. Please sign in again.",
        "INVALID_REFRESH_TOKEN": "Your saved login session is no longer valid. Please sign in again.",
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

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 30px !important;
            border: 1px solid rgba(102, 0, 148, 0.12) !important;
            box-shadow:
                0 26px 80px rgba(35, 25, 66, 0.16),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
            background: rgba(255, 255, 255, 0.94) !important;
            backdrop-filter: blur(18px) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 36px 40px 32px 40px !important;
            box-sizing: border-box;
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
            padding-bottom: 0;
        }

        .mode-active {
            position: relative;
            font-family: Arial, sans-serif;
            font-size: 15px;
            font-weight: 900;
            color: #231942;
            background: transparent;
            border: none;
            padding: 0 2px 13px 2px;
            letter-spacing: -0.1px;
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
            margin-bottom: 4px !important;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 14px !important;
            min-height: 46px !important;
            font-size: 13px !important;
            border: 1px solid #e5d9eb !important;
            background: #fcfbfd !important;
            box-shadow: inset 0 1px 2px rgba(35,25,66,0.03) !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #660094 !important;
            box-shadow: 0 0 0 3px rgba(102,0,148,0.10) !important;
        }

        div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
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

        button:hover {
            border-color: #660094 !important;
            box-shadow: 0 8px 20px rgba(35,25,66,0.10) !important;
        }

        .small-footer {
            text-align: center;
            color: #91869b;
            font-size: 10.8px;
            font-family: Arial, sans-serif;
            margin-top: 18px;
        }

        @media (max-width: 900px) {
            .block-container {
                max-width: 100% !important;
                padding: 1rem !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: 24px !important;
            }

            .auth-pill {
                justify-content: center;
                width: 100%;
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
            value=st.session_state.get("auth_remember", True),
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
            name = email.split("@")[0].replace(".", " ").title()

            _set_authenticated_session(
                email=email,
                name=name,
                verified=verified,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
            )

            st.session_state.auth_remember = remember
            st.session_state.auth_view = False
            st.session_state.restored = True

            _save_cookie_session(
                email=email,
                name=name,
                verified=verified,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
                remember=remember,
            )

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
                if st.button(
                    "← Dashboard",
                    use_container_width=True,
                    key="premium_back_dashboard",
                ):
                    _back_to_dashboard()

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