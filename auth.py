# auth.py
import json
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript

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


LS_EMAIL = "eusee_email"
LS_NAME = "eusee_name"
LS_ROLE = "eusee_role"
LS_VERIFIED = "eusee_email_verified"
LS_ID_TOKEN = "eusee_id_token"
LS_REFRESH_TOKEN = "eusee_refresh_token"


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
    if not HAS_PYREBASE:
        st.error("❌ Add `pyrebase4` to requirements.txt.")
        return None, None

    cfg_raw = st.secrets.get("firebase", {})
    if not cfg_raw:
        st.error("❌ Firebase config missing in secrets.toml.")
        return None, None

    cfg = dict(cfg_raw)

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

    cfg.setdefault("databaseURL", "")

    try:
        firebase = pyrebase.initialize_app(cfg)
        return firebase, firebase.auth()
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
        "auth_view": False,
        "id_token": None,
        "refresh_token": None,
        "browser_session_data": {},
        "browser_session_read_done": False,
        "_auth_js_counter": 0,
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _js_escape(value):
    return json.dumps(str(value or ""))


def _next_js_key(prefix: str) -> str:
    st.session_state["_auth_js_counter"] = st.session_state.get("_auth_js_counter", 0) + 1
    return f"{prefix}_{st.session_state['_auth_js_counter']}"


def save_browser_session(email, name, verified, role, id_token, refresh_token):
    js = f"""
    localStorage.setItem("{LS_EMAIL}", {_js_escape(email)});
    localStorage.setItem("{LS_NAME}", {_js_escape(name)});
    localStorage.setItem("{LS_ROLE}", {_js_escape(role)});
    localStorage.setItem("{LS_VERIFIED}", {_js_escape(str(bool(verified)))});
    localStorage.setItem("{LS_ID_TOKEN}", {_js_escape(id_token)});
    localStorage.setItem("{LS_REFRESH_TOKEN}", {_js_escape(refresh_token)});
    "ok";
    """
    return st_javascript(js, key=_next_js_key("save_eusee_auth"))


def save_browser_session_and_reload(email, name, verified, role, id_token, refresh_token):
    html = f"""
    <script>
    localStorage.setItem("{LS_EMAIL}", {_js_escape(email)});
    localStorage.setItem("{LS_NAME}", {_js_escape(name)});
    localStorage.setItem("{LS_ROLE}", {_js_escape(role)});
    localStorage.setItem("{LS_VERIFIED}", {_js_escape(str(bool(verified)))});
    localStorage.setItem("{LS_ID_TOKEN}", {_js_escape(id_token)});
    localStorage.setItem("{LS_REFRESH_TOKEN}", {_js_escape(refresh_token)});

    setTimeout(function() {{
        window.parent.location.reload();
    }}, 300);
    </script>
    """
    components.html(html, height=0, width=0)


def clear_browser_session():
    html = f"""
    <script>
    localStorage.removeItem("{LS_EMAIL}");
    localStorage.removeItem("{LS_NAME}");
    localStorage.removeItem("{LS_ROLE}");
    localStorage.removeItem("{LS_VERIFIED}");
    localStorage.removeItem("{LS_ID_TOKEN}");
    localStorage.removeItem("{LS_REFRESH_TOKEN}");

    setTimeout(function() {{
        window.parent.location.reload();
    }}, 250);
    </script>
    """
    components.html(html, height=0, width=0)


def read_browser_session():
    if st.session_state.get("browser_session_read_done"):
        return st.session_state.get("browser_session_data", {})

    js = f"""
    JSON.stringify({{
        email: localStorage.getItem("{LS_EMAIL}") || "",
        name: localStorage.getItem("{LS_NAME}") || "",
        role: localStorage.getItem("{LS_ROLE}") || "",
        email_verified: localStorage.getItem("{LS_VERIFIED}") || "",
        id_token: localStorage.getItem("{LS_ID_TOKEN}") || "",
        refresh_token: localStorage.getItem("{LS_REFRESH_TOKEN}") || ""
    }});
    """

    raw = st_javascript(js, key=_next_js_key("read_eusee_auth"))

    if not raw or raw in [0, "0", None]:
        return {}

    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    st.session_state.browser_session_data = data
    st.session_state.browser_session_read_done = True

    return data


def refresh_firebase_token(refresh_token: str):
    api_key = st.secrets.get("firebase", {}).get("apiKey")

    if not api_key or not refresh_token:
        return None

    url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    try:
        response = requests.post(url, data=payload, timeout=15)

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


def restore_session():
    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.restored = True
        return

    data = read_browser_session()

    if not data:
        return

    email = str(data.get("email") or "").lower().strip()
    name = data.get("name")
    role = data.get("role")
    verified = str(data.get("email_verified") or "").lower() == "true"
    refresh_token = data.get("refresh_token")

    if not email or not verified or not refresh_token:
        return

    refreshed = refresh_firebase_token(refresh_token)

    if not refreshed:
        return

    id_token = refreshed.get("id_token")
    new_refresh_token = refreshed.get("refresh_token", refresh_token)

    st.session_state.user = True
    st.session_state.email = email
    st.session_state.name = name or email.split("@")[0].replace(".", " ").title()
    st.session_state.role = role or "privileged"
    st.session_state.email_verified = True
    st.session_state.auth_view = False
    st.session_state.id_token = id_token
    st.session_state.refresh_token = new_refresh_token
    st.session_state.restored = True

    save_browser_session(
        email=email,
        name=st.session_state.name,
        verified=True,
        role=st.session_state.role,
        id_token=id_token,
        refresh_token=new_refresh_token,
    )


def logout():
    clear_browser_session()

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
        "browser_session_data",
        "browser_session_read_done",
    ]:
        if key in st.session_state:
            del st.session_state[key]


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

    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
        and st.session_state.get("role") == "privileged"
    )


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
            role = "privileged" if verified else "restricted"
            name = email.split("@")[0].replace(".", " ").title()

            st.session_state.user = True
            st.session_state.email = email
            st.session_state.name = name
            st.session_state.email_verified = verified
            st.session_state.role = role
            st.session_state.auth_view = False
            st.session_state.restored = True
            st.session_state.id_token = user.get("idToken")
            st.session_state.refresh_token = user.get("refreshToken")

            st.session_state.browser_session_data = {
                "email": email,
                "name": name,
                "role": role,
                "email_verified": str(bool(verified)),
                "id_token": user.get("idToken") or "",
                "refresh_token": user.get("refreshToken") or "",
            }
            st.session_state.browser_session_read_done = True

            st.success("Signed in successfully. Redirecting to dashboard...")

            save_browser_session_and_reload(
                email=email,
                name=name,
                verified=verified,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
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