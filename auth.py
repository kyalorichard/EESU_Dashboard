# auth.py
from __future__ import annotations

import json
import hashlib
import re
import hmac
import base64
from pathlib import Path
from datetime import datetime, timedelta

import requests
import streamlit as st

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
    HAS_COOKIE_MANAGER = True
except ImportError:
    EncryptedCookieManager = None
    HAS_COOKIE_MANAGER = False


AUTH_COOKIE_NAME = "eusee_firebase_session_v4"
AUTH_COOKIE_DAYS = 30
DEBUG = False

CHAT_HISTORY_KEY = "eusee_chat_history"
CHAT_HISTORY_ALIASES = [
    "messages",
    "chat_messages",
    "chat_history",
    "eusee_messages",
    "eusee_chat_messages",
    "copilot_messages",
    "ai_copilot_messages",
]
CHAT_HISTORY_MAX_MESSAGES = 80
CHAT_HISTORY_DIR = Path(
    st.secrets.get("chatbot", {}).get("history_dir", ".eusee_chat_history")
)


def _cookie_password() -> str:
    return (
        st.secrets.get("auth", {}).get("cookie_password")
        or st.secrets.get("firebase", {}).get("apiKey")
        or "CHANGE_ME_EUSEE_COOKIE_PASSWORD"
    )


def get_cookie_manager():
    if not HAS_COOKIE_MANAGER:
        return None

    if "_eusee_cookie_manager" not in st.session_state:
        st.session_state["_eusee_cookie_manager"] = EncryptedCookieManager(
            prefix="eusee_auth_cookie/",
            password=_cookie_password(),
        )

    return st.session_state["_eusee_cookie_manager"]


def _cookies_ready():
    manager = get_cookie_manager()

    if manager is None:
        st.error("Persistent login requires streamlit-cookies-manager.")
        st.stop()

    if not manager.ready():
        st.stop()

    return manager

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


def _safe_user_key(email: str) -> str:
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return "guest"

    readable = re.sub(r"[^a-z0-9]+", "_", clean_email).strip("_")[:42]
    digest = hashlib.sha256(clean_email.encode("utf-8")).hexdigest()[:16]
    return f"{readable}_{digest}"


def _chat_history_path(email: str | None = None) -> Path | None:
    target_email = str(email or st.session_state.get("email") or "").strip().lower()
    if not target_email:
        return None

    return CHAT_HISTORY_DIR / f"{_safe_user_key(target_email)}.json"


def _normalise_chat_message(message: dict) -> dict | None:
    if not isinstance(message, dict):
        return None

    role = str(message.get("role") or message.get("sender") or "").strip().lower()
    content = str(
        message.get("content")
        or message.get("message")
        or message.get("text")
        or ""
    ).strip()

    if role not in {"user", "assistant", "system"} or not content:
        return None

    return {
        "role": role,
        "content": content,
        "timestamp": str(
            message.get("timestamp")
            or datetime.utcnow().isoformat(timespec="seconds") + "Z"
        ),
    }


def _normalise_chat_history(messages) -> list[dict]:
    if not isinstance(messages, list):
        return []

    cleaned = []
    for msg in messages:
        item = _normalise_chat_message(msg)
        if item:
            cleaned.append(item)

    return cleaned[-CHAT_HISTORY_MAX_MESSAGES:]


def _get_current_session_chat_history() -> list[dict]:
    for key in [CHAT_HISTORY_KEY, *CHAT_HISTORY_ALIASES]:
        value = st.session_state.get(key)
        if isinstance(value, list) and value:
            return _normalise_chat_history(value)

    return []


def _sync_chat_history_aliases(messages: list[dict]):
    cleaned = _normalise_chat_history(messages)
    st.session_state[CHAT_HISTORY_KEY] = cleaned

    for key in CHAT_HISTORY_ALIASES:
        if key in st.session_state:
            st.session_state[key] = cleaned


def load_user_chat_history(email: str | None = None) -> list[dict]:
    path = _chat_history_path(email)

    if path is None:
        _sync_chat_history_aliases([])
        return []

    try:
        if not path.exists():
            _sync_chat_history_aliases([])
            return []

        data = json.loads(path.read_text(encoding="utf-8"))
        messages = _normalise_chat_history(data.get("messages", []))
        _sync_chat_history_aliases(messages)
        st.session_state.chat_history_loaded = True
        return messages

    except Exception as e:
        if DEBUG:
            st.warning(f"Could not load chatbot history: {e}")
        _sync_chat_history_aliases([])
        return []


def save_user_chat_history(
    messages: list[dict] | None = None,
    email: str | None = None,
) -> bool:
    target_email = str(email or st.session_state.get("email") or "").strip().lower()
    if not target_email:
        return False

    cleaned = _normalise_chat_history(
        messages if messages is not None else _get_current_session_chat_history()
    )

    try:
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        path = _chat_history_path(target_email)
        if path is None:
            return False

        payload = {
            "email": target_email,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "messages": cleaned,
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _sync_chat_history_aliases(cleaned)
        return True

    except Exception as e:
        if DEBUG:
            st.warning(f"Could not save chatbot history: {e}")
        return False


def append_user_chat_message(role: str, content: str) -> list[dict]:
    current = _get_current_session_chat_history()
    item = _normalise_chat_message({"role": role, "content": content})

    if item:
        current.append(item)

    current = current[-CHAT_HISTORY_MAX_MESSAGES:]
    _sync_chat_history_aliases(current)
    save_user_chat_history(current)
    return current


def clear_user_chat_history(email: str | None = None):
    _sync_chat_history_aliases([])

    path = _chat_history_path(email)
    if path and path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def ensure_user_chat_history_loaded():
    if not st.session_state.get("user") or not st.session_state.get("email_verified"):
        return []

    loaded_for = st.session_state.get("chat_history_loaded_for")
    current_email = str(st.session_state.get("email") or "").strip().lower()

    if loaded_for != current_email:
        messages = load_user_chat_history(current_email)
        st.session_state.chat_history_loaded_for = current_email
        return messages

    return st.session_state.get(CHAT_HISTORY_KEY, [])


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
        CHAT_HISTORY_KEY: [],
        "chat_history_loaded": False,
        "chat_history_loaded_for": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _sign_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(
        _cookie_password().encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    token = json.dumps({"payload": payload, "sig": sig}, separators=(",", ":"))
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("utf-8")


def _unsign_payload(token: str) -> dict:
    try:
        decoded = base64.urlsafe_b64decode(str(token).encode("utf-8")).decode("utf-8")
        wrapped = json.loads(decoded)

        payload = wrapped.get("payload", {})
        sig = wrapped.get("sig", "")

        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        expected = hmac.new(
            _cookie_password().encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, expected):
            return {}

        return payload if isinstance(payload, dict) else {}

    except Exception:
        return {}


def _session_payload(email, name, verified, role, refresh_token):
    return {
        "email": str(email or "").lower().strip(),
        "name": str(name or ""),
        "email_verified": bool(verified),
        "role": str(role or "guest"),
        "refresh_token": str(refresh_token or ""),
        "expires_at": (
            datetime.utcnow() + timedelta(days=AUTH_COOKIE_DAYS)
        ).isoformat(timespec="seconds"),
    }


def _write_persistent_auth(payload: dict):
    manager = _cookies_ready()
    if manager is None:
        return

    manager[AUTH_COOKIE_NAME] = _sign_payload(payload)
    manager.save()


def _read_persistent_auth() -> dict:
    manager = _cookies_ready()
    if manager is None:
        return {}

    token = manager.get(AUTH_COOKIE_NAME)
    if not token:
        return {}

    payload = _unsign_payload(token)
    if not payload:
        _delete_persistent_auth()
        return {}

    expires_at = payload.get("expires_at")
    if expires_at:
        try:
            if datetime.fromisoformat(expires_at) < datetime.utcnow():
                _delete_persistent_auth()
                return {}
        except Exception:
            _delete_persistent_auth()
            return {}

    return payload
def _delete_persistent_auth():
    manager = get_cookie_manager()
    if manager is None:
        return

    try:
        if manager.ready():
            manager[AUTH_COOKIE_NAME] = ""
            del manager[AUTH_COOKIE_NAME]
            manager.save()
    except Exception:
        pass


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


def _verify_firebase_email(id_token: str, expected_email: str) -> bool:
    if not firebase_auth or not id_token or not expected_email:
        return False

    try:
        info = firebase_auth.get_account_info(id_token)
        firebase_email = (
            info.get("users", [{}])[0]
            .get("email", "")
            .lower()
            .strip()
        )

        firebase_verified = bool(
            info.get("users", [{}])[0].get("emailVerified", False)
        )

        return (
            firebase_email == str(expected_email or "").lower().strip()
            and firebase_verified
        )

    except Exception:
        return False


def _apply_authenticated_state(email, name, verified, role, id_token, refresh_token):
    email = str(email or "").lower().strip()

    st.session_state.user = bool(email and verified)
    st.session_state.email = email
    st.session_state.name = name or email.split("@")[0].replace(".", " ").title()
    st.session_state.email_verified = bool(verified)
    st.session_state.role = role or "privileged"
    st.session_state.id_token = id_token
    st.session_state.refresh_token = refresh_token
    st.session_state.auth_view = False
    st.session_state.restored = True

    ensure_user_chat_history_loaded()


def restore_session():
    init_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.restored = True
        return True

    cookie_data = _read_persistent_auth()
    if not cookie_data:
        st.session_state.restored = True
        return False

    email = str(cookie_data.get("email") or "").lower().strip()
    name = cookie_data.get("name") or ""
    role = cookie_data.get("role") or "privileged"
    verified = bool(cookie_data.get("email_verified"))
    refresh_token = cookie_data.get("refresh_token") or ""

    if not email or not verified or not refresh_token:
        _delete_persistent_auth()
        st.session_state.restored = True
        return False

    refreshed = refresh_firebase_token(refresh_token)
    if not refreshed:
        _delete_persistent_auth()
        st.session_state.restored = True
        return False

    id_token = refreshed.get("id_token")
    new_refresh_token = refreshed.get("refresh_token", refresh_token)

    if not _verify_firebase_email(id_token, email):
        _delete_persistent_auth()
        st.session_state.restored = True
        return False

    _apply_authenticated_state(
        email=email,
        name=name,
        verified=True,
        role=role,
        id_token=id_token,
        refresh_token=new_refresh_token,
    )

    _write_persistent_auth(
        _session_payload(
            email=email,
            name=st.session_state.name,
            verified=True,
            role=role,
            refresh_token=new_refresh_token,
        )
    )

    return True

def is_authenticated():
    init_session()

    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
    )


def is_privileged():
    init_session()

    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
        and st.session_state.get("role") in ["privileged", "admin"]
    )

def logout():
    save_user_chat_history()
    _delete_persistent_auth()

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
        CHAT_HISTORY_KEY,
        "chat_history_loaded",
        "chat_history_loaded_for",
        *CHAT_HISTORY_ALIASES,
    ]:
        st.session_state.pop(key, None)

    init_session()
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
                st.warning(
                    "Your account exists, but the email is not verified. "
                    "Please verify your email first."
                )
                return

            role = "privileged"
            name = email.split("@")[0].replace(".", " ").title()

            _apply_authenticated_state(
                email=email,
                name=name,
                verified=True,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
            )

            _write_persistent_auth(
                _session_payload(
                    email=email,
                    name=name,
                    verified=True,
                    role=role,
                    refresh_token=user.get("refreshToken"),
                )
            )

            st.session_state.auth_view = False
            st.success("Signed in successfully.")
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
            st.success(
                "Registration successful. Check your email to verify your account, then sign in."
            )
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
        ensure_user_chat_history_loaded()
        st.session_state.auth_view = False
        return

    _render_premium_auth_page()