# auth.py
from __future__ import annotations

import json
import hashlib
import re
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
    import extra_streamlit_components as stx
    HAS_COOKIE_MANAGER = True
except ImportError:
    stx = None
    HAS_COOKIE_MANAGER = False


COOKIE_NAME = "eusee_auth_session"
COOKIE_DAYS = 30
DEBUG = False
COOKIE_RESTORE_MAX_ATTEMPTS = 8


# -----------------------------------------------------------------------------
# Per-user chatbot history persistence
# -----------------------------------------------------------------------------
# Streamlit session_state is lost on browser refresh/new session. These helpers
# persist chatbot messages on the server, separated by authenticated user email.
# Use append_user_chat_message() whenever the chatbot adds a message, or call
# save_user_chat_history() after updating st.session_state[CHAT_HISTORY_KEY].

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



def get_cookie_manager():
    """
    Return one CookieManager per Streamlit browser session.

    Do NOT cache this object globally with @st.cache_resource or a module-level
    singleton. Streamlit server globals are shared by all connected users, and a
    globally cached CookieManager can cause one browser session to reuse another
    user's authentication cookie/state.
    """
    if not HAS_COOKIE_MANAGER:
        return None

    if "_eusee_cookie_manager" not in st.session_state:
        st.session_state["_eusee_cookie_manager"] = stx.CookieManager(
            key="eusee_cookie_manager_main"
        )

    return st.session_state["_eusee_cookie_manager"]


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

def _secret_list(section: str, key: str) -> list:
    """Safely read a list-like value from Streamlit secrets."""
    value = st.secrets.get(section, {}).get(key, [])

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    try:
        return list(value)
    except Exception:
        return []


ADMIN_EMAILS = set(
    str(e).lower().strip()
    for e in (
        _secret_list("auth", "admin_emails")
        + _secret_list("access", "admin_emails")
    )
    if str(e).strip()
)


def get_domain(email: str) -> str:
    return str(email or "").strip().split("@")[-1].lower()


def _email_allowed_to_authenticate(email: str) -> bool:
    """Allow configured admin emails and, when domain rules exist, approved domains.

    If PRIVILEGED_DOMAINS is empty, Firebase-authenticated users are allowed and
    their role falls back to viewer unless they are an admin.
    """
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return False

    if clean_email in ADMIN_EMAILS:
        return True

    if not PRIVILEGED_DOMAINS:
        return True

    return get_domain(clean_email) in PRIVILEGED_DOMAINS


def _role_for_email(email: str) -> str:
    """Resolve role from the verified Firebase email only.

    Never derive role from browser cookies or previously stored Streamlit state.
    Admin is email-based. Privileged is domain-based. Everyone else is viewer.
    """
    clean_email = str(email or "").strip().lower()
    domain = get_domain(clean_email)

    if clean_email in ADMIN_EMAILS:
        return "admin"

    if domain in PRIVILEGED_DOMAINS:
        return "privileged"

    return "viewer"


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
    content = str(message.get("content") or message.get("message") or message.get("text") or "").strip()

    if role not in {"user", "assistant", "system"} or not content:
        return None

    item = {
        "role": role,
        "content": content,
    }

    if message.get("timestamp"):
        item["timestamp"] = str(message.get("timestamp"))
    else:
        item["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return item


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

    # Keep common existing chatbot keys synchronized so the rest of app.py can
    # continue using its current variable name without breaking.
    for key in CHAT_HISTORY_ALIASES:
        if key in st.session_state:
            st.session_state[key] = cleaned


def load_user_chat_history(email: str | None = None) -> list[dict]:
    """Load saved chatbot history for the authenticated user into session_state."""
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


def save_user_chat_history(messages: list[dict] | None = None, email: str | None = None) -> bool:
    """Save chatbot history for the authenticated user."""
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
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _sync_chat_history_aliases(cleaned)
        return True

    except Exception as e:
        if DEBUG:
            st.warning(f"Could not save chatbot history: {e}")
        return False


def append_user_chat_message(role: str, content: str) -> list[dict]:
    """Append one chatbot message and persist it immediately."""
    current = _get_current_session_chat_history()
    item = _normalise_chat_message({"role": role, "content": content})

    if item:
        current.append(item)

    current = current[-CHAT_HISTORY_MAX_MESSAGES:]
    _sync_chat_history_aliases(current)
    save_user_chat_history(current)
    return current


def clear_user_chat_history(email: str | None = None):
    """Clear the current user's saved chatbot history."""
    _sync_chat_history_aliases([])

    path = _chat_history_path(email)
    if path and path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def ensure_user_chat_history_loaded():
    """Call this once after authentication to restore the user's chatbot memory."""
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
        "cookie_restore_attempts": 0,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _session_payload(email, name, verified, role, id_token, refresh_token):
    return {
        "email": str(email or "").lower().strip(),
        "name": str(name or ""),
        "email_verified": bool(verified),
        "role": str(role or "guest"),
        "id_token": str(id_token or ""),
        "refresh_token": str(refresh_token or ""),
    }


def _write_cookie(payload: dict):
    manager = get_cookie_manager()
    if manager is None:
        st.error("❌ Add `extra-streamlit-components` to requirements.txt.")
        return

    manager.set(
        COOKIE_NAME,
        json.dumps(payload),
        expires_at=datetime.now() + timedelta(days=COOKIE_DAYS),
    )
    st.session_state.cookie_restore_attempts = 0


def _read_cookie() -> dict:
    manager = get_cookie_manager()
    if manager is None:
        return {}

    raw = manager.get(COOKIE_NAME)
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        return {}


def _delete_cookie():
    manager = get_cookie_manager()
    if manager is not None:
        try:
            manager.delete(COOKIE_NAME)
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


def _clear_auth_state_only():
    """Clear only authentication/user-scoped keys before applying another user."""
    for key in [
        "user",
        "email",
        "name",
        "role",
        "email_verified",
        "id_token",
        "refresh_token",
        CHAT_HISTORY_KEY,
        "chat_history_loaded",
        "chat_history_loaded_for",
        "cookie_restore_attempts",
        *CHAT_HISTORY_ALIASES,
    ]:
        st.session_state.pop(key, None)


def _apply_authenticated_state(email, name, verified, role, id_token, refresh_token):
    clean_email = str(email or "").lower().strip()

    # If a different account signs in in the same browser tab, remove all
    # previous user-scoped state first so names, roles, permissions and chatbot
    # memory cannot bleed into the new account.
    previous_email = str(st.session_state.get("email") or "").lower().strip()
    if previous_email and previous_email != clean_email:
        _clear_auth_state_only()

    st.session_state.user = bool(clean_email and verified)
    st.session_state.email = clean_email
    st.session_state.name = name or clean_email.split("@")[0].replace(".", " ").title()
    st.session_state.email_verified = bool(verified)
    st.session_state.role = role or _role_for_email(clean_email)
    st.session_state.id_token = id_token
    st.session_state.refresh_token = refresh_token
    st.session_state.auth_view = False
    st.session_state.restored = True
    st.session_state.cookie_restore_attempts = 0
    ensure_user_chat_history_loaded()


def restore_session():
    init_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        st.session_state.restored = True
        return True

    cookie_data = _read_cookie()
    if not cookie_data:
        # On a hard browser refresh, extra_streamlit_components.CookieManager
        # may need one or two Streamlit reruns before browser cookies are
        # available to Python. Do not finalize the user as logged out on the
        # first empty read; otherwise the login page appears even though the
        # Firebase refresh token cookie still exists.
        attempts = int(st.session_state.get("cookie_restore_attempts", 0))
        if attempts < COOKIE_RESTORE_MAX_ATTEMPTS:
            st.session_state.cookie_restore_attempts = attempts + 1
            st.session_state.restored = False
            st.rerun()

        st.session_state.restored = True
        return False

    refresh_token = cookie_data.get("refresh_token") or ""
    if not refresh_token:
        _delete_cookie()
        st.session_state.restored = True
        return False

    refreshed = refresh_firebase_token(refresh_token)
    if not refreshed:
        st.session_state.restored = False
        st.stop()

    id_token = refreshed.get("id_token")
    new_refresh_token = refreshed.get("refresh_token", refresh_token)

    # Critical: never trust email/name/role from the browser cookie. The cookie is
    # client-side and can be stale or tampered with. After refreshing the token,
    # ask Firebase which user this token belongs to, then apply that user only.
    try:
        info = firebase_auth.get_account_info(id_token)
        firebase_user = info["users"][0]
        email = str(firebase_user.get("email") or "").lower().strip()
        verified = bool(firebase_user.get("emailVerified", False))
        name = firebase_user.get("displayName") or email.split("@")[0].replace(".", " ").title()
    except Exception:
        _delete_cookie()
        _clear_auth_state_only()
        st.session_state.restored = True
        return False

    if not email or not verified:
        _delete_cookie()
        _clear_auth_state_only()
        st.session_state.restored = True
        return False

    if not _email_allowed_to_authenticate(email):
        _delete_cookie()
        _clear_auth_state_only()
        st.session_state.restored = True
        return False

    role = _role_for_email(email)

    _apply_authenticated_state(
        email=email,
        name=name,
        verified=True,
        role=role,
        id_token=id_token,
        refresh_token=new_refresh_token,
    )

    _write_cookie(
        _session_payload(
            email=email,
            name=st.session_state.name,
            verified=True,
            role=role,
            id_token=id_token,
            refresh_token=new_refresh_token,
        )
    )

    return True


def is_authenticated():
    init_session()

    if not st.session_state.get("restored"):
        restore_session()

    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
    )


def is_privileged():
    init_session()

    if not st.session_state.get("restored"):
        restore_session()

    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
        and st.session_state.get("role") in ["privileged", "admin"]
    )
def logout():
    save_user_chat_history()
    _delete_cookie()

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
        "cookie_restore_attempts",
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

        if not _email_allowed_to_authenticate(email):
            st.error("Access is restricted to approved EUSEE partner accounts.")
            return

        try:
            user = firebase_auth.sign_in_with_email_and_password(email, password)
            info = firebase_auth.get_account_info(user["idToken"])

            verified = bool(info["users"][0].get("emailVerified", False))

            if not verified:
                st.warning("Your account exists, but the email is not verified. Please verify your email first.")
                return

            role = _role_for_email(email)
            name = email.split("@")[0].replace(".", " ").title()

            _apply_authenticated_state(
                email=email,
                name=name,
                verified=True,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
            )

            _write_cookie(
                _session_payload(
                    email=email,
                    name=name,
                    verified=True,
                    role=role,
                    id_token=user.get("idToken"),
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

        if not _email_allowed_to_authenticate(email):
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

        if not _email_allowed_to_authenticate(reset_email):
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
    if not st.session_state.get("restored"):
        restore_session()

    if st.session_state.get("user") and st.session_state.get("email_verified"):
        ensure_user_chat_history_loaded()
        st.session_state.auth_view = False
        return

    _render_premium_auth_page()