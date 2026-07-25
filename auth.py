# auth.py
from __future__ import annotations

import json
import hashlib
import re
import time
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
COOKIE_WRITE_WAIT_SECONDS = 1.20
TOKEN_REFRESH_ATTEMPTS = 3
DEBUG = False

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



# CookieManager is a browser component. The Python object does not contain a
# user's authentication cookie; each visitor's browser supplies its own value.
# Keep one component instance to avoid duplicate Streamlit component keys.
_COOKIE_MANAGER = None

def get_cookie_manager():
    global _COOKIE_MANAGER

    if not HAS_COOKIE_MANAGER:
        return None

    if _COOKIE_MANAGER is None:
        _COOKIE_MANAGER = stx.CookieManager(
            key="eusee_cookie_manager_main"
        )

    return _COOKIE_MANAGER


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
    str(d).lower().strip().lstrip("@")
    for d in st.secrets.get("access", {}).get("privileged_domains", [])
    if str(d).strip()
)


def get_temporary_shared_account() -> dict:
    """
    Temporary shared-account configuration.

    Add this to .streamlit/secrets.toml:

    [temporary_shared_account]
    enabled = true
    email = "dashboard.access@eusee.global"
    role = "privileged"
    bypass_email_verification = true
    """
    config = st.secrets.get("temporary_shared_account", {})

    role = str(config.get("role", "privileged")).strip().lower()
    if role not in {"privileged", "viewer"}:
        role = "privileged"

    return {
        "enabled": bool(config.get("enabled", False)),
        "email": str(config.get("email", "")).strip().lower(),
        "role": role,
        "bypass_email_verification": bool(
            config.get("bypass_email_verification", True)
        ),
    }


def is_temporary_shared_account(email: str | None) -> bool:
    config = get_temporary_shared_account()
    candidate = str(email or "").strip().lower()

    return bool(
        config["enabled"]
        and config["email"]
        and candidate == config["email"]
    )


def should_bypass_email_verification(email: str | None) -> bool:
    config = get_temporary_shared_account()
    return bool(
        is_temporary_shared_account(email)
        and config["bypass_email_verification"]
    )


def get_domain(email: str) -> str:
    return str(email or "").strip().split("@")[-1].lower()


def is_approved_login_email(email: str | None) -> bool:
    """
    Approve either:
    - an account from an approved partner domain; or
    - the explicitly configured temporary shared account.
    """
    normalized = str(email or "").strip().lower()
    if not normalized:
        return False

    if is_temporary_shared_account(normalized):
        return True

    return bool(
        PRIVILEGED_DOMAINS
        and get_domain(normalized) in PRIVILEGED_DOMAINS
    )


def get_login_role(email: str | None) -> str:
    if is_temporary_shared_account(email):
        return get_temporary_shared_account()["role"]

    return "privileged"


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


def _write_cookie(payload: dict) -> bool:
    """Write the browser cookie and allow the frontend component to commit it."""
    manager = get_cookie_manager()
    if manager is None:
        st.error("❌ Add `extra-streamlit-components` to requirements.txt.")
        return False

    try:
        manager.set(
            COOKIE_NAME,
            json.dumps(payload),
            expires_at=datetime.now() + timedelta(days=COOKIE_DAYS),
        )

        # CookieManager writes in the browser through a Streamlit component.
        # An immediate st.rerun() can cancel that frontend write. Give the
        # component enough time to commit the cookie before rerunning.
        time.sleep(COOKIE_WRITE_WAIT_SECONDS)
        return True
    except Exception as e:
        if DEBUG:
            st.warning(f"Cookie write failed: {e}")
        return False


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

    for attempt in range(TOKEN_REFRESH_ATTEMPTS):
        try:
            response = requests.post(
                url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                timeout=15,
            )

            if response.status_code == 200:
                return response.json()

            # Retry only server/rate-limit failures. A 400 response generally
            # means the refresh token is genuinely invalid or revoked.
            if response.status_code < 500 and response.status_code != 429:
                return None

        except requests.RequestException:
            pass

        if attempt + 1 < TOKEN_REFRESH_ATTEMPTS:
            time.sleep(0.5 * (attempt + 1))

    return None


def _apply_authenticated_state(email, name, verified, role, id_token, refresh_token):
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

    cookie_data = _read_cookie()
    if not cookie_data:
        # This is a normal logged-out browser. Allow the login page to render.
        # On a hard refresh CookieManager may trigger its own frontend rerun;
        # auth_ui() calls restore_session() again on that rerun.
        st.session_state.restored = True
        return False


    email = str(cookie_data.get("email") or "").lower().strip()
    name = cookie_data.get("name") or ""
    role = cookie_data.get("role") or "privileged"
    verified = bool(cookie_data.get("email_verified"))
    refresh_token = cookie_data.get("refresh_token") or ""

    if not email or not verified or not refresh_token:
        _delete_cookie()
        st.session_state.restored = True
        return False

    refreshed = refresh_firebase_token(refresh_token)
    if not refreshed:
        # Do not erase a valid long-lived cookie because of a temporary network
        # or Firebase endpoint failure. The next reload can try restoration again.
        st.session_state.restored = True
        return False

    id_token = refreshed.get("id_token")
    new_refresh_token = refreshed.get("refresh_token", refresh_token)

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
            background: rgba(255, 250, 240, 0.72);
            border-left: 3px solid #FFDB58;
            border-radius: 6px;
            padding: 10px 12px;
            color: #4b3b14;
            font-size: 11.8px;
            line-height: 1.5;
            font-family: Arial, sans-serif;
            margin-top: 18px;
        }

        .auth-topbar {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 18px;
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

        if not is_approved_login_email(email):
            st.error("Access is restricted to approved EUSEE partner accounts.")
            return

        try:
            user = firebase_auth.sign_in_with_email_and_password(email, password)
            info = firebase_auth.get_account_info(user["idToken"])

            firebase_verified = bool(
                info["users"][0].get("emailVerified", False)
            )
            verification_bypassed = should_bypass_email_verification(email)

            if not firebase_verified and not verification_bypassed:
                st.warning(
                    "Your account exists, but the email is not verified. "
                    "Please verify your email first."
                )
                return

            # Internally mark the session as verified when the explicitly
            # configured temporary shared account is allowed to bypass
            # Firebase email verification. Existing session and cookie logic
            # requires this value to be True.
            session_verified = firebase_verified or verification_bypassed
            role = get_login_role(email)
            name = email.split("@")[0].replace(".", " ").title()

            _apply_authenticated_state(
                email=email,
                name=name,
                verified=session_verified,
                role=role,
                id_token=user.get("idToken"),
                refresh_token=user.get("refreshToken"),
            )

            _write_cookie(
                _session_payload(
                    email=email,
                    name=name,
                    verified=session_verified,
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

        if not is_approved_login_email(reset_email):
            st.error(
                "Password reset is restricted to approved EUSEE partner accounts."
            )
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

    _, center, _ = st.columns([0.18, 0.64, 0.18])

    with center:
        # Keep the page visually open: no outer bordered container and no
        # duplicated "Authorized users only" badge.
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