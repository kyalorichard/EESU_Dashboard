import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Firebase/Auth remains responsible for login and identity.
# Admin elevation is controlled through .streamlit/secrets.toml.
# Non-admin visibility can be managed from the Admin Panel and stored in JSON.

DEFAULT_ROLE_PERMISSIONS = {
    "admin": "ALL",
    "analyst": [
        "view_dashboard",
        "view_overview",
        "view_negative_alerts",
        "view_map",
        "view_manual",
        "view_country_counts",
        "use_ai_copilot",
        "download_data",
        "view_admin_summary",
    ],
    "viewer": [
        "view_dashboard",
        "view_overview",
        "view_map",
        "view_manual",
    ],
    "restricted": [
        "view_dashboard",
        "view_overview",
    ],
    "guest": [
        "view_dashboard",
        "view_overview",
    ],
}

ALL_PERMISSION_KEYS = [
    "view_dashboard",
    "view_overview",
    "view_negative_alerts",
    "view_map",
    "view_manual",
    "view_country_counts",
    "use_ai_copilot",
    "download_data",
    "view_admin_summary",
]

DEFAULT_FEATURE_FLAGS = {
    "dashboard_enabled": True,
    "overview_enabled": True,
    "negative_alerts_enabled": True,
    "map_enabled": True,
    "manual_enabled": True,
    "ai_copilot_enabled": True,
    "downloads_enabled": True,
    "country_counts_enabled": True,
}

CONFIG_FILE_NAME = "admin_access_config.json"


def _clean_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _get_secret_list(section: str, key: str) -> List[str]:
    try:
        value = st.secrets.get(section, {}).get(key, [])
    except Exception:
        value = []
    if isinstance(value, str):
        value = [value]
    return [_clean_email(v) for v in value if _clean_email(v)]


def get_current_email() -> str:
    """Return current Firebase-authenticated email from Streamlit session state."""
    for key in ["email", "user_email", "firebase_email", "auth_email"]:
        email = _clean_email(st.session_state.get(key, ""))
        if email:
            return email

    user = st.session_state.get("user") or st.session_state.get("firebase_user") or {}
    if isinstance(user, dict):
        for key in ["email", "user_email"]:
            email = _clean_email(user.get(key, ""))
            if email:
                return email
    return ""


def get_admin_emails() -> List[str]:
    return _get_secret_list("auth", "admin_emails")


def get_power_user_emails() -> List[str]:
    return _get_secret_list("auth", "power_users")


def get_power_users() -> List[str]:
    return get_power_user_emails()


def is_logged_in() -> bool:
    return bool(get_current_email())


def is_authenticated_user() -> bool:
    return is_logged_in()


def is_admin() -> bool:
    return get_current_email() in get_admin_emails()


def is_power_user() -> bool:
    return get_current_email() in get_power_user_emails()


def _config_path() -> Path:
    """Prefer /exports for Render/Docker persistence, otherwise project root."""
    export_dir = Path("/exports") if Path("/exports").exists() else Path(__file__).resolve().parent
    return export_dir / CONFIG_FILE_NAME


def default_access_config() -> Dict[str, Any]:
    return {
        "feature_flags": DEFAULT_FEATURE_FLAGS.copy(),
        "users": {},
        "audit_log": [],
    }


def load_access_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return default_access_config()
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return default_access_config()

    config.setdefault("feature_flags", DEFAULT_FEATURE_FLAGS.copy())
    config.setdefault("users", {})
    config.setdefault("audit_log", [])
    for key, value in DEFAULT_FEATURE_FLAGS.items():
        config["feature_flags"].setdefault(key, value)
    return config


def save_access_config(config: Dict[str, Any]) -> bool:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return True


def get_managed_user(email: str = None) -> Dict[str, Any]:
    email = _clean_email(email or get_current_email())
    if not email:
        return {}
    return load_access_config().get("users", {}).get(email, {})


def get_current_role() -> str:
    if is_admin():
        return "admin"
    email = get_current_email()
    managed = get_managed_user(email)
    if managed and managed.get("active", True) is False:
        return "guest"
    if managed.get("role"):
        return str(managed.get("role")).lower()
    if is_power_user():
        return "analyst"
    if is_logged_in():
        return "viewer"
    return "guest"


def get_role() -> str:
    return get_current_role()


def is_privileged() -> bool:
    return get_current_role() in ["admin", "analyst"]


def get_global_feature_flag(flag_name: str) -> bool:
    flags = load_access_config().get("feature_flags", {})
    return bool(flags.get(flag_name, DEFAULT_FEATURE_FLAGS.get(flag_name, True)))


def has_permission(permission_name: str) -> bool:
    if is_admin():
        return True

    feature_dependency = {
        "view_dashboard": "dashboard_enabled",
        "view_overview": "overview_enabled",
        "view_negative_alerts": "negative_alerts_enabled",
        "view_map": "map_enabled",
        "view_manual": "manual_enabled",
        "use_ai_copilot": "ai_copilot_enabled",
        "download_data": "downloads_enabled",
        "view_country_counts": "country_counts_enabled",
    }
    flag = feature_dependency.get(permission_name)
    if flag and not get_global_feature_flag(flag):
        return False

    email = get_current_email()
    managed = get_managed_user(email)
    if managed and managed.get("active", True) is False:
        return False

    if managed.get("permissions") is not None:
        return bool(managed.get("permissions", {}).get(permission_name, False))

    role = get_current_role()
    permissions = DEFAULT_ROLE_PERMISSIONS.get(role, [])
    if permissions == "ALL":
        return True
    return permission_name in permissions


def get_allowed_scope_for_current_user() -> Dict[str, Any]:
    if is_admin():
        return {}
    managed_scope = get_managed_user().get("data_scope", {})
    if managed_scope:
        return managed_scope
    email = get_current_email()
    if not email:
        return {}
    try:
        return dict(st.secrets.get("access_scope", {}).get(email, {}))
    except Exception:
        return {}


def apply_data_scope(df):
    """Filter dataframe by optional user-level scope. Admins are unrestricted."""
    if df is None or getattr(df, "empty", True) or is_admin():
        return df

    scope = get_allowed_scope_for_current_user()
    if not scope:
        return df

    scoped = df.copy()
    regions = scope.get("regions", []) or []
    countries = scope.get("countries", []) or []
    years = scope.get("years", []) or []

    if regions and "region" in scoped.columns:
        scoped = scoped[scoped["region"].isin(regions)]
    if countries and "alert-country" in scoped.columns:
        scoped = scoped[scoped["alert-country"].isin(countries)]
    if years and "year" in scoped.columns:
        scoped = scoped[scoped["year"].isin(years)]

    return scoped


def upsert_managed_user(email: str, role: str, active: bool, permissions: Dict[str, bool], data_scope: Dict[str, Any], actor_email: str = None) -> bool:
    email = _clean_email(email)
    if not email:
        raise ValueError("Email is required.")
    config = load_access_config()
    config.setdefault("users", {})
    config["users"][email] = {
        "email": email,
        "role": role,
        "active": bool(active),
        "permissions": permissions,
        "data_scope": data_scope,
    }
    config.setdefault("audit_log", []).append({
        "action": "upsert_user_visibility",
        "target": email,
        "actor": actor_email or get_current_email(),
    })
    save_access_config(config)
    return True


def delete_managed_user(email: str, actor_email: str = None) -> bool:
    email = _clean_email(email)
    config = load_access_config()
    if email in config.get("users", {}):
        del config["users"][email]
        config.setdefault("audit_log", []).append({
            "action": "delete_user_visibility",
            "target": email,
            "actor": actor_email or get_current_email(),
        })
        save_access_config(config)
    return True


def update_feature_flags(flags: Dict[str, bool], actor_email: str = None) -> bool:
    config = load_access_config()
    config["feature_flags"] = {**DEFAULT_FEATURE_FLAGS, **flags}
    config.setdefault("audit_log", []).append({
        "action": "update_feature_flags",
        "actor": actor_email or get_current_email(),
    })
    save_access_config(config)
    return True


def render_access_badge():
    role = get_current_role().title()
    email = get_current_email() or "Not signed in"
    st.sidebar.caption(f"Access: **{role}**")
    st.sidebar.caption(email)
