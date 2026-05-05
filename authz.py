from __future__ import annotations

import json
from pathlib import Path
import streamlit as st


DEFAULT_ROLE_PERMISSIONS = {
    "guest": {
        "view_public_summary": True,
        "view_dashboard": True,
        "view_overview": True,
        "view_coverage_monitored_countries": True,
        "view_country_counts": True,
        "view_maps": True,
        "view_negative_alerts": True,
        "view_negative_relationship_intelligence": True,
        "view_analytical_flow_panel": True,
        "view_data_table": True,
        "download_data": False,
        "use_ai_copilot": False,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "viewer": {
        "view_public_summary": True,
        "view_dashboard": True,
        "view_overview": True,
        "view_coverage_monitored_countries": True,
        "view_country_counts": True,
        "view_maps": True,
        "view_negative_alerts": True,
        "view_negative_relationship_intelligence": True,
        "view_analytical_flow_panel": True,
        "view_data_table": True,
        "download_data": False,
        "use_ai_copilot": False,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "privileged": {
        "view_public_summary": True,
        "view_dashboard": True,
        "view_overview": True,
        "view_coverage_monitored_countries": True,
        "view_country_counts": True,
        "view_maps": True,
        "view_negative_alerts": True,
        "view_negative_relationship_intelligence": True,
        "view_analytical_flow_panel": True,
        "view_data_table": True,
        "download_data": True,
        "use_ai_copilot": True,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "admin": {"ALL": True},
}


def _default_base_dir() -> Path:
    return Path("/exports") if Path("/exports").exists() else Path(__file__).resolve().parent


def get_access_config_path() -> Path:
    configured = st.secrets.get("access_control", {}).get("config_path", "")
    return Path(configured) if configured else _default_base_dir() / "eusee_access_config.json"


def default_access_config() -> dict:
    return {
        role: {
            "features": {k: bool(v) for k, v in perms.items() if k != "ALL"},
            "regions": [],
            "countries": [],
            "years": [],
        }
        for role, perms in DEFAULT_ROLE_PERMISSIONS.items()
        if role != "admin"
    }


def _merge_with_defaults(loaded: dict) -> dict:
    """Keep saved admin choices while adding any newly introduced permission keys."""
    defaults = default_access_config()
    if not isinstance(loaded, dict):
        return defaults

    for role, role_cfg in defaults.items():
        loaded.setdefault(role, {})
        loaded[role].setdefault("features", {})
        loaded[role].setdefault("regions", [])
        loaded[role].setdefault("countries", [])
        loaded[role].setdefault("years", [])

        for feature, default_value in role_cfg["features"].items():
            loaded[role]["features"].setdefault(feature, default_value)

    return loaded


def load_access_config() -> dict:
    path = get_access_config_path()
    if not path.exists():
        return default_access_config()

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_access_config()

    return _merge_with_defaults(loaded)


def save_access_config(config: dict) -> bool:
    path = get_access_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_merge_with_defaults(config), indent=2), encoding="utf-8")
        tmp.replace(path)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Could not save access config to {path}: {e}")
        return False


def reset_access_config() -> bool:
    """Delete stale JSON and rebuild it with the current permission schema."""
    path = get_access_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        return save_access_config(default_access_config())
    except Exception as e:
        st.error(f"Could not reset access config at {path}: {e}")
        return False


def get_current_email() -> str:
    return str(
        st.session_state.get("email")
        or st.session_state.get("user_email")
        or st.session_state.get("firebase_email")
        or ""
    ).lower().strip()


def get_admin_emails() -> list[str]:
    return [
        str(e).lower().strip()
        for e in st.secrets.get("auth", {}).get("admin_emails", [])
        if str(e).strip()
    ]


def get_privileged_domains() -> list[str]:
    return [
        str(d).lower().strip()
        for d in st.secrets.get("access", {}).get("privileged_domains", [])
        if str(d).strip()
    ]


def get_email_domain(email: str | None = None) -> str:
    email = str(email or get_current_email()).lower().strip()
    return email.split("@")[-1] if "@" in email else ""


def is_authenticated_session() -> bool:
    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
        and get_current_email()
    )


def is_admin() -> bool:
    return bool(is_authenticated_session() and get_current_email() in get_admin_emails())


def is_privileged_domain() -> bool:
    return bool(is_authenticated_session() and get_email_domain() in get_privileged_domains())


def get_current_role() -> str:
    if is_admin():
        return "admin"
    if is_privileged_domain():
        return "privileged"
    if is_authenticated_session():
        return "viewer"
    return "guest"


def has_permission(permission: str) -> bool:
    role = get_current_role()
    if role == "admin":
        return True
    return bool(load_access_config().get(role, {}).get("features", {}).get(permission, False))


def apply_data_scope(df):
    if df is None:
        return df

    role = get_current_role()
    if role == "admin":
        return df

    scope = load_access_config().get(role, {})
    scoped = df.copy()

    if scope.get("regions") and "region" in scoped.columns:
        scoped = scoped[scoped["region"].isin(scope["regions"])]

    if scope.get("countries") and "alert-country" in scoped.columns:
        scoped = scoped[scoped["alert-country"].isin(scope["countries"])]

    if scope.get("years") and "year" in scoped.columns:
        scoped = scoped[scoped["year"].isin(scope["years"])]

    return scoped
