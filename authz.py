# authz.py
"""
EU SEE Dashboard role and permission resolver.

Role model:
- guest: not logged in; public-only access
- viewer: logged in and verified, but not from approved privileged domain
- privileged: logged in and verified from approved domain
- admin: logged in admin email listed in [auth].admin_emails

Admin-selected visibility is stored in admin_visibility_config.json when edited from the Admin page.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

CONFIG_PATH = Path(__file__).resolve().parent / "admin_visibility_config.json"

DEFAULT_ROLE_PERMISSIONS: dict[str, dict[str, bool]] = {
    "guest": {
        "view_public_summary": True,
        "view_overview": False,
        "view_negative_alerts": False,
        "view_map": False,
        "view_manual": True,
        "view_data_table": False,
        "view_country_counts": False,
        "download_data": False,
        "use_ai_copilot": False,
        "view_admin_page": False,
    },
    "viewer": {
        "view_public_summary": True,
        "view_overview": True,
        "view_negative_alerts": False,
        "view_map": True,
        "view_manual": True,
        "view_data_table": False,
        "view_country_counts": False,
        "download_data": False,
        "use_ai_copilot": False,
        "view_admin_page": False,
    },
    "privileged": {
        "view_public_summary": True,
        "view_overview": True,
        "view_negative_alerts": True,
        "view_map": True,
        "view_manual": True,
        "view_data_table": True,
        "view_country_counts": True,
        "download_data": True,
        "use_ai_copilot": True,
        "view_admin_page": False,
    },
    "admin": {
        "view_public_summary": True,
        "view_overview": True,
        "view_negative_alerts": True,
        "view_map": True,
        "view_manual": True,
        "view_data_table": True,
        "view_country_counts": True,
        "download_data": True,
        "use_ai_copilot": True,
        "view_admin_page": True,
    },
}

DEFAULT_DATA_SCOPE: dict[str, dict[str, list[Any]]] = {
    "guest": {"regions": [], "countries": [], "years": []},
    "viewer": {"regions": [], "countries": [], "years": []},
    "privileged": {"regions": [], "countries": [], "years": []},
    "admin": {"regions": [], "countries": [], "years": []},
}


def _deepcopy_defaults() -> dict[str, Any]:
    return {
        "permissions": json.loads(json.dumps(DEFAULT_ROLE_PERMISSIONS)),
        "data_scope": json.loads(json.dumps(DEFAULT_DATA_SCOPE)),
    }


def load_visibility_config() -> dict[str, Any]:
    cfg = _deepcopy_defaults()
    try:
        if CONFIG_PATH.exists():
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for role, perms in saved.get("permissions", {}).items():
                if role in cfg["permissions"] and isinstance(perms, dict):
                    cfg["permissions"][role].update(perms)
            for role, scope in saved.get("data_scope", {}).items():
                if role in cfg["data_scope"] and isinstance(scope, dict):
                    cfg["data_scope"][role].update(scope)
    except Exception:
        pass
    return cfg


def save_visibility_config(config: dict[str, Any]) -> bool:
    try:
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Could not save admin visibility config: {e}")
        return False


def get_current_email() -> str:
    email = (
        st.session_state.get("email")
        or st.session_state.get("user_email")
        or st.session_state.get("firebase_email")
        or ""
    )
    return str(email).lower().strip()


def get_admin_emails() -> list[str]:
    return [
        str(e).lower().strip()
        for e in st.secrets.get("auth", {}).get("admin_emails", [])
        if str(e).strip()
    ]


def get_privileged_domains() -> list[str]:
    return [
        str(d).lower().strip().lstrip("@")
        for d in st.secrets.get("access", {}).get("privileged_domains", [])
        if str(d).strip()
    ]


def is_logged_in() -> bool:
    return bool(st.session_state.get("user") and st.session_state.get("email_verified") and get_current_email())


def is_admin() -> bool:
    return bool(is_logged_in() and get_current_email() in get_admin_emails())


def is_privileged_domain(email: str | None = None) -> bool:
    email = (email or get_current_email()).lower().strip()
    if not email or "@" not in email:
        return False
    domain = email.split("@")[-1]
    return domain in get_privileged_domains()


def get_current_role() -> str:
    if not is_logged_in():
        return "guest"
    if is_admin():
        return "admin"
    if is_privileged_domain():
        return "privileged"
    return "viewer"


def has_permission(permission_name: str) -> bool:
    role = get_current_role()
    if role == "admin":
        return True
    cfg = load_visibility_config()
    return bool(cfg.get("permissions", {}).get(role, {}).get(permission_name, False))


def apply_data_scope(df):
    if df is None:
        return df
    role = get_current_role()
    if role == "admin":
        return df

    cfg = load_visibility_config()
    scope = cfg.get("data_scope", {}).get(role, {})
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
