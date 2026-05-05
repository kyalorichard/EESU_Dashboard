# authz.py
"""
Persistent EU SEE Dashboard access control.

Access tiers:
- guest: not logged in; public baseline only
- viewer: logged in, but email domain is not in [access].privileged_domains
- privileged: logged in and email domain is approved
- admin: logged in and exact email is in [auth].admin_emails

IMPORTANT:
Admin-selected visibility must be persistent. This module reads/writes a JSON
config file instead of relying only on st.session_state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


DEFAULT_ROLE_PERMISSIONS = {
    "guest": {
        "view_public_summary": True,
        "view_dashboard": True,
        "view_overview": True,
        "view_maps": False,
        "view_country_counts": False,
        "view_negative_alerts": False,
        "view_data_table": False,
        "download_data": False,
        "use_ai_copilot": False,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "viewer": {
        "view_public_summary": True,
        "view_dashboard": True,
        "view_overview": True,
        "view_maps": True,
        "view_country_counts": False,
        "view_negative_alerts": False,
        "view_data_table": False,
        "download_data": False,
        "use_ai_copilot": False,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "privileged": {
        "view_public_summary": True,
        "view_dashboard": True,
        "view_overview": True,
        "view_maps": True,
        "view_country_counts": True,
        "view_negative_alerts": True,
        "view_data_table": True,
        "download_data": True,
        "use_ai_copilot": True,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "admin": {
        "ALL": True,
    },
}


def _base_dir() -> Path:
    # On Render/Docker, /exports is often the persistent volume.
    if Path("/exports").exists():
        return Path("/exports")
    return Path(__file__).resolve().parent


def get_access_config_path() -> Path:
    configured = st.secrets.get("access_control", {}).get("config_path", "")
    if configured:
        return Path(configured)
    return _base_dir() / "eusee_access_config.json"


def default_access_config() -> dict:
    return {
        role: {
            "features": {
                k: bool(v)
                for k, v in perms.items()
                if k != "ALL"
            },
            "regions": [],
            "countries": [],
            "years": [],
        }
        for role, perms in DEFAULT_ROLE_PERMISSIONS.items()
        if role != "admin"
    }


def load_access_config() -> dict:
    path = get_access_config_path()
    default_cfg = default_access_config()

    if not path.exists():
        return default_cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return default_cfg

    # Merge loaded config with defaults so new keys never break.
    for role, role_cfg in default_cfg.items():
        loaded.setdefault(role, role_cfg)
        loaded[role].setdefault("features", {})
        loaded[role].setdefault("regions", [])
        loaded[role].setdefault("countries", [])
        loaded[role].setdefault("years", [])
        for feature, value in role_cfg["features"].items():
            loaded[role]["features"].setdefault(feature, value)

    return loaded


def save_access_config(config: dict) -> bool:
    path = get_access_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        tmp.replace(path)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Could not save access config to {path}: {e}")
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
        str(d).lower().strip()
        for d in st.secrets.get("access", {}).get("privileged_domains", [])
        if str(d).strip()
    ]


def get_email_domain(email: str | None = None) -> str:
    email = str(email or get_current_email()).lower().strip()
    return email.split("@")[-1] if "@" in email else ""


def is_authenticated_session() -> bool:
    return bool(st.session_state.get("user") and st.session_state.get("email_verified") and get_current_email())


def is_admin() -> bool:
    email = get_current_email()
    return bool(is_authenticated_session() and email in get_admin_emails())


def is_privileged_domain() -> bool:
    domain = get_email_domain()
    return bool(is_authenticated_session() and domain in get_privileged_domains())


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

    config = load_access_config()
    role_features = config.get(role, {}).get("features", {})
    return bool(role_features.get(permission, False))


def apply_data_scope(df):
    if df is None:
        return df

    role = get_current_role()
    if role == "admin":
        return df

    config = load_access_config()
    scope = config.get(role, {})
    scoped_df = df.copy()

    regions = scope.get("regions", [])
    countries = scope.get("countries", [])
    years = scope.get("years", [])

    if regions and "region" in scoped_df.columns:
        scoped_df = scoped_df[scoped_df["region"].isin(regions)]

    if countries and "alert-country" in scoped_df.columns:
        scoped_df = scoped_df[scoped_df["alert-country"].isin(countries)]

    if years and "year" in scoped_df.columns:
        scoped_df = scoped_df[scoped_df["year"].isin(years)]

    return scoped_df
