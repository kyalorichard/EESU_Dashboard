# authz.py
"""
EU SEE Dashboard authorization helpers.

Access model:
- Firebase Auth handles login identity in auth.py.
- [auth].admin_emails controls who can open the Admin page.
- [access].privileged_domains controls which logged-in domains become privileged users.
- Admin page runtime settings control what Viewer and Privileged users can see.

Recommended secrets.toml:
[auth]
admin_emails = ["admin@icarda.org"]

[access]
privileged_domains = ["icarda.org", "cgiar.org"]
"""

from __future__ import annotations

import streamlit as st


FEATURE_DEFAULTS = {
    "guest": {
        "view_dashboard": True,
        "view_overview": True,
        "view_negative_alerts": False,
        "view_maps": True,
        "view_data_table": False,
        "view_country_counts": False,
        "download_data": False,
        "use_ai_copilot": False,
        "view_admin_page": False,
    },
    "viewer": {
        "view_dashboard": True,
        "view_overview": True,
        "view_negative_alerts": False,
        "view_maps": True,
        "view_data_table": False,
        "view_country_counts": False,
        "download_data": False,
        "use_ai_copilot": False,
        "view_admin_page": False,
    },
    "privileged": {
        "view_dashboard": True,
        "view_overview": True,
        "view_negative_alerts": True,
        "view_maps": True,
        "view_data_table": True,
        "view_country_counts": True,
        "download_data": True,
        "use_ai_copilot": True,
        "view_admin_page": False,
    },
}


def get_current_email() -> str:
    email = (
        st.session_state.get("email")
        or st.session_state.get("user_email")
        or st.session_state.get("firebase_email")
        or ""
    )
    return str(email).lower().strip()


def get_email_domain(email: str | None = None) -> str:
    email = str(email or get_current_email()).lower().strip()
    return email.split("@")[-1] if "@" in email else ""


def get_admin_emails() -> list[str]:
    return [
        str(e).lower().strip()
        for e in st.secrets.get("auth", {}).get("admin_emails", [])
        if str(e).strip()
    ]


def get_privileged_domains() -> list[str]:
    # Primary source: [access].privileged_domains, matching auth.py.
    domains = st.secrets.get("access", {}).get("privileged_domains", [])
    # Optional alias if you prefer storing under [auth].allowed_domains.
    if not domains:
        domains = st.secrets.get("auth", {}).get("allowed_domains", [])
    return [str(d).lower().strip().lstrip("@") for d in domains if str(d).strip()]


def is_logged_in() -> bool:
    return bool(get_current_email() and st.session_state.get("user", False) and st.session_state.get("email_verified", False))


def is_admin() -> bool:
    email = get_current_email()
    return bool(is_logged_in() and email in get_admin_emails())


def is_domain_approved() -> bool:
    """True when the logged-in user's email domain is listed in privileged_domains."""
    if not is_logged_in():
        return False
    domains = get_privileged_domains()
    if not domains:
        # If no domain list is configured, do not automatically grant privileged rights.
        return False
    return get_email_domain() in domains


def is_privileged_user() -> bool:
    return bool(is_admin() or is_domain_approved())


def get_current_role() -> str:
    if is_admin():
        return "admin"
    if is_domain_approved():
        return "privileged"
    if is_logged_in():
        return "viewer"
    return "guest"


def get_role() -> str:
    return get_current_role()


def _runtime_features_for_role(role: str) -> dict:
    """Read Admin-page runtime visibility settings, falling back to defaults."""
    config = st.session_state.get("admin_runtime_config", {}) or {}
    if role in config and isinstance(config.get(role), dict):
        features = config[role].get("features", {}) or {}
        merged = FEATURE_DEFAULTS.get(role, {}).copy()
        merged.update({k: bool(v) for k, v in features.items()})
        return merged
    return FEATURE_DEFAULTS.get(role, {}).copy()


def has_permission(permission: str) -> bool:
    if is_admin():
        return True
    role = get_current_role()
    return bool(_runtime_features_for_role(role).get(permission, False))


def apply_data_scope(df):
    """Apply role-based data scope. Admins see all data."""
    if df is None or is_admin():
        return df

    try:
        scoped_df = df.copy()
        role = get_current_role()

        # Runtime Admin page config takes priority.
        runtime_cfg = st.session_state.get("admin_runtime_config", {}) or {}
        role_cfg = runtime_cfg.get(role, {}) if isinstance(runtime_cfg.get(role, {}), dict) else {}

        regions = role_cfg.get("regions", None)
        countries = role_cfg.get("countries", None)
        years = role_cfg.get("years", None)

        # Fall back to secrets if runtime values are absent.
        if regions is None or countries is None or years is None:
            scope = st.secrets.get("data_scope", {}).get(role, {})
            regions = scope.get("regions", []) if regions is None else regions
            countries = scope.get("countries", []) if countries is None else countries
            years = scope.get("years", []) if years is None else years

        if regions and "region" in scoped_df.columns:
            scoped_df = scoped_df[scoped_df["region"].isin(regions)]
        if countries and "alert-country" in scoped_df.columns:
            scoped_df = scoped_df[scoped_df["alert-country"].isin(countries)]
        if years and "year" in scoped_df.columns:
            scoped_df = scoped_df[scoped_df["year"].isin(years)]
        return scoped_df
    except Exception:
        return df


def access_summary() -> dict:
    return {
        "email": get_current_email(),
        "domain": get_email_domain(),
        "role": get_current_role(),
        "is_logged_in": is_logged_in(),
        "is_admin": is_admin(),
        "is_domain_approved": is_domain_approved(),
        "privileged_domains": get_privileged_domains(),
        "admin_email_count": len(get_admin_emails()),
    }
