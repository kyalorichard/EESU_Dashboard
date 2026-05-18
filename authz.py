from __future__ import annotations

import json
from pathlib import Path
import streamlit as st


# ============================================================
# EU SEE Dashboard Access Control
# ============================================================
# Key fix:
# - Removed legacy privileges:
#   view_public_summary, view_country_counts, view_negative_relationship_intelligence.
# - Summary cards are controlled by view_coverage_monitored_countries.
# - The monitored countries number is controlled separately by view_monitored_countries_value.
# - Admin choices for CSV/XLSX downloads and AI Copilot are preserved for all roles.
# - Saved JSON configs are merged with defaults while stale permission keys are removed.
# - has_permission() always reads the repaired config.
# ============================================================


DEFAULT_ROLE_PERMISSIONS = {
    "guest": {
        "view_dashboard": True,
        "view_overview": True,
        "view_coverage_monitored_countries": False,
        "view_monitored_countries_value": False,
        "view_maps": True,
        "view_negative_alerts": True,
        "view_analytical_flow_panel": True,
        "view_data_table": True,
        "download_data": False,
        "use_ai_copilot": False,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "viewer": {
        "view_dashboard": True,
        "view_overview": True,
        "view_coverage_monitored_countries": False,
        "view_monitored_countries_value": False,
        "view_maps": True,
        "view_negative_alerts": True,
        "view_analytical_flow_panel": True,
        "view_data_table": True,
        "download_data": False,
        "use_ai_copilot": False,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "privileged": {
        "view_dashboard": True,
        "view_overview": True,
        "view_coverage_monitored_countries": True,
        "view_monitored_countries_value": True,
        "view_maps": True,
        "view_negative_alerts": True,
        "view_analytical_flow_panel": True,
        "view_data_table": True,
        "download_data": True,
        "use_ai_copilot": True,
        "view_user_manual": True,
        "view_admin_page": False,
    },
    "admin": {"ALL": True},
}



REMOVED_PERMISSIONS = {
    "view_public_summary",
    "view_country_counts",
    "view_negative_relationship_intelligence",
}


PUBLIC_FEATURES_ALWAYS_ALLOWED_FOR_GUEST = {
    "view_dashboard",
    "view_overview",
    "view_maps",
    "view_negative_alerts",
    "view_analytical_flow_panel",
    "view_data_table",
    "view_user_manual",
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


def _normalise_role_config(role_cfg: dict | None) -> dict:
    """Ensure each role config has the expected shape."""
    if not isinstance(role_cfg, dict):
        role_cfg = {}

    features = role_cfg.get("features", {})
    if not isinstance(features, dict):
        features = {}

    for removed_permission in REMOVED_PERMISSIONS:
        features.pop(removed_permission, None)

    return {
        "features": features,
        "regions": role_cfg.get("regions", []) if isinstance(role_cfg.get("regions", []), list) else [],
        "countries": role_cfg.get("countries", []) if isinstance(role_cfg.get("countries", []), list) else [],
        "years": role_cfg.get("years", []) if isinstance(role_cfg.get("years", []), list) else [],
    }


def _merge_with_defaults(loaded: dict | None) -> dict:
    """
    Keep saved admin choices while repairing stale/missing config keys.

    Important:
    - Existing saved True/False choices are preserved.
    - Newly introduced permission keys receive the default value.
    - Missing guest public features are restored as True.
    """
    defaults = default_access_config()

    if not isinstance(loaded, dict):
        return defaults

    repaired: dict = {}

    for role, default_role_cfg in defaults.items():
        current_role_cfg = _normalise_role_config(loaded.get(role, {}))

        repaired[role] = {
            "features": {},
            "regions": current_role_cfg["regions"],
            "countries": current_role_cfg["countries"],
            "years": current_role_cfg["years"],
        }

        # Merge default schema with existing admin choices.
        for feature, default_value in default_role_cfg["features"].items():
            if feature in current_role_cfg["features"]:
                repaired[role]["features"][feature] = bool(current_role_cfg["features"][feature])
            else:
                repaired[role]["features"][feature] = bool(default_value)

        # Preserve any extra future/custom permissions saved by admin UI,
        # but never preserve removed legacy permissions.
        for feature, value in current_role_cfg["features"].items():
            if feature in REMOVED_PERMISSIONS:
                continue
            if feature not in repaired[role]["features"]:
                repaired[role]["features"][feature] = bool(value)

    return repaired


@st.cache_data(show_spinner=False)
def load_access_config() -> dict:
    path = get_access_config_path()

    if not path.exists():
        return default_access_config()

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_access_config()

    repaired = _merge_with_defaults(loaded)

    # Self-heal stale JSON silently when schema changed.
    try:
        if repaired != loaded:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass

    return repaired


def save_access_config(config: dict) -> bool:
    path = get_access_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        repaired = _merge_with_defaults(config)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
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
        st.cache_data.clear()
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
    """
    Return permission status for the active role.

    Guest public dashboard sections are explicitly allowed to avoid stale JSON
    accidentally blocking public content after logout.
    """
    role = get_current_role()

    if role == "admin":
        return True

    config = load_access_config()
    role_features = config.get(role, {}).get("features", {})

    # Hard safety for public guest sections. Admin can still restrict private
    # actions such as download_data and use_ai_copilot.
    if role == "guest" and permission in PUBLIC_FEATURES_ALWAYS_ALLOWED_FOR_GUEST:
        return bool(role_features.get(permission, True))

    return bool(role_features.get(permission, False))


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


def render_access_debug_panel():
    """Optional helper for sidebar debugging."""
    with st.sidebar.expander("🔎 Access status", expanded=False):
        role = get_current_role()
        config = load_access_config()
        st.caption(f"Role: {role}")
        st.caption(f"Email: {get_current_email() or 'Guest / not signed in'}")
        st.caption(f"Config path: {get_access_config_path()}")
        st.caption(f"Visualization map: {has_permission('view_maps')}")
        st.caption(f"User manual: {has_permission('view_user_manual')}")
        st.caption(f"Data table: {has_permission('view_data_table')}")
        st.caption(f"Summary cards: {has_permission('view_coverage_monitored_countries')}")
        st.caption(f"Monitored countries value: {has_permission('view_monitored_countries_value')}")
        st.caption(f"CSV/XLSX downloads: {has_permission('download_data')}")
        st.caption(f"AI Copilot: {has_permission('use_ai_copilot')}")
        st.caption(f"Analytical flow panel: {has_permission('view_analytical_flow_panel')}")
        if role in config:
            st.caption(f"Guest/View features loaded: {len(config[role].get('features', {}))}")
