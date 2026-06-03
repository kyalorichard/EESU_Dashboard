# authz.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


FEATURE_KEYS = [
    "view_dashboard",
    "view_overview",
    "view_coverage_monitored_countries",
    "view_monitored_countries_value",
    "view_maps",
    "view_negative_alerts",
    "view_analytical_flow_panel",
    "view_data_table",
    "download_data",
    "use_ai_copilot",
    "view_user_manual",
    "view_admin_page",
    "view_chart_overview_alert_type",
    "view_chart_overview_enabling_principles",
    "view_chart_overview_regions",
    "view_chart_overview_countries",
    "view_chart_negative_restrictive_actors",
    "view_chart_negative_affected_actors",
    "view_chart_negative_restrictive_mechanisms",
    "view_chart_negative_event_types",
    "view_chart_negative_alert_types",
    "view_chart_negative_enabling_principles",
    "view_chart_heatmap_actor_mechanism",
    "view_chart_heatmap_subject_mechanism",
    "view_chart_heatmap_actor_subject",
    "view_chart_sankey_flow",
    "view_chart_geospatial_map",
    "view_chart_ai_copilot_plots",
]

REMOVED_FEATURE_KEYS = {
    "view_public_summary",
    "view_country_counts",
    "view_negative_relationship_intelligence",
}


def _secrets_get(section: str, key: str, default: Any = None) -> Any:
    """Safely read nested Streamlit secrets without failing local runs."""
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def get_access_config_path() -> Path:
    configured = _secrets_get("access_control", "config_path", None)
    if configured:
        return Path(str(configured))
    if Path("/exports").exists():
        return Path("/exports/eusee_access_config.json")
    return Path(__file__).resolve().parent / "exports" / "eusee_access_config.json"


def get_admin_emails() -> list[str]:
    return [x.lower() for x in _as_list(_secrets_get("auth", "admin_emails", []))]


def get_privileged_domains() -> list[str]:
    return [x.lower().lstrip("@") for x in _as_list(_secrets_get("access", "privileged_domains", []))]


def get_current_email() -> str:
    user = st.session_state.get("user") or {}
    candidates = [
        st.session_state.get("email"),
        st.session_state.get("user_email"),
        user.get("email") if isinstance(user, dict) else None,
    ]
    for value in candidates:
        if value:
            return str(value).strip().lower()
    return ""


def is_admin() -> bool:
    email = get_current_email()
    return bool(email and email in get_admin_emails())


def get_current_role() -> str:
    """Resolve role from admin email, authenticated email domain, or public guest."""
    if is_admin():
        return "admin"

    email = get_current_email()
    if email:
        domain = email.split("@")[-1].lower() if "@" in email else ""
        if domain in get_privileged_domains():
            return "privileged"
        return "viewer"

    return "guest"


def default_access_config() -> dict[str, Any]:
    """Default role permissions. Admins always receive all permissions dynamically."""
    return {
        "guest": {
            "features": {
                "view_dashboard": True,
                "view_overview": True,
                "view_coverage_monitored_countries": True,
                "view_monitored_countries_value": False,
                "view_maps": True,
                "view_negative_alerts": False,
                "view_analytical_flow_panel": False,
                "view_data_table": False,
                "download_data": False,
                "use_ai_copilot": False,
                "view_user_manual": True,
                "view_admin_page": False,
                "view_chart_overview_alert_type": False,
                "view_chart_overview_enabling_principles": False,
                "view_chart_overview_regions": False,
                "view_chart_overview_countries": False,
                "view_chart_negative_restrictive_actors": False,
                "view_chart_negative_affected_actors": False,
                "view_chart_negative_restrictive_mechanisms": False,
                "view_chart_negative_event_types": False,
                "view_chart_negative_alert_types": False,
                "view_chart_negative_enabling_principles": False,
                "view_chart_heatmap_actor_mechanism": False,
                "view_chart_heatmap_subject_mechanism": False,
                "view_chart_heatmap_actor_subject": False,
                "view_chart_sankey_flow": False,
                "view_chart_geospatial_map": False,
                "view_chart_ai_copilot_plots": False,
            },
            "regions": [],
            "countries": [],
            "years": [],
        },
        "viewer": {
            "features": {
                "view_dashboard": True,
                "view_overview": True,
                "view_coverage_monitored_countries": True,
                "view_monitored_countries_value": False,
                "view_maps": True,
                "view_negative_alerts": True,
                "view_analytical_flow_panel": False,
                "view_data_table": True,
                "download_data": False,
                "use_ai_copilot": False,
                "view_user_manual": True,
                "view_admin_page": False,
                "view_chart_overview_alert_type": False,
                "view_chart_overview_enabling_principles": False,
                "view_chart_overview_regions": False,
                "view_chart_overview_countries": False,
                "view_chart_negative_restrictive_actors": False,
                "view_chart_negative_affected_actors": False,
                "view_chart_negative_restrictive_mechanisms": False,
                "view_chart_negative_event_types": False,
                "view_chart_negative_alert_types": False,
                "view_chart_negative_enabling_principles": False,
                "view_chart_heatmap_actor_mechanism": False,
                "view_chart_heatmap_subject_mechanism": False,
                "view_chart_heatmap_actor_subject": False,
                "view_chart_sankey_flow": False,
                "view_chart_geospatial_map": False,
                "view_chart_ai_copilot_plots": False,
            },
            "regions": [],
            "countries": [],
            "years": [],
        },
        "privileged": {
            "features": {
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
                "view_chart_overview_alert_type": True,
                "view_chart_overview_enabling_principles": True,
                "view_chart_overview_regions": True,
                "view_chart_overview_countries": True,
                "view_chart_negative_restrictive_actors": True,
                "view_chart_negative_affected_actors": True,
                "view_chart_negative_restrictive_mechanisms": True,
                "view_chart_negative_event_types": True,
                "view_chart_negative_alert_types": True,
                "view_chart_negative_enabling_principles": True,
                "view_chart_heatmap_actor_mechanism": True,
                "view_chart_heatmap_subject_mechanism": True,
                "view_chart_heatmap_actor_subject": True,
                "view_chart_sankey_flow": True,
                "view_chart_geospatial_map": True,
                "view_chart_ai_copilot_plots": True,
            },
            "regions": [],
            "countries": [],
            "years": [],
        },
    }


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    base = default_access_config()
    if not isinstance(config, dict):
        return base

    for role in ["guest", "viewer", "privileged"]:
        config.setdefault(role, {})
        config[role].setdefault("features", {})
        config[role].setdefault("regions", [])
        config[role].setdefault("countries", [])
        config[role].setdefault("years", [])

        for removed_key in REMOVED_FEATURE_KEYS:
            config[role]["features"].pop(removed_key, None)

        for key in FEATURE_KEYS:
            default_value = base[role]["features"].get(key, False)
            config[role]["features"].setdefault(key, default_value)

    return config


def load_access_config() -> dict[str, Any]:
    path = get_access_config_path()
    if not path.exists():
        config = default_access_config()
        save_access_config(config)
        return config

    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        config = default_access_config()

    normalized = _normalize_config(config)
    if normalized != config:
        save_access_config(normalized)
    return normalized


def save_access_config(config: dict[str, Any]) -> bool:
    path = get_access_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = _normalize_config(config)
        with path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
        return True
    except Exception as exc:
        st.error(f"Failed to save access config: {exc}")
        return False


def reset_access_config() -> bool:
    path = get_access_config_path()
    try:
        if path.exists():
            path.unlink()
        return save_access_config(default_access_config())
    except Exception as exc:
        st.error(f"Failed to reset access config: {exc}")
        return False


def has_permission(permission: str) -> bool:
    if is_admin():
        return True

    role = get_current_role()
    if role not in {"guest", "viewer", "privileged"}:
        role = "guest"

    config = load_access_config()
    features = config.get(role, {}).get("features", {})
    return bool(features.get(permission, False))


def apply_data_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Apply role-level region/country/year restrictions from the admin page."""
    if df is None or getattr(df, "empty", True) or is_admin():
        return df

    role = get_current_role()
    config = load_access_config().get(role, {})
    scoped = df.copy()

    regions = config.get("regions", []) or []
    countries = config.get("countries", []) or []
    years = config.get("years", []) or []

    if regions and "region" in scoped.columns:
        scoped = scoped[scoped["region"].astype(str).isin([str(x) for x in regions])]

    if countries and "alert-country" in scoped.columns:
        scoped = scoped[scoped["alert-country"].astype(str).isin([str(x) for x in countries])]

    if years and "year" in scoped.columns:
        scoped = scoped[pd.to_numeric(scoped["year"], errors="coerce").isin([int(x) for x in years])]

    return scoped
