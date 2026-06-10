from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None


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
            "features": {key: True for key in FEATURE_KEYS},
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
            config[role]["features"].setdefault(
                key,
                base[role]["features"].get(key, False),
            )

    return config


@st.cache_resource(show_spinner=False)
def _get_firestore_client():
    if firebase_admin is None:
        return None

    try:
        if firebase_admin._apps:
            return firestore.client()

        try:
            service_account_info = dict(st.secrets["firebase_admin"])
        except Exception:
            service_account_info = _secrets_get(
                "firebase",
                "service_account",
                None
            )

        if service_account_info:
            if isinstance(service_account_info, str):
                service_account_info = json.loads(service_account_info)

            cred = credentials.Certificate(dict(service_account_info))
            firebase_admin.initialize_app(cred)
            return firestore.client()

    except Exception as exc:
        st.error(f"Firebase initialization failed: {exc}")

    return None


def _firestore_doc_ref():
    db = _get_firestore_client()
    if db is None:
        return None

    collection = _secrets_get(
        "access_control",
        "firestore_collection",
        "dashboard_settings",
    )
    document = _secrets_get(
        "access_control",
        "firestore_document",
        "access_control",
    )

    return db.collection(collection).document(document)


def load_access_config() -> dict[str, Any]:
    doc_ref = _firestore_doc_ref()

    if doc_ref is None:
        return default_access_config()

    try:
        snapshot = doc_ref.get()

        if not snapshot.exists:
            config = default_access_config()
            doc_ref.set(config)
            return config

        config = snapshot.to_dict()
        normalized = _normalize_config(config)

        if normalized != config:
            doc_ref.set(normalized)

        return normalized

    except Exception as exc:
        st.error(f"Failed to load access config from Firestore: {exc}")
        return default_access_config()


def save_access_config(config: dict[str, Any]) -> bool:
    doc_ref = _firestore_doc_ref()

    if doc_ref is None:
        st.error("Firestore is not configured. Access settings were not saved.")
        return False

    try:
        normalized = _normalize_config(config)
        doc_ref.set(normalized)
        st.cache_data.clear()
        return True

    except Exception as exc:
        st.error(f"Failed to save access config to Firestore: {exc}")
        return False


def reset_access_config() -> bool:
    return save_access_config(default_access_config())


def get_access_config_path() -> str:
    return "Firestore: dashboard_settings/access_control"


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
        scoped = scoped[
            pd.to_numeric(scoped["year"], errors="coerce").isin([int(x) for x in years])
        ]

    return scoped
