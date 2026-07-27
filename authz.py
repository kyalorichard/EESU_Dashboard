from __future__ import annotations

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


ROLES = ["guest", "viewer", "privileged"]

FEATURE_REGISTRY = {
    "view_dashboard": ("Core access", "Dashboard access"),
    "view_overview": ("Core access", "Overview tab"),
    "view_coverage_monitored_countries": ("Core access", "Summary cards"),
    "view_monitored_countries_value": ("Core access", "Monitored Countries value"),
    "view_maps": ("Core access", "Visualization Map"),
    "view_negative_alerts": ("Core access", "Negative Alerts tab"),
    "view_negative_alert_filters": ("Administration", "Negative Alerts filter panel"),
    "view_analytical_flow_panel": ("Core access", "Analytical Flow Panels"),
    "view_data_table": ("Core access", "Summary data preview"),
    "download_data": ("Core access", "CSV/XLSX downloads"),
    "use_ai_copilot": ("AI Copilot", "AI Copilot"),
    "view_user_manual": ("Core access", "User manual"),
    "view_admin_page": ("Administration", "Admin page"),

    "view_chart_overview_alert_type": ("Overview charts", "Alert type distribution"),
    "view_chart_overview_enabling_principles": ("Overview charts", "Enabling-principle distribution"),
    "view_chart_overview_regions": ("Overview charts", "Regional distribution"),
    "view_chart_overview_countries": ("Overview charts", "Country distribution"),

    "view_chart_negative_restrictive_actors": ("Negative alerts charts", "Restrictive actors"),
    "view_chart_negative_affected_actors": ("Negative alerts charts", "Civil society actors affected"),
    "view_chart_negative_restrictive_mechanisms": ("Negative alerts charts", "Restrictive mechanisms"),
    "view_chart_negative_event_types": ("Negative alerts charts", "Negative event types"),
    "view_chart_negative_alert_types": ("Negative alerts charts", "Negative alert types"),
    "view_chart_negative_enabling_principles": ("Negative alerts charts", "Negative enabling principles"),

    "view_chart_heatmap_actor_mechanism": ("Analytical charts", "Actor × mechanism heatmap"),
    "view_chart_heatmap_subject_mechanism": ("Analytical charts", "Affected actor × mechanism heatmap"),
    "view_chart_heatmap_actor_subject": ("Analytical charts", "Actor × affected actor heatmap"),
    "view_chart_sankey_flow": ("Analytical charts", "Analytical Sankey flow"),
    "view_chart_negative_flow_diagram": ("Analytical charts", "Negative-alert flow diagram"),
    "view_chart_negative_key_links": ("Analytical charts", "Negative-alert key links"),
    "view_chart_negative_follow_pathway": ("Analytical charts", "Negative-alert follow pathway"),
    "view_chart_negative_top_n_selector": ("Analytical charts", "Negative-alert Top-N selector"),
    "view_chart_negative_detail_level": ("Analytical charts", "Negative-alert detail-level control"),
    "view_chart_geospatial_map": ("Analytical charts", "Geospatial intelligence map"),
    "view_chart_ai_copilot_plots": ("AI Copilot", "AI Copilot generated plots"),
}

FEATURE_KEYS = list(FEATURE_REGISTRY.keys())

REMOVED_FEATURE_KEYS = {
    "view_public_summary",
    "view_country_counts",
    "view_negative_relationship_intelligence",
}

ROLE_PRESETS = {
    "guest": {
        "view_dashboard",
        "view_overview",
        "view_coverage_monitored_countries",
        "view_user_manual",
    },
    "viewer": {
        "view_dashboard",
        "view_overview",
        "view_coverage_monitored_countries",
        "view_negative_alerts",
        "view_data_table",
        "view_user_manual",
    },
    # Privileged users retain all normal dashboard permissions except the
    # Visualization Map permissions, which are locked to Admin only below.
    "privileged": set(FEATURE_KEYS),
}

# Permissions that are permanently restricted by role. Admin users always have
# full access because has_permission() returns True for is_admin(). Privileged
# users are allowed to view every dashboard plot, including the map.
ADMIN_ONLY_FEATURES = {
    "view_admin_page",
    "view_negative_alert_filters",
}

LOCKED_FALSE = {
    "guest": {
        "view_admin_page",
        "view_negative_alert_filters",
        "view_maps",
        "view_chart_geospatial_map",
    },
    "viewer": {
        "view_admin_page",
        "view_negative_alert_filters",
        "view_maps",
        "view_chart_geospatial_map",
    },
    "privileged": set(ADMIN_ONLY_FEATURES),
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


def get_temporary_shared_account() -> dict[str, Any]:
    """Return a normalized temporary shared-account configuration."""
    enabled = bool(_secrets_get("temporary_shared_account", "enabled", False))
    email = str(_secrets_get("temporary_shared_account", "email", "") or "").strip().lower()
    role = str(_secrets_get("temporary_shared_account", "role", "viewer") or "viewer").strip().lower()
    bypass_email_verification = bool(
        _secrets_get("temporary_shared_account", "bypass_email_verification", False)
    )

    if role not in ROLES:
        role = "viewer"

    return {
        "enabled": enabled,
        "email": email,
        "role": role,
        "bypass_email_verification": bypass_email_verification,
    }


def is_temporary_shared_account(email: str | None = None) -> bool:
    """Return True when the supplied/current email is the enabled shared account."""
    account = get_temporary_shared_account()
    candidate = str(email or get_current_email() or "").strip().lower()

    return bool(
        account["enabled"]
        and account["email"]
        and candidate
        and candidate == account["email"]
    )


def can_bypass_email_verification(email: str | None = None) -> bool:
    """Allow verification bypass only for the explicitly configured shared account."""
    account = get_temporary_shared_account()
    return bool(
        account["bypass_email_verification"]
        and is_temporary_shared_account(email)
    )


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
    """Resolve the effective role using admin, shared-account and domain rules."""
    if is_admin():
        return "admin"

    email = get_current_email()
    if not email:
        return "guest"

    # Explicit shared-account assignment takes precedence over domain rules.
    temporary_account = get_temporary_shared_account()
    if is_temporary_shared_account(email):
        return temporary_account["role"]

    domain = email.rsplit("@", 1)[-1].strip().lower() if "@" in email else ""

    secret_domains = {
        str(item).strip().lower().lstrip("@")
        for item in get_privileged_domains()
        if str(item).strip()
    }

    firestore_domains: set[str] = set()
    try:
        config = load_access_config()
        firestore_domains = {
            str(item).strip().lower().lstrip("@")
            for item in config.get("privileged_domains", [])
            if str(item).strip()
        }
    except Exception:
        # Secrets remain a safe fallback when Firestore is unavailable.
        firestore_domains = set()

    if domain and domain in (secret_domains | firestore_domains):
        return "privileged"

    return "viewer"


def default_access_config() -> dict[str, Any]:
    config = {}

    for role in ROLES:
        enabled = ROLE_PRESETS.get(role, set())

        features = {
            key: key in enabled
            for key in FEATURE_KEYS
        }

        for locked_key in LOCKED_FALSE.get(role, set()):
            features[locked_key] = False

        config[role] = {
            "features": features,
            "regions": [],
            "countries": [],
            "years": [],
        }

    return config


def normalize_access_config(config: dict[str, Any] | None) -> dict[str, Any]:
    base = default_access_config()

    if not isinstance(config, dict):
        return base

    config["privileged_domains"] = sorted({
        str(item).strip().lower().lstrip("@")
        for item in _as_list(config.get("privileged_domains", get_privileged_domains()))
        if str(item).strip()
    })

    for role in ROLES:
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

        for locked_key in LOCKED_FALSE.get(role, set()):
            config[role]["features"][locked_key] = False

    return config


@st.cache_resource(show_spinner=False)
def _get_firestore_client():
    if firebase_admin is None:
        st.error("firebase-admin is not installed. Add firebase-admin to requirements.txt and redeploy.")
        return None

    try:
        if firebase_admin._apps:
            return firestore.client()

        service_account_info = None

        if "firebase_admin" in st.secrets:
            service_account_info = dict(st.secrets["firebase_admin"])
        elif "firebase" in st.secrets and "service_account" in st.secrets["firebase"]:
            service_account_info = dict(st.secrets["firebase"]["service_account"])

        if not service_account_info:
            st.error(
                "Firebase secrets not found. Expected [firebase_admin] or [firebase.service_account]."
            )
            return None

        required_keys = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "auth_provider_x509_cert_url",
            "client_x509_cert_url",
        ]

        missing = [key for key in required_keys if not service_account_info.get(key)]

        if missing:
            st.error(f"Firebase secrets missing required keys: {missing}")
            return None

        service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate(service_account_info)
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
        normalized = normalize_access_config(config)

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
        normalized = normalize_access_config(config)
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

    if role not in ROLES:
        role = "guest"

    config = load_access_config()
    features = config.get(role, {}).get("features", {})

    return bool(features.get(permission, False))


def get_access_diagnostics() -> dict[str, Any]:
    """Return a compact runtime snapshot for troubleshooting access problems."""
    account = get_temporary_shared_account()
    return {
        "current_email": get_current_email(),
        "current_role": get_current_role(),
        "is_admin": is_admin(),
        "temporary_account_enabled": account["enabled"],
        "temporary_account_match": is_temporary_shared_account(),
        "bypass_email_verification": can_bypass_email_verification(),
        "permissions": {key: has_permission(key) for key in FEATURE_KEYS},
    }


def apply_data_scope(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True) or is_admin():
        return df

    role = get_current_role()

    if role not in ROLES:
        role = "guest"

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