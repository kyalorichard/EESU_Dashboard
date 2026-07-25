from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import streamlit as st


# =============================================================================
# ROLE DEFINITIONS
# =============================================================================

ROLES = ["guest", "viewer", "privileged"]

# Each permission is registered as:
# "permission_key": ("Permission group", "Human-readable label")
FEATURE_REGISTRY: dict[str, tuple[str, str]] = {
    # Core access
    "view_overview": ("Core access", "View Overview"),
    "view_negative_alerts": ("Core access", "View Negative Alerts"),
    "view_analytical_flow_panel": ("Core access", "View Analytical Flow Panels"),
    "view_data_table": ("Core access", "View Data Table"),
    "view_user_manual": ("Core access", "View User Manual"),

    # Overview charts
    "view_coverage_monitored_countries": (
        "Overview charts",
        "View monitored-country coverage",
    ),
    "view_monitored_countries_value": (
        "Overview charts",
        "View monitored-country value",
    ),
    "view_chart_overview_alert_type": (
        "Overview charts",
        "View alert-type chart",
    ),
    "view_chart_overview_enabling_principles": (
        "Overview charts",
        "View enabling-principles chart",
    ),
    "view_chart_overview_regions": (
        "Overview charts",
        "View regions chart",
    ),
    "view_chart_overview_countries": (
        "Overview charts",
        "View countries chart",
    ),

    # Negative-alert charts
    "view_chart_negative_restrictive_actors": (
        "Negative-alert charts",
        "View restrictive actors chart",
    ),
    "view_chart_negative_affected_actors": (
        "Negative-alert charts",
        "View affected actors chart",
    ),
    "view_chart_negative_restrictive_mechanisms": (
        "Negative-alert charts",
        "View restrictive mechanisms chart",
    ),
    "view_chart_negative_event_types": (
        "Negative-alert charts",
        "View event types chart",
    ),
    "view_chart_negative_alert_types": (
        "Negative-alert charts",
        "View negative alert types chart",
    ),
    "view_chart_negative_enabling_principles": (
        "Negative-alert charts",
        "View negative enabling-principles chart",
    ),

    # Analytical charts
    "view_chart_heatmap_actor_mechanism": (
        "Analytical charts",
        "Actor–mechanism heatmap",
    ),
    "view_chart_heatmap_subject_mechanism": (
        "Analytical charts",
        "Subject–mechanism heatmap",
    ),
    "view_chart_heatmap_actor_subject": (
        "Analytical charts",
        "Actor–subject heatmap",
    ),
    "view_chart_sankey_flow": (
        "Analytical charts",
        "Sankey flow",
    ),
    "view_chart_negative_flow_diagram": (
        "Analytical charts",
        "Negative-alert flow diagram",
    ),
    "view_chart_negative_key_links": (
        "Analytical charts",
        "Negative-alert key links",
    ),
    "view_chart_negative_follow_pathway": (
        "Analytical charts",
        "Negative-alert follow pathway",
    ),
    "view_chart_negative_top_n_selector": (
        "Analytical charts",
        "Top-N selector",
    ),
    "view_chart_negative_detail_level": (
        "Analytical charts",
        "Detail-level selector",
    ),

    # AI
    "use_ai_copilot": ("AI Copilot", "Use AI Copilot"),
    "view_chart_ai_copilot_plots": (
        "AI Copilot",
        "View AI Copilot plots",
    ),

    # Exports
    "download_data": ("Data and exports", "Download data"),
}

FEATURE_KEYS = list(FEATURE_REGISTRY.keys())


# Permissions that must remain disabled for selected ordinary roles.
# Admins do not use these role-level restrictions.
LOCKED_FALSE: dict[str, set[str]] = {
    "guest": {
        "download_data",
        "use_ai_copilot",
        "view_chart_ai_copilot_plots",
    },
    "viewer": {
        "download_data",
    },
    "privileged": set(),
}


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def _normalize_email(email: Any) -> str:
    return str(email or "").strip().lower()


def _normalize_domain(domain: Any) -> str:
    value = str(domain or "").strip().lower()
    value = value.replace("https://", "").replace("http://", "")
    value = value.replace("www.", "").replace("@", "")
    return value.split("/")[0].strip()


def _secret_section(name: str) -> dict:
    """
    Safely convert a Streamlit secrets section into a regular dictionary.
    """
    try:
        section = st.secrets.get(name, {})
        return dict(section) if section else {}
    except Exception:
        return {}


def _as_email_set(values: Any) -> set[str]:
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, (list, tuple, set)):
        return set()

    return {
        normalized
        for value in values
        if (normalized := _normalize_email(value))
    }


def _as_domain_set(values: Any) -> set[str]:
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, (list, tuple, set)):
        return set()

    return {
        normalized
        for value in values
        if (normalized := _normalize_domain(value))
    }


# =============================================================================
# TEMPORARY SHARED ACCOUNT
# =============================================================================

def get_temporary_account_config() -> dict[str, Any]:
    """
    Read the temporary account configuration from .streamlit/secrets.toml.

    Expected format:

    [temporary_shared_account]
    enabled = true
    email = "dashboard.access@eusee.global"
    role = "privileged"
    bypass_email_verification = true
    expires_at = "2026-08-15T23:59:59+03:00"

    The password MUST remain in Firebase Authentication and must not be stored
    in Streamlit secrets or source code.
    """
    raw = _secret_section("temporary_shared_account")

    role = str(raw.get("role", "privileged")).strip().lower()
    if role not in ROLES:
        role = "privileged"

    return {
        "enabled": bool(raw.get("enabled", False)),
        "email": _normalize_email(raw.get("email")),
        "role": role,
        "bypass_email_verification": bool(
            raw.get("bypass_email_verification", True)
        ),
        "expires_at": str(raw.get("expires_at", "")).strip(),
    }


def is_temporary_shared_account(email: str | None = None) -> bool:
    config = get_temporary_account_config()

    if not config["enabled"] or not config["email"]:
        return False

    candidate = _normalize_email(
        email if email is not None else get_current_email()
    )

    if candidate != config["email"]:
        return False

    expires_at = config.get("expires_at", "")
    if not expires_at:
        return True

    # Avoid adding a hard dependency on python-dateutil.
    try:
        from datetime import datetime

        expiry = datetime.fromisoformat(expires_at)
        now = datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.now()
        return now <= expiry
    except Exception:
        # Fail closed when an expiry was supplied but cannot be parsed.
        return False


def bypass_email_verification(email: str | None = None) -> bool:
    config = get_temporary_account_config()
    return (
        is_temporary_shared_account(email)
        and bool(config.get("bypass_email_verification", False))
    )


def should_allow_authenticated_user(
    email: str,
    email_verified: bool,
) -> tuple[bool, str]:
    """
    Central email-verification decision.

    Returns:
        (allowed, message)

    Use this immediately after Firebase sign-in.
    """
    normalized_email = _normalize_email(email)

    if not normalized_email:
        return False, "No authenticated email address was returned."

    if email_verified:
        return True, ""

    if bypass_email_verification(normalized_email):
        return True, ""

    return (
        False,
        "Please verify your email address before accessing the dashboard.",
    )


# =============================================================================
# CURRENT SESSION
# =============================================================================

def get_current_email() -> str:
    """
    Return the authenticated email stored by the login layer.

    The function supports several common session-state keys so that it can be
    used with existing Firebase/Streamlit login implementations.
    """
    direct_keys = (
        "user_email",
        "email",
        "authenticated_email",
        "firebase_email",
    )

    for key in direct_keys:
        value = st.session_state.get(key)
        if value:
            return _normalize_email(value)

    nested_keys = ("user", "firebase_user", "auth_user")

    for key in nested_keys:
        user = st.session_state.get(key)
        if isinstance(user, dict):
            value = user.get("email")
            if value:
                return _normalize_email(value)

    return ""


def set_current_user_session(
    *,
    email: str,
    email_verified: bool,
    uid: str = "",
    id_token: str = "",
) -> None:
    """
    Store the authenticated Firebase user in Streamlit session state.

    Call this only after should_allow_authenticated_user(...) returns True.
    """
    normalized_email = _normalize_email(email)

    st.session_state["authenticated"] = True
    st.session_state["user_email"] = normalized_email
    st.session_state["email_verified"] = bool(email_verified)
    st.session_state["firebase_uid"] = str(uid or "")
    st.session_state["firebase_id_token"] = str(id_token or "")
    st.session_state["user_role"] = get_role_for_email(normalized_email)


def clear_current_user_session() -> None:
    keys = (
        "authenticated",
        "user_email",
        "email",
        "authenticated_email",
        "firebase_email",
        "email_verified",
        "firebase_uid",
        "firebase_id_token",
        "user_role",
        "user",
        "firebase_user",
        "auth_user",
    )

    for key in keys:
        st.session_state.pop(key, None)


# =============================================================================
# IDENTITY AND ROLE CONFIGURATION
# =============================================================================

def get_admin_emails() -> set[str]:
    auth = _secret_section("auth")
    return _as_email_set(auth.get("admin_emails", []))


def get_shared_privileged_emails() -> set[str]:
    """
    Optional permanent/shared identities from the [auth] section.

    For the temporary account, prefer [temporary_shared_account].
    """
    auth = _secret_section("auth")
    return _as_email_set(auth.get("shared_privileged_emails", []))


def get_privileged_domains() -> set[str]:
    access = _secret_section("access")
    return _as_domain_set(access.get("privileged_domains", []))


def get_role_for_email(email: str) -> str:
    normalized_email = _normalize_email(email)

    if not normalized_email:
        return "guest"

    if normalized_email in get_admin_emails():
        return "admin"

    if is_temporary_shared_account(normalized_email):
        return get_temporary_account_config()["role"]

    if normalized_email in get_shared_privileged_emails():
        return "privileged"

    domain = (
        normalized_email.rsplit("@", 1)[1]
        if "@" in normalized_email
        else ""
    )

    if domain and domain in get_privileged_domains():
        return "privileged"

    return "viewer"


def get_current_role() -> str:
    """
    Return admin, guest, viewer or privileged.
    """
    email = get_current_email()
    return get_role_for_email(email)


def is_admin() -> bool:
    return get_current_role() == "admin"


# =============================================================================
# DEFAULT ACCESS CONFIGURATION
# =============================================================================

def _all_false() -> dict[str, bool]:
    return {key: False for key in FEATURE_KEYS}


def _all_true() -> dict[str, bool]:
    return {key: True for key in FEATURE_KEYS}


def default_access_config() -> dict[str, Any]:
    guest = _all_false()
    guest.update(
        {
            "view_overview": True,
            "view_coverage_monitored_countries": True,
            "view_monitored_countries_value": True,
            "view_chart_overview_alert_type": True,
            "view_chart_overview_enabling_principles": True,
            "view_chart_overview_regions": True,
            "view_chart_overview_countries": True,
            "view_user_manual": True,
        }
    )

    viewer = deepcopy(guest)
    viewer.update(
        {
            "view_negative_alerts": True,
            "view_chart_negative_restrictive_actors": True,
            "view_chart_negative_affected_actors": True,
            "view_chart_negative_restrictive_mechanisms": True,
            "view_chart_negative_event_types": True,
            "view_chart_negative_alert_types": True,
            "view_chart_negative_enabling_principles": True,
            "view_data_table": True,
        }
    )

    privileged = _all_true()

    return {
        "guest": {
            "features": guest,
            "regions": [],
            "countries": [],
            "years": [],
        },
        "viewer": {
            "features": viewer,
            "regions": [],
            "countries": [],
            "years": [],
        },
        "privileged": {
            "features": privileged,
            "regions": [],
            "countries": [],
            "years": [],
        },
        "privileged_domains": sorted(get_privileged_domains()),
    }


def normalize_access_config(config: dict | None) -> dict[str, Any]:
    defaults = default_access_config()
    source = config if isinstance(config, dict) else {}

    normalized = deepcopy(defaults)

    for role in ROLES:
        role_source = source.get(role, {})
        if not isinstance(role_source, dict):
            role_source = {}

        feature_source = role_source.get("features", {})
        if not isinstance(feature_source, dict):
            feature_source = {}

        normalized[role]["features"] = {
            key: bool(
                feature_source.get(
                    key,
                    defaults[role]["features"].get(key, False),
                )
            )
            for key in FEATURE_KEYS
        }

        for locked_key in LOCKED_FALSE.get(role, set()):
            if locked_key in normalized[role]["features"]:
                normalized[role]["features"][locked_key] = False

        for scope_key in ("regions", "countries", "years"):
            value = role_source.get(scope_key, defaults[role][scope_key])
            normalized[role][scope_key] = (
                list(value)
                if isinstance(value, (list, tuple, set))
                else []
            )

    domains = source.get("privileged_domains")
    if domains is None:
        domains = get_privileged_domains()

    normalized["privileged_domains"] = sorted(_as_domain_set(domains))
    return normalized


# =============================================================================
# ACCESS-CONFIG STORAGE
# =============================================================================

def get_access_config_path() -> str:
    access_control = _secret_section("access_control")
    configured = str(access_control.get("local_config_path", "")).strip()

    if configured:
        return configured

    return str(Path("data") / "eusee_access_config.json")


def load_access_config() -> dict[str, Any]:
    """
    Load the access configuration from a local JSON file.

    Replace these storage functions with your Firestore implementation if your
    deployed app already persists the configuration in Firestore.
    """
    path = Path(get_access_config_path())

    if not path.exists():
        return default_access_config()

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return normalize_access_config(data)
    except Exception:
        return default_access_config()


def save_access_config(config: dict) -> bool:
    path = Path(get_access_config_path())

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = normalize_access_config(config)

        with path.open("w", encoding="utf-8") as file:
            json.dump(normalized, file, indent=2, ensure_ascii=False)

        return True
    except Exception as exc:
        st.error(f"Unable to save access configuration: {exc}")
        return False


def reset_access_config() -> bool:
    return save_access_config(default_access_config())


# =============================================================================
# PERMISSION AND DATA-SCOPE CHECKS
# =============================================================================

def has_permission(
    feature_key: str,
    role: str | None = None,
) -> bool:
    if feature_key not in FEATURE_KEYS:
        return False

    effective_role = str(role or get_current_role()).strip().lower()

    if effective_role == "admin":
        return True

    if effective_role not in ROLES:
        effective_role = "guest"

    config = load_access_config()
    return bool(
        config.get(effective_role, {})
        .get("features", {})
        .get(feature_key, False)
    )


def get_role_scope(role: str | None = None) -> dict[str, list]:
    effective_role = str(role or get_current_role()).strip().lower()

    if effective_role == "admin":
        return {
            "regions": [],
            "countries": [],
            "years": [],
        }

    if effective_role not in ROLES:
        effective_role = "guest"

    config = load_access_config()
    role_config = config.get(effective_role, {})

    return {
        "regions": list(role_config.get("regions", []) or []),
        "countries": list(role_config.get("countries", []) or []),
        "years": list(role_config.get("years", []) or []),
    }


def apply_role_scope(dataframe):
    """
    Filter a pandas DataFrame using the current role's configured scope.

    Empty scope lists mean unrestricted access for that dimension.
    """
    if dataframe is None or getattr(dataframe, "empty", True):
        return dataframe

    if is_admin():
        return dataframe

    scope = get_role_scope()
    result = dataframe.copy()

    if scope["regions"] and "region" in result.columns:
        result = result[
            result["region"].astype(str).isin(
                {str(value) for value in scope["regions"]}
            )
        ]

    if scope["countries"] and "alert-country" in result.columns:
        result = result[
            result["alert-country"].astype(str).isin(
                {str(value) for value in scope["countries"]}
            )
        ]

    if scope["years"] and "year" in result.columns:
        allowed_years = {str(value) for value in scope["years"]}
        result = result[
            result["year"].astype(str).isin(allowed_years)
        ]

    return result
