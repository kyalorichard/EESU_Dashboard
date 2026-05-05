# admin_page.py
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from authz import (
    DEFAULT_ROLE_PERMISSIONS,
    get_access_config_path,
    get_admin_emails,
    get_current_email,
    get_current_role,
    get_privileged_domains,
    has_permission,
    is_admin,
    load_access_config,
    save_access_config,
)


FEATURE_LABELS = {
    "view_public_summary": "Public summary",
    "view_dashboard": "Dashboard access",
    "view_overview": "Overview tab",
    "view_maps": "Maps / spatial views",
    "view_country_counts": "Country count KPIs",
    "view_negative_alerts": "Negative Alerts tab",
    "view_data_table": "Data tables",
    "download_data": "CSV/data downloads",
    "use_ai_copilot": "AI Copilot",
    "view_user_manual": "User manual",
    "view_admin_page": "Admin page",
}


def render_admin_sidebar_navigation():
    if not is_admin():
        return "Dashboard"

    st.sidebar.markdown("""
    <style>
    .admin-nav-card {
        margin-top: 14px;
        padding: 14px;
        border-radius: 17px;
        background:
            radial-gradient(circle at 92% 0%, rgba(255,219,88,.22), transparent 30%),
            linear-gradient(135deg,#FFFFFF 0%,#F4EAF8 72%,#EFFBFE 100%);
        border: 1px solid rgba(102,0,148,.18);
        box-shadow: 0 12px 28px rgba(16,24,40,.08);
        font-family: Arial, sans-serif;
    }
    .admin-nav-card .eyebrow {
        font-size: 9px;
        font-weight: 950;
        color: #660094;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .admin-nav-card .title {
        font-size: 14px;
        font-weight: 950;
        color: #23152F;
        margin-top: 4px;
    }
    .admin-nav-card .note {
        font-size: 10.8px;
        color: #667085;
        line-height: 1.35;
        margin-top: 5px;
    }
    </style>
    <div class="admin-nav-card">
        <div class="eyebrow">Admin workspace</div>
        <div class="title">🔐 Administration</div>
        <div class="note">Control what guest, viewer and privileged users can see.</div>
    </div>
    """, unsafe_allow_html=True)

    return st.sidebar.radio(
        "Admin navigation",
        ["Dashboard", "Admin"],
        index=0,
        key="admin_navigation_choice",
        label_visibility="collapsed",
    )


def _admin_css():
    st.markdown("""
    <style>
    .admin-hero {
        background:
            radial-gradient(circle at 95% 5%, rgba(255,219,88,.22), transparent 30%),
            linear-gradient(135deg, #FFFFFF 0%, #F7ECFB 62%, #EFFBFE 100%);
        border: 1px solid rgba(102,0,148,.16);
        border-radius: 22px;
        padding: 20px 22px;
        box-shadow: 0 16px 38px rgba(16,24,40,.08);
        font-family: Arial, sans-serif;
        margin-bottom: 16px;
    }
    .admin-eyebrow {
        font-size: 10px;
        font-weight: 950;
        color: #660094;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .admin-title {
        font-size: 28px;
        font-weight: 950;
        color: #23152F;
        letter-spacing: -.03em;
        line-height: 1.05;
        font-family: Arial Black, Arial, sans-serif;
    }
    .admin-note {
        font-size: 12px;
        color: #667085;
        max-width: 850px;
        line-height: 1.45;
        margin-top: 8px;
    }
    .admin-pill-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
    }
    .admin-pill {
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 10.5px;
        font-weight: 900;
        background: #F4EAF8;
        color: #660094;
        border: 1px solid #E7D4F1;
    }
    </style>
    """, unsafe_allow_html=True)


def _safe_default_list(config, role, key):
    return config.get(role, {}).get(key, [])


def render_admin_page(data=None):
    _admin_css()

    if not is_admin():
        st.error("Access restricted. This page is only available to configured admin emails.")
        st.stop()

    config = load_access_config()

    st.markdown(f"""
    <div class="admin-hero">
        <div class="admin-eyebrow">Admin workspace</div>
        <div class="admin-title">EU SEE Dashboard Administration</div>
        <div class="admin-note">
            Configure what guest, viewer, and privileged users can see. Settings are saved to:
            <code>{get_access_config_path()}</code>
        </div>
        <div class="admin-pill-row">
            <span class="admin-pill">Signed in: {get_current_email()}</span>
            <span class="admin-pill">Role: {get_current_role()}</span>
            <span class="admin-pill">Admin emails: {len(get_admin_emails())}</span>
            <span class="admin-pill">Privileged domains: {len(get_privileged_domains())}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_visibility, tab_scope, tab_users, tab_system = st.tabs(
        ["🎛️ Visibility", "🌍 Data scope", "👤 Users", "⚙️ Diagnostics"]
    )

    with tab_visibility:
        role = st.selectbox("Configure role", ["guest", "viewer", "privileged"], index=0)

        st.warning(
            "After changing permissions, click 'Save visibility settings'. "
            "Then log out/open a new session to test as guest/viewer/privileged."
        )

        features = config[role]["features"]
        cols = st.columns(2)

        for i, (key, label) in enumerate(FEATURE_LABELS.items()):
            with cols[i % 2]:
                features[key] = st.checkbox(
                    label,
                    value=bool(features.get(key, False)),
                    key=f"persist_feature_{role}_{key}",
                    disabled=(role == "guest" and key in ["download_data", "use_ai_copilot", "view_admin_page"]),
                )

        config[role]["features"] = features

        save_col, reset_col = st.columns([1, 1])
        with save_col:
            if st.button("💾 Save visibility settings", use_container_width=True, type="primary"):
                if save_access_config(config):
                    st.success("Visibility settings saved persistently.")
                    st.rerun()

        with reset_col:
            if st.button("↩ Reset to defaults", use_container_width=True):
                from authz import default_access_config
                if save_access_config(default_access_config()):
                    st.success("Defaults restored.")
                    st.rerun()

        st.download_button(
            "⬇️ Download current visibility config",
            data=json.dumps(config, indent=2),
            file_name="eusee_access_visibility_config.json",
            mime="application/json",
            use_container_width=True,
        )

    with tab_scope:
        role = st.selectbox("Configure data scope for role", ["guest", "viewer", "privileged"], index=0, key="scope_role")
        st.caption("Leave empty to allow all available values for that role.")

        regions, countries, years = [], [], []
        if data is not None and not getattr(data, "empty", True):
            if "region" in data.columns:
                regions = sorted(data["region"].dropna().astype(str).unique())
            if "alert-country" in data.columns:
                countries = sorted(data["alert-country"].dropna().astype(str).unique())
            if "year" in data.columns:
                years = sorted([int(y) for y in data["year"].dropna().unique()])

        config[role]["regions"] = st.multiselect(
            "Allowed regions",
            regions,
            default=[x for x in _safe_default_list(config, role, "regions") if x in regions],
            key=f"persist_regions_{role}",
        )
        config[role]["countries"] = st.multiselect(
            "Allowed countries",
            countries,
            default=[x for x in _safe_default_list(config, role, "countries") if x in countries],
            key=f"persist_countries_{role}",
        )
        config[role]["years"] = st.multiselect(
            "Allowed years",
            years,
            default=[x for x in _safe_default_list(config, role, "years") if x in years],
            key=f"persist_years_{role}",
        )

        if st.button("💾 Save data scope", use_container_width=True, type="primary"):
            if save_access_config(config):
                st.success("Data scope saved persistently.")
                st.rerun()

        st.json(config[role])

    with tab_users:
        rows = []
        for email in get_admin_emails():
            rows.append({"identity": email, "role": "admin", "source": "secrets.toml"})
        for domain in get_privileged_domains():
            rows.append({"identity": f"*@{domain}", "role": "privileged", "source": "[access].privileged_domains"})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No admin emails or privileged domains configured.")

        st.code("""
[auth]
admin_emails = ["admin@example.org"]

[access]
privileged_domains = ["icarda.org", "cgiar.org"]

[access_control]
# optional. Defaults to /exports/eusee_access_config.json if /exports exists.
config_path = "/exports/eusee_access_config.json"
""", language="toml")

    with tab_system:
        st.json({
            "current_email": get_current_email(),
            "current_role": get_current_role(),
            "is_admin": is_admin(),
            "config_path": str(get_access_config_path()),
            "can_download": has_permission("download_data"),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "session_email_verified": st.session_state.get("email_verified"),
            "session_user": st.session_state.get("user"),
        })
        st.json(load_access_config())
