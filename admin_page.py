from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from authz import (
    FEATURE_KEYS,
    FEATURE_REGISTRY,
    LOCKED_FALSE,
    ROLES,
    default_access_config,
    get_access_config_path,
    get_admin_emails,
    get_current_email,
    get_current_role,
    get_privileged_domains,
    has_permission,
    is_admin,
    load_access_config,
    normalize_access_config,
    reset_access_config,
    save_access_config,
)


def _clear_app_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

    try:
        st.cache_resource.clear()
    except Exception:
        pass


def inject_admin_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 0.75rem !important;
            max-width: 1500px !important;
        }

        .admin-hero {
            background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
            border: 1px solid #e6e8ef;
            border-radius: 20px;
            padding: 24px 26px;
            box-shadow: 0 12px 32px rgba(16,24,40,.06);
            margin-bottom: 16px;
        }

        .admin-eyebrow {
            font-size: 10px;
            font-weight: 900;
            color: #475467;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }

        .admin-title {
            font-size: 28px;
            font-weight: 900;
            color: #101828;
            line-height: 1.1;
            margin-bottom: 7px;
        }

        .admin-subtitle {
            font-size: 13px;
            color: #667085;
            line-height: 1.5;
            max-width: 1050px;
        }

        .admin-card {
            background: #ffffff;
            border: 1px solid #e6e8ef;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 8px 22px rgba(16,24,40,.045);
            height: 100%;
        }

        .metric-label {
            font-size: 11px;
            font-weight: 800;
            color: #667085;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .metric-value {
            font-size: 26px;
            font-weight: 900;
            color: #101828;
            margin-top: 3px;
        }

        .metric-note {
            font-size: 12px;
            color: #667085;
            margin-top: 3px;
        }

        .admin-info {
            background: #eff8ff;
            border: 1px solid #b2ddff;
            border-radius: 13px;
            padding: 11px 13px;
            color: #175cd3;
            font-size: 12px;
            margin: 12px 0 14px 0;
        }

        .admin-footer {
            text-align: center;
            font-size: 11px;
            color: #98a2b3;
            margin-top: 18px;
        }

        .stButton > button {
            border-radius: 11px !important;
            font-weight: 800 !important;
            height: 42px !important;
        }

        .stDownloadButton > button {
            border-radius: 11px !important;
            font-weight: 800 !important;
            height: 42px !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 11px !important;
            min-height: 42px !important;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px !important;
            border: 1px solid #e6e8ef !important;
            box-shadow: 0 6px 18px rgba(16,24,40,.04) !important;
            overflow: hidden !important;
        }

        div[data-testid="stExpander"] summary {
            font-weight: 900 !important;
            color: #101828 !important;
            background: #ffffff !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px !important;
            overflow: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_admin_sidebar_navigation():
    if not is_admin():
        return "Dashboard"

    st.sidebar.markdown(
        """
        <div style="
            margin-top:14px;
            padding:14px;
            border-radius:16px;
            background:#ffffff;
            border:1px solid #e6e8ef;
            box-shadow:0 8px 20px rgba(16,24,40,.05);
            font-family:Arial,sans-serif;
        ">
            <div style="font-size:9px;font-weight:900;color:#667085;letter-spacing:.12em;text-transform:uppercase;">
                Workspace
            </div>
            <div style="font-size:14px;font-weight:900;color:#101828;margin-top:5px;">
                ⚙️ Administration
            </div>
            <div style="font-size:11px;color:#667085;line-height:1.4;margin-top:5px;">
                Manage visibility, data scope and access roles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.sidebar.radio(
        "Admin navigation",
        ["Dashboard", "Admin"],
        index=0,
        key="admin_navigation_choice",
        label_visibility="collapsed",
    )


def _render_header():
    st.markdown(
        f"""
        <div class="admin-hero">
            <div class="admin-eyebrow">Admin workspace</div>
            <div class="admin-title">EU SEE Dashboard Administration</div>
            <div class="admin-subtitle">
                Manage access roles, feature visibility, data scope, users and dashboard governance from one professional control center.
                <br>
                Saved config: <code>{get_access_config_path()}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str | int, note: str = ""):
    st.markdown(
        f"""
        <div class="admin-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_label(feature_key: str) -> str:
    return FEATURE_REGISTRY.get(feature_key, ("Other", feature_key))[1]


def _feature_group(feature_key: str) -> str:
    return FEATURE_REGISTRY.get(feature_key, ("Other", feature_key))[0]


def _feature_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}

    for key in FEATURE_KEYS:
        group = _feature_group(key)
        groups.setdefault(group, []).append(key)

    return groups


def _role_enabled_count(config: dict, role: str) -> int:
    features = config.get(role, {}).get("features", {})
    return sum(1 for key in FEATURE_KEYS if bool(features.get(key, False)))


def _build_permission_matrix(config: dict) -> pd.DataFrame:
    rows = []

    for key in FEATURE_KEYS:
        row = {
            "Group": _feature_group(key),
            "Permission": _feature_label(key),
            "Key": key,
        }

        for role in ROLES:
            row[role.capitalize()] = "✅" if config.get(role, {}).get("features", {}).get(key, False) else "—"

        rows.append(row)

    return pd.DataFrame(rows)


def _render_overview_tab(config: dict):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _metric_card("Current role", get_current_role().capitalize(), get_current_email() or "Not signed in")

    with c2:
        _metric_card("Admin emails", len(get_admin_emails()), "Configured in secrets")

    with c3:
        _metric_card("Privileged domains", len(get_privileged_domains()), "Domain-based access")

    with c4:
        _metric_card("Permissions", len(FEATURE_KEYS), "Centralized registry")

    st.markdown("### Role access summary")

    summary_rows = []

    for role in ROLES:
        summary_rows.append(
            {
                "Role": role.capitalize(),
                "Enabled permissions": _role_enabled_count(config, role),
                "Total permissions": len(FEATURE_KEYS),
                "Regions scope": "All" if not config.get(role, {}).get("regions") else len(config[role]["regions"]),
                "Countries scope": "All" if not config.get(role, {}).get("countries") else len(config[role]["countries"]),
                "Years scope": "All" if not config.get(role, {}).get("years") else len(config[role]["years"]),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("### Permission matrix")
    st.caption("This matrix gives a quick professional overview of Guest, Viewer and Privileged access.")

    matrix_df = _build_permission_matrix(config)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)


def _render_roles_tab(config: dict):
    st.markdown("### Configure role permissions")

    role = st.selectbox(
        "Select role",
        ROLES,
        format_func=lambda x: x.capitalize(),
        key="admin_role_selector",
    )

    config = normalize_access_config(config)
    features = config[role]["features"]

    st.markdown(
        """
        <div class="admin-info">
            Permissions are grouped from a single central registry. This avoids duplicate definitions across the admin page and access-control backend.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

    with col_a:
        if st.button("Enable default preset", use_container_width=True):
            default_config = default_access_config()
            config[role]["features"] = default_config[role]["features"]
            if save_access_config(config):
                _clear_app_cache()
                st.success(f"{role.capitalize()} preset restored.")
                st.rerun()

    with col_b:
        if role != "guest":
            if st.button("Enable all permissions", use_container_width=True):
                config[role]["features"] = {key: True for key in FEATURE_KEYS}
                for locked_key in LOCKED_FALSE.get(role, set()):
                    config[role]["features"][locked_key] = False
                if save_access_config(config):
                    _clear_app_cache()
                    st.success(f"All allowed permissions enabled for {role}.")
                    st.rerun()
        else:
            st.button("Enable all permissions", disabled=True, use_container_width=True)

    with col_c:
        if st.button("Disable all optional permissions", use_container_width=True):
            config[role]["features"] = {key: False for key in FEATURE_KEYS}
            for locked_key in LOCKED_FALSE.get(role, set()):
                config[role]["features"][locked_key] = False
            if save_access_config(config):
                _clear_app_cache()
                st.success(f"Permissions disabled for {role}.")
                st.rerun()

    groups = _feature_groups()

    for group_name, keys in groups.items():
        enabled_in_group = sum(1 for key in keys if features.get(key, False))

        with st.expander(f"{group_name} ({enabled_in_group}/{len(keys)} enabled)", expanded=group_name == "Core access"):
            group_cols = st.columns(2)

            for index, feature_key in enumerate(keys):
                with group_cols[index % 2]:
                    disabled = feature_key in LOCKED_FALSE.get(role, set())

                    features[feature_key] = st.checkbox(
                        _feature_label(feature_key),
                        value=bool(features.get(feature_key, False)),
                        key=f"feature_{role}_{feature_key}",
                        disabled=disabled,
                    )

                    if disabled:
                        features[feature_key] = False

    config[role]["features"] = features

    save_col, reset_col, download_col = st.columns(3)

    with save_col:
        if st.button("💾 Save role permissions", type="primary", use_container_width=True):
            if save_access_config(config):
                _clear_app_cache()
                st.success("Role permissions saved.")
                st.rerun()

    with reset_col:
        if st.button("↩ Reset all roles", use_container_width=True):
            if reset_access_config():
                _clear_app_cache()
                st.success("All roles reset to default permissions.")
                st.rerun()

    with download_col:
        st.download_button(
            "⬇ Download config",
            data=json.dumps(config, indent=2),
            file_name="eusee_access_config.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_dashboard_tab(config: dict):
    st.markdown("### Dashboard visibility manager")
    st.caption("Quickly control major dashboard areas without scrolling through every chart permission.")
    st.info("Visualization Map is locked to Admin users only and is intentionally not configurable for Guest, Viewer, or Privileged roles.")

    role = st.selectbox(
        "Configure dashboard visibility for role",
        ROLES,
        format_func=lambda x: x.capitalize(),
        key="dashboard_visibility_role",
    )

    features = config[role]["features"]

    section_map = {
        "Overview": [
            "view_overview",
            "view_coverage_monitored_countries",
            "view_monitored_countries_value",
            "view_chart_overview_alert_type",
            "view_chart_overview_enabling_principles",
            "view_chart_overview_regions",
            "view_chart_overview_countries",
        ],
        "Negative Alerts": [
            "view_negative_alerts",
            "view_chart_negative_restrictive_actors",
            "view_chart_negative_affected_actors",
            "view_chart_negative_restrictive_mechanisms",
            "view_chart_negative_event_types",
            "view_chart_negative_alert_types",
            "view_chart_negative_enabling_principles",
        ],
        "Analytical Flow Panels": [
            "view_analytical_flow_panel",
            "view_chart_heatmap_actor_mechanism",
            "view_chart_heatmap_subject_mechanism",
            "view_chart_heatmap_actor_subject",
            "view_chart_sankey_flow",
        ],
        "Data and exports": [
            "view_data_table",
            "download_data",
        ],
        "AI Copilot": [
            "use_ai_copilot",
            "view_chart_ai_copilot_plots",
        ],
        "User Manual": [
            "view_user_manual",
        ],
    }

    cols = st.columns(2)

    for idx, (section, keys) in enumerate(section_map.items()):
        with cols[idx % 2]:
            with st.container(border=True):
                enabled_count = sum(1 for key in keys if features.get(key, False))
                st.markdown(f"#### {section}")
                st.caption(f"{enabled_count}/{len(keys)} permissions enabled")

                enable_all = st.checkbox(
                    f"Enable {section}",
                    value=enabled_count == len(keys),
                    key=f"section_toggle_{role}_{section}",
                )

                for key in keys:
                    if key in LOCKED_FALSE.get(role, set()):
                        features[key] = False
                    else:
                        features[key] = bool(enable_all)

                with st.expander("Included permissions", expanded=False):
                    for key in keys:
                        st.write(("✅ " if features.get(key, False) else "— ") + _feature_label(key))

    config[role]["features"] = features

    if st.button("💾 Save dashboard visibility", type="primary", use_container_width=True):
        if save_access_config(config):
            _clear_app_cache()
            st.success("Dashboard visibility saved.")
            st.rerun()


def _render_scope_tab(config: dict, data=None):
    st.markdown("### Data scope")
    st.caption("Leave selections empty to allow all available values for that role.")

    role = st.selectbox(
        "Configure data scope for role",
        ROLES,
        format_func=lambda x: x.capitalize(),
        key="scope_role",
    )

    config = normalize_access_config(config)

    regions, countries, years = [], [], []

    if data is not None and not getattr(data, "empty", True):
        if "region" in data.columns:
            regions = sorted(data["region"].dropna().astype(str).unique())

        if "alert-country" in data.columns:
            countries = sorted(data["alert-country"].dropna().astype(str).unique())

        if "year" in data.columns:
            years = sorted([int(y) for y in pd.to_numeric(data["year"], errors="coerce").dropna().unique()])

    c1, c2, c3 = st.columns(3)

    with c1:
        config[role]["regions"] = st.multiselect(
            "Allowed regions",
            regions,
            default=[x for x in config[role].get("regions", []) if x in regions],
            key=f"persist_regions_{role}",
        )

    with c2:
        config[role]["countries"] = st.multiselect(
            "Allowed countries",
            countries,
            default=[x for x in config[role].get("countries", []) if x in countries],
            key=f"persist_countries_{role}",
        )

    with c3:
        config[role]["years"] = st.multiselect(
            "Allowed years",
            years,
            default=[x for x in config[role].get("years", []) if x in years],
            key=f"persist_years_{role}",
        )

    if st.button("💾 Save data scope", use_container_width=True, type="primary"):
        if save_access_config(config):
            _clear_app_cache()
            st.success("Data scope saved.")
            st.rerun()

    with st.expander("Current role scope JSON", expanded=False):
        st.json(config[role])


def _render_users_tab():
    st.markdown("### Access identities")
    st.caption("Admins are configured by email. Privileged users can also be assigned by email domain.")

    rows = []

    for email in get_admin_emails():
        rows.append(
            {
                "Identity": email,
                "Effective role": "Admin",
                "Source": "[auth].admin_emails",
                "Access type": "Direct email",
            }
        )

    for domain in get_privileged_domains():
        rows.append(
            {
                "Identity": f"*@{domain}",
                "Effective role": "Privileged",
                "Source": "[access].privileged_domains",
                "Access type": "Domain rule",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No admin emails or privileged domains configured.")

    st.markdown("### Current session")
    st.json(
        {
            "current_email": get_current_email(),
            "current_role": get_current_role(),
            "is_admin": is_admin(),
            "timestamp": datetime.now().isoformat(),
        }
    )

    with st.expander("Secrets format reference", expanded=False):
        st.code(
            """
[auth]
admin_emails = ["kyalorichard11@gmail.com"]

[access]
privileged_domains = ["cgiar.org", "icarda.org"]

[access_control]
firestore_collection = "dashboard_settings"
firestore_document = "access_control"

[firebase_admin]
type = "service_account"
project_id = "YOUR_PROJECT_ID"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n"
client_email = "YOUR_FIREBASE_CLIENT_EMAIL"
client_id = "YOUR_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "YOUR_CLIENT_CERT_URL"
""",
            language="toml",
        )


def _render_system_tab(config: dict):
    st.markdown("### System diagnostics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _metric_card("Firestore config", "Active", get_access_config_path())

    with c2:
        _metric_card("Current role", get_current_role().capitalize(), get_current_email() or "No email")

    with c3:
        _metric_card("AI permission", "Yes" if has_permission("use_ai_copilot") else "No", "Current session")

    with c4:
        _metric_card("Download permission", "Yes" if has_permission("download_data") else "No", "Current session")

    st.markdown("### Loaded access configuration")

    with st.expander("View raw access config", expanded=False):
        st.json(config)

    st.markdown("### Maintenance")

    st.warning(
        "Use reset only if permissions are corrupted or you intentionally want to restore Guest, Viewer and Privileged defaults."
    )

    if st.button("🧹 Reset access config and rebuild permissions", type="primary", use_container_width=True):
        if reset_access_config():
            _clear_app_cache()
            st.success("Access config reset successfully.")
            st.rerun()


def render_admin_page(data=None):
    if not is_admin():
        st.error("Access restricted. This page is only available to configured admin emails.")
        st.stop()

    inject_admin_css()

    config = normalize_access_config(load_access_config())

    _render_header()

    tab_overview, tab_roles, tab_dashboard, tab_scope, tab_users, tab_system = st.tabs(
        [
            "Overview",
            "Roles",
            "Dashboard visibility",
            "Data scope",
            "Users",
            "System",
        ]
    )

    with tab_overview:
        _render_overview_tab(config)

    with tab_roles:
        _render_roles_tab(config)

    with tab_dashboard:
        _render_dashboard_tab(config)

    with tab_scope:
        _render_scope_tab(config, data=data)

    with tab_users:
        _render_users_tab()

    with tab_system:
        _render_system_tab(config)

    st.markdown(
        """
        <div class="admin-footer">
            © 2026 EU SEE Project. Administration Center.
        </div>
        """,
        unsafe_allow_html=True,
    )