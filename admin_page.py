from __future__ import annotations

import json
import re
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
    get_temporary_shared_account,
    is_temporary_shared_account,
    can_bypass_email_verification,
    has_permission,
    is_admin,
    load_access_config,
    normalize_access_config,
    reset_access_config,
    save_access_config,
)


ANALYTICAL_FLOW_PARENT = "view_analytical_flow_panel"

ANALYTICAL_FLOW_CHILDREN = [
    "view_chart_heatmap_actor_mechanism",
    "view_chart_heatmap_subject_mechanism",
    "view_chart_heatmap_actor_subject",
    "view_chart_sankey_flow",
    "view_chart_negative_flow_diagram",
    "view_chart_negative_key_links",
    "view_chart_negative_follow_pathway",
    "view_chart_negative_top_n_selector",
    "view_chart_negative_detail_level",
]


def _clear_app_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

    try:
        st.cache_resource.clear()
    except Exception:
        pass


def _clean_domain(domain: str) -> str:
    domain = str(domain or "").strip().lower()
    domain = domain.replace("https://", "").replace("http://", "")
    domain = domain.replace("www.", "")
    domain = domain.replace("@", "")
    domain = domain.split("/")[0].strip()
    return domain


def _is_valid_domain(domain: str) -> bool:
    if not domain:
        return False

    if " " in domain:
        return False

    if domain.startswith(".") or domain.endswith("."):
        return False

    pattern = r"^(?!-)[a-z0-9-]+(\.[a-z0-9-]+)+$"
    return bool(re.match(pattern, domain))


def _get_config_privileged_domains(config: dict) -> list[str]:
    domains = config.get("privileged_domains")

    if domains is None:
        domains = get_privileged_domains()

    return sorted({_clean_domain(d) for d in domains if _clean_domain(d)})


def _save_config_privileged_domains(config: dict, domains: list[str]) -> bool:
    config["privileged_domains"] = sorted({_clean_domain(d) for d in domains if _clean_domain(d)})
    config = _sanitize_access_config(config) if "_sanitize_access_config" in globals() else config
    return save_access_config(config)


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

        .admin-warning {
            background: #fff8e6;
            border: 1px solid #fedf89;
            border-radius: 13px;
            padding: 11px 13px;
            color: #93370d;
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
                Manage role permissions, data scope, privileged domains and dashboard governance from one control center.
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


def _analytical_keys() -> set[str]:
    return {ANALYTICAL_FLOW_PARENT, *[key for key in ANALYTICAL_FLOW_CHILDREN if key in FEATURE_KEYS]}


def _analytical_child_keys() -> list[str]:
    return [key for key in ANALYTICAL_FLOW_CHILDREN if key in FEATURE_KEYS]


def _is_locked_for_role(role: str, feature_key: str) -> bool:
    # Analytical Flow Panels and its children are intentionally available to every role.
    # Children are controlled only by the parent dependency, not by LOCKED_FALSE.
    if feature_key in _analytical_keys():
        return False

    return feature_key in LOCKED_FALSE.get(role, set())


def _sanitize_role_features(role: str, features: dict | None) -> dict:
    raw_features = features if isinstance(features, dict) else {}

    clean_features = {
        key: bool(raw_features.get(key, False))
        for key in FEATURE_KEYS
    }

    # Enforce locked permissions for ordinary permissions only. Analytical permissions
    # remain selectable for all roles, subject to the parent being enabled first.
    for locked_key in LOCKED_FALSE.get(role, set()):
        if locked_key in clean_features and locked_key not in _analytical_keys():
            clean_features[locked_key] = False

    parent_enabled = bool(clean_features.get(ANALYTICAL_FLOW_PARENT, False))

    if not parent_enabled:
        for child_key in _analytical_child_keys():
            clean_features[child_key] = False

    return clean_features


def _sanitize_access_config(config: dict) -> dict:
    config = normalize_access_config(config)

    for role in ROLES:
        config.setdefault(role, {})
        config[role]["features"] = _sanitize_role_features(
            role,
            config.get(role, {}).get("features", {}),
        )
        config[role]["regions"] = list(config[role].get("regions", []) or [])
        config[role]["countries"] = list(config[role].get("countries", []) or [])
        config[role]["years"] = list(config[role].get("years", []) or [])

    config["privileged_domains"] = _get_config_privileged_domains(config)

    return config


def _role_enabled_count(config: dict, role: str) -> int:
    features = _sanitize_role_features(role, config.get(role, {}).get("features", {}))
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
            features = _sanitize_role_features(role, config.get(role, {}).get("features", {}))
            row[role.capitalize()] = "✅" if features.get(key, False) else "—"

        rows.append(row)

    return pd.DataFrame(rows)


def _enforce_analytical_flow_dependency(features: dict) -> dict:
    # Backward-compatible wrapper used by existing app imports/calls.
    clean_features = dict(features or {})

    if not bool(clean_features.get(ANALYTICAL_FLOW_PARENT, False)):
        for child_key in _analytical_child_keys():
            clean_features[child_key] = False

    return clean_features


def _render_overview_tab(config: dict):
    privileged_domains = _get_config_privileged_domains(config)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _metric_card("Current role", get_current_role().capitalize(), get_current_email() or "Not signed in")

    with c2:
        _metric_card("Admin emails", len(get_admin_emails()), "Configured in secrets")

    with c3:
        _metric_card("Privileged domains", len(privileged_domains), "Admin-managed domain access")

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
    st.caption("This matrix gives a quick overview of Guest, Viewer and Privileged access.")

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

    config = _sanitize_access_config(config)
    features = _sanitize_role_features(role, config[role]["features"])

    st.markdown(
        """
        <div class="admin-info">
            Roles are the single editable source of truth for dashboard visibility and functionality.
            Analytical Flow Panels is placed under Core access. Analytical charts remain disabled until that parent permission is enabled.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

    with col_a:
        if st.button("Enable default preset", use_container_width=True):
            default_config = default_access_config()
            config[role]["features"] = _sanitize_role_features(role, default_config[role]["features"])

            config = _sanitize_access_config(config)

            if save_access_config(config):
                _clear_app_cache()
                st.success(f"{role.capitalize()} preset restored.")
                st.rerun()

    with col_b:
        if st.button("Enable all permissions", use_container_width=True):
            config[role]["features"] = {key: True for key in FEATURE_KEYS}

            for locked_key in LOCKED_FALSE.get(role, set()):
                if locked_key in FEATURE_KEYS and locked_key not in _analytical_keys():
                    config[role]["features"][locked_key] = False

            config[role]["features"][ANALYTICAL_FLOW_PARENT] = True

            for child_key in ANALYTICAL_FLOW_CHILDREN:
                if child_key in FEATURE_KEYS:
                    config[role]["features"][child_key] = True

            config = _sanitize_access_config(config)

            if save_access_config(config):
                _clear_app_cache()
                st.success(f"All allowed permissions enabled for {role}.")
                st.rerun()

    with col_c:
        if st.button("Disable all optional permissions", use_container_width=True):
            config[role]["features"] = {key: False for key in FEATURE_KEYS}

            for locked_key in LOCKED_FALSE.get(role, set()):
                if locked_key in FEATURE_KEYS and locked_key not in _analytical_keys():
                    config[role]["features"][locked_key] = False

            config = _sanitize_access_config(config)

            if save_access_config(config):
                _clear_app_cache()
                st.success(f"Permissions disabled for {role}.")
                st.rerun()

    groups = _feature_groups()
    analytical_set = {ANALYTICAL_FLOW_PARENT, *ANALYTICAL_FLOW_CHILDREN}

    # ------------------------------------------------------------------
    # CORE ACCESS
    # Analytical Flow Panels parent is rendered here for every role.
    # It is intentionally NOT controlled by LOCKED_FALSE so every role can
    # enable the parent before selecting analytical charts.
    # ------------------------------------------------------------------
    core_keys = [
        key for key in groups.get("Core access", [])
        if key not in analytical_set
    ]

    core_enabled_count = sum(1 for key in core_keys if features.get(key, False))
    if features.get(ANALYTICAL_FLOW_PARENT, False):
        core_enabled_count += 1

    with st.expander(
        f"Core access ({core_enabled_count}/{len(core_keys) + 1} enabled)",
        expanded=True,
    ):
        core_cols = st.columns(2)

        for index, feature_key in enumerate(core_keys):
            with core_cols[index % 2]:
                disabled = _is_locked_for_role(role, feature_key)

                features[feature_key] = st.checkbox(
                    _feature_label(feature_key),
                    value=bool(features.get(feature_key, False)),
                    key=f"feature_{role}_{feature_key}",
                    disabled=disabled,
                )

                if disabled:
                    features[feature_key] = False

        parent_col_index = len(core_keys) % 2
        with core_cols[parent_col_index]:
            parent_enabled = st.checkbox(
                _feature_label(ANALYTICAL_FLOW_PARENT),
                value=bool(features.get(ANALYTICAL_FLOW_PARENT, False)),
                key=f"feature_{role}_{ANALYTICAL_FLOW_PARENT}",
                disabled=False,
                help="Enable this parent permission before selecting any analytical chart.",
            )

        features[ANALYTICAL_FLOW_PARENT] = parent_enabled

    # ------------------------------------------------------------------
    # ANALYTICAL CHARTS
    # Child analytical chart permissions are rendered in their own pane.
    # They remain disabled until the Core access parent is enabled.
    # ------------------------------------------------------------------
    analytical_chart_keys = [
        key for key in ANALYTICAL_FLOW_CHILDREN
        if key in FEATURE_KEYS
    ]

    enabled_analytical_count = sum(
        1 for key in analytical_chart_keys
        if features.get(key, False) and features.get(ANALYTICAL_FLOW_PARENT, False)
    )

    with st.expander(
        f"Analytical Charts ({enabled_analytical_count}/{len(analytical_chart_keys)} enabled)",
        expanded=True,
    ):
        if not features.get(ANALYTICAL_FLOW_PARENT, False):
            st.markdown(
                """
                <div class="admin-warning">
                    Analytical chart permissions are disabled. Enable Analytical Flow Panels under Core access first.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="admin-info">
                    Analytical Flow Panels is enabled. You can now select individual analytical charts.
                </div>
                """,
                unsafe_allow_html=True,
            )

        analytical_cols = st.columns(2)

        for index, feature_key in enumerate(analytical_chart_keys):
            with analytical_cols[index % 2]:
                child_enabled = st.checkbox(
                    _feature_label(feature_key),
                    value=bool(features.get(feature_key, False)) if features.get(ANALYTICAL_FLOW_PARENT, False) else False,
                    key=f"feature_{role}_{feature_key}",
                    disabled=not features.get(ANALYTICAL_FLOW_PARENT, False),
                    help="Enable Analytical Flow Panels under Core access first."
                    if not features.get(ANALYTICAL_FLOW_PARENT, False)
                    else None,
                )

                features[feature_key] = child_enabled if features.get(ANALYTICAL_FLOW_PARENT, False) else False

    # ------------------------------------------------------------------
    # OTHER PERMISSION GROUPS
    # Render all remaining groups, excluding Core access and analytical keys.
    # ------------------------------------------------------------------
    for group_name, keys in groups.items():
        if group_name == "Core access":
            continue

        normal_keys = [key for key in keys if key not in analytical_set]

        if not normal_keys:
            continue

        enabled_in_group = sum(1 for key in normal_keys if features.get(key, False))

        with st.expander(
            f"{group_name} ({enabled_in_group}/{len(normal_keys)} enabled)",
            expanded=False,
        ):
            group_cols = st.columns(2)

            for index, feature_key in enumerate(normal_keys):
                with group_cols[index % 2]:
                    disabled = _is_locked_for_role(role, feature_key)

                    features[feature_key] = st.checkbox(
                        _feature_label(feature_key),
                        value=bool(features.get(feature_key, False)),
                        key=f"feature_{role}_{feature_key}",
                        disabled=disabled,
                    )

                    if disabled:
                        features[feature_key] = False

    features = _enforce_analytical_flow_dependency(features)
    config[role]["features"] = features

    save_col, reset_col, download_col = st.columns(3)

    with save_col:
        if st.button("💾 Save role permissions", type="primary", use_container_width=True):
            config[role]["features"] = _sanitize_role_features(role, config[role]["features"])
            config = _sanitize_access_config(config)

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


def _render_dashboard_visibility_tab(config: dict):
    st.markdown("### Dashboard visibility overview")
    st.caption("Read-only view. Edit permissions from the Roles tab only.")

    st.info(
        "Dashboard visibility is now controlled only from Roles to avoid conflicting functionality."
    )

    role = st.selectbox(
        "View dashboard visibility for role",
        ROLES,
        format_func=lambda x: x.capitalize(),
        key="dashboard_visibility_readonly_role",
    )

    features = _sanitize_role_features(role, config[role]["features"])

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
            "view_chart_negative_flow_diagram",
            "view_chart_negative_key_links",
            "view_chart_negative_follow_pathway",
            "view_chart_negative_top_n_selector",
            "view_chart_negative_detail_level",
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

    rows = []

    for section, keys in section_map.items():
        for key in keys:
            if key in FEATURE_KEYS:
                rows.append(
                    {
                        "Section": section,
                        "Permission": _feature_label(key),
                        "Key": key,
                        "Enabled": "✅ Yes" if features.get(key, False) else "— No",
                    }
                )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_scope_tab(config: dict, data=None):
    st.markdown("### Data scope")
    st.caption("Leave selections empty to allow all available values for that role.")

    role = st.selectbox(
        "Configure data scope for role",
        ROLES,
        format_func=lambda x: x.capitalize(),
        key="scope_role",
    )

    config = _sanitize_access_config(config)

    regions, countries, years = [], [], []

    if data is not None and not getattr(data, "empty", True):
        if "region" in data.columns:
            regions = sorted(data["region"].dropna().astype(str).unique())

        if "alert-country" in data.columns:
            countries = sorted(data["alert-country"].dropna().astype(str).unique())

        if "year" in data.columns:
            years = sorted(
                [int(y) for y in pd.to_numeric(data["year"], errors="coerce").dropna().unique()]
            )

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
        config = _sanitize_access_config(config)

        if save_access_config(config):
            _clear_app_cache()
            st.success("Data scope saved.")
            st.rerun()

    with st.expander("Current role scope JSON", expanded=False):
        st.json(config[role])


def _render_users_tab(config: dict):
    st.markdown("### Access identities")
    st.caption(
        "Admins and the temporary shared account are configured in secrets. "
        "Additional privileged users can be managed by email domain."
    )

    config = _sanitize_access_config(config)
    privileged_domains = _get_config_privileged_domains(config)

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

    temporary_account = get_temporary_shared_account()
    if temporary_account.get("enabled") and temporary_account.get("email"):
        rows.append(
            {
                "Identity": temporary_account["email"],
                "Effective role": temporary_account.get("role", "viewer").capitalize(),
                "Source": "[temporary_shared_account]",
                "Access type": "Direct shared account",
            }
        )

    for domain in privileged_domains:
        rows.append(
            {
                "Identity": f"*@{domain}",
                "Effective role": "Privileged",
                "Source": "Admin-managed access config",
                "Access type": "Domain rule",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No admin emails or privileged domains configured.")

    st.markdown("### Manage privileged email domains")

    st.info(
        "Adding a domain gives Privileged access to users whose email ends with that domain, for example: user@icarda.org."
    )

    with st.container(border=True):
        new_domain = st.text_input(
            "Add privileged domain",
            placeholder="example: icarda.org",
            key="new_privileged_domain_input",
        )

        if st.button("➕ Add privileged domain", type="primary", use_container_width=True):
            cleaned = _clean_domain(new_domain)

            if not _is_valid_domain(cleaned):
                st.error("Enter a valid domain, for example: icarda.org")
            elif cleaned in privileged_domains:
                st.warning(f"{cleaned} is already assigned as privileged.")
            else:
                updated_domains = sorted(set(privileged_domains + [cleaned]))

                if _save_config_privileged_domains(config, updated_domains):
                    _clear_app_cache()
                    st.success(f"{cleaned} added as a privileged domain.")
                    st.rerun()

    if privileged_domains:
        st.markdown("#### Edit or remove existing domains")

        for idx, domain in enumerate(privileged_domains):
            c1, c2, c3 = st.columns([3, 1, 1])

            with c1:
                edited_domain = st.text_input(
                    "Domain",
                    value=domain,
                    label_visibility="collapsed",
                    key=f"edit_privileged_domain_{domain}_{idx}",
                )

            with c2:
                if st.button("Save", key=f"save_privileged_domain_{domain}_{idx}", use_container_width=True):
                    cleaned = _clean_domain(edited_domain)

                    if not _is_valid_domain(cleaned):
                        st.error("Enter a valid domain.")
                    elif cleaned != domain and cleaned in privileged_domains:
                        st.warning(f"{cleaned} already exists. Use a unique domain.")
                    else:
                        updated_domains = [
                            cleaned if existing_domain == domain else existing_domain
                            for existing_domain in privileged_domains
                        ]

                        updated_domains = sorted(set(updated_domains))

                        if _save_config_privileged_domains(config, updated_domains):
                            _clear_app_cache()
                            st.success("Privileged domain updated.")
                            st.rerun()

            with c3:
                if st.button("Remove", key=f"remove_privileged_domain_{domain}_{idx}", use_container_width=True):
                    updated_domains = [
                        existing_domain
                        for existing_domain in privileged_domains
                        if existing_domain != domain
                    ]

                    if _save_config_privileged_domains(config, updated_domains):
                        _clear_app_cache()
                        st.success(f"{domain} removed.")
                        st.rerun()

    st.markdown("### Current session")
    st.json(
        {
            "current_email": get_current_email(),
            "current_role": get_current_role(),
            "is_admin": is_admin(),
            "temporary_account_match": is_temporary_shared_account(),
            "bypass_email_verification": can_bypass_email_verification(),
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

[temporary_shared_account]
enabled = true
email = "dashboard.access@eusee.global"
role = "privileged"
bypass_email_verification = true

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

    config = _sanitize_access_config(load_access_config())

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
        _render_dashboard_visibility_tab(config)

    with tab_scope:
        _render_scope_tab(config, data=data)

    with tab_users:
        _render_users_tab(config)

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