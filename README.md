# EU SEE Dashboard

**A Streamlit-based data exploration, reporting, and AI-assisted analysis platform for the EU SEE project**

[EU SEE website](https://eusee.hivos.org/)

---

## Table of contents

1. [Overview](#1-overview)
2. [Key capabilities](#2-key-capabilities)
3. [Application architecture](#3-application-architecture)
4. [Repository structure](#4-repository-structure)
5. [Technology stack](#5-technology-stack)
6. [Data sources and expected files](#6-data-sources-and-expected-files)
7. [Installation](#7-installation)
8. [Configuration and secrets](#8-configuration-and-secrets)
9. [Running the application](#9-running-the-application)
10. [Using the dashboard](#10-using-the-dashboard)
11. [AI Copilot](#11-ai-copilot)
12. [Authentication and access control](#12-authentication-and-access-control)
13. [Administration workspace](#13-administration-workspace)
14. [Data processing and refresh behaviour](#14-data-processing-and-refresh-behaviour)
15. [Deployment considerations](#15-deployment-considerations)
16. [Troubleshooting](#16-troubleshooting)
17. [Security and operational guidance](#17-security-and-operational-guidance)
18. [Maintenance and contribution guidelines](#18-maintenance-and-contribution-guidelines)
19. [Known implementation boundaries](#19-known-implementation-boundaries)
20. [Citation and acknowledgement](#20-citation-and-acknowledgement)

---

## 1. Overview

The **EU SEE Dashboard** is an interactive application for exploring and analysing information about changes in the enabling environment for civil society, as documented by the EU SEE network. The application is implemented in Python using Streamlit and combines structured alert data, country-level Country Focus Report (CFR) scores, interactive visualisations, geospatial presentation, access management, and an AI Copilot.

The dashboard is designed to support evidence-based exploration of the records available in its underlying datasets. Users can filter information by geography, alert characteristics, enabling principles, impact, and time; inspect distributions and relationships; and, where their permissions allow, review or download data.

The AI Copilot is deliberately **dataset-grounded**. It uses a language model to interpret a question and select a structured query, while numerical results are calculated from the dashboard's in-memory pandas dataframes. It is not intended to act as an unrestricted general-purpose research assistant.

> **Interpretation note:** The dashboard documentation highlights that partner reporting practices differ between countries. Direct country comparisons using alert counts should therefore be treated with caution. Some alerts may also relate to more than one enabling environment principle.

## 2. Key capabilities

### 2.1 Alert exploration

- Interactive dashboard views for alert overview and negative-alert analysis.
- Global filters for region, country, alert type, enabling principle, alert impact, month, and year.
- Summary cards and distributions that respond to the active filters.
- Exploration of actors, affected subjects, mechanisms, and event types associated with negative alerts.
- Searchable data preview and data export controls where enabled by the user's permissions.

### 2.2 Analytical visualisations

The application uses Plotly for interactive charts and includes:

- Bar and stacked-bar charts.
- Distribution and comparison charts.
- Heatmaps for relationships between actors, affected subjects, and mechanisms.
- Sankey and flow visualisations.
- Top-N controls and analytical detail controls.
- A geospatial visualisation view, subject to access permissions.
- Responsive chart layouts and dashboard styling for smaller screens.

Chart percentages and counts should be interpreted in the context of the active filters and the source data. Refer to chart annotations and explanatory notes in the application for the relevant denominator or methodology.

### 2.3 Country Focus Report (CFR) scores

The CFR module presents country-level scores across the six enabling environment principles. The implementation:

- Loads a CFR CSV export.
- Validates required country, principle-score, and last-modified fields.
- Uses the source's **1–5 scoring scale**.
- Provides aggregated principle-score and over-time views.
- Provides country-level score tables and regional summaries.
- Uses country metadata to support country enrichment and presentation.

CFR scores are country-level records; they are not individual alert records. The Copilot can query the alert and CFR datasets separately or together when a question explicitly asks to relate them.

### 2.4 AI Copilot

- Conversational questions about the dashboard's alert and CFR data.
- Summary, count, grouping, trend, comparison, ranking, and record-retrieval workflows.
- Structured filters and topic searches.
- Plotly chart generation from returned grouped query results.
- Per-user conversation history persistence.
- A configurable OpenAI model.

The Copilot must not be used to infer causes, fill missing data, or answer questions that require information outside the authorized dashboard datasets.

### 2.5 Authentication and administration

- Firebase Authentication integration for account registration, login, password reset, and session handling.
- Browser-cookie session restoration.
- Guest, viewer, and privileged access roles, plus configured administrator accounts.
- Feature-level permissions and optional region, country, and year data scopes.
- Firestore-backed access configuration.
- An administrator workspace for managing roles, feature visibility, data scope, users/domains, and system settings.

---

## 3. Application architecture

The application is organised into five principal Python modules:

```text
                    ┌───────────────────────────┐
                    │       Streamlit UI        │
                    │         app.py            │
                    └─────────────┬─────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
    ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ Authentication │  │ Authorization &  │  │ Administration   │
    │ auth.py        │  │ data scope        │  │ admin_page.py    │
    │                │  │ authz.py          │  │                  │
    └────────────────┘  └──────────────────┘  └──────────────────┘
             │                    │
             ▼                    ▼
    ┌────────────────┐  ┌──────────────────┐
    │ Firebase Auth  │  │ Firestore access │
    │ and cookies    │  │ configuration    │
    └────────────────┘  └──────────────────┘

                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │       AI Copilot          │
                    │   eusee_ai_copilot.py     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             OpenAI Responses API        Local pandas query
             question interpretation     deterministic results
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit entry point; application layout, data loading, filters, dashboard tabs, visualisations, CFR module, user manual, and Copilot integration. |
| `auth.py` | Firebase client/admin initialisation, authentication UI and operations, cookie persistence, session restoration, logout, and per-user chat-history persistence. |
| `authz.py` | Role resolution, feature permissions, Firestore access configuration, and row-level data-scope filtering. |
| `admin_page.py` | Administrator interface for role configuration, dashboard visibility, data scopes, user/domain settings, and system diagnostics. |
| `eusee_ai_copilot.py` | Dataset-grounded AI workflow, query schema, pandas query execution, conversational history, and Copilot chart rendering. |

The dashboard imports the Copilot as a separate module and renders it in a Streamlit fragment to isolate many Copilot interactions from full dashboard reruns.

---

## 4. Repository structure

The following is the recommended layout for the application. The Python modules listed below correspond to the supplied implementation files; supporting data, documentation, and deployment files must also be included in the deployed repository or mounted volumes.

```text
eusee-dashboard/
├── app.py
├── auth.py
├── authz.py
├── admin_page.py
├── eusee_ai_copilot.py
├── requirements.txt
├── README.md
│
├── data/
│   └── ...                         # Optional source/intermediate data
│
├── exports/
│   ├── output_final.parquet        # Main alert dataset (required)
│   ├── countries_metadata.json     # Country metadata (expected)
│   └── CFR_Export_YYYY_MM_DD.csv   # CFR export (required for CFR features)
│
├── docs/
│   ├── EU_SEE_Dashboard_Quick_Start_Executive.pdf
│   └── EU SEE Dashboard user manual.pdf
│
└── .streamlit/
    └── secrets.toml                # Local secrets; do not commit
```

**Important:** The application uses `/exports` when that directory exists; otherwise it uses the repository's `exports/` directory. In container deployments, `/exports` can therefore be mounted as a persistent data volume.

The README describes the runtime expectations visible in the supplied application modules. It does not imply that every supporting file was included with the five Python source files.

---

## 5. Technology stack

| Component | Role in the application |
|---|---|
| Python | Application language and data-processing runtime. |
| Streamlit | Interactive web application framework and session/UI state. |
| pandas | Tabular data loading, filtering, aggregation, and deterministic Copilot queries. |
| NumPy | Numerical and data-processing support. |
| Plotly | Interactive charts and analytical visualisations. |
| PyArrow | Parquet support for the primary alert dataset (install a compatible Parquet engine). |
| Firebase Authentication / Pyrebase | User authentication and token operations. |
| Firebase Admin SDK | Administrative Firebase integration and Firestore access. |
| Cloud Firestore | Persistent role and feature-access configuration. |
| `extra-streamlit-components` | Browser cookie manager used for persistent authentication sessions. |
| OpenAI Python SDK | Copilot question interpretation and response generation through the Responses API. |
| Requests | HTTP requests used by authentication/token workflows. |

The main dashboard source also imports Paramiko. Confirm the deployed dependency set against the complete application and the environment in which it runs. Pin tested package versions in `requirements.txt`; the supplied Python files do not establish a definitive version lock.

### Suggested dependency baseline

Create a `requirements.txt` containing compatible, tested versions of the packages used by the application. A starting package list is:

```text
streamlit
pandas
numpy
plotly
pyarrow
openai
requests
paramiko
pyrebase4
firebase-admin
extra-streamlit-components
```

This is a **package checklist, not a tested lockfile**. Resolve and pin versions in the target deployment environment, then validate imports and application behaviour before release.

---

## 6. Data sources and expected files

### 6.1 Main alerts dataset

The main dashboard expects:

```text
exports/output_final.parquet
```

The application reads this file with `pandas.read_parquet()` and caches the result for up to one hour. It expects alert-related fields including:

- `alert-country`
- `alert-impact`
- `alert-type`
- `Actor of repression`
- `Subject of repression`
- `Mechanism of repression`
- `Type of event`

Other fields used by the dashboard and Copilot include region, enabling principle, year, and month. The exact available fields depend on the supplied Parquet export.

If a number of required alert fields are absent, the main loader may add empty columns for selected fields so the application can continue. This does not restore missing source information; charts and queries may consequently have limited results.

### 6.2 Country metadata

The application expects:

```text
exports/countries_metadata.json
```

This metadata is used in country enrichment and geospatial presentation. Ensure it is consistent with the country labels in the alert and CFR exports.

### 6.3 CFR export

The CFR loader searches for a dated CSV export matching:

```text
CFR_Export_YYYY_MM_DD.csv
```

Selection priority:

1. A valid path configured through the `EUSEE_CFR_CSV` environment variable.
2. The newest matching dated CFR export in the active export directory.

The date is derived from the filename for selecting the newest export. The CFR loader validates the presence of `Country`, the six principle score columns, and `Last Modified`. The exact principle column names are defined in `app.py`; new exports should preserve the expected schema or the loader will report missing columns.

The configured score range is 1–5. Values outside that range trigger a warning; the application does not silently rewrite the source scores.

### 6.4 Documentation PDFs

The application references these files:

```text
docs/EU_SEE_Dashboard_Quick_Start_Executive.pdf
docs/EU SEE Dashboard user manual.pdf
```

They are used by the dashboard's PDF download helpers. Include the files in the deployed `docs/` directory if those download links are expected to work.

---

## 7. Installation

### 7.1 Prerequisites

- A supported Python runtime compatible with the pinned Streamlit and Firebase dependencies.
- Access to the dashboard repository.
- The main alert Parquet export.
- Country metadata and CFR export if those modules are required.
- Firebase project credentials for authentication and administrative access.
- An OpenAI API key if the AI Copilot is to be enabled.
- Access to the target deployment environment and its persistent storage.

### 7.2 Clone or prepare the repository

```bash
git clone <repository-url>
cd eusee-dashboard
```

If the source files are being deployed manually, place the five application modules in the repository root using the filenames shown in the repository structure.

### 7.3 Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

### 7.4 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you do not yet have a requirements file, create one from the package checklist in [Technology stack](#5-technology-stack), resolve compatible versions, and pin them before production deployment.

### 7.5 Add the data and documentation files

Ensure the expected directories and files exist:

```text
exports/output_final.parquet
exports/countries_metadata.json
exports/CFR_Export_YYYY_MM_DD.csv
docs/EU_SEE_Dashboard_Quick_Start_Executive.pdf
docs/EU SEE Dashboard user manual.pdf
```

The CFR CSV and documentation PDFs are needed only for the corresponding features. The main alert Parquet file is required for the primary dashboard.

---

## 8. Configuration and secrets

The application reads sensitive configuration from Streamlit secrets and selected environment variables. Do not hard-code credentials in Python files or commit secrets to source control.

Create `.streamlit/secrets.toml` for local development. Configure the values for your own Firebase project and deployment; the example below is a **template only**.

```toml
[openai]
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENAI_MODEL = "gpt-4.1-mini"

[firebase]
apiKey = "YOUR_FIREBASE_WEB_API_KEY"
authDomain = "YOUR_FIREBASE_AUTH_DOMAIN"
projectId = "YOUR_FIREBASE_PROJECT_ID"
storageBucket = "YOUR_FIREBASE_STORAGE_BUCKET"
messagingSenderId = "YOUR_FIREBASE_MESSAGING_SENDER_ID"
appId = "YOUR_FIREBASE_APP_ID"
databaseURL = ""

[firebase_admin]
type = "service_account"
project_id = "YOUR_FIREBASE_PROJECT_ID"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
client_email = "YOUR_SERVICE_ACCOUNT_EMAIL"
client_id = "YOUR_SERVICE_ACCOUNT_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "YOUR_CLIENT_CERTIFICATE_URL"

[auth]
admin_emails = ["admin@example.org"]

[access]
privileged_domains = ["example.org"]

[access_control]
firestore_collection = "dashboard_settings"
firestore_document = "access_control"

[temporary_shared_account]
enabled = false
email = ""
role = "viewer"
bypass_email_verification = false

[chatbot]
history_dir = ".eusee_chat_history"
```

### 8.1 Configuration reference

| Setting | Purpose |
|---|---|
| `[openai].OPENAI_API_KEY` | API key used by the AI Copilot. |
| `[openai].OPENAI_MODEL` | Model name passed to the Copilot. The code defaults to `gpt-4.1-mini` when no model is supplied. |
| `[firebase]` | Firebase client configuration used by the authentication module. |
| `[firebase_admin]` | Firebase service-account configuration used for administrative Firebase operations. |
| `[auth].admin_emails` | Explicit administrator email addresses. |
| `[access].privileged_domains` | Email domains used by the role-resolution logic for privileged access. |
| `[access_control]` | Firestore collection/document location for access configuration. |
| `[temporary_shared_account]` | Optional explicitly configured shared account, including role and verification-bypass setting. Keep disabled unless there is a documented operational need. |
| `[chatbot].history_dir` | Server-side directory used by `auth.py` to persist per-user chat history. |

The access-control module can also use Firebase service-account credentials from the supported Firebase secrets structure. Keep the credential format consistent across `auth.py` and `authz.py`.

### 8.2 Environment variables

| Variable | Purpose |
|---|---|
| `EUSEE_CFR_CSV` | Optional explicit path to the CFR CSV file. If valid, it takes precedence over automatic dated-file discovery. |

### 8.3 Firebase setup checklist

1. Create or select the Firebase project used by the deployment.
2. Enable the Firebase Authentication methods required by the login and registration flows.
3. Register the application and populate the `[firebase]` client settings.
4. Create a service account with the minimum permissions required for Firestore access.
5. Populate `[firebase_admin]` with the service-account credentials.
6. Confirm that the Firestore database is available.
7. Confirm the configured access-control collection and document.
8. Add administrator email addresses and approved partner domains.
9. Test registration, login, password reset, session restoration, and logout using non-production test accounts.

The dashboard should not be exposed with production credentials until authentication and Firestore permissions have been tested.

---

## 9. Running the application

From the repository root, run:

```bash
streamlit run app.py
```

Streamlit will print the local application URL in the terminal. Open that URL in a browser.

For a non-default port:

```bash
streamlit run app.py --server.port 8502
```

For a deployment, configure the process manager or platform to run the same Streamlit entry point and make the data, documentation, and secrets available to the application process.

---

## 10. Using the dashboard

### 10.1 Global filters

The global filters are used to refine the active alert dataframe. The implemented filter dimensions include:

| Filter | Description |
|---|---|
| Region | Regional grouping used by the dashboard. |
| Country | Country associated with the alert record. |
| Alert type | Type/category of alert. |
| Enabling principle | One or more enabling environment principles associated with the alert. |
| Alert impact | Impact classification, including positive and negative. |
| Month | Month associated with the alert date. |
| Year | Year associated with the alert date. |

Selecting or changing filters updates the dashboard's active filtered dataset and the views built from it. A result shown under a restricted data scope is further constrained by the user's configured access.

### 10.2 Alert Overview

The Alert Overview presents high-level distributions across the filtered records, including alert type, enabling principles, region, and country. Summary cards and charts should be interpreted using the currently selected filters.

### 10.3 Country Focus Report (CFR) Scores

The CFR view provides a country-level perspective across the six enabling environment principles. It includes aggregated principle scores, scores over time, and country-level tables. The CFR score scale is 1–5.

### 10.4 Negative Alerts Analysis

The Negative Alerts Analysis focuses on records whose `alert-impact` is classified as `Negative`. It supports analysis of:

- Actors of repression.
- Subjects/actors affected.
- Mechanisms of repression.
- Types of events.
- Alert types.
- Enabling principles.
- Relationships between these dimensions through heatmaps and flow visualisations.

Some analytical panels include Top-N and detail-level controls. These controls affect the presentation of the selected records and should not be interpreted as changing the underlying source dataset.

### 10.5 Visualization Map

The Visualization Map presents geographic information for records where the necessary country/geospatial metadata is available. Access to the map is permission-controlled. A missing country code or incomplete country metadata can limit the records that can be mapped.

### 10.6 Data preview and downloads

The application contains a searchable data preview and download helpers. Availability is controlled by feature permissions. Exported data should be handled according to the project's data governance requirements and any restrictions associated with the user's role or data scope.

### 10.7 User Manual and feedback

The dashboard includes an in-app user guide and references PDF documentation in the `docs/` directory. A feedback link is also provided in the interface for questions, suggestions, and issue reports.

---

## 11. AI Copilot

### 11.1 Design principle

The Copilot separates language interpretation from data computation:

1. The user asks a question in natural language.
2. The model receives the available dataset schemas and recent conversation context.
3. The model selects a dataset and constructs a structured `query_dataset` request.
4. The application applies the request to the authorized pandas dataframe.
5. Numerical summaries and grouped results are calculated by pandas.
6. The model explains the returned evidence.
7. If a chart is requested and the result is suitable, the application renders a Plotly visualisation from the query output.

The language model does not directly calculate authoritative dashboard totals from unverified narrative assumptions. The application supplies the computed result for explanation.

### 11.2 Supported datasets

| Dataset selector | Contents |
|---|---|
| `alerts` | Alert-level records from the main dashboard dataset. |
| `cfr` | Country-level CFR scores. |
| `both` | Both datasets, when the user asks to relate or compare them. |

For questions about CFR scores, country scores, the six principles, Overall CFR, or CFR report years, the Copilot is instructed to use the CFR dataset. CFR data must not be described as individual alert records.

### 11.3 Supported query operations

| Operation | Purpose |
|---|---|
| `summary` | Summarise the filtered records and key fields. |
| `count` | Count records matching the query. |
| `group` | Aggregate records by one or more fields. |
| `trend` | Group data for time-oriented analysis. |
| `records` | Retrieve matching records, subject to the configured result limit. |

The query schema also supports metrics such as count, unique count, mean, and sum, where a suitable column exists.

### 11.4 Filtering and topic search

Structured filters support equality, inequality, containment, inclusion/exclusion, numeric comparisons, and ranges. The Copilot resolves common aliases such as country, region, impact, alert type, principle, actor, event type, date, year, and month to available dataframe columns.

For topic searches, the model is instructed to generate relevant spelling variants, synonyms, abbreviations, alternate names, and specific related concepts. It should avoid overly broad terms that could create false positives.

### 11.5 Charts

The Copilot can infer chart preferences from a question and create Plotly figures from grouped query results. Supported chart requests include, among others:

- Bar and grouped/stacked bar charts.
- Line and area charts.
- Pie and donut charts.
- Scatter and bubble charts.
- Histograms.
- Box and violin plots.
- Heatmaps and scatter-matrix views.
- Treemaps, sunbursts, funnels, and waterfall charts.

A chart is only produced when the query result contains suitable grouped categorical and numeric data. The application must not fabricate missing values to force a chart.

### 11.6 Scope and evidence rules

The Copilot is restricted to information in the authorized alerts and CFR datasets. It must:

- Query the dataset before giving factual answers.
- Use computed values returned by the query engine.
- Mention the number of records covered by a summary.
- State when required fields or records are unavailable.
- Distinguish a zero-result search from proof that the dataset contains no relevant information.
- Avoid inventing totals, percentages, trends, causes, countries, or categories.
- Decline questions outside the dashboard's data scope using the configured scope message.
- Append the EU SEE website footer to its answers.

**Recommended user practice:** Ask focused questions, specify a country or time period where relevant, and request a table or chart when a comparison would be clearer. Verify the selected filters and record counts before using results in a report.

### 11.7 Conversation history

The Copilot module maintains its own conversation history. The authentication module also provides server-side per-user chat-history persistence, storing messages under a user-specific key derived from the authenticated email. The authentication history helper retains up to 80 messages. The Copilot module's own local history helper retains up to 100 messages. These are distinct persistence mechanisms in the supplied implementation; deployment maintainers should confirm which history path is active for their user experience and avoid assuming the two stores are automatically identical.

---

## 12. Authentication and access control

### 12.1 Authentication flow

The authentication module integrates Firebase Authentication and a browser cookie manager. The application is designed to render the public-facing dashboard without forcing every visitor into a login page. Sign-in and registration are initiated through the dashboard's user privilege interface when a protected feature requires authentication.

The authentication module includes:

- Login.
- Registration.
- Password-reset workflow.
- Email verification handling.
- Firebase token refresh.
- Persistent browser session cookie.
- Session restoration after a browser refresh.
- Explicit logout and cookie deletion.

The authentication cookie is named `eusee_auth_session` and is configured for a 30-day lifetime in the supplied code. Actual browser persistence also depends on HTTPS, browser policy, and the deployment environment.

### 12.2 Roles

The authorization module defines these standard roles:

| Role | Intended access |
|---|---|
| Guest | Limited public/dashboard access according to the guest feature preset. |
| Viewer | Authenticated viewing access, including the AI Copilot in the supplied role preset. |
| Privileged | Broad dashboard and analytical access, subject to administrator-only restrictions. |
| Admin | Explicitly configured administrator email; administrator checks grant full feature access and permit the administration workspace. |

The role returned by the authorization logic is determined from administrator email configuration, temporary shared-account configuration, and privileged email domains. Users who do not match administrator or privileged-domain rules resolve to the viewer role when authenticated.

### 12.3 Feature-level permissions

The feature registry controls access to dashboard sections and individual visualisations. Examples include:

- Dashboard and overview access.
- Summary cards and monitored-country values.
- Negative-alert analysis and filters.
- Analytical flow panels and their individual charts.
- Map and geospatial visualisations.
- Data preview and downloads.
- AI Copilot and Copilot plots.
- User manual.
- Administration page.

The administrator interface can configure feature visibility by role. Some permissions are locked for particular roles in the authorization logic. Administrator-only capabilities include the administration page and the Negative Alerts filter panel.

### 12.4 Data scopes

The authorization module supports role-specific restrictions for:

- Regions.
- Countries.
- Years.

The `apply_data_scope()` function applies the configured scope to a dataframe for non-admin users. Administrators bypass this row-scope filtering. Configure and test data scopes carefully because they affect the records available to dashboard views and analysis.

### 12.5 Important shared-account warning

The code supports an optional temporary shared account. Shared accounts reduce individual accountability and should not be enabled by default. If a shared account is operationally necessary:

- Use a dedicated account rather than a staff member's personal account.
- Assign the minimum required role.
- Set a review/expiry date outside the application if required by policy.
- Do not enable email-verification bypass unless explicitly approved.
- Rotate credentials and disable the account when the access period ends.

---

## 13. Administration workspace

The administration workspace is restricted to configured administrator emails.

It contains six areas:

| Area | Main purpose |
|---|---|
| Overview | Summary of the current access configuration. |
| Roles | Configure role-level access. |
| Dashboard visibility | Enable or disable dashboard features and visualisations by role. |
| Data scope | Configure permitted regions, countries, and years. |
| Users | Manage privileged domains and related access settings exposed by the admin interface. |
| System | Review system/access diagnostics and reset the access configuration when necessary. |

Access settings are stored in Firestore using the configured access-control collection and document. If Firestore is unavailable, the authorization module can fall back to default access configuration for loading; however, saving configuration requires Firestore and should be verified after each administrative change.

### Administrative change procedure

1. Sign in using an explicitly configured administrator account.
2. Open the Admin workspace.
3. Review the current configuration before editing.
4. Change one role or scope at a time.
5. Save the configuration.
6. Test the affected permission using a separate test account with the target role.
7. Confirm that both allowed and restricted views behave as expected.
8. Record material permission changes according to the project's governance process.

Use the reset-access-config function only when the current permission configuration is corrupted or when a deliberate reset to role defaults is required.

---

## 14. Data processing and refresh behaviour

### 14.1 Main alert data

The main alert dataframe is loaded from Parquet and cached with a one-hour time-to-live. The application performs basic validation and normalisation, including required-column checks and country-label cleaning. The dashboard then creates an authoritative dataframe and applies user data-scope restrictions before presenting the filtered dashboard views.

### 14.2 Global filtering

The active dashboard filters are applied to the alert dataframe. The resulting filtered dataframe is stored in Streamlit session state for downstream visualisations and related UI components.

The global filter dimensions are applied as selections against the corresponding alert fields. Enabling-principle values may be multi-valued and are handled accordingly.

### 14.3 CFR data

The CFR loader caches data based on the source path and file modification metadata. The source is resolved from the environment variable or the newest dated export in the export directory. Country metadata is also used in the CFR load/enrichment process.

### 14.4 Cache invalidation

When replacing a dataset, ensure the running application sees the updated file and refreshes the relevant Streamlit cache. The dashboard uses Streamlit data/resource caching in several places. For an operational refresh, use the deployment's approved restart or cache-clear procedure if a new file is not reflected after the expected cache lifetime.

### 14.5 Data quality checks before publication

Before publishing a new export, verify:

- The Parquet file opens successfully.
- Required alert columns are present.
- Country labels are valid and consistent.
- Region, month, and year fields are populated as expected.
- Enabling-principle values use the expected labels and delimiters.
- Alert impact labels are consistent.
- The country metadata JSON is valid and aligned with the dataset.
- The CFR export includes all required fields and uses the 1–5 scale.
- The latest CFR filename follows the expected date pattern.

---

## 15. Deployment considerations

The application can be deployed in any environment that supports the required Python packages, Streamlit, persistent data files, and secure secret injection.

### 15.1 Runtime requirements

- Start the application with `streamlit run app.py`.
- Expose the configured Streamlit port through the hosting platform or reverse proxy.
- Configure HTTPS for production use, particularly for persistent authentication cookies.
- Provide the required secrets securely through the platform's secret manager.
- Mount or deploy the `exports/` directory and ensure the application process can read it.
- Include the `docs/` directory if the PDF downloads are required.
- Ensure the chatbot history directory is writable and persistent if history must survive container restarts.

### 15.2 Container and volume note

The application selects `/exports` when that path exists, otherwise it uses the local `exports/` directory. A container deployment should mount the persistent export volume at `/exports` or package the data under the repository's `exports/` directory.

For example, a conceptual Docker volume mapping could be:

```yaml
volumes:
  - ./exports:/exports
```

The actual Compose/Docker configuration must also define the application image, working directory, port, environment, secrets, and restart policy for the chosen hosting environment.

### 15.3 Persistent storage

Do not assume that a container's writable filesystem is durable. Persist at least:

- The main and CFR data exports if they are not supplied by an external data pipeline.
- Per-user chat-history files if history persistence is required.
- Any other operational artefacts explicitly written by the application.

### 15.4 Production release checklist

- [ ] All five Python modules are present and import correctly.
- [ ] Dependencies are pinned and installed.
- [ ] Main Parquet file is available and readable.
- [ ] Country metadata is present and valid.
- [ ] CFR export is available if CFR features are required.
- [ ] User manual and executive brief PDFs are present if downloads are enabled.
- [ ] Firebase client and service-account secrets are configured.
- [ ] Firestore access-control collection/document is configured.
- [ ] Administrator email addresses and privileged domains are correct.
- [ ] OpenAI API key and model are configured if Copilot is enabled.
- [ ] HTTPS and cookie behaviour are verified.
- [ ] Export and chat-history storage are persistent where required.
- [ ] Guest, viewer, privileged, and admin access have been tested.
- [ ] Dashboard filters, visualisations, map, CFR, downloads, and Copilot have been smoke-tested.
- [ ] Logs and error reporting do not expose secrets or tokens.

---

## 16. Troubleshooting

| Symptom | Likely area to inspect | Recommended checks |
|---|---|---|
| Dashboard reports that `output_final.parquet` is missing | Export path or volume mount | Confirm `/exports/output_final.parquet` or `<repository>/exports/output_final.parquet` exists and is readable by the application process. |
| Parquet loading fails | File integrity or Parquet engine | Validate the file independently and confirm a compatible `pyarrow` or other pandas Parquet engine is installed. |
| Country names or map locations are missing | Country metadata / ISO mapping | Validate `countries_metadata.json`, country labels, and the expected metadata keys. |
| CFR tab reports no data | CFR export discovery | Confirm a valid `CFR_Export_YYYY_MM_DD.csv` exists, or set `EUSEE_CFR_CSV` to a valid file path. |
| CFR loader reports missing columns | CFR export schema | Compare the CSV headers with the required fields and six principle columns defined in `app.py`. |
| CFR score warning appears | Source score values | Check that scores are within the expected 1–5 range; correct the source export rather than silently altering values. |
| Firebase initialisation fails | Firebase secrets or dependencies | Verify the `[firebase]` client settings, service-account credentials, enabled authentication methods, and installed Firebase packages. |
| Login works but refresh loses the session | Cookie/session configuration | Check HTTPS, browser cookie policy, CookieManager installation, and Firebase refresh-token validity. |
| User sees the wrong access level | Role configuration | Check administrator emails, privileged domains, temporary shared-account settings, and Firestore access configuration. |
| Admin changes do not persist | Firestore configuration | Verify the service account, Firestore database, collection/document settings, and write permissions. |
| AI Copilot reports missing OpenAI configuration | OpenAI secrets | Confirm `[openai].OPENAI_API_KEY` is configured and visible to the Streamlit process. |
| Copilot returns no matching records | Query scope or search terms | Review the topic wording, active dataset, available columns, and search specificity. A zero-row result is not proof that no relevant information exists. |
| Copilot cannot create a chart | Query result shape | Request a grouping with a categorical field and a numeric metric; a chart cannot be built from unsuitable or empty results. |
| User history is not retained | History directory | Check the configured history directory, filesystem write permissions, persistent storage, and authenticated user identity. |
| PDF download is unavailable | Documentation files | Confirm the expected PDF filename and `docs/` path. |
| Changes to a data file are not visible | Streamlit cache | Allow the configured cache TTL to expire or use the deployment's approved cache-clear/restart process. |

When troubleshooting authentication or API failures, inspect server logs without printing API keys, Firebase tokens, refresh tokens, private keys, or cookie values.

---

## 17. Security and operational guidance

1. **Protect secrets.** Keep `.streamlit/secrets.toml`, service-account JSON/private keys, API keys, and tokens out of Git.
2. **Use least privilege.** Give service accounts only the Firebase/Firestore permissions they require.
3. **Use individual accounts.** Avoid shared accounts where individual authentication is possible.
4. **Review privileged domains.** A privileged-domain rule can grant broad access to every matching authenticated user. Keep the domain list narrow and current.
5. **Restrict administrator accounts.** Explicitly configure trusted administrator emails and protect those accounts with the strongest available authentication controls.
6. **Use HTTPS.** Persistent browser authentication depends on secure cookie handling and should be tested behind the production proxy.
7. **Protect exported data.** Data download permissions should align with the project's data governance and partner agreements.
8. **Protect chat history.** The authentication module stores per-user chat history in server-side JSON files. Restrict filesystem access and define a retention/deletion policy.
9. **Validate uploaded/source data.** Treat Parquet and CSV exports as data inputs; validate schema and content before publishing.
10. **Avoid sensitive logs.** Do not log credentials, tokens, private keys, or full authentication cookies.
11. **Review access changes.** Test role and row-scope changes with representative accounts before production release.
12. **Back up configuration and data.** Maintain controlled backups of source exports and Firestore access configuration.

---

## 18. Maintenance and contribution guidelines

### 18.1 General principles

- Keep data loading, query logic, authorization, and UI rendering separated.
- Preserve permission checks when adding new dashboard features.
- Use unique Streamlit widget keys, especially inside repeated components and fragments.
- Keep visualisations responsive and use the shared Plotly styling helpers.
- Do not bypass `authz.py` when exposing protected features or data.
- Do not make the Copilot answer from model knowledge when the question requires dashboard evidence.
- Keep the expected data schema and documentation aligned with the deployed application.

### 18.2 Adding a dashboard feature

1. Define the feature and the data fields it requires.
2. Add a permission key to `FEATURE_REGISTRY` in `authz.py`.
3. Set appropriate defaults in `ROLE_PRESETS` and role restrictions in `LOCKED_FALSE`.
4. Add the feature to the relevant UI section in `app.py`.
5. Wrap protected rendering in the corresponding permission check.
6. If necessary, expose the permission in the admin feature registry and confirm the admin page groups it correctly.
7. Test the feature for each applicable role.
8. Test behaviour with restricted region/country/year scopes.
9. Update this README and the user manual when the user-facing workflow changes.

### 18.3 Changing the Copilot

When changing the Copilot:

- Update the structured tool schema and query executor together.
- Confirm column resolution works for both alert and CFR schemas.
- Keep numerical calculations in pandas.
- Preserve the scope-control rules and zero-result handling.
- Validate chart generation against empty, categorical-only, and numeric query results.
- Test summaries, counts, groupings, trends, record retrieval, and cross-dataset questions.
- Ensure user history is not mixed across authenticated users.

### 18.4 Release validation

Run the application locally or in a staging environment with representative data and accounts before production deployment. At minimum, test the release checklist in [Deployment considerations](#15-deployment-considerations).

---

## 19. Known implementation boundaries

The following boundaries are important when interpreting or maintaining the supplied implementation:

- The dashboard's alert data is dependent on the contents and quality of the supplied Parquet export.
- CFR analysis is dependent on a correctly formatted dated CSV export and country metadata.
- Country-level CFR scores and alert-level records are different data units and should not be conflated.
- Country comparisons based on alert counts may be affected by differences in partner reporting practices.
- An alert can be associated with multiple enabling principles; principle-level totals may therefore not behave like mutually exclusive categories.
- The Copilot is limited to the available authorized alert and CFR data. It cannot reliably answer questions about external events or context not represented in those datasets.
- A failed or overly narrow topic search can return zero records even when related information exists under other wording.
- Map coverage depends on the availability and quality of geographic metadata.
- Authentication and role assignment depend on Firebase configuration, the configured email/domain rules, and Firestore availability.
- Access-configuration loading can fall back to defaults when Firestore is unavailable; saving administrative changes requires a working Firestore connection.
- The supplied source files do not include a dependency lockfile, automated test suite, CI/CD configuration, Dockerfile, or data-ingestion pipeline specification. These should be maintained separately if required by the deployment.
- The exact production infrastructure, hosting provider, scheduled data refresh process, and operational backup policy are not established by the five supplied Python modules and should be documented by the deployment owner.

---

## 20. Citation and acknowledgement

When using dashboard data, charts, or findings in a report or presentation, cite the EU SEE Dashboard and identify the relevant visualisation and date of access.

**Suggested citation format:**

> EU SEE Dashboard, “[Name of visualisation]”, accessed [DD Month YYYY].

For project information and related source research, visit the [EU SEE website](https://eusee.hivos.org/).

---

*README prepared from the supplied EU SEE Dashboard application modules. Update this document whenever application features, data schemas, access rules, configuration, or deployment procedures change.*
