s#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import posixpath
import random
import smtplib
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
import pandas as pd
import paramiko
from dotenv import load_dotenv
from langdetect import LangDetectException, detect
from tqdm.asyncio import tqdm_asyncio

import openai

# ---------------- LOAD ENVIRONMENT VARIABLES ----------------
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not set! Please add it to your .env file as OPENAI_API_KEY")


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

# --- SFTP CONFIG ---
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = int(os.getenv("SFTP_PORT") or 22)
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("REMOTE_DIR", "exports")
SFTP_HOST = os.getenv("SFTP_HOST")


# --- SMTP / NOTIFICATIONS ---
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# --- LOCAL PATHS ---
LOCAL_DIR = BASE_DIR / os.getenv("LOCAL_DIR", "exports")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FOLDER = BASE_DIR / "exports"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_CSV = OUTPUT_FOLDER / "output_final.csv"
THEMES_FILE = OUTPUT_FOLDER / "themes.json"
OUTPUT_PARQUET = OUTPUT_FOLDER / "output_final.parquet"
OUTPUT_CSV = OUTPUT_FOLDER / "output_final.csv"
PERMANENTLY_FAILED_FILE = OUTPUT_FOLDER / "permanently_failed_batches.json"

# --- PROCESSING CONFIG ---
MAX_BATCH_TOKENS = 4000
MAX_BATCH_SIZE = 100
CONCURRENT_BATCHES = 5
MAX_RETRIES = 2
TEST_ROWS = None

FIELDS = [
    "Actor of repression",
    "Subject of repression",
    "Mechanism of repression",
    "Type of event",
]


# ---------------- TOKEN ESTIMATION ----------------
try:
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-5-mini")

    def estimate_tokens(text: str) -> int:
        return len(encoding.encode(text or "")) + 50

except ImportError:
    def estimate_tokens(text: str) -> int:
        text = text or ""
        return max(1, len(text) // 4 + 50)


# ---------------- SFTP HELPERS ----------------
def sftp_enabled() -> bool:
    return all([SFTP_HOST, SFTP_USERNAME, SFTP_PASSWORD, REMOTE_DIR])


def create_sftp_client():
    if not sftp_enabled():
        raise ValueError("Missing SFTP configuration.")

    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    """
    Recursively create remote directory if it does not exist.
    """
    parts = remote_directory.strip("/").split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def download_file_from_sftp(remote_filename: str, local_path: Path) -> None:
    transport = None
    sftp = None
    try:
        transport, sftp = create_sftp_client()
        remote_path = posixpath.join(REMOTE_DIR, remote_filename)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {remote_path} -> {local_path}")
        sftp.get(remote_path, str(local_path))
        print(f"Downloaded: {remote_filename}")
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


def upload_file_to_sftp(local_path: Path, remote_filename: str | None = None) -> None:
    transport = None
    sftp = None
    try:
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Upload skipped, file does not exist: {local_path}")
            return

        transport, sftp = create_sftp_client()
        ensure_remote_dir(sftp, REMOTE_DIR)

        remote_filename = remote_filename or local_path.name
        remote_path = posixpath.join(REMOTE_DIR, remote_filename)

        print(f"Uploading {local_path} -> {remote_path}")
        sftp.put(str(local_path), remote_path)
        print(f"Uploaded: {remote_filename}")
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


def fetch_required_input_files() -> None:
    """
    Download required input files from the remote SFTP folder to local exports folder.
    """
    if not sftp_enabled():
        print("SFTP not configured. Using local files only.")
        return

    required_files = {
        "output_final.csv": INPUT_CSV,
        "themes.json": THEMES_FILE,
    }

    for remote_name, local_path in required_files.items():
        download_file_from_sftp(remote_name, local_path)


def upload_output_files() -> None:
    """
    Upload generated outputs back to the remote SFTP folder.
    """
    if not sftp_enabled():
        print("SFTP not configured. Skipping remote upload.")
        return

    files_to_upload = [
        OUTPUT_CSV,
        OUTPUT_PARQUET,
        PERMANENTLY_FAILED_FILE,
    ]

    for file_path in files_to_upload:
        if file_path.exists():
            upload_file_to_sftp(file_path)
        else:
            print(f"Output not found, skipping upload: {file_path}")


# ---------------- EMAIL NOTIFIER ----------------
def send_email(subject: str, body: str, to_email: str) -> None:
    if not all([subject, body, to_email, SMTP_USER, SMTP_PASS, SMTP_HOST]):
        print("Email not sent: Missing credentials or recipient.")
        return

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = str(subject)
        msg["From"] = str(SMTP_USER)
        msg["To"] = str(to_email)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(f"Email successfully sent to {to_email}")
    except Exception as e:
        print(f"Email failed: {e}")


# ---------------- LOAD HELPERS ----------------
def load_themes(themes_path: Path) -> dict:
    if not themes_path.exists():
        raise FileNotFoundError(f"Themes file not found: {themes_path}")

    with open(themes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_input_dataframe(input_csv: Path, test_rows: int | None = None) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if test_rows:
        df = df.head(test_rows).copy()

    for col in FIELDS:
        if col not in df.columns:
            df[col] = ""

    return df


# ---------------- HELPERS ----------------
def format_theme_options(theme_list: dict, lang: str) -> str:
    options = theme_list.get(lang, theme_list["en"])
    return ", ".join([t["label"] for t in options])


def to_comma_separated(item) -> str:
    if isinstance(item, list):
        return ", ".join(str(i) for i in item[:2])
    if isinstance(item, str) and item.strip():
        return item.strip()
    return ""


def pick_random_themes(theme_map: dict, lang: str, n: int = 2) -> list[str]:
    if lang not in theme_map:
        lang = "en"
    options = theme_map[lang]
    selected = random.sample(options, n) if len(options) >= n else options
    return [c["label"] for c in selected]


def build_prompt(
    batch_summaries: list[str],
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
) -> str:
    numbered_texts = []

    for idx, summary in enumerate(batch_summaries):
        try:
            lang = detect(summary) if str(summary).strip() else "en"
        except LangDetectException:
            lang = "en"

        numbered_texts.append(
            f"{idx + 1}. Summary: {summary}\n"
            f"Language: {lang}\n"
            f"Actor options: {format_theme_options(actor_themes, lang)}\n"
            f"Subject options: {format_theme_options(subject_themes, lang)}\n"
            f"Mechanism options: {format_theme_options(mechanism_themes, lang)}\n"
            f"Type options: {format_theme_options(type_themes, lang)}"
        )

    numbered_text = "\n\n".join(numbered_texts)

    prompt = f"""
Extract repression info from each text below. Return a JSON array of objects in the same order.
- Return only valid JSON, do not include explanations or extra text.
- Each object must contain:
{json.dumps({field: "" for field in FIELDS}, indent=4)}

- Use ONLY the provided options for each field. Do NOT invent or add new labels.
- Select the minimum number of labels necessary to reflect the summary accurately.
- Add a SECOND label only when the summary contains clear, distinct, and explicit evidence of another actor/category/mechanism.
- Never return more than TWO labels in any field.
- Return multiple labels as a comma-separated string, never use `;`.
- Do NOT assign labels based on weak implication, background context, or speculation.

ACTOR / SUBJECT RULES
- Identify the primary affected actor.
- If more than two actor groups are affected, or the restriction applies broadly across civic space, use "All civil society (indiscriminate)".
- Use both "All civil society (indiscriminate)" and a specific actor label only when the summary describes a broad restriction and also explicitly highlights a specific targeted actor.

SUBJECT OF REPRESSION RULES

Identify the group directly targeted, restricted, punished, intimidated, surveilled, excluded, or otherwise affected by the repressive action.
Choose the MOST SPECIFIC subject group explicitly mentioned in the summary.
Do not choose a broad category when a more specific target is clearly identified.
Add a SECOND subject label only when the summary clearly describes two distinct targeted groups and both are central to the event.
Never return more than TWO subject labels.
Do not assign subject labels based only on general background, possible future impact, or broad implications for society.
Choose the group that is directly affected in the event itself.
USE THE MOST SPECIFIC TARGET

Examples:
- If the summary is about the arrest, harassment, or prosecution of a journalist, blogger, commentator, editor, media house, or influencer
  → "Journalists, media and influencers"
- If the summary is about women’s rights groups, feminist movements, LGBTQ+ groups, indigenous peoples, ethnic minorities, religious minorities, migrants, refugees, or persons with disabilities
  → "Minority groups and their rights"
- If the summary is about environmental defenders, land defenders, climate activists, anti-extractives campaigners, or communities mobilizing around environmental harm
  → "Environment Justice"
- If the summary is about trade unions, labour leaders, workers’ movements, strikes over wages or working conditions, or labour organizing
  → "Socio-Economic Rights"
- If the summary is about access to health, housing, education, food, water, land, livelihoods, social welfare, or service delivery
  → "Socio-Economic Rights"
- If the summary is about civil society organizations, NGOs, associations, community groups, activists, or defenders as civic actors
  → use the most relevant specific label if one is clearly stated; otherwise use:
  → "All civil society (indiscriminate)" only when the restriction broadly affects multiple civic actors and no single primary target is clearly identified.

WHEN TO USE "ALL CIVIL SOCIETY (INDISCRIMINATE)"

Use "All civil society (indiscriminate)" only when:
- the restriction is broad and applies across civic space, or
- the event affects multiple civic actors without one clearly primary group, or
- the measure targets CSOs, activists, associations, and public-interest actors generally.

Examples:
- A foreign agents law applying to all NGOs
  → "All civil society (indiscriminate)"
- A nationwide internet shutdown affecting activists, journalists, and CSOs broadly, with no single main target
  → "All civil society (indiscriminate)"
- A rule restricting registration or funding for all associations
  → "All civil society (indiscriminate)"

WHEN NOT TO USE "ALL CIVIL SOCIETY (INDISCRIMINATE)"
Do NOT use it when a specific group is clearly targeted.
Examples:
- An editor arrested over a corruption story
  → "Journalists, media and influencers"
  NOT "All civil society (indiscriminate)"
- Women denied hospital access under gender-based restrictions
  → "Minority groups and their rights"
  and, if the event is explicitly about health access, possibly "Socio-Economic Rights" as a second label only if both are central
- Indigenous organizations’ accounts frozen during a protest
  → "Minority groups and their rights"
  or "Environment Justice" if the event is clearly about environmental defense
  NOT automatically "All civil society (indiscriminate)"

SECOND SUBJECT LABEL RULE
Add a second subject label only when the summary clearly identifies two distinct target groups and both are central.
Examples:
- Journalists and LGBTQ+ activists both explicitly targeted
  → "Journalists, media and influencers, Minority groups and their rights"
- Indigenous environmental defenders targeted for anti-mining advocacy
  → "Environment Justice, Minority groups and their rights"
- A law broadly affecting all CSOs but explicitly highlighting journalists as a major target
  → "All civil society (indiscriminate), Journalists, media and influencers"
Do not add a second label just because another group may be indirectly affected.

DISAMBIGUATION EXAMPLES
- Arrest of a blogger for online criticism
  → "Journalists, media and influencers"
- Investigation of an LGBTQI+ community organization
  → "Minority groups and their rights"
- Ban on an environmental protest group or land defenders
  → "Environment Justice"
- Suspension of a trade union or repression of a workers’ strike
  → "Socio-Economic Rights"
- Restrictions on NGOs broadly through registration, funding, or foreign agent laws
  → "All civil society (indiscriminate)"
- Gender equality rollback affecting women’s organizations
  → "Minority groups and their rights"
- Restriction on health access for women
  → "Minority groups and their rights"
  Add "Socio-Economic Rights" only if the summary explicitly centers the denial of health rights, not just gender discrimination.

MECHANISM RULES
- Identify the primary mechanism used.
- Add a second mechanism only when the text clearly describes a separate and independently applied mechanism.
- Do not stack mechanisms that are merely related parts of the same event unless both are central and explicit.

Example:
An activist investigated and then arrested → classify as:
"Incarceration and Legal Repression" (the arrest is the main repression action).

ADMINISTRATIVE REPRESSION

Definition:
Restrictions or penalties imposed through administrative procedures, permits, licensing systems, registration rules, fines, or bureaucratic decisions.
Typical signals:
permit denial, organization deregistration, asset freezing, funding restrictions, suspension of licenses, administrative fines.
Examples:
Example 1
Authorities refused to renew the operating license of an independent media outlet.
→ Administrative Repression
Example 2
The government froze the bank accounts of several NGOs under new financial regulations.
→ Administrative Repression
Example 3
Police denied permission for a protest organized by civil society groups.
→ Administrative Repression

INCARCERATION AND LEGAL REPRESSION
Definition:
Use of criminal law or judicial processes to detain, prosecute, or imprison individuals.
Typical signals:
arrest, detention, criminal charges, prosecution, court trials, imprisonment.

Examples:
Example 1
Police arrested a journalist after publishing corruption allegations.
→ Incarceration and Legal Repression
Example 2
Authorities charged an activist with sedition and brought the case to court.
→ Incarceration and Legal Repression
Example 3
A blogger was sentenced to two years in prison for insulting public officials.
→ Incarceration and Legal Repression

DIGITAL REPRESSION
Definition:
Repression carried out through digital technologies, internet regulation, or online censorship.
Typical signals:
internet shutdowns, website blocking, social media censorship, cybercrime laws, online surveillance.
Examples:
Example 1
Authorities blocked access to several independent news websites.
→ Digital Repression
Example 2
The government shut down internet services during protests.
→ Digital Repression

Example 3
A new cybercrime law criminalizes criticism of public officials online.
→ Digital Repression

PSYCHOLOGICAL INTIMIDATION ON INDIVIDUALS
Definition:
Threats, harassment, intimidation, surveillance, or pressure intended to discourage civic participation without direct physical violence.
Typical signals:
threats, harassment campaigns, intimidation by police, stalking, coercive questioning.

Examples:
Example 1
Police repeatedly summoned an activist for questioning to pressure them to stop organizing protests.
→ Psychological intimidation on individuals
Example 2
Journalists received threats after reporting on corruption.
→ Psychological intimidation on individuals
Example 3
Security agents followed and monitored a human rights defender.
→ Psychological intimidation on individuals

PHYSICAL VIOLENCE
Definition:
Use of physical force causing harm, injury, or bodily violence against individuals.
Typical signals:
beatings, shootings, assaults, violent attacks, excessive use of force.

Examples:
Example 1
Police beat protesters during a demonstration.
→ Physical Violence
Example 2
Unknown attackers assaulted a journalist outside their home.
→ Physical Violence
Example 3
Security forces fired live ammunition at protesters.
→ Physical Violence

DISCOURSES AND BEHAVIOUR
Definition:
Public rhetoric, stigmatization, or hostile narratives by authorities or influential actors that delegitimize or attack civic actors.
Typical signals:
public accusations, smear campaigns, hostile rhetoric, labeling groups as enemies or criminals.
Examples:
Example 1
Government officials publicly accused NGOs of being foreign agents.
→ Discourses and Behaviour
Example 2
A minister described human rights defenders as traitors during a press conference.
→ Discourses and Behaviour
Example 3
State media repeatedly portrayed activists as terrorists.
→ Discourses and Behaviour

FINAL MECHANISM DECISION RULE
Choose the mechanism that represents the MAIN repression action described in the summary.
Examples:
Journalist arrested for social media post
→ Incarceration and Legal Repression
Activist threatened by police
→ Psychological intimidation on individuals
Police beat protesters
→ Physical Violence
NGO license revoked
→ Administrative Repression
Website blocked by authorities
→ Digital Repression
Government publicly labels activists as enemies
→ Discourses and Behaviour

TYPE OF EVENT CLASSIFICATION

Identify the civic context where repression occurs.
Examples:
Journalist arrested for social media posts
→ Online activities
Police beat protesters during demonstration
→ Freedom of assembly
NGO deregistered by authorities
→ Freedom of association
Police raid media office
→ Media Freedom
Opposition candidate arrested during election campaign
→ Electoral Process

RIGHTS DISAMBIGUATION
- Use "Socio-Economic Rights" only when the text explicitly concerns economic, social, or cultural rights, such as labor, housing, health, education, land, livelihoods, or social welfare.
- Use "Minority groups and their rights" for women’s rights, feminist organizations, LGBTQ+ groups, ethnic or religious minorities, migrants, refugees, indigenous groups, persons with disabilities, and similar identity-based groups.
- Do not assign "Socio-Economic Rights" without textual support.

IMPORTANT:
Subject labels identify WHO is targeted.
They should not be used to describe the general theme or rights area unless that rights-bearing group is itself the target.

Texts:
{numbered_text}
"""
    return prompt


# ---------------- BATCH BUILDER ----------------
def build_batches(
    df_input: pd.DataFrame,
    max_tokens: int = MAX_BATCH_TOKENS,
    max_rows: int | None = None,
) -> list[tuple[list[int], list[str]]]:
    """
    Build batches of rows from df_input based on token limits and max_rows.
    Only includes rows that are truly blank in the FIELDS columns.
    Returns a list of tuples: (row_indices, batch_summaries)
    """
    batches = []
    i = 0

    def is_row_filled(row, fields=FIELDS) -> bool:
        for col in fields:
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip() != "":
                return True
        return False

    while i < len(df_input):
        if is_row_filled(df_input.iloc[i]):
            i += 1
            continue

        batch_summaries = []
        batch_indices = []
        batch_tokens = 0

        while i < len(df_input):
            row = df_input.iloc[i]
            summary = str(row.get("summary", "") or "")
            est_tokens = estimate_tokens(summary)

            if batch_tokens + est_tokens > max_tokens and batch_summaries:
                break

            batch_summaries.append(summary)
            batch_indices.append(df_input.index[i])
            batch_tokens += est_tokens
            i += 1

            if max_rows and len(batch_summaries) >= max_rows:
                break

        if batch_summaries:
            batches.append((batch_indices, batch_summaries))

    return batches


# ---------------- MOCK EXTRACTOR ----------------
async def mock_extract_batch(
    batch_summaries: list[str],
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
    batch_indices: list[int] | None = None,
):
    await asyncio.sleep(random.uniform(0.1, 0.4))

    if batch_indices is not None:
        print(f"[MOCK] Processing rows with indices: {batch_indices}")
    else:
        print(f"[MOCK] Processing batch of size: {len(batch_summaries)}")

    results = []
    for _summary in batch_summaries:
        lang = "en"
        result = {
            "Actor of repression": pick_random_themes(actor_themes, lang, n=2),
            "Subject of repression": pick_random_themes(subject_themes, lang, n=2),
            "Mechanism of repression": pick_random_themes(mechanism_themes, lang, n=2),
            "Type of event": pick_random_themes(type_themes, lang, n=2),
        }
        for key in FIELDS:
            result[key] = to_comma_separated(result[key])
        results.append(result)

    return results, None


# ---------------- OPENAI EXTRACTOR ----------------
async def extract_batch(
    batch_summaries: list[str],
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
    mock_mode: bool = False,
    batch_indices: list[int] | None = None,
):
    if mock_mode:
        return await mock_extract_batch(
            batch_summaries=batch_summaries,
            actor_themes=actor_themes,
            subject_themes=subject_themes,
            mechanism_themes=mechanism_themes,
            type_themes=type_themes,
            batch_indices=batch_indices,
        )

    if batch_indices is not None:
        print(f"[OPENAI] Processing rows with indices: {batch_indices}")
    else:
        print(f"[OPENAI] Processing batch of size: {len(batch_summaries)}")

    prompt = build_prompt(
        batch_summaries=batch_summaries,
        actor_themes=actor_themes,
        subject_themes=subject_themes,
        mechanism_themes=mechanism_themes,
        type_themes=type_themes,
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content.strip()

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("[")
                end = content.rfind("]") + 1
                if start == -1 or end <= 0:
                    raise ValueError("Model response did not contain a valid JSON array.")
                data = json.loads(content[start:end])

            if not isinstance(data, list):
                raise ValueError("Model response is not a JSON array.")

            for res in data:
                for key in FIELDS:
                    res[key] = to_comma_separated(res.get(key, ""))

            return data, None

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)

    return [{k: "Error" for k in FIELDS} for _ in batch_summaries], None


# ---------------- MAIN PROCESS ----------------
async def process_all(
    df_source: pd.DataFrame,
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
    mock_mode: bool = False,
) -> pd.DataFrame:
    # Load previous output if exists, else use fresh source dataframe
    df_out = df_source.copy()
    print("Starting processing from latest output_final.csv")

    for col in FIELDS:
        if col not in df_out.columns:
            df_out[col] = ""

    permanently_failed = []
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)

    def is_row_fully_blank(row, fields=FIELDS) -> bool:
        for col in fields:
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip() != "":
                return False
        return True

    rows_to_process = df_out[df_out.apply(is_row_fully_blank, axis=1)]

    total_rows = len(df_out)
    blank_rows_count = len(rows_to_process)
    skipped_rows_count = total_rows - blank_rows_count

    print(f"Total rows in dataset: {total_rows}")
    print(f"Fully blank rows to process: {blank_rows_count}")
    print(f"Rows skipped (already filled): {skipped_rows_count}")

    if rows_to_process.empty:
        print("No fully blank rows to process. Nothing to do.")
        return df_out

    batches = build_batches(rows_to_process, max_rows=MAX_BATCH_SIZE)
    print(f"Total batches to process (fully blank rows): {len(batches)}")

    async def process_batch(batch_indices: list[int], batch_summaries: list[str]):
        async with semaphore:
            retries = 0
            last_exception = None

            while retries <= MAX_RETRIES:
                try:
                    results, _ = await extract_batch(
                        batch_summaries=batch_summaries,
                        actor_themes=actor_themes,
                        subject_themes=subject_themes,
                        mechanism_themes=mechanism_themes,
                        type_themes=type_themes,
                        mock_mode=mock_mode,
                        batch_indices=batch_indices,
                    )
                    break
                except Exception as exc:
                    retries += 1
                    last_exception = exc
                    print(f"Batch starting at {batch_indices[0]} failed (attempt {retries}): {exc}")
                    await asyncio.sleep(2 ** retries)
            else:
                permanently_failed.append(
                    {"start_idx": batch_indices[0], "error": str(last_exception)}
                )
                results = [{k: "Error" for k in FIELDS} for _ in batch_summaries]

            for j, res in enumerate(results):
                idx = batch_indices[j]
                row_status = []

                for key in FIELDS:
                    value = res.get(key, "")
                    df_out.loc[idx, key] = value
                    if value == "Error" or not str(value).strip():
                        row_status.append(f"{key}=ERROR")

                if row_status:
                    print(f"[ROW {idx}] Failed fields: {', '.join(row_status)}")
                else:
                    print(f"[ROW {idx}] Successfully filled all fields.")

    for i in range(0, len(batches), CONCURRENT_BATCHES):
        chunk = batches[i:i + CONCURRENT_BATCHES]
        tasks = [process_batch(*b) for b in chunk]
        await tqdm_asyncio.gather(*tasks, desc="Processing batches", total=len(chunk))

        # Save intermediate outputs
        df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
        df_out.to_csv(OUTPUT_CSV, index=False)

    # Final save
    df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    df_out.to_csv(OUTPUT_CSV, index=False)

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(permanently_failed, f, indent=2)
    elif PERMANENTLY_FAILED_FILE.exists():
        PERMANENTLY_FAILED_FILE.unlink()

    print(f"Processing complete! Fully blank rows processed: {blank_rows_count}")
    return df_out


def send_summary_update_email(
    to_email,
    total_rows,
    processed_rows,
    skipped_rows,
    output_csv,
    output_parquet,
    permanently_failed_count=0,
    mock_mode=False,
    smtp_host=SMTP_HOST,
    smtp_port=SMTP_PORT,
    smtp_user=SMTP_USER,
    smtp_pass=SMTP_PASS,
):
    """
    Send a formatted HTML email summarizing dataset update results.
    """

    if not all([to_email, smtp_host, smtp_port, smtp_user, smtp_pass]):
        print("Email not sent: missing SMTP configuration.")
        return

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_csv = Path(output_csv)
    output_parquet = Path(output_parquet)

    csv_status = "Created" if output_csv.exists() else "Not found"
    parquet_status = "Created" if output_parquet.exists() else "Not found"

    subject = f"Dataset Summary Update Completed | {run_time}"

    plain_text = f"""
Dataset Summary Update Completed

Run time: {run_time}
Mock mode: {mock_mode}

Processing results:
- Total rows in dataset: {total_rows}
- Fully blank rows processed: {processed_rows}
- Rows skipped (already filled): {skipped_rows}
- Permanently failed batches: {permanently_failed_count}

Output files:
- CSV: {output_csv.name} ({csv_status})
- Parquet: {output_parquet.name} ({parquet_status})

This notification confirms that the dataset summary update pipeline has completed.
"""

    html = f"""
    <html>
      <body style="margin:0;padding:0;background-color:#f4f6f8;font-family:Arial,Helvetica,sans-serif;">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f4f6f8;padding:24px 0;">
          <tr>
            <td align="center">
              <table role="presentation" width="700" cellspacing="0" cellpadding="0" style="background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #dfe3e8;">
                
                <tr>
                  <td style="background:#1f4e79;color:#ffffff;padding:24px 32px;">
                    <h1 style="margin:0;font-size:24px;">Dataset Summary Update Completed</h1>
                    <p style="margin:8px 0 0 0;font-size:14px;opacity:0.95;">
                      Automated processing report for summary field updates
                    </p>
                  </td>
                </tr>

                <tr>
                  <td style="padding:28px 32px;">
                    <p style="margin:0 0 18px 0;font-size:15px;color:#222;">
                      Hello,
                    </p>
                    <p style="margin:0 0 20px 0;font-size:15px;line-height:1.6;color:#222;">
                      The dataset update pipeline has finished processing summary-based classification fields.
                      Below is the execution summary and output status.
                    </p>

                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-bottom:24px;">
                      <tr>
                        <td colspan="2" style="padding:12px 14px;background:#eef4f8;border:1px solid #dfe3e8;font-weight:bold;color:#1f2937;">
                          Run Details
                        </td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;width:40%;font-weight:bold;">Run time</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{run_time}</td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;font-weight:bold;">Mock mode</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{mock_mode}</td>
                      </tr>
                    </table>

                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-bottom:24px;">
                      <tr>
                        <td colspan="2" style="padding:12px 14px;background:#eef4f8;border:1px solid #dfe3e8;font-weight:bold;color:#1f2937;">
                          Processing Summary
                        </td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;width:40%;font-weight:bold;">Total rows in dataset</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{total_rows}</td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;font-weight:bold;">Fully blank rows processed</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{processed_rows}</td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;font-weight:bold;">Rows skipped</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{skipped_rows}</td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;font-weight:bold;">Permanently failed batches</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{permanently_failed_count}</td>
                      </tr>
                    </table>

                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-bottom:24px;">
                      <tr>
                        <td colspan="2" style="padding:12px 14px;background:#eef4f8;border:1px solid #dfe3e8;font-weight:bold;color:#1f2937;">
                          Output Files
                        </td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;width:40%;font-weight:bold;">CSV output</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{output_csv.name} — {csv_status}</td>
                      </tr>
                      <tr>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;font-weight:bold;">Parquet output</td>
                        <td style="padding:12px 14px;border:1px solid #dfe3e8;">{output_parquet.name} — {parquet_status}</td>
                      </tr>
                    </table>

                    <p style="margin:0 0 10px 0;font-size:15px;line-height:1.6;color:#222;">
                      This message confirms that the summary update job completed and the output dataset was written successfully.
                    </p>

                    <p style="margin:20px 0 0 0;font-size:15px;color:#222;">
                      Regards,<br>
                      Automated Dataset Update Pipeline
                    </p>
                  </td>
                </tr>

                <tr>
                  <td style="padding:16px 32px;background:#f8fafc;color:#6b7280;font-size:12px;border-top:1px solid #e5e7eb;">
                    This is an automated notification generated by the dataset processing workflow.
                  </td>
                </tr>

              </table>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg.set_content(plain_text)
        msg.add_alternative(html, subtype="html")

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"Summary update email sent to {to_email}")

    except Exception as e:
        print(f"Failed to send summary update email: {e}")


# ---------------- RUN SCRIPT ----------------
if __name__ == "__main__":
    mock_mode = False  # Set True for testing without API calls

    # 1) Pull latest input files from remote SFTP
    fetch_required_input_files()

    # 2) Load themes and source dataframe after SFTP download
    themes = load_themes(THEMES_FILE)
    ACTOR_THEMES = themes["ACTOR_THEMES"]
    SUBJECT_THEMES = themes["SUBJECT_THEMES"]
    MECHANISM_THEMES = themes["MECHANISM_THEMES"]
    TYPE_THEMES = themes["TYPE_THEMES"]

    df_source = load_input_dataframe(INPUT_CSV, test_rows=TEST_ROWS)

    # 3) Pre-run summary based on current input/output state
    df_prev = pd.read_csv(INPUT_CSV)

    for col in FIELDS:
        if col not in df_prev.columns:
            df_prev[col] = ""

    def is_row_fully_blank_for_summary(row) -> bool:
        for col in FIELDS:
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip() != "":
                return False
        return True

    blank_rows = df_prev[df_prev.apply(is_row_fully_blank_for_summary, axis=1)]
    total_rows = len(df_prev)
    skipped_rows = total_rows - len(blank_rows)

    print(
        f"Total rows: {total_rows} | "
        f"Fully blank rows to process: {len(blank_rows)} | "
        f"Skipped rows: {skipped_rows}"
    )

    # 4) Run extraction
    df_out = asyncio.run(
        process_all(
            df_source=df_source,
            actor_themes=ACTOR_THEMES,
            subject_themes=SUBJECT_THEMES,
            mechanism_themes=MECHANISM_THEMES,
            type_themes=TYPE_THEMES,
            mock_mode=mock_mode,
        )
    )

    summary_message = (
        f"Processing complete! Total rows: {len(df_out)} | "
        f"Fully blank rows processed: {len(blank_rows)} | "
        f"Skipped rows: {skipped_rows} | "
        f"Mock mode: {mock_mode}"
    )
    print(summary_message)

    # 5) Push updated outputs back to the same remote SFTP folder
    upload_output_files()

    # 6) Optional notification
    if NOTIFY_EMAIL:
        permanently_failed_count = 0
        if PERMANENTLY_FAILED_FILE.exists():
            with open(PERMANENTLY_FAILED_FILE, "r", encoding="utf-8") as f:
                permanently_failed = json.load(f)
            permanently_failed_count = len(permanently_failed)

        send_summary_update_email(
            to_email=NOTIFY_EMAIL,
            total_rows=len(df_out),
            processed_rows=len(blank_rows),
            skipped_rows=skipped_rows,
            output_csv=OUTPUT_CSV,
            output_parquet=OUTPUT_PARQUET,
            permanently_failed_count=permanently_failed_count,
            mock_mode=mock_mode,
        )
