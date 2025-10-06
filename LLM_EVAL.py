from langfuse import Langfuse
import requests 
import base64
import os
from langfuse import get_client
import re
import pandas as pd
from datetime import date
from typing import List, Optional
from datetime import datetime, timezone, timedelta
import json
import bisect
from typing import Optional, Tuple
import numpy as np
from Meeting_notes import meeting_notes
#from gpt_cleaning import build_df_with_models
import uuid
from copy import deepcopy
from Writing_df_to_LF import  upsert_dataset_from_df, as_text, is_gpt5,is_gpt41,  split_by_model

from clean_df import  hydrate_traces, build_df_from_traces

langfuse = Langfuse(
  secret_key="sk-lf-43e4b161-aecf-415f-84d4-03464e2e8371",
  public_key="pk-lf-87e59847-a8b8-45f8-aa2e-d15cd64f15c3",
  host="http://10.48.26.167:443"
)

secret_key="sk-lf-43e4b161-aecf-415f-84d4-03464e2e8371"
public_key="pk-lf-87e59847-a8b8-45f8-aa2e-d15cd64f15c3"
host="http://10.48.26.167:443"

    
########################################################################################################################

# Set these in your notebook environment or via your runtime secrets
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-87e59847-a8b8-45f8-aa2e-d15cd64f15c3"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-43e4b161-aecf-415f-84d4-03464e2e8371"
os.environ["LANGFUSE_HOST"] = "http://10.48.26.167:443"

langfuse = get_client()
assert langfuse.auth_check(), "Langfuse auth failed — check keys/host."



import argparse


parser = argparse.ArgumentParser(description="Build datasets with external source_data file")
parser.add_argument("--source-file", required=True, help="Path to .py or .json with meeting notes")
parser.add_argument("--user_id", required=True, help="Path to .py or .json with meeting notes")
parser.add_argument("--bot", required= True,
                    help="choose the name of the bot you want to read the traces of. (e.g. 'digibot', 'wikibot', etc.)")

parser.add_argument("--on-or-before", action="store_true",
                    help="If a date is missing, use the latest note on or before the date")
args, _ = parser.parse_known_args()  # parse_known_args so existing args/env still work

SOURCE_FILE = args.source_file #This is where the meeting notes are stored
USE_ON_OR_BEFORE = args.on_or_before
USER_ID = args.user_id
bot_name = args.bot

#################################################################################################
####### LOAD IN SOURCE_DATA WITH EITHER PYTHON OR JSON FILE
#################################################################################################
# 1) Load meeting notes from either a Python or JSON file
import importlib.util
def _load_meeting_notes_from_py(path: str) -> dict[date, str]:
    """Python file must define a dict named `meeting_notes` with 'YYYY-MM-DD' keys."""
    spec = importlib.util.spec_from_file_location("source_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "meeting_notes") or not isinstance(mod.meeting_notes, dict):
        raise ValueError(f"{path} must define a dict variable `meeting_notes` mapping 'YYYY-MM-DD' -> text")

    return {
        pd.to_datetime(k, format="%Y-%m-%d", errors="raise").date(): str(v)
        for k, v in mod.meeting_notes.items()
    }

def _load_meeting_notes_from_json(path: str) -> dict[date, str]:
    """JSON file must contain a top-level object mapping 'YYYY-MM-DD' -> text."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object (dict) of date->text")
    return {
        pd.to_datetime(k, format="%Y-%m-%d", errors="raise").date(): str(v)
        for k, v in data.items()
    }

def load_source_mapping(source_file: str) -> dict[date, str]:
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    ext = os.path.splitext(source_file)[1].lower()
    if ext == ".py":
        return _load_meeting_notes_from_py(source_file)
    if ext == ".json":
        return _load_meeting_notes_from_json(source_file)
    raise ValueError("Unsupported source file type. Use .py (with `meeting_notes`) or .json")

SOURCE_FILE = load_source_mapping(SOURCE_FILE)

################################################################################################
####### READS IN TRACES BY USERNAME
###############################################################################################

DATASET_NAME = "06-10"

#USER_ID = "caimin.ayres@sefe.eu"  # <-- set your user id here
PAGE_LIMIT = 100
MAX_PAGES = 15

def fetch_by_user_id(user_id: str, limit_per_page: int = 100, max_pages: int = 50):
    items: List = []
    page = 1
    while page <= max_pages:
        resp = langfuse.api.trace.list(limit=limit_per_page, page=page, user_id=user_id)
        batch = resp.data
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit_per_page:
            break
        page += 1
    return items

traces_user = fetch_by_user_id(USER_ID, PAGE_LIMIT, MAX_PAGES)
print("Traces for user:", USER_ID, "->", len(traces_user))


##############################################################################
#WRITE DATAFRAME OF TRACES BY USER_ID TO LANGFUSE S
##############################################################################
try:
    _ = langfuse.get_dataset(DATASET_NAME)
except Exception:
    langfuse.create_dataset(name=DATASET_NAME, description="Offline eval set built from digibot traces")

def _list_existing_dataset_entries(dataset_name: str, page_size: int = 100):
    """Return (existing_item_ids, existing_source_trace_ids) for the dataset."""
    existing_item_ids = set()
    existing_source_trace_ids = set()
    page = 1
    while True:
        resp = langfuse.api.dataset_items.list(dataset_name=dataset_name, page=page, limit=page_size)
        data = getattr(resp, "data", None) or []
        if not data:
            break
        for it in data:
            it_id = getattr(it, "id", None)
            if it_id:
                existing_item_ids.add(it_id)
            stid = getattr(it, "source_trace_id", None)
            if stid is not None:
                existing_source_trace_ids.add(str(stid))
        if len(data) < page_size:
            break
        page += 1
    return existing_item_ids, existing_source_trace_ids

def pick_best_io(trace):
    ti, to = getattr(trace, "input", None), getattr(trace, "output", None)
    if ti is not None or to is not None:
        return ti, to, None
    obs = getattr(trace, "observations", None) or []
    gens = [o for o in obs if (getattr(o,"type","") or "").upper() == "GENERATION"]
    for o in reversed(gens):
        oi, oo = getattr(o, "input", None), getattr(o, "output", None)
        if oi is not None or oo is not None:
            return oi, oo, getattr(o, "id", None)
    for o in reversed(obs):
        oi, oo = getattr(o, "input", None), getattr(o, "output", None)
        if oi is not None or oo is not None:
            return oi, oo, getattr(o, "id", None)
    return None, None, None

created = skipped_no_io = skipped_existing = 0
## THIS IS WHERE I'M CREATING THE DATASET (PUT CODE IN TO HELP WITH CHECKING IF THE TRACE ALREADY EXISTS IN THE DATASET)
_existing_item_ids, _existing_source_trace_ids = _list_existing_dataset_entries(DATASET_NAME)

for t in traces_user:  # <-- use the filtered set here
    i, o, obs_id = pick_best_io(t)
    if i is None and o is None:
        skipped_no_io += 1
        continue
    ds_item_id = f"{DATASET_NAME}_dsitem_{t.id}"
    if ds_item_id in _existing_item_ids or str(t.id) in _existing_source_trace_ids:
        skipped_existing += 1
        continue
    # High-level helper is fine (snake_case, supports id in recent SDKs)
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        id=ds_item_id,         # upsert key to avoid duplicates
        input=i,
        expected_output=o,
        source_trace_id=t.id,
        source_observation_id=obs_id,
        metadata={"name": getattr(t, "name", None)}
    )
    created += 1
    # Keep sets updated in case of duplicate traces in the same run
    _existing_item_ids.add(ds_item_id)
    _existing_source_trace_ids.add(str(t.id))

print(f"Created: {created} | Skipped (no I/O): {skipped_no_io} | Skipped (already exists): {skipped_existing}")

##############################################################################
##############################################################################

#Goes through the dataset and creates a dataframe with the input, source_data and model_output
def paginate_items(dataset_name, page_size=100):
    page = 1
    while True:
        resp = langfuse.api.dataset_items.list(dataset_name=dataset_name, page=page, limit=page_size)
        if not resp.data:
            break
        for it in resp.data:
            yield it
        if len(resp.data) < page_size:
            break
        page += 1


########################################################


# 0) You already fetched the user's traces once
#traces_user = fetch_by_user_id(USER_ID, limit_per_page=100, max_pages=15)

# 1) Build items from dataset with a back-link to the original trace
items = []
for it in paginate_items(DATASET_NAME, page_size=100):
    # ----- INPUT -----
    input_text = ""
    source_text = ""
    model_output = ""

    raw_inp = it.input
    if isinstance(raw_inp, dict):
        # if your app shaped inputs earlier: {"input": "...", "source_data": "..."}
        raw_input_val = raw_inp.get("input")
        source_text = raw_inp.get("source_data") or ""
        input_text = raw_input_val if isinstance(raw_input_val, str) or raw_input_val is None \
                     else json.dumps(raw_input_val, ensure_ascii=False, default=str)
    elif isinstance(raw_inp, str):
        input_text = raw_inp
    else:
        # list-of-messages or other structure → stringify (or use a helper to pull last user message)
        input_text = json.dumps(raw_inp, ensure_ascii=False, default=str)

    # ----- EXPECTED OUTPUT -----
    exp = getattr(it, "expected_output", None)
    if isinstance(exp, dict) and "model_output" in exp:
        mo = exp["model_output"]
        model_output = mo if isinstance(mo, str) else json.dumps(mo, ensure_ascii=False, default=str)
    elif isinstance(exp, str):
        model_output = exp
    elif exp is not None:
        model_output = json.dumps(exp, ensure_ascii=False, default=str)

    stid = getattr(it, "source_trace_id", None)  # ← keep the link back to the trace

    items.append({
        "id": it.id,
        "source_trace_id": stid,          # ← NEW COLUMN used for joining
        "input_text": input_text,
        "source_data": source_text,
        "model_output": model_output,
        "metadata": (getattr(it, "metadata", None) or {}).copy(),
    })

df = pd.DataFrame(items).drop_duplicates(subset=["id"]).reset_index(drop=True)

# 2) Filter dataset items to traces from THIS run/user (avoid older items in dataset)
current_trace_ids = {t.id for t in traces_user}
df = df[df["source_trace_id"].isin(current_trace_ids)].copy()
df = df[df["source_trace_id"].notna()].copy()  # drop unlinked items if any

# 3) Hydrate exactly the same traces; build model lookup
traces_full = hydrate_traces(traces_user, langfuse)
df_models = (
    build_df_from_traces(traces_full)[["trace_id", "model"]]
    .dropna(subset=["trace_id"])                      # defensive
    .drop_duplicates("trace_id", keep="last")
)

# (important) ensure merge keys have the same dtype (string-to-string)
df["source_trace_id"]    = df["source_trace_id"].astype(str)
df_models["trace_id"]    = df_models["trace_id"].astype(str)


# Build a mapping from trace id -> trace name
id_to_name = {str(t.id): getattr(t, "name", None) for t in traces_user}

# Add a human-friendly column with the trace name for each dataset row
df["Bot"] = df["source_trace_id"].map(id_to_name)

#drop columns whihc don't associate to 
df.drop(df[df["Bot"] != bot_name].index, inplace=True)
# 4) Join model by trace id (recommended) – keeps alignment perfect
df = df.merge(
    df_models.rename(columns={"trace_id": "source_trace_id"}),
    on="source_trace_id",
    how="left",
)

# Debug
missing = int(df["model"].isna().sum())
print(f"Models mapped: {len(df)-missing}/{len(df)}; missing: {missing}")

###########################################################################################
################ EXTRACTING THE DATE FROM THE OUTPUT
############################################################################################


# ----------------- YOUR LOOKUP (replace with your real mapping) -----------------
my_lookup = {
    date(2025, 9, 18): {"flag": "A", "note": "Example for 18-Sep-2025"},
    date(2025, 9, 26): {"flag": "C", "note": "Most recent on 26-Sep-2025"},
    date(2025, 9, 30): {"flag": "D", "note": "Source doc dated 30-Sep-2025"},
    # ...
}
available_dates_sorted = sorted(my_lookup.keys())

def next_most_recent_on_or_before(target: Optional[date], today_: date) -> Optional[date]:
    """
    Given a 'target' date (or None), return the greatest available date <= (target|today_).
    If none exists, return None.
    """
    if target is None or target > today_:
        target = today_
    i = bisect.bisect_right(available_dates_sorted, target) - 1
    return available_dates_sorted[i] if i >= 0 else None

# ----------------- DATE PATTERNS (from model_output only) -----------------
PAT_TODAY_YDAY = re.compile(r'\b(?:today|yesterday)\b\s*\(\s*([^)]+?)\s*\)', re.IGNORECASE)

# "Latest available:" (high priority)
PAT_LATEST_PAREN = re.compile(r'\(\s*[^)]*\bLatest\s+available\s*[:\-–—]\s*([^)]+?)\s*\)', re.IGNORECASE)
PAT_LATEST_ANY   = re.compile(r'\bLatest\s+available\s*[:\-–—]\s*([^\n\)\]\|]+)', re.IGNORECASE)

# Month names for long DMY
MONTHS = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DMY_LONG = rf'[0-3]?\d(?:st|nd|rd|th)?\s+{MONTHS}\s+20\d{{2}}'
DMY_SLASH = r'\d{1,2}/\d{1,2}/\d{4}'

# NEW: Title with en/em-dash/hyphen then date
PAT_TITLE_DASH_LONG  = re.compile(rf'Morning\s+Call[^\n]*[–\-—]\s*({DMY_LONG})', re.IGNORECASE)
PAT_TITLE_DASH_SLASH = re.compile(rf'Morning\s+Call[^\n]*[–\-—]\s*({DMY_SLASH})', re.IGNORECASE)

# NEW: Parentheses with "Dated ..." (both slash and long month tolerated)
PAT_PAREN_DATED = re.compile(rf'\(\s*Dated\s*[:\-–—]?\s*({DMY_SLASH}|{DMY_LONG})\s*\)', re.IGNORECASE)

# Existing helpers
PAT_MOST_RECENT_1 = re.compile(r'\bmost\s+recent\s+available\s+is\s+dated\s+([^\n:]+?)(?:[\.:,\n]|$)', re.IGNORECASE)
PAT_MOST_RECENT_2 = re.compile(r'\bmost\s+recent\s+available\s*(?:is|:)\s*([^\n:]+?)(?:[\.:,\n]|$)', re.IGNORECASE)
PAT_SOURCE_ISO    = re.compile(r'\bSource:\s*\[[^\]]*,\s*(20\d{2}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}:\d{2}Z?)?', re.IGNORECASE)
PAT_TITLE_DMY     = re.compile(rf'Morning\s+Call[^\n()]*\(\s*({DMY_SLASH})\s*\)', re.IGNORECASE)  # Generic paren DMY
PAT_SENT_ISO      = re.compile(r'\bsent\s+(20\d{2}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}:\d{2}Z?)?', re.IGNORECASE)
PAT_ANY_ISO       = re.compile(r'\b(20\d{2}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}:\d{2}Z?)?\b')

def _strip_today_yesterday_wrappers(text: str) -> str:
    return PAT_TODAY_YDAY.sub('', text or '')

def _try_parse_date_like(s: str) -> Optional[pd.Timestamp]:
    """
    Parse a date-like string into tz-aware Timestamp (Europe/London).
    Supports long DMY ('25 September 2025'), DD/MM/YYYY, ISO.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip().strip('"\':,.)(').strip()
    ts = pd.to_datetime(s, errors='coerce', dayfirst=True)
    if pd.isna(ts):
        return None
    return ts.tz_localize("Europe/London") if ts.tzinfo is None else ts.tz_convert("Europe/London")

def parse_effective_datetime_from_output(
    text: str,
    prefer_explicit_most_recent: bool = True
) -> Tuple[Optional[pd.Timestamp], Optional[str], Optional[str]]:
    """
    Returns (timestamp_tz_aware, parse_reason, matched_snippet)

    Priority:
      1) 'Latest available: ...' inside parentheses (LATP)
      2) 'Latest available: ...' anywhere          (LAT)
      3) Title '... Morning Call – <date>' long    (TDASHL)
      4) Title '... Morning Call – <date>' d/m/Y   (TDASHS)
      5) Parentheses '(Dated <date>)'              (PDATED)
      6) 'most recent available is dated ...'      (MR1)
      7) 'most recent available is/:'              (MR2)
      8) Source line ISO date                      (SRC)
      9) Title '(dd/mm/yyyy)'                      (TTL)
     10) 'sent YYYY-MM-DD ...'                     (SENT)
     11) any ISO date                              (ISO)
    """
    if not isinstance(text, str) or not text.strip():
        return None, None, None

    cleaned = _strip_today_yesterday_wrappers(text)

    def _probe(pat, reason_code) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
        m = pat.search(cleaned)
        if not m:
            return None, None
        s = m.group(1)
        ts = _try_parse_date_like(s)
        return (ts, s) if ts is not None else (None, None)

    # 1–2) Latest available
    ts, s = _probe(PAT_LATEST_PAREN, "LATP")
    if ts is not None:
        return ts, "LATP", s
    ts, s = _probe(PAT_LATEST_ANY, "LAT")
    if ts is not None:
        return ts, "LAT", s

    # 3–4) Title with dash before date
    ts, s = _probe(PAT_TITLE_DASH_LONG, "TDASHL")
    if ts is not None:
        return ts, "TDASHL", s
    ts, s = _probe(PAT_TITLE_DASH_SLASH, "TDASHS")
    if ts is not None:
        return ts, "TDASHS", s

    # 5) Parenthesis "Dated ..."
    ts, s = _probe(PAT_PAREN_DATED, "PDATED")
    if ts is not None:
        return ts, "PDATED", s

    # 6–7) Most recent available ...
    if prefer_explicit_most_recent:
        ts, s = _probe(PAT_MOST_RECENT_1, "MR1")
        if ts is not None:
            return ts, "MR1", s
        ts, s = _probe(PAT_MOST_RECENT_2, "MR2")
        if ts is not None:
            return ts, "MR2", s

    # 8) Source line ISO
    ts, s = _probe(PAT_SOURCE_ISO, "SRC")
    if ts is not None:
        return ts, "SRC", s

    # 9) Title '(dd/mm/yyyy)'
    ts, s = _probe(PAT_TITLE_DMY, "TTL")
    if ts is not None:
        return ts, "TTL", s

    # 10) Sent timestamp (ISO-ish)
    ts, s = _probe(PAT_SENT_ISO, "SENT")
    if ts is not None:
        return ts, "SENT", s

    # 11) Any ISO anywhere
    ts, s = _probe(PAT_ANY_ISO, "ISO")
    if ts is not None:
        return ts, "ISO", s

    return None, None, None

# ----------------- BUILD THE DATAFRAME (keep model input as well) -----------------
def _to_text(val) -> str:
    """Serialize non-strings to a readable JSON string."""
    if isinstance(val, str) or val is None:
        return val or ""
    try:
        return json.dumps(val, ensure_ascii=False, default=str)
    except Exception:
        return str(val)

items = []
for it in paginate_items(DATASET_NAME, page_size=100):
    # INPUT: may be shaped (dict) or raw string
    if isinstance(it.input, dict):
        input_text  = _to_text(it.input.get("input"))
        source_data = _to_text(it.input.get("source_data"))
    else:
        input_text  = _to_text(it.input)
        source_data = ""  # not present for unshaped inputs

    # EXPECTED OUTPUT: unwrap model_output
    mo = getattr(it, "expected_output", None)
    if isinstance(mo, dict) and "model_output" in mo:
        model_output = _to_text(mo["model_output"])
    else:
        model_output = _to_text(mo)

    items.append({
        "id": it.id,
        "input_text": input_text,         # <-- kept
        "source_data": source_data,       # optional but handy for auditing
        "model_output": model_output,     # parse from this only
        "metadata": (getattr(it, "metadata", None) or {}).copy(),
    })

#df = pd.DataFrame(items)
#df['model'] = [df_models['model'].iloc[i] for i in df_models.index]

# Parse once and unpack columns
triples = df["model_output"].apply(lambda t: parse_effective_datetime_from_output(t, prefer_explicit_most_recent=True))
df[["call_datetime", "parse_reason", "matched_snippet"]] = pd.DataFrame(triples.tolist(), index=df.index)

# Derived dates
df["parsed_call_date"] = df["call_datetime"].apply(lambda x: x.date() if (x is not None and not pd.isna(x)) else None)

# Resolve effective date via your dictionary with 'next most recent' fallback
today_london = pd.Timestamp.now(tz="Europe/London").date()



df["effective_date"] = df["parsed_call_date"]
df["lookup_value"]   = df["effective_date"].map(my_lookup)

print('------------------------This is the dataframe before i access the meeting notes------------------------',df.head(5))
################################################################################################################################
################### NOW I NEED TO ACCESS THE MEETING NOTES
################################################################################################################################





# 2) Normalize dict keys -> datetime.date (so it matches df['effective_date'] when we normalize it)
meeting_notes_by_date = {
    pd.to_datetime(k, format="%Y-%m-%d", errors="raise").date(): v
    for k, v in SOURCE_FILE.items()
}

# 3) Normalize df['effective_date'] -> datetime.date (handles strings / Timestamp / object)
df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date

# 4) Quick diagnostics: confirm types and a direct dictionary get()
print("effective_date dtype:", df["effective_date"].dtype)
sample_idx = df["effective_date"].first_valid_index()
if sample_idx is not None:
    sample_date = df.at[sample_idx, "effective_date"]
    print("Sample value and type:", sample_date, type(sample_date))
    print("Direct dict get:", meeting_notes_by_date.get(sample_date))
else:
    print("No non-null effective_date values in df — check your parsing earlier.")

# 5) Map the notes (types now match)
df["mapped_minutes"] = df["effective_date"].map(meeting_notes_by_date)

# 6) Choose your replacement policy
REPLACE_POLICY = "replace_when_available" 
# - "replace_when_available": only replace when we have notes (safer while debugging)
# - "strict_replace": overwrite with mapped values even if NaN (will blank rows without notes)
# - "leave_original_if_missing": alias of replace_when_available

if REPLACE_POLICY in ("replace_when_available", "leave_original_if_missing"):
    df["source_data"] = df["mapped_minutes"].combine_first(df["source_data"])
elif REPLACE_POLICY == "strict_replace":
    df["source_data"] = df["mapped_minutes"]
else:
    raise ValueError("Unknown REPLACE_POLICY")

# 7) Keep only the columns you asked for
df = df[["id", "input_text", "source_data", "model_output", "effective_date","model"]]

# 8) Optional: show any rows where mapping unexpectedly failed
unmatched = df.loc[df["effective_date"].notna() & df["source_data"].isna(), ["id", "effective_date", "model_output"]]
print("Unmatched rows after mapping (should be 0 if keys exist):", len(unmatched))
if len(unmatched):
    print(unmatched.head(10))


print("The dataframe before we read it into langfuse",df.head(10))  


################################################################################################################################
################### REWRITE THE TRACES TO THE DATASET
################################################################################################################################

#Write a function here to check whether the trace is in the dataset or not


# --- Model family detection
_PAT_41 = re.compile(r"\bgpt[-_]?4\.1(\b|-|_)", re.IGNORECASE)
_PAT_5  = re.compile(r"\bgpt[-_]?5(\b|[^0-9])", re.IGNORECASE)


# 1) Split your full df by model family
df_41, df_5, df_unknown = split_by_model(df)
print("Counts:", len(df_41), "gpt-4.1 |", len(df_5), "gpt-5 |", len(df_unknown), "unknown")


# 3) Upsert 4.1 subset into a dedicated dataset
DATASET_41 = f"{DATASET_NAME}__gpt-4.1"
upsert_dataset_from_df(
    df=df_41,
    target_dataset=DATASET_41,
    dry_run=True,               # start with a preview
    id_col="id",                # if df has stable ids; else function will generate stable UUID5
    schema_version=1,
    extra_input_keys_from=None, # or e.g. ["conversation_id", "user_id"]
    source_trace_col=None,      # set if your df has a 'source_trace_id' column
    source_obs_col=None,        # set if your df has a 'source_observation_id' column
)

# When the preview looks good, flip to dry_run=False
upsert_dataset_from_df(
    df=df_41,
    target_dataset=DATASET_41,
    dry_run=False,
)

# 4) Upsert 5 subset into its dataset
DATASET_5 = f"{DATASET_NAME}__gpt-5"
upsert_dataset_from_df(
    df=df_5,
    target_dataset=DATASET_5,
    dry_run=False,
)
