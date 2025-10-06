
import re
import json
import uuid
import numpy as np
import pandas as pd
from copy import deepcopy
from langfuse import get_client
import uuid
from langfuse import Langfuse


langfuse = Langfuse(
  secret_key="",
  public_key="",
  host=""
)

# --- Safe stringification (keeps your behavior)
def as_text(x) -> str:
    """Return a safe string for preview/upsert. NaN/None -> ''."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return str(x)

def preview(x, n=120) -> str:
    """Short, single-line preview."""
    s = as_text(x)
    return s.replace("\n", " ")[:n]


# --- Model family detection
_PAT_41 = re.compile(r"\bgpt[-_]?4\.1(\b|-|_)", re.IGNORECASE)
_PAT_5  = re.compile(r"\bgpt[-_]?5(\b|[^0-9])", re.IGNORECASE)

def is_gpt41(m: str) -> bool:
    return bool(_PAT_41.search((m or "").lower()))

def is_gpt5(m: str) -> bool:
    return bool(_PAT_5.search((m or "").lower()))

def split_by_model(df: pd.DataFrame):
    """Return df_41, df_5, df_unknown based on df['model']."""
    def _tag(m):
        if is_gpt41(m): return "gpt-4.1"
        if is_gpt5(m):  return "gpt-5"
        return "unknown"

    if "model" not in df.columns:
        raise ValueError("split_by_model: DataFrame must contain a 'model' column.")

    tags = df["model"].map(_tag)
    return (
        df[tags == "gpt-4.1"].copy(),
        df[tags == "gpt-5"].copy(),
        df[tags == "unknown"].copy()
    )


# --- Langfuse dataset helpers (Python SDK surface)
def ensure_dataset(dataset_name: str, description: str = None, metadata: dict = None):
    """Create dataset if missing (Python SDK: create_dataset / get_dataset)."""
    try:
        langfuse.get_dataset(dataset_name)
    except Exception:
        langfuse.create_dataset(name=dataset_name, description=description, metadata=metadata or {})


# --- Shaping to your desired schema
def set_source_data_on_input_for_new(input_text: str, source_data: str, extra: dict | None = None):
    """
    Build the 'input' payload for dataset_item:
      {
        "input": "<input_text as string>",
        "source_data": "<source_data as string>",
        ... any extra keys ...
      }
    """
    obj = {"input": as_text(input_text), "source_data": as_text(source_data)}
    if extra:
        obj.update(extra)
    return obj

def shape_expected_output(model_output: str):
    """Wrap expected_output to the schema you showed."""
    return {"model_output": as_text(model_output)}


# --- Stable id generation for new datasets
import uuid

def make_item_id(dataset_name: str, row_id_value, row_index: int, strategy: str = "namespaced") -> str:
    """
    Strategies:
      - 'namespaced' (default): stable UUID5 derived from (dataset_name, base_id)
      - 'prefix'            : f"{dataset_name}::{base_id}"
      - 'reuse'             : use the raw id from df (DANGEROUS when splitting)
    base_id = df[id_col] if present, else "row:{row_index}"
    """
    base_id = None
    if row_id_value is not None and str(row_id_value) != "nan":
        base_id = str(row_id_value)
    else:
        base_id = f"row:{row_index}"

    if strategy == "namespaced":
        ns = uuid.uuid5(uuid.NAMESPACE_URL, f"langfuse:{dataset_name}")
        return str(uuid.uuid5(ns, base_id))
    elif strategy == "prefix":
        return f"{dataset_name}::{base_id}"
    elif strategy == "reuse":
        # only safe when writing to a single dataset; collisions likely across datasets
        if row_id_value is None or str(row_id_value) == "nan":
            raise ValueError("id_strategy='reuse' requires non-null ids in the df")
        return str(row_id_value)
    else:
        raise ValueError(f"Unknown id strategy: {strategy}")

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

def existing_trace_ids(dataset_name: str) -> set[str]:
    """Collect all source_trace_id values already present in the dataset."""
    ids = set()
    for it in paginate_items(dataset_name, page_size=200):
        stid = getattr(it, "source_trace_id", None)
        if stid:
            ids.add(str(stid))
    return ids



# --- Core function: take a DataFrame, transform, and upsert to a (new) dataset
# def upsert_dataset_from_df(
#     df: pd.DataFrame,
#     target_dataset: str,
#     *,
#     dry_run: bool = True,
#     id_col: str = "id",
#     include_backup: bool = False,
#     schema_version: int = 1,
#     extra_input_keys_from: list[str] | None = None,
#     source_trace_col: str | None = None,
#     source_obs_col: str | None = None,
# ):
#     """
#     Create/Upsert Langfuse dataset items from a DataFrame.

#     Expects df to contain at least:
#       - 'input_text'      (stringifiable)
#       - 'source_data'     (stringifiable)
#       - 'model_output'    (stringifiable)
#     Optional:
#       - 'effective_date'  (string/date) -> stored in metadata
#       - id_col            (default 'id') -> used as item id, else stable UUID5
#       - 'model'           -> copied into metadata['model']
#       - parse/debug cols (e.g., 'parse_reason', 'matched_snippet') -> copied into metadata
#       - `source_trace_col` and `source_obs_col` if linking to production traces

#     Transformation:
#       input  := {"input": <input_text>, "source_data": <source_data>, **extras}
#       output := {"model_output": <model_output>}
#       metadata includes: schema_version, effective_date (if present), model (if present), etc.
#     """
#     required = ["input_text", "source_data", "model_output"]
#     missing_cols = [c for c in required if c not in df.columns]
#     if missing_cols:
#         raise ValueError(f"DataFrame missing required columns: {missing_cols}")

#     # Normalize the three text columns to strings
#     for col in required:
#         df[col] = df[col].astype(object).apply(as_text)

#     # Preview
#     # print(f"\nPreparing to upsert {len(df):,} items into dataset '{target_dataset}' (dry_run={dry_run})")
#     # print("Preview of first 5 rows:")
#     # for i, row in df.head(5).iterrows():
#     #     print(f"- idx={i}  id={row.get(id_col)}")
#     #     print("  input_text:", preview(row.get("input_text")))
#     #     print("  source_data:", preview(row.get("source_data")))
#     #     print("  model_output:", preview(row.get("model_output")))
#     #     if "effective_date" in row:
#     #         print("  effective_date:", row.get("effective_date"))

#     # if dry_run:
#     #     print("\nDry run only. Set dry_run=False to write to Langfuse.")
#     #     return

#     # Ensure dataset exists
#     ensure_dataset(target_dataset, description=f"Auto-created from DataFrame split")

#     # Upsert loop
#     created = 0
#     for i, row in df.iterrows():
#         item_id = make_item_id(target_dataset, row.get(id_col), i)

#         # Optional extra input keys to propagate from df -> input (e.g., 'conversation_id', etc.)
#         extras = {}
#         if extra_input_keys_from:
#             for k in extra_input_keys_from:
#                 if k in df.columns:
#                     extras[k] = row.get(k)

#         input_obj = set_source_data_on_input_for_new(
#             input_text=row.get("input_text"),
#             source_data=row.get("source_data"),
#             extra=extras
#         )
#         expected_obj = shape_expected_output(row.get("model_output"))

#         # Build metadata
#         meta = {}
#         meta["schema_version"] = int(schema_version)
#         eff = row.get("effective_date")
#         if eff is not None and not (isinstance(eff, float) and np.isnan(eff)):
#             meta["effective_date"] = str(eff)

#         # Copy optional diagnostic fields if present
#         for opt in ["model", "parse_reason", "matched_snippet"]:
#             if opt in df.columns and row.get(opt) is not None:
#                 meta[opt] = row.get(opt)

#         # If requested, store backup of original columns in metadata (use sparingly)
#         if include_backup:
#             backup = {}
#             for k in ["input_text", "source_data", "model_output"]:
#                 backup[k] = row.get(k)
#             meta["migration_backup"] = backup

#         # Links to production trace/observation (if provided)
#         src_trace = str(row.get(source_trace_col)) if source_trace_col and pd.notna(row.get(source_trace_col)) else None
#         src_obs   = str(row.get(source_obs_col))   if source_obs_col and pd.notna(row.get(source_obs_col))   else None

#         # Python SDK upsert call (idempotent if id is stable)
#         langfuse.create_dataset_item(
#             dataset_name=target_dataset,
#             id=item_id,
#             input=input_obj,
#             expected_output=expected_obj,
#             metadata=meta,
#             source_trace_id=src_trace,
#             source_observation_id=src_obs,
#         )
#         created += 1

#     print(f"\nDone. Upserted {created:,} items into '{target_dataset}'.")



def upsert_dataset_from_df(
    df: pd.DataFrame,
    target_dataset: str,
    *,
    dry_run: bool = True,
    id_col: str = "id",
    include_backup: bool = False,
    schema_version: int = 1,
    extra_input_keys_from: list[str] | None = None,
    source_trace_col: str | None = None,
    source_obs_col: str | None = None,
    skip_if_trace_exists: bool = True,    # NEW: guard by source_trace_id
    skip_if_id_exists: bool = False,      # Optional: guard by dataset item id
):
    required = ["input_text", "source_data", "model_output"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Normalize core text columns
    for col in required:
        df[col] = df[col].astype(object).apply(as_text)

    # Ensure the dataset exists
    ensure_dataset(target_dataset, description="Auto-created from DataFrame split")

    # --- Build existing indexes ONCE (O(N) over dataset, O(1) checks per row) ---
    existing_item_ids: set[str] = set()
    existing_trace_ids: set[str] = set()

    if skip_if_trace_exists or skip_if_id_exists:
        for it in paginate_items(target_dataset, page_size=200):
            if skip_if_id_exists and getattr(it, "id", None):
                existing_item_ids.add(str(it.id))
            if skip_if_trace_exists and getattr(it, "source_trace_id", None):
                existing_trace_ids.add(str(it.source_trace_id))

    created = skipped = 0
    for i, row in df.iterrows():
        # Stable application-defined id
        item_id = make_item_id(target_dataset, row.get(id_col), i)

        # Optional: guard by dataset-item id
        if skip_if_id_exists and item_id in existing_item_ids:
            skipped += 1
            continue

        # Optional: guard by source_trace_id (preferred for your use case)
        src_trace = (
            str(row.get(source_trace_col))
            if source_trace_col and pd.notna(row.get(source_trace_col))
            else None
        )
        if skip_if_trace_exists and src_trace and src_trace in existing_trace_ids:
            skipped += 1
            continue

        # Prepare extra keys (if any)
        extras = {}
        if extra_input_keys_from:
            for k in extra_input_keys_from:
                if k in df.columns:
                    extras[k] = row.get(k)

        input_obj = set_source_data_on_input_for_new(
            input_text=row.get("input_text"),
            source_data=row.get("source_data"),
            extra=extras,
        )
        expected_obj = shape_expected_output(row.get("model_output"))

        # Metadata
        meta = {"schema_version": int(schema_version)}
        eff = row.get("effective_date")
        if eff is not None and not (isinstance(eff, float) and np.isnan(eff)):
            meta["effective_date"] = str(eff)
        for opt in ["model", "parse_reason", "matched_snippet"]:
            if opt in df.columns and row.get(opt) is not None:
                meta[opt] = row.get(opt)
        if include_backup:
            meta["migration_backup"] = {
                k: row.get(k) for k in ["input_text", "source_data", "model_output"]
            }

        # Source links
        src_obs = (
            str(row.get(source_obs_col))
            if source_obs_col and pd.notna(row.get(source_obs_col))
            else None
        )

        if dry_run:
            created += 1
            continue

        # Create (idempotent if your id is stable; we still guarded to avoid hitting API)
        langfuse.create_dataset_item(
            dataset_name=target_dataset,
            id=item_id,
            input=input_obj,
            expected_output=expected_obj,
            metadata=meta,
            source_trace_id=src_trace,
            source_observation_id=src_obs,
        )
        created += 1

        # Keep indexes hot if we’re running long jobs
        if skip_if_id_exists:
            existing_item_ids.add(item_id)
        if skip_if_trace_exists and src_trace:
            existing_trace_ids.add(src_trace)

    print(f"\nDone. Upserted {created:,} items into '{target_dataset}'. Skipped (existing): {skipped:,}.")



#as_text, is_gpt5,is_gpt41,  split_by_model
