# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 04 · Sync real collection from Unicorn Auctions
# MAGIC
# MAGIC Two independent parts, run in order:
# MAGIC
# MAGIC **Part 1** pulls the partner API (`/api/partner/v1/collection/items`) and
# MAGIC upserts by `uuid` into a **new, dedicated** bronze table, `bronze.collection_items`.
# MAGIC This deliberately does NOT write into `bronze.bourbon_inventory` /
# MAGIC `bronze.agave_inventory` — those are `WRITE_MODE="overwrite"` targets for the
# MAGIC LLM mock generators (`mock_data_generator/`); sharing a table would let a mock
# MAGIC run silently wipe real collection data. Re-running this notebook is always
# MAGIC safe: it's a MERGE (insert new / update changed), never an overwrite, which
# MAGIC also matches the API's own guidance ("upsert on uuid and a duplicate is
# MAGIC harmless" — a bottle edited mid-sync can legitimately reappear on a later page).
# MAGIC
# MAGIC **Part 2** flags `gold.bourbon_catalog` / `gold.agave_catalog` with an
# MAGIC `app_status` column (`available` / `in_app` / `retired`) so nothing downstream
# MAGIC has to guess whether a bottle is already live or already gone:
# MAGIC - `in_app`: currently present in `app.bourbons`/`app.bourbons_weighted`
# MAGIC   (or the agave equivalents) — already placed on a floor right now.
# MAGIC - `retired`: NOT currently present there, but has a `app.rips` row with
# MAGIC   `status IN ('shipped', 'stored')` — physically left the building (shipped to
# MAGIC   a winner) or is sitting in a winner's vault. Note `status = 'sold'`
# MAGIC   (sellback for store credit) is deliberately excluded: `decideRip`'s `sell`
# MAGIC   branch never calls `retireBottle()`, so a sold-back bottle stays on the
# MAGIC   floor — it's still `in_app`, not `retired`.
# MAGIC - `available`: neither — sitting in gold, never yet placed.
# MAGIC
# MAGIC This needs no new plumbing: Databricks already has both signals today via the
# MAGIC existing `pushRipToDatabricks`/`deleteBottleFromDatabricks` sync in
# MAGIC `celr/src/lib/databricks.server.ts`.
# MAGIC
# MAGIC ## Known gap — retail_value
# MAGIC The API's response fields (identity/physical, wine, spirits) have no price or
# MAGIC valuation field. `01_classify_and_enrich.py`'s tier/band placement is driven
# MAGIC entirely by `retail_value`, and silently drops any bottle missing one. This
# MAGIC notebook stores `retail_value = NULL` for every real bottle and prints a loud
# MAGIC warning — those rows will sync into bronze/gold fine but won't reach an app
# MAGIC floor until priced. See the note at the bottom.
# MAGIC
# MAGIC ## Known gap — incremental cursor field
# MAGIC Docs describe ordering/`updated_after` filtering by "last-modified time," but
# MAGIC no such field is named in the response field list. This notebook probes a
# MAGIC short list of likely key names on the first real page and prints what it
# MAGIC finds — confirm that against the actual API response before trusting
# MAGIC incremental mode for anything but a manual spot-check.
# MAGIC
# MAGIC `DRY_RUN=True` fetches and reports without writing anything.

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Environment (prod/dev)")
dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("bronze_schema", "bronze", "Bronze schema")
dbutils.widgets.text("gold_schema", "gold", "Gold schema")
dbutils.widgets.text("app_schema", "app", "App schema")
dbutils.widgets.text("collection_table", "collection_items", "Bronze table for real collection inventory")
dbutils.widgets.text("sync_state_table", "collection_sync_state", "Bronze table tracking the sync high-water mark")
dbutils.widgets.text("api_base", "https://api.unicornauctions.net", "Partner API base URL")
dbutils.widgets.text("secret_scope", "unicorn-auctions", "Databricks secret scope holding the API token")
dbutils.widgets.text("secret_key", "partner-api-token", "Secret key name within that scope")
dbutils.widgets.text("page_size", "500", "Items per page (API max 500)")
dbutils.widgets.dropdown("incremental", "true", ["true", "false"], "Use updated_after from last sync (falls back to full pull if no high-water mark found)")
dbutils.widgets.dropdown("full_resync", "false", ["true", "false"], "Force a full pull, ignoring any stored high-water mark")
dbutils.widgets.text("max_retries", "5", "Max retries on HTTP 429 before giving up")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (fetch + report only, no writes)")

ENV              = dbutils.widgets.get("environment")
CATALOG          = dbutils.widgets.get("catalog")
BRONZE           = dbutils.widgets.get("bronze_schema")
GOLD             = dbutils.widgets.get("gold_schema")
APP              = dbutils.widgets.get("app_schema")
COLLECTION_TBL   = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('collection_table')}"
SYNC_STATE_TBL   = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('sync_state_table')}"
BOURBON_GOLD_TBL = f"{CATALOG}.{GOLD}.bourbon_catalog"
AGAVE_GOLD_TBL   = f"{CATALOG}.{GOLD}.agave_catalog"
BOURBON_APP_TBLS = [f"{CATALOG}.{APP}.bourbons", f"{CATALOG}.{APP}.bourbons_weighted"]
AGAVE_APP_TBLS   = [f"{CATALOG}.{APP}.agave", f"{CATALOG}.{APP}.agave_weighted"]
RIPS_TBL         = f"{CATALOG}.{APP}.rips"

API_BASE     = dbutils.widgets.get("api_base").rstrip("/")
ITEMS_URL    = f"{API_BASE}/api/partner/v1/collection/items/"
SECRET_SCOPE = dbutils.widgets.get("secret_scope")
SECRET_KEY   = dbutils.widgets.get("secret_key")
PAGE_SIZE    = int(dbutils.widgets.get("page_size"))
INCREMENTAL  = dbutils.widgets.get("incremental") == "true"
FULL_RESYNC  = dbutils.widgets.get("full_resync") == "true"
MAX_RETRIES  = int(dbutils.widgets.get("max_retries"))
DRY_RUN      = dbutils.widgets.get("dry_run") == "true"

# Never hardcode the token — it lives in a Databricks secret scope, set once via
# `databricks secrets put-secret <scope> <key>`, not pasted into this notebook.
try:
    API_TOKEN = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
except Exception as e:
    raise ValueError(
        f"Couldn't read the API token from secret scope '{SECRET_SCOPE}' key '{SECRET_KEY}': {e}. "
        f"Store it with `databricks secrets put-secret {SECRET_SCOPE} {SECRET_KEY}` first."
    )

print(f"catalog={CATALOG} collection_table={COLLECTION_TBL} incremental={INCREMENTAL} "
      f"full_resync={FULL_RESYNC} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md ## Part 1a. HTTP client — pagination + 429/Retry-After handling

# COMMAND ----------

import time
import requests

def _request(url, params=None):
    """GET with Bearer auth, retrying on 429 per the documented Retry-After header."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", "5"))
            print(f"  [429] rate limited, sleeping {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        if resp.status_code == 400:
            raise ValueError(f"API rejected the request (400) — likely a bad updated_after value: {resp.text}")
        if not resp.ok:
            raise RuntimeError(f"API request failed ({resp.status_code}): {resp.text}")
        return resp.json()
    raise RuntimeError(f"Gave up after {MAX_RETRIES} retries on 429 for {url}")

def fetch_all_items(updated_after=None):
    """Yields every item across all pages. Follows `next` verbatim — never
    constructs or re-derives the cursor, per the API's own instructions."""
    params = {"page_size": PAGE_SIZE}
    if updated_after:
        params["updated_after"] = updated_after
    url = ITEMS_URL
    page = 0
    while url:
        page += 1
        body = _request(url, params=params if page == 1 else None)
        results = body.get("results", [])
        print(f"  page {page}: {len(results)} items")
        for item in results:
            yield item
        url = body.get("next")
        params = None  # `next` already carries page_size + cursor

# COMMAND ----------

# MAGIC %md ## Part 1b. Category mapping + defensive field extraction
# MAGIC Extend `TYPE_TO_CATEGORY` once you see what real `type`/`category` values the
# MAGIC API returns — unmapped rows still land in bronze (nothing is dropped), just
# MAGIC flagged so `01_classify_and_enrich.py` won't union them into WHISKEY/AGAVE gold
# MAGIC until mapped.

# COMMAND ----------

import json

TYPE_TO_CATEGORY = {
    "whiskey": "WHISKEY", "whisky": "WHISKEY", "bourbon": "WHISKEY", "rye": "WHISKEY",
    "scotch": "WHISKEY",
    "tequila": "AGAVE", "mezcal": "AGAVE", "raicilla": "AGAVE", "sotol": "AGAVE",
    "bacanora": "AGAVE",
}

def map_category(raw_type, raw_category):
    for candidate in (raw_type, raw_category):
        if candidate and str(candidate).strip().lower() in TYPE_TO_CATEGORY:
            return TYPE_TO_CATEGORY[str(candidate).strip().lower()]
    return None

# Candidate key names for the "last-modified" field the docs describe but never
# name explicitly. First match wins; printed once below so you can confirm/adjust.
MODIFIED_KEY_CANDIDATES = ["last_modified", "updated_at", "modified_at", "modified", "date_modified"]

def extract_modified(item):
    for k in MODIFIED_KEY_CANDIDATES:
        if item.get(k):
            return k, item[k]
    return None, None

def shape_item(item):
    raw_type = item.get("type")
    raw_category = item.get("category")
    mod_key, mod_val = extract_modified(item)
    return {
        "uuid": item.get("uuid"),
        "title": item.get("title"),
        "source_type": raw_type,
        "source_category": raw_category,
        "mapped_category": map_category(raw_type, raw_category),
        "size": item.get("size"),
        "quantity": int(item["quantity"]) if item.get("quantity") is not None else 1,
        "condition": item.get("condition"),
        "fill_level": item.get("fill_level"),
        "photos": item.get("photos") or [],
        "added_on": item.get("added_on"),
        "year": item.get("year"),
        "vintage": item.get("vintage"),
        "varietal": item.get("varietal"),
        "region": item.get("region"),
        "sub_region": item.get("sub_region"),
        "appellation": item.get("appellation"),
        "country": item.get("country"),
        "abv": float(item["abv"]) if item.get("abv") is not None else None,
        "age_statement": item.get("age_statement"),
        "proof": float(item["proof"]) if item.get("proof") is not None else None,
        "barrel_number": item.get("barrel_number"),
        "bottle_number": item.get("bottle_number"),
        "bottles_produced": item.get("bottles_produced"),
        "batch_description": item.get("batch_description"),
        "bottle_pick_description": item.get("bottle_pick_description"),
        "retail_value": None,   # not provided by this API — see notebook header
        "source_modified_at": str(mod_val) if mod_val is not None else None,
        "raw_json": json.dumps(item, default=str),
    }

# COMMAND ----------

# MAGIC %md ## Part 1c. Pull everything (or everything since the last high-water mark)

# COMMAND ----------

from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               DoubleType, ArrayType)

COLLECTION_SCHEMA = StructType([
    StructField("uuid", StringType()),
    StructField("title", StringType()),
    StructField("source_type", StringType()),
    StructField("source_category", StringType()),
    StructField("mapped_category", StringType()),
    StructField("size", StringType()),
    StructField("quantity", IntegerType()),
    StructField("condition", StringType()),
    StructField("fill_level", StringType()),
    StructField("photos", ArrayType(StringType())),
    StructField("added_on", StringType()),
    StructField("year", StringType()),
    StructField("vintage", StringType()),
    StructField("varietal", StringType()),
    StructField("region", StringType()),
    StructField("sub_region", StringType()),
    StructField("appellation", StringType()),
    StructField("country", StringType()),
    StructField("abv", DoubleType()),
    StructField("age_statement", StringType()),
    StructField("proof", DoubleType()),
    StructField("barrel_number", StringType()),
    StructField("bottle_number", StringType()),
    StructField("bottles_produced", StringType()),
    StructField("batch_description", StringType()),
    StructField("bottle_pick_description", StringType()),
    StructField("retail_value", DoubleType()),
    StructField("source_modified_at", StringType()),
    StructField("raw_json", StringType()),
])

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {SYNC_STATE_TBL} (
        sync_key STRING,
        last_modified_value STRING,
        last_run_at TIMESTAMP,
        items_seen_total BIGINT
    ) USING DELTA
""")

high_water_mark = None
if INCREMENTAL and not FULL_RESYNC:
    row = spark.table(SYNC_STATE_TBL).where("sync_key = 'unicorn_collection'").select("last_modified_value").limit(1).collect()
    high_water_mark = row[0]["last_modified_value"] if row else None
    if high_water_mark:
        print(f"Incremental: pulling items with updated_after={high_water_mark}")
    else:
        print("Incremental requested but no stored high-water mark yet — doing a full pull.")

items = []
mod_key_seen = None
for i, raw in enumerate(fetch_all_items(updated_after=high_water_mark)):
    if i == 0:
        mod_key_seen, mod_val = extract_modified(raw)
        if mod_key_seen:
            print(f"[info] last-modified field detected as '{mod_key_seen}' (sample value: {mod_val!r})")
        else:
            print(f"[warn] none of {MODIFIED_KEY_CANDIDATES} found on the first item — "
                  "incremental sync cannot engage until the real field name is confirmed. "
                  "Top-level keys on that item: " + ", ".join(sorted(raw.keys())))
    items.append(shape_item(raw))

print(f"\nFetched {len(items)} items total.")

unmapped = {}
for it in items:
    if it["mapped_category"] is None:
        key = (it["source_type"], it["source_category"])
        unmapped[key] = unmapped.get(key, 0) + 1
if unmapped:
    print(f"\n[warn] {sum(unmapped.values())} items have an unmapped type/category "
          f"(stored in bronze regardless, just excluded from gold until mapped):")
    for (t, c), n in sorted(unmapped.items(), key=lambda kv: -kv[1]):
        print(f"  type={t!r} category={c!r}: {n} items")

n_no_price = sum(1 for it in items if it["retail_value"] is None)
if n_no_price:
    print(f"\n[warn] {n_no_price} of {len(items)} items have no retail_value "
          "(expected — this API doesn't provide one) and won't reach an app floor "
          "until priced. See the notebook header.")

# COMMAND ----------

# MAGIC %md ## Part 1d. Upsert into bronze by uuid

# COMMAND ----------

from delta.tables import DeltaTable

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {COLLECTION_TBL} (
        uuid STRING,
        title STRING,
        source_type STRING,
        source_category STRING,
        mapped_category STRING,
        size STRING,
        quantity INT,
        condition STRING,
        fill_level STRING,
        photos ARRAY<STRING>,
        added_on STRING,
        year STRING,
        vintage STRING,
        varietal STRING,
        region STRING,
        sub_region STRING,
        appellation STRING,
        country STRING,
        abv DOUBLE,
        age_statement STRING,
        proof DOUBLE,
        barrel_number STRING,
        bottle_number STRING,
        bottles_produced STRING,
        batch_description STRING,
        bottle_pick_description STRING,
        retail_value DOUBLE,
        source_modified_at STRING,
        raw_json STRING,
        synced_at TIMESTAMP
    ) USING DELTA
""")

if not items:
    print("Nothing to write.")
elif DRY_RUN:
    print(f"DRY RUN: would upsert {len(items)} rows into {COLLECTION_TBL} (nothing written).")
    for it in items[:5]:
        print(json.dumps({k: v for k, v in it.items() if k != "raw_json"}, indent=2, default=str))
else:
    from pyspark.sql import functions as F
    updates = (spark.createDataFrame(items, COLLECTION_SCHEMA)
               .withColumn("synced_at", F.current_timestamp()))

    target = DeltaTable.forName(spark, COLLECTION_TBL)
    (target.alias("t")
        .merge(updates.alias("s"), "t.uuid = s.uuid")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute())

    hist = spark.sql(f"DESCRIBE HISTORY {COLLECTION_TBL} LIMIT 1").collect()[0]
    metrics = hist["operationMetrics"] or {}
    n_inserted = int(metrics.get("numTargetRowsInserted", 0))
    n_updated = int(metrics.get("numTargetRowsUpdated", 0))
    print(f"Upserted into {COLLECTION_TBL}: {n_inserted} new, {n_updated} updated "
          f"({len(items)} items processed).")

    if mod_key_seen:
        new_high_water = max((it["source_modified_at"] for it in items if it["source_modified_at"]), default=None)
        if new_high_water:
            spark.sql(f"""
                MERGE INTO {SYNC_STATE_TBL} t
                USING (SELECT 'unicorn_collection' AS sync_key, '{new_high_water}' AS last_modified_value,
                              current_timestamp() AS last_run_at, {len(items)}L AS items_seen_total) s
                ON t.sync_key = s.sync_key
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
            """)
            print(f"Sync state updated: high-water mark = {new_high_water}")
    else:
        print("[warn] sync state NOT updated — last-modified field wasn't found, so the "
              "next run will fall back to a full pull again (safe, just not incremental).")

# COMMAND ----------

# MAGIC %md ## Part 2. Flag gold with app_status
# MAGIC Additive, safe to run anytime — `ALTER TABLE ADD COLUMN IF NOT EXISTS` and
# MAGIC then a full recompute of the flag from current `app`/`rips` state.

# COMMAND ----------

from pyspark.sql import functions as F

def flag_gold(gold_table, app_tables, spirit_column_on_rips):
    spark.sql(f"ALTER TABLE {gold_table} ADD COLUMN IF NOT EXISTS app_status STRING")

    in_app_ids = None
    for t in app_tables:
        try:
            df = spark.table(t).select("id").withColumnRenamed("id", "bottle_serial")
            in_app_ids = df if in_app_ids is None else in_app_ids.unionByName(df)
        except Exception as e:
            print(f"  [info] {t} not readable ({e}); skipping for in_app detection")
    if in_app_ids is None:
        raise ValueError(f"None of {app_tables} were readable — can't compute app_status for {gold_table}")
    in_app_ids = in_app_ids.distinct()

    try:
        retired_ids = (spark.table(RIPS_TBL)
            .where(f"{spirit_column_on_rips} IS NOT NULL AND status IN ('shipped', 'stored')")
            .select(F.col(spirit_column_on_rips).alias("bottle_serial"))
            .distinct())
    except Exception as e:
        print(f"  [info] {RIPS_TBL} not readable ({e}); treating retired set as empty")
        retired_ids = spark.createDataFrame([], "bottle_serial string")

    gold = spark.table(gold_table)
    status = (gold.select("bottle_serial")
        .join(in_app_ids.withColumn("_in_app", F.lit(True)), "bottle_serial", "left")
        .join(retired_ids.withColumn("_retired", F.lit(True)), "bottle_serial", "left")
        .withColumn("app_status",
            F.when(F.col("_in_app"), "in_app")
             .when(F.col("_retired"), "retired")
             .otherwise("available"))
        .select("bottle_serial", "app_status"))

    if DRY_RUN:
        counts = status.groupBy("app_status").count().collect()
        print(f"DRY RUN: {gold_table} app_status would be — " +
              ", ".join(f"{r['app_status']}={r['count']}" for r in counts))
        return

    target = DeltaTable.forName(spark, gold_table)
    (target.alias("t")
        .merge(status.alias("s"), "t.bottle_serial = s.bottle_serial")
        .whenMatchedUpdate(set={"app_status": "s.app_status"})
        .execute())
    counts = spark.table(gold_table).groupBy("app_status").count().collect()
    print(f"{gold_table} app_status — " + ", ".join(f"{r['app_status']}={r['count']}" for r in counts))

flag_gold(BOURBON_GOLD_TBL, BOURBON_APP_TBLS, "bourbon_id")
flag_gold(AGAVE_GOLD_TBL, AGAVE_APP_TBLS, "agave_id")

# COMMAND ----------

# MAGIC %md ## Next steps (not done here)
# MAGIC 1. **retail_value.** Decide how real bottles get priced (manual appraisal
# MAGIC    field, LLM estimate, external pricing feed) — until then they sit in
# MAGIC    `bronze.collection_items` / gold but never reach an app floor.
# MAGIC 2. **Wire into gold.** `01_classify_and_enrich.py` currently only unions
# MAGIC    `bronze.bourbon_inventory` + `bronze.agave_inventory`. It needs a third
# MAGIC    source reading `bronze.collection_items` (split on `mapped_category`,
# MAGIC    `uuid` as `bottle_serial`, `title` as `name`) once retail_value is solved —
# MAGIC    say the word and I'll wire that union in.
# MAGIC 3. **Exclude retired bottles from future floor builds.** Once `app_status`
# MAGIC    exists, `02_build_bourbon_app_floor_weighted.py` / `03_...agave...` should
# MAGIC    filter `WHERE app_status != 'retired'` on their `SOURCE_TABLE` read so a
# MAGIC    shipped/stored bottle never gets re-placed on a rebuilt floor. Small,
# MAGIC    additive change — I held off only because this notebook is what creates
# MAGIC    the column in the first place.
