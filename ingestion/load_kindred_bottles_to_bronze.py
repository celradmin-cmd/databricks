# Databricks notebook source
# MAGIC %md
# MAGIC # Load Kindred bottle list into bronze
# MAGIC
# MAGIC Replaces the mock generators (`mock_data_generator/`) with the **real** bottle list
# MAGIC from Kindred — `Kindred — Whiskey & Premium Tequila Online Pricing.csv`, uploaded to
# MAGIC a Unity Catalog volume.
# MAGIC
# MAGIC Writes two things:
# MAGIC 1. **`bronze.kindred_bottles_raw`** — every CSV row, columns renamed to snake_case but
# MAGIC    values untouched, plus a deterministic `bottle_serial` and load metadata. This is the
# MAGIC    source `scrape_bottle_images.py` reads (it needs `source_url`).
# MAGIC 2. **`bronze.bourbon_inventory` / `bronze.agave_inventory`** — the same
# MAGIC    `bottle_serial, name, category, description, rarity, retail_value, distillery` shape
# MAGIC    the mock generators wrote, so `01_classify_and_enrich.py` runs unchanged.
# MAGIC    `description`/`rarity` are left null for `01`'s LLM step to fill.
# MAGIC
# MAGIC ## Routing
# MAGIC - Category mentions tequila / agave / mezcal → `agave_inventory` (`AGAVE`)
# MAGIC - Category mentions whiskey / whisky / bourbon / rye → `bourbon_inventory` (`WHISKEY`)
# MAGIC - Anything else (the Copper & Kings brandies) → `other_spirits` widget: `skip` (default)
# MAGIC   or `whiskey` to put them on the bourbon floor anyway.
# MAGIC - Rows with **no price** stay in the raw table but are kept out of inventory —
# MAGIC   `01` tiers bottles by `retail_value` and can't place one without it.
# MAGIC
# MAGIC ## bottle_serial
# MAGIC A `uuid5` of `brand|product|size`, not a random `uuid4` like the mock generators. Re-loading
# MAGIC the same CSV (or a re-priced copy) gives every bottle the same serial, so the `merge`
# MAGIC write mode updates prices in place and leaves the `ever_placed` / `app_status` flags
# MAGIC `sync_collection_from_unicorn.py` writes back to bronze alone.
# MAGIC
# MAGIC **One row per SKU, unlimited supply.** These come direct from the distributor, so a row is a
# MAGIC product we can always order more of, not one physical bottle. Note the rest of the pipeline
# MAGIC (`ever_placed`, the reserve pool in `04`, the buy list in `05`) still assumes each row is a
# MAGIC single bottle that is used up once won.

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("bronze_schema", "bronze", "Bronze schema")
dbutils.widgets.text("csv_path", "/Volumes/prod_celr/bronze/raw_files/kindred_bottles.csv", "CSV path (UC volume)")
dbutils.widgets.text("raw_table", "kindred_bottles_raw", "Bronze raw table")
dbutils.widgets.text("whiskey_table", "bourbon_inventory", "Bronze whiskey inventory table")
dbutils.widgets.text("agave_table", "agave_inventory", "Bronze agave inventory table")
dbutils.widgets.dropdown("other_spirits", "skip", ["skip", "whiskey"], "Brandy / other: skip, or load as whiskey")
dbutils.widgets.dropdown("write_mode", "merge", ["merge", "overwrite"],
                         "merge = upsert by bottle_serial; overwrite = replace inventory (drops mock bottles)")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (no write)")

CATALOG       = dbutils.widgets.get("catalog")
BRONZE        = dbutils.widgets.get("bronze_schema")
CSV_PATH      = dbutils.widgets.get("csv_path").strip()
RAW_TABLE     = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('raw_table')}"
WHISKEY_TABLE = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('whiskey_table')}"
AGAVE_TABLE   = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('agave_table')}"
OTHER_SPIRITS = dbutils.widgets.get("other_spirits")
WRITE_MODE    = dbutils.widgets.get("write_mode")
DRY_RUN       = dbutils.widgets.get("dry_run") == "true"

print(f"csv={CSV_PATH}\nraw={RAW_TABLE}\nwhiskey={WHISKEY_TABLE}\nagave={AGAVE_TABLE}\n"
      f"other_spirits={OTHER_SPIRITS} write_mode={WRITE_MODE} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the CSV
# MAGIC Read as all-strings: bronze keeps what the file said (`$2,700.00`, `750ml`). Parsing happens
# MAGIC only on the way into the inventory tables. `multiLine` + `escape='"'` because product
# MAGIC names contain quoted commas (`"1953, 17-Year Straight Bourbon Whiskey, ..."`).

# COMMAND ----------

from pyspark.sql import functions as F

# CSV header -> bronze column. Delta rejects spaces/parens/slashes in column names.
COLUMNS = {
    "Brand": "brand",
    "Product / Expression": "product",
    "Category": "source_category",
    "Size": "size",
    "Online Price (USD)": "online_price",
    "Source URL": "source_url",
    "Checked": "price_checked",
}

raw = (spark.read
       .option("header", "true")
       .option("multiLine", "true")
       .option("escape", '"')
       .option("encoding", "UTF-8")
       .option("inferSchema", "false")
       .csv(CSV_PATH))

missing = set(COLUMNS) - set(raw.columns)
assert not missing, f"CSV is missing expected columns {sorted(missing)}; found {raw.columns}"

raw = raw.select([F.trim(F.col(f"`{src}`")).alias(dst) for src, dst in COLUMNS.items()])
raw = raw.select([F.when(F.col(c) == "", None).otherwise(F.col(c)).alias(c) for c in raw.columns])
raw = raw.filter(F.col("brand").isNotNull() & F.col("product").isNotNull())

print(f"{raw.count()} bottles in CSV")
display(raw)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Serial, parsed fields, routing

# COMMAND ----------

import uuid
from pyspark.sql.types import StringType

# Fixed namespace so the same bottle always hashes to the same serial, run after run.
SERIAL_NS = uuid.UUID("6c3f4c1e-2b1a-4d5e-9a7f-6b0e1c2d3a4f")

@F.udf(returnType=StringType())
def bottle_serial(brand, product, size):
    key = "|".join((s or "").strip().lower() for s in (brand, product, size))
    return str(uuid.uuid5(SERIAL_NS, key))

cat = F.lower("source_category")
spirit = (F.when(cat.rlike("tequila|agave|mezcal"), F.lit("AGAVE"))
           .when(cat.rlike("whisk|bourbon|rye"), F.lit("WHISKEY")))
if OTHER_SPIRITS == "whiskey":
    spirit = spirit.otherwise(F.lit("WHISKEY"))

bottles = (raw
    .withColumn("size_norm", F.regexp_replace(F.lower("size"), r"\s+", ""))   # "750 mL" == "750ml"
    .withColumn("bottle_serial", bottle_serial("brand", "product", "size_norm"))
    .withColumn("retail_value", F.regexp_replace("online_price", r"[^0-9.]", "").cast("double"))
    .withColumn("size_ml", F.regexp_extract("size_norm", r"(\d+)", 1).cast("int"))
    .withColumn("category", spirit)
    # "Black Emerald 7 Year Old" already names its brand; "Dàn Yǎ" doesn't.
    .withColumn("name", F.when(F.instr(F.lower("product"), F.lower("brand")) > 0, F.col("product"))
                         .otherwise(F.concat_ws(" ", "brand", "product")))
    .drop("size_norm"))

dupes = bottles.groupBy("bottle_serial").count().filter("count > 1")
assert dupes.count() == 0, "duplicate brand|product|size rows in CSV:\n" + \
    "\n".join(str(r) for r in bottles.join(dupes, "bottle_serial").select("brand", "product", "size").collect())

no_price = bottles.filter(F.col("retail_value").isNull())
unrouted = bottles.filter(F.col("category").isNull())
loadable = bottles.filter(F.col("retail_value").isNotNull() & F.col("category").isNotNull())

print(f"loadable={loadable.count()}  no_price={no_price.count()}  unrouted={unrouted.count()}")
if no_price.count():
    print("\nNo price — raw table only, not inventory:")
    no_price.select("brand", "product").show(truncate=False)
if unrouted.count():
    print(f"\nNot whiskey or agave (other_spirits={OTHER_SPIRITS}) — raw table only:")
    unrouted.select("brand", "product", "source_category").show(truncate=False)

display(loadable.groupBy("category").agg(F.count("*").alias("bottles"),
                                         F.min("retail_value").alias("min_price"),
                                         F.max("retail_value").alias("max_price")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write the raw table
# MAGIC Always a full overwrite — it's a snapshot of the file, nothing downstream writes to it.

# COMMAND ----------

raw_out = (bottles
    .select("bottle_serial", "brand", "product", "source_category", "size", "online_price",
            "source_url", "price_checked", "retail_value", "size_ml",
            F.col("category").alias("routed_category"))
    .withColumn("_source_file", F.lit(CSV_PATH))
    .withColumn("_ingested_at", F.current_timestamp()))

if DRY_RUN:
    print(f"DRY RUN - would overwrite {RAW_TABLE} with {raw_out.count()} rows")
else:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE}")
    (raw_out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(RAW_TABLE))
    print(f"overwrote {RAW_TABLE} with {raw_out.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write the inventory tables
# MAGIC - **merge** (default): upsert by `bottle_serial`. Existing Kindred rows get the new price
# MAGIC   and name; bronze-only columns (`ever_placed`, `app_status`, `first_placed_at`) and any
# MAGIC   mock bottles already in the table are left alone.
# MAGIC - **overwrite**: the table becomes exactly this CSV. Use once to clear out the mock
# MAGIC   bottles — it also drops their `ever_placed` ledger, which is fine for mock rows but
# MAGIC   means **never overwrite after real bottles have been placed on a floor.**

# COMMAND ----------

INVENTORY_COLS = ["bottle_serial", "name", "category", "description", "rarity", "retail_value", "distillery"]

def inventory_rows(category):
    return (loadable.filter(F.col("category") == category)
            .select("bottle_serial", "name", "category",
                    F.lit(None).cast("string").alias("description"),   # 01's LLM fills these
                    F.lit(None).cast("string").alias("rarity"),
                    "retail_value",
                    F.col("brand").alias("distillery")))

def write_inventory(df, table):
    n = df.count()
    if DRY_RUN:
        print(f"DRY RUN - would {WRITE_MODE} {n} rows into {table}")
        return
    if WRITE_MODE == "overwrite" or not spark.catalog.tableExists(table):
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        print(f"overwrote {table} with {n} rows")
        return
    df.createOrReplaceTempView("_kindred_src")
    spark.sql(f"""
        MERGE INTO {table} t
        USING _kindred_src s
        ON t.bottle_serial = s.bottle_serial
        WHEN MATCHED THEN UPDATE SET
            t.name = s.name, t.category = s.category,
            t.retail_value = s.retail_value, t.distillery = s.distillery
        WHEN NOT MATCHED THEN INSERT ({', '.join(INVENTORY_COLS)})
            VALUES ({', '.join('s.' + c for c in INVENTORY_COLS)})
    """)
    print(f"merged {n} rows into {table}")

write_inventory(inventory_rows("WHISKEY"), WHISKEY_TABLE)
write_inventory(inventory_rows("AGAVE"), AGAVE_TABLE)

if DRY_RUN:
    print("\nDRY RUN - set dry_run=false to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Next steps (not run here)
# MAGIC 1. `ingestion/scrape_bottle_images.py` — pull each bottle's photo from its `source_url`.
# MAGIC 2. `01_classify_and_enrich.py` — tiers, bands, LLM rarity + description into gold.
# MAGIC 3. `02_build_bourbon_app_floor_weighted.py` / `03_build_agave_app_floor_weighted.py`.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prod_celr.bronze.kindred_bottles_raw order by brand, product
