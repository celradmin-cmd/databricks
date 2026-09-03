# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Classify + Enrich
# MAGIC
# MAGIC **Input:** the source-system inventory (`silver.source_inventory`). **Output:** a curated
# MAGIC `bourbon_catalog` Gold table where every bottle has (a) its eligible tier(s) and band, and
# MAGIC (b) an LLM-generated **rarity** (`common` / `rare` / `legendary`) and **one-sentence
# MAGIC description**, produced by **Llama 4 Maverick** via Databricks Foundation Model APIs.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Read source inventory.
# MAGIC 2. Assign each bottle a **primary tier** (closest-to-par) + all eligible tiers.
# MAGIC 3. **Enrich** each bottle with rarity + description (Llama 4 Maverick, structured JSON out).
# MAGIC 4. Write `bourbon_catalog` + a per-tier **odds breakdown** view.
# MAGIC
# MAGIC No image generation. `image_url` is left null here — populate it from the retailer's real
# MAGIC product photography downstream (the safer choice for trademarked bottles anyway). Llama 4
# MAGIC Maverick is a Databricks-hosted pay-per-token endpoint: no AWS credentials, no Bedrock quota.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Environment (prod/dev)")
dbutils.widgets.text("llm_endpoint", "databricks-llama-4-maverick", "Foundation Model endpoint")
dbutils.widgets.text("llm_temperature", "0.2", "LLM temperature")
dbutils.widgets.dropdown("enrich", "true", ["true", "false"], "Run LLM enrichment?")

ENV         = dbutils.widgets.get("environment")
CATALOG     = f"{ENV}_celr"                       # Unity Catalog: celr_prod / celr_dev
SRC_TABLE   = f"{CATALOG}.bronze.inventory"
BOURBON_CATALOG_TBL = f"{CATALOG}.gold.bourbon_catalog"
AGAVE_CATALOG_TBL = f"{CATALOG}.gold.agave_catalog"
BOURBON_ODDS_VIEW   = f"{CATALOG}.gold.bourbon_tier_odds_breakdown"
AGAVE_ODDS_VIEW   = f"{CATALOG}.gold.agave_tier_odds_breakdown"

LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint").strip()
LLM_TEMP     = float(dbutils.widgets.get("llm_temperature"))
DO_ENRICH    = dbutils.widgets.get("enrich") == "true"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read source inventory
# MAGIC Expected columns: `source_sku, name, distillery, description, rarity, retail_value, quantity`.
# MAGIC (`rarity`/`description` may be null coming in — enrichment fills them.)

# COMMAND ----------

from pyspark.sql import functions as F

src = (spark.table(SRC_TABLE)
       .withColumn("name", F.trim("product_name"))\
       .withColumn('retail_value', F.coalesce(F.col('avg_price'), F.col('median_price'))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Assign tiers + bands
# MAGIC `primary_tier` = where the bottle sits closest to par. `eligible_tiers` = every tier whose
# MAGIC curve can host it (the replacement engine uses this to fill a gap in any tier).

# COMMAND ----------

from pyspark.sql.types import IntegerType, ArrayType, StructType, StructField, DoubleType

# Helpers from 00_celr_odds_config (primary_tier, eligible_tiers, band_index).
@F.udf(returnType=IntegerType())
def udf_primary_tier(rv):
    return primary_tier(rv)

@F.udf(returnType=ArrayType(StructType([
    StructField("tier", IntegerType()),
    StructField("multiple", DoubleType()),
    StructField("band_idx", IntegerType()),
])))
def udf_eligible(rv):
    return [(int(t), float(m), int(b)) for (t, m, b) in eligible_tiers(rv)]

@F.udf(returnType=IntegerType())
def udf_primary_band(rv):
    t = primary_tier(rv)
    return band_index(rv, t) if t is not None else None

catalog = (src
    .withColumn("primary_tier", udf_primary_tier("retail_value"))
    .withColumn("primary_band", udf_primary_band("retail_value"))
    .withColumn("eligible_tiers", udf_eligible("retail_value"))
    .withColumn("placeable", F.col("primary_tier").isNotNull()))

unplaceable = catalog.where(~F.col("placeable"))
if unplaceable.count() > 0:
    print("Bottles that fit no tier curve (review these):")
    unplaceable.select("name", "retail_value").show(truncate=False)

catalog = catalog.where(F.col("placeable"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Enrich: rarity + one-sentence description (Llama 4 Maverick)
# MAGIC For each distinct bottle we ask Llama 4 Maverick for a collectibility `rarity`
# MAGIC (`common`/`rare`/`legendary`) and a single vivid sentence. **Structured output** pins the
# MAGIC response to a JSON schema, so rarity is always one of the three enum values — no parsing
# MAGIC guesswork. This rarity is the LLM's judgment of real-world scarcity; it is independent of
# MAGIC the tier/band math (which stays purely retail-value driven).
# MAGIC
# MAGIC Auth is automatic inside Databricks via the MLflow deployments client — no token to manage.

# COMMAND ----------

import json
import mlflow.deployments

_deploy = mlflow.deployments.get_deploy_client("databricks")

SYSTEM = (
    "You are an expert on Liquor. For the given bottle, decide its "
    "collectibility rarity and write one vivid sentence describing it.\n"
    "- rarity: exactly one of common, rare, legendary. Judge by real-world availability and "
    "secondary-market demand: widely-available shelf bottles are common; allocated or "
    "hard-to-find bottles are rare; highly sought, very limited, or iconic bottles "
    "(e.g. Pappy Van Winkle, the Buffalo Trace Antique Collection) are legendary.\n"
    "- description: exactly ONE sentence, 30 words or fewer, focused on flavor and character. "
    "No price, no marketing fluff, no unverifiable claims."
    "- distillery: What distillery is this bottle from? If unknown, say so"
)

# Structured output: rarity constrained to the enum, description free text.
RESP_FMT = {
    "type": "json_schema",
    "json_schema": {
        "name": "Liquor_enrichment",
        "schema": {
            "type": "object",
            "properties": {
                "rarity": {"type": "string", "enum": ["common", "rare", "legendary"]},
                "description": {"type": "string"},
                "distillery":{"type": "string"}
            },
            "required": ["rarity", "description", "distillery"],
        },
        "strict": True,
    },
}

def _price_rarity(rv):
    return "common" if rv < 50 else ("rare" if rv < 250 else "legendary")

def enrich(name, distillery, retail_value):
    """Return (rarity, description, distillery). Falls back to price-derived rarity + generic text on error,
    so one bad LLM call never fails the whole batch."""
    user = f"Bottle: {name}"
    if distillery:
        user += f" by {distillery}"
    user += f". Approximate retail ${int(retail_value)}."
    try:
        resp = _deploy.predict(endpoint=LLM_ENDPOINT, inputs={
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "max_tokens": 200,
            "temperature": LLM_TEMP,
            "response_format": RESP_FMT,
        })
        out = json.loads(resp["choices"][0]["message"]["content"])
        rarity = out["rarity"].strip().lower()
        if rarity not in ("common", "rare", "legendary"):
            rarity = _price_rarity(retail_value)
        return rarity, out["description"].strip(), out["distillery"].strip()
    except Exception as e:
        print(f"  ! enrich failed for {name}: {e} — using price-derived fallback")
        return _price_rarity(retail_value), None, None

# COMMAND ----------

import pandas as pd

rows = (catalog.select("name", "category", "retail_value").distinct()
        .toPandas().to_dict("records"))

results = []
for r in rows:
    if DO_ENRICH:
        rarity, desc, distillery = enrich(r["name"], r.get("category"), r["retail_value"])
    else:
        rarity, desc, distillery = _price_rarity(r["retail_value"]), None, None
    results.append({"name": r["name"], "llm_rarity": rarity, "llm_description": desc, "llm_distillery": distillery})
    print(f"[{rarity:9}] {r['name']}: {desc}")

enrich_df = spark.createDataFrame(pd.DataFrame(results))

# Overlay LLM values; keep any incoming rarity/description only where the LLM returned nothing.
catalog = catalog.join(enrich_df, on="name", how="left")\
    .withColumn("rarity", F.col("llm_rarity"))\
    .withColumn("description", F.col("llm_description"))\
    .withColumn("distillery", F.col("llm_distillery"))\
    .drop("llm_rarity", "llm_description", "llm_distillery")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: enrich the whole table in one shot with `ai_query` (serverless only)
# MAGIC If you're on serverless compute (DBR 18.2+), you can skip the Python loop and let Databricks
# MAGIC fan the calls out in a single SQL statement. Same endpoint, same schema. Left as reference.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT *,
# MAGIC   ai_query(
# MAGIC     'databricks-llama-4-maverick',
# MAGIC     CONCAT('Bottle: ', name, ' by ', distillery, '. Approximate retail $', CAST(retail_value AS INT),
# MAGIC            '. Give collectibility rarity (common/rare/legendary) and one vivid sentence.'),
# MAGIC     responseFormat => '{"type":"json_schema","json_schema":{"name":"e","schema":{"type":"object",
# MAGIC       "properties":{"rarity":{"type":"string","enum":["common","rare","legendary"]},
# MAGIC       "description":{"type":"string"}},"required":["rarity","description"]},"strict":true}}',
# MAGIC     failOnError => false
# MAGIC   ) AS enrichment
# MAGIC FROM celr_dev.silver.source_inventory;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write the Gold catalog + odds breakdown
# MAGIC `bourbon_catalog` is the curated pool the floor and replacement engine draw from. It is
# MAGIC **derived/reference data**, not operational truth. `image_url` is null — fill from the
# MAGIC retailer's product photos downstream.

# COMMAND ----------

whiskeyCatalog = catalog.filter(F.upper("category") == 'WHISKEY')
agaveCatalog = catalog.filter(F.upper("category") == 'AGAVE')

# COMMAND ----------

whiskeyCatalog.display()

# COMMAND ----------

(agaveCatalog.select(
    "bottle_serial", "name", F.col("category").alias("category"), "description", "rarity",
    F.col("retail_value").cast("double").alias("retail_value"),
    F.col("distillery"),
    "primary_tier", "primary_band", "eligible_tiers",
    F.lit(None).cast("string").alias("image_url"),   # populate from retailer photos later
    F.current_timestamp().alias("cataloged_at"),
 )
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(AGAVE_CATALOG_TBL))

print(f"Wrote {spark.table(AGAVE_CATALOG_TBL).count()} bottles to {AGAVE_CATALOG_TBL}")
display(spark.table(AGAVE_CATALOG_TBL).select("name", "retail_value", "primary_tier", "rarity", "description"))

# COMMAND ----------

(whiskeyCatalog.select(
    "bottle_serial", "name", F.col("category").alias("category"), "description", "rarity",
    F.col("retail_value").cast("double").alias("retail_value"),
    F.col("distillery"),
    "primary_tier", "primary_band", "eligible_tiers",
    F.lit(None).cast("string").alias("image_url"),   # populate from retailer photos later
    F.current_timestamp().alias("cataloged_at"),
 )
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(BOURBON_CATALOG_TBL))

print(f"Wrote {spark.table(BOURBON_CATALOG_TBL).count()} bottles to {BOURBON_CATALOG_TBL}")
display(spark.table(BOURBON_CATALOG_TBL).select("name", "retail_value", "primary_tier", "rarity", "description"))

# COMMAND ----------

# Odds breakdown per tier/band — the data behind "Where your $X lands".
from pyspark.sql import Row

breakdown_rows = []
for tier in sorted(TIER_PRICE):
    for bi in range(len(ODDS_CURVE)):
        breakdown_rows.append(Row(
            tier=tier, pull_price=TIER_PRICE[tier], band_idx=bi,
            value_range=band_label(tier, bi),
            target_pct=float(target_prob(bi) * 100),
        ))
odds_meta = spark.createDataFrame(breakdown_rows)

placed = (whiskeyCatalog.select("primary_tier", "primary_band")
          .groupBy("primary_tier","primary_band").count()
          .withColumnRenamed("primary_tier", "tier")
          .withColumnRenamed("primary_band", "band_idx")
          .withColumnRenamed("count", "bottles_in_band"))

(odds_meta.join(placed, on=["tier", "band_idx"], how="left")
    .fillna({"bottles_in_band": 0})
    .orderBy("tier", "band_idx")
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(BOURBON_ODDS_VIEW))

display(spark.table(BOURBON_ODDS_VIEW).orderBy("tier", "band_idx"))

# COMMAND ----------

breakdown_rows = []
for tier in sorted(TIER_PRICE):
    for bi in range(len(ODDS_CURVE)):
        breakdown_rows.append(Row(
            tier=tier, pull_price=TIER_PRICE[tier], band_idx=bi,
            value_range=band_label(tier, bi),
            target_pct=float(target_prob(bi) * 100),
        ))
odds_meta = spark.createDataFrame(breakdown_rows)

placed = (agaveCatalog.select("primary_tier", "primary_band")
          .groupBy("primary_tier","primary_band").count()
          .withColumnRenamed("primary_tier", "tier")
          .withColumnRenamed("primary_band", "band_idx")
          .withColumnRenamed("count", "bottles_in_band"))

(odds_meta.join(placed, on=["tier", "band_idx"], how="left")
    .fillna({"bottles_in_band": 0})
    .orderBy("tier", "band_idx")
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(AGAVE_ODDS_VIEW))

display(spark.table(AGAVE_ODDS_VIEW).orderBy("tier", "band_idx"))

# COMMAND ----------

