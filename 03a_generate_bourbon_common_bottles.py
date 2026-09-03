# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Generate Common Bottles
# MAGIC
# MAGIC Fills every (tier, band) cell up to the target floor using LLM-written common
# MAGIC whiskey records. Writes into `prod_celr.gold.bourbon_catalog`, matching its schema:
# MAGIC `bottle_serial, name, category, description, rarity, retail_value, primary_tier,
# MAGIC primary_band, eligible_tiers<array<struct<tier,multiple,band_idx>>>, image_url,
# MAGIC cataloged_at` (+ optional `source`).
# MAGIC
# MAGIC **Coverage is counted by eligibility, not primary.** One bottle is eligible for
# MAGIC several tiers at once (a $400 bottle = 0.8x common in the $500 tier AND an 8x
# MAGIC jackpot in the $50 tier). We explode `eligible_tiers` to count what each cell
# MAGIC already has, and credit each new bottle to every cell it qualifies for.
# MAGIC
# MAGIC Odds truth (`BANDS`, `PROBS`, `TIER_PRICE`) comes only from `00_celr_odds_config`.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Environment (prod/dev)")
dbutils.widgets.text("per_tier", "1000", "Target bottles per tier")
dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("gold_schema", "gold", "Gold schema")
dbutils.widgets.text("catalog_table", "bourbon_catalog", "Gold catalog table")
dbutils.widgets.text("category", "WHISKEY", "Category value for synthetic rows")
dbutils.widgets.dropdown("naming_mode", "real_commodity", ["real_commodity", "fictional_house"], "Naming mode")
dbutils.widgets.text("llm_endpoint", "databricks-llama-4-maverick", "Foundation Model endpoint")
dbutils.widgets.text("gen_batch", "20", "Records per LLM call")
dbutils.widgets.dropdown("mark_source", "true", ["true", "false"], "Tag rows source='synthetic' (adds column)")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (no write)")

ENV        = dbutils.widgets.get("environment")
PER_TIER   = int(dbutils.widgets.get("per_tier"))
CATALOG    = dbutils.widgets.get("catalog")
GOLD       = dbutils.widgets.get("gold_schema")
CAT_TABLE  = f"{CATALOG}.{GOLD}.{dbutils.widgets.get('catalog_table')}"
CATEGORY   = dbutils.widgets.get("category")
NAMING     = dbutils.widgets.get("naming_mode")
LLM        = dbutils.widgets.get("llm_endpoint")
GEN_BATCH  = int(dbutils.widgets.get("gen_batch"))
MARK_SRC   = dbutils.widgets.get("mark_source") == "true"
DRY_RUN    = dbutils.widgets.get("dry_run") == "true"

print(f"env={ENV} per_tier={PER_TIER} table={CAT_TABLE} naming={NAMING} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Classification (mirror of 01) and target counts

# COMMAND ----------

# From 00_celr_odds_config: TIER_PRICE {int:float}, ODDS_CURVE [(lo_mult, hi_mult, prob)]
BANDS = [(lo, hi) for (lo, hi, _p) in ODDS_CURVE]
PROBS = [p for (_lo, _hi, p) in ODDS_CURVE]
LO_MULT, HI_MULT = BANDS[0][0], BANDS[-1][1]   # overall eligibility window, e.g. 0.5x..16x

def band_index(mult):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= mult < hi:
            return i
    return None

def classify(retail_value):
    """Return (eligible_tiers, primary_tier, primary_band) for a retail value.
    eligible_tiers = list of (tier, multiple, band_idx)."""
    elig = []
    for t, price in TIER_PRICE.items():
        mult = retail_value / price
        b = band_index(mult)
        if b is not None:
            elig.append((int(t), round(float(mult), 4), int(b)))
    if not elig:
        return [], None, None
    # primary = eligible tier whose multiple is closest to par (1.0x)
    primary = min(elig, key=lambda e: abs(e[1] - 1.0))
    return elig, primary[0], primary[2]

def target_count(band):
    return round(PROBS[band] * PER_TIER)

targets = {(t, b): target_count(b) for t in TIER_PRICE for b in range(len(BANDS))}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Existing coverage per cell (explode eligible_tiers)

# COMMAND ----------

from pyspark.sql import functions as F

have = {}
try:
    exploded = (spark.table(CAT_TABLE)
                .select(F.explode("eligible_tiers").alias("e"))
                .select(F.col("e.tier").alias("tier"), F.col("e.band_idx").alias("band")))
    have = {(r["tier"], r["band"]): r["c"]
            for r in exploded.groupBy("tier", "band").agg(F.count("*").alias("c")).collect()}
    print(f"existing bottles contribute coverage to {len(have)} cells")
except Exception as e:
    print(f"(catalog not readable / empty: {e}); assuming zero coverage")

need = {}
total = 0
print("\ntier band  target  have  need")
for t in TIER_PRICE:
    for b in range(len(BANDS)):
        tg = targets[(t, b)]
        hv = have.get((t, b), 0)
        gap = max(tg - hv, 0)
        need[(t, b)] = gap
        total += gap
        print(f"  {t}   {b}   {tg:>5}  {hv:>4}  {gap:>4}")
print(f"\nremaining cell-coverage to fill: {total}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. LLM text generation (names + descriptions only; we assign price)

# COMMAND ----------

import json, uuid, random
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
RARITY_BY_BAND = {0: "common", 1: "common", 2: "common", 3: "rare", 4: "rare", 5: "legendary"}

def text_prompt(n, lo_d, hi_d):
    if NAMING == "real_commodity":
        rule = (f"Use REAL, actual American whiskey bottles that genuinely retail around "
                f"${lo_d}-${hi_d} and are widely available at national retailers. Use their "
                "true product and distillery names. Repeats are fine if few options exist "
                "at this price.")
    else:
        rule = ("Invent plausible but ENTIRELY FICTIONAL American whiskey and distillery "
                "names (no real brands). Craft/small-batch styling. No duplicate names.")
    return (f"List {n} American whiskey bottles. {rule} "
            "Return ONLY a JSON array, no prose, no markdown. Each element: "
            '{"name": string, "distillery": string, '
            '"description": one sentence under 30 words, tasting-note tone}.'
            '"distillery": What distillery is this bottle from? If unknown, say so')

def call_llm_text(n, lo_d, hi_d):
    resp = client.predict(endpoint=LLM, inputs={
        "messages": [
            {"role": "system", "content": "You output strictly valid JSON and nothing else."},
            {"role": "user", "content": text_prompt(n, lo_d, hi_d)},
        ],
        "temperature": 0.9, "max_tokens": 2000,
    })
    text = resp["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`")
    return json.loads(text[text.find("["): text.rfind("]") + 1])

def clean_desc(d):
    d = " ".join(str(d).split())
    return " ".join(d.split()[:30]) if d else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fill loop
# MAGIC Repeatedly target the neediest cell, pick a retail value inside that cell's window
# MAGIC for that tier (guarantees eligibility), classify, and credit every cell the bottle
# MAGIC qualifies for. Terminates because each iteration reduces the neediest cell.

# COMMAND ----------

# DBTITLE 1,Cell 11
def retail_for_cell(tier, band):
    lo, hi = BANDS[band]
    price = TIER_PRICE[tier]
    lo_d, hi_d = int(round(lo * price)), int(round(hi * price))
    return random.randint(lo_d, max(lo_d, hi_d - 1))

rows = []
used_names = set()   # only enforced for fictional naming
MAX_ITERS = total * 3 + 1000
iters = 0
while total > 0 and iters < MAX_ITERS:
    iters += 1
    (t, b), gap = max(need.items(), key=lambda kv: kv[1])
    if gap <= 0:
        break
    want = min(GEN_BATCH, gap)
    lo, hi = BANDS[b]
    lo_d, hi_d = int(round(lo * TIER_PRICE[t])), int(round(hi * TIER_PRICE[t]))
    try:
        recs = call_llm_text(want, lo_d, hi_d)
    except Exception as e:
        print(f"[warn] LLM call failed ({e}); retrying")
        continue
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        key = name.lower()
        if not name:
            continue
        if NAMING == "fictional_house" and key in used_names:
            continue  # real names may repeat across physical bottles
        rv = float(retail_for_cell(t, b))
        elig, ptier, pband = classify(rv)
        if not elig:
            continue
        used_names.add(key)
        bid = str(uuid.uuid4())
        rows.append({
            "bottle_serial": bid,
            "name": name,
            "category": CATEGORY,
            "description": clean_desc(rec.get("description")),
            "distillery": rec.get("distillery"),
            "rarity": RARITY_BY_BAND[pband],
            "retail_value": rv,
            "primary_tier": ptier,
            "primary_band": pband,
            "eligible_tiers": elig,
            "image_url": None,   # no image generated; retailer photo backfilled later
        })
        # credit every needy cell this bottle covers
        for (et, em, eb) in elig:
            if need.get((et, eb), 0) > 0:
                need[(et, eb)] -= 1
                total -= 1

print(f"generated {len(rows)} bottles over {iters} iterations; remaining need={total}")
leftover = {k: v for k, v in need.items() if v > 0}
if leftover:
    print(f"[warn] cells still short: {leftover}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write to Gold (append, schema-matched)

# COMMAND ----------

from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               IntegerType, ArrayType)

elig_type = ArrayType(StructType([
    StructField("tier", IntegerType()),
    StructField("multiple", DoubleType()),
    StructField("band_idx", IntegerType()),
]))
schema = StructType([
    StructField("bottle_serial", StringType()),
    StructField("name", StringType()),
    StructField("category", StringType()),
    StructField("description", StringType()),
    StructField("distillery", StringType()),
    StructField("rarity", StringType()),
    StructField("retail_value", DoubleType()),
    StructField("primary_tier", IntegerType()),
    StructField("primary_band", IntegerType()),
    StructField("eligible_tiers", elig_type),
    StructField("image_url", StringType()),
])

def to_tuple(r):
    return (r["bottle_serial"], r["name"], r["category"], r["description"], r["distillery"], r["rarity"],
            r["retail_value"], r["primary_tier"], r["primary_band"],
            [(t, m, b) for (t, m, b) in r["eligible_tiers"]], r["image_url"])

if not rows:
    print("nothing to write")
elif DRY_RUN:
    print("DRY RUN - sample of what would be written:")
    for r in rows[:5]:
        print(json.dumps(r, indent=2))
    print(f"... {len(rows)} rows total (set dry_run=false to write)")
else:
    df = spark.createDataFrame([to_tuple(r) for r in rows], schema)
    df = df.withColumn("cataloged_at", F.current_timestamp())
    if MARK_SRC:
        df = df.withColumn("source", F.lit("synthetic"))
    (df.write.mode("append").option("mergeSchema", "true").saveAsTable(CAT_TABLE))
    print(f"appended {len(rows)} rows to {CAT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify the floor (coverage per cell after write)

# COMMAND ----------

if not DRY_RUN:
    chk = (spark.table(CAT_TABLE)
           .select(F.explode("eligible_tiers").alias("e"))
           .select(F.col("e.tier").alias("tier"), F.col("e.band_idx").alias("band"))
           .groupBy("tier", "band").agg(F.count("*").alias("have"))
           .orderBy("tier", "band"))
    display(chk)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prod_celr.gold.bourbon_catalog

# COMMAND ----------

