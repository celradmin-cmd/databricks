# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Build `app.agave` floor from the gold catalog
# MAGIC
# MAGIC Reads `prod_celr.gold.agave_catalog` and writes the pullable floor to
# MAGIC `prod_celr.app.agave` so that **odds live in the floor composition**.
# MAGIC The app pull stays uniform and trivial:
# MAGIC ```sql
# MAGIC select * from agave where tier = :t order by random() limit 1;
# MAGIC ```
# MAGIC For every tier we fill the six value bands with the exact target row
# MAGIC **counts**, so a uniform pull inside a tier reproduces 37/39/20/3/0.8/0.2.
# MAGIC
# MAGIC **Band math is NOT recomputed here.** Each bottle's `(tier, band_idx)`
# MAGIC placements are read straight from the catalog's `eligible_tiers` struct,
# MAGIC which `00_celr_odds_config` already computed (same odds config as whiskey).
# MAGIC One catalog bottle serves several tiers, so each floor row is a
# MAGIC `(bottle, tier)` placement and a bottle is replicated within a cell to
# MAGIC reach the target count.
# MAGIC
# MAGIC `DRY_RUN=True` reports the composition without writing. Flip to `False` to
# MAGIC write. `WRITE_MODE="overwrite"` **replaces the entire floor** each run.

# COMMAND ----------

import math
from pyspark.sql import functions as F, Window

# ============================== CONFIG =======================================
SOURCE_TABLE = "prod_celr.gold.agave_catalog"
TARGET_TABLE = "prod_celr.app.agave"
BOTTLES_PER_TIER = 1000          # min size that makes 0.2% an integer count (1/500)
DRY_RUN = False                  # report only; set False to write
WRITE_MODE = "overwrite"        # overwrite = full floor rebuild (replaces all rows)
FILL_GAP_PLACEHOLDERS = True    # inject flagged rows for empty (tier, band) cells

# Odds targets, band index -> probability. Composition target only; the band
# ASSIGNMENT of each bottle comes from the catalog, not from here.
TARGET_PROBS = [0.370, 0.390, 0.200, 0.030, 0.008, 0.002]
# Pull price + band multiplier ranges are used ONLY for placeholder labels.
PULL_PRICE = {1: 50, 2: 100, 3: 500, 4: 1000}
BAND_MULTS = [(0.00, 0.75), (0.75, 1.00), (1.00, 1.50),
              (1.50, 3.00), (3.00, 6.00), (6.00, float("inf"))]
# =============================================================================


def target_counts(n):
    """Largest-remainder rounding so band counts always sum to n."""
    raw = [(i, n * p) for i, p in enumerate(TARGET_PROBS)]
    floors = {i: math.floor(x) for i, x in raw}
    leftover = n - sum(floors.values())
    for i, _ in sorted(raw, key=lambda t: t[1] - math.floor(t[1]),
                       reverse=True)[:leftover]:
        floors[i] += 1
    return floors


COUNTS = target_counts(BOTTLES_PER_TIER)   # {0:185, 1:195, 2:100, 3:15, 4:4, 5:1}
TIERS = sorted(PULL_PRICE)
print("Target counts per tier:", COUNTS, "sum:", sum(COUNTS.values()))

# COMMAND ----------

# MAGIC %md ### 1. Read catalog and explode eligible tiers

# COMMAND ----------

cat = spark.table(SOURCE_TABLE)

# Schema guard: the agave catalog is assumed to mirror the whiskey catalog.
# If the producer column is not named `distillery` (or another field differs),
# update the select mapping below and this required set.
REQUIRED = {"name", "distillery", "description", "rarity", "retail_value",
            "image_url", "bottle_serial", "eligible_tiers"}
missing_cols = REQUIRED - set(cat.columns)
if missing_cols:
    raise ValueError(
        f"{SOURCE_TABLE} is missing expected columns: {sorted(missing_cols)}. "
        f"Available: {sorted(cat.columns)}. If the agave schema differs, update "
        "the select mapping in this cell to match."
    )

exploded = (cat
    .withColumn("et", F.explode("eligible_tiers"))
    .select(
        "name", "distillery", "description", "rarity",
        F.col("retail_value").cast("double").alias("retail_value"),
        "image_url", "bottle_serial",
        F.col("et.tier").cast("int").alias("tier"),
        F.col("et.band_idx").cast("int").alias("band_idx"),
    ))

# Precondition: the catalog must be enriched with eligible_tiers.
if exploded.limit(1).count() == 0:
    raise ValueError(
        f"{SOURCE_TABLE} produced no (bottle, tier) placements. "
        "eligible_tiers is empty/NULL. Run the odds-enrichment step "
        "(00_celr_odds_config) before building the floor."
    )

# COMMAND ----------

# MAGIC %md ### 2. Compose each (tier, band) cell to its target count
# MAGIC Round-robin the eligible bottles so no single bottle carries a cell more
# MAGIC than it has to: each bottle gets `need // m` copies, and the first
# MAGIC `need % m` (by serial) get one extra.

# COMMAND ----------

target_df = spark.createDataFrame(
    [(b, c) for b, c in COUNTS.items()], "band_idx int, need int")

cell = Window.partitionBy("tier", "band_idx")
order = cell.orderBy("bottle_serial", "name")

composed = (exploded
    .withColumn("rn", F.row_number().over(order) - 1)
    .withColumn("m", F.count(F.lit(1)).over(cell))
    .join(F.broadcast(target_df), "band_idx", "inner")
    .withColumn("base", (F.col("need") / F.col("m")).cast("int"))     # floor, need>=0
    .withColumn("extra", F.col("need") % F.col("m"))
    .withColumn("copies",
                F.col("base") + F.when(F.col("rn") < F.col("extra"), 1).otherwise(0))
    .where("copies > 0")
    .withColumn("_r", F.explode(F.array_repeat(F.lit(1), F.col("copies"))))
    .select("name", "distillery", "description", "rarity",
            "retail_value", "image_url", "tier", "band_idx"))

# COMMAND ----------

# MAGIC %md ### 3. Placeholders for any empty (tier, band) cell
# MAGIC A cell with no eligible bottle cannot be composed. Rather than silently
# MAGIC skew a tier's odds, fill it with clearly-flagged rows so the structure is
# MAGIC correct the moment a real bottle in that retail range is added.

# COMMAND ----------

expected = spark.createDataFrame(
    [(t, b) for t in TIERS for b in COUNTS], "tier int, band_idx int")
available = exploded.select("tier", "band_idx").distinct()
missing = [(r["tier"], r["band_idx"])
           for r in expected.join(available, ["tier", "band_idx"], "left_anti").collect()]

placeholder_rows = []
for t, b in missing:
    lo, hi = BAND_MULTS[b]
    p = PULL_PRICE[t]
    hi_v = (lo + 1) * p if hi == float("inf") else hi * p
    mid = float(round((lo * p + hi_v) / 2))
    label = f"[TIER{t} BAND{b+1} PLACEHOLDER - ADD ${lo*p:.0f}-{hi_v:.0f} BOTTLE]"
    for _ in range(COUNTS[b]):
        placeholder_rows.append(
            (label, "PLACEHOLDER",
             "Reserve gap. Do not open this tier to real pulls until replaced.",
             "placeholder", mid, "", t, b))

if missing:
    print("RESERVE GAPS (placeholder-filled):")
    for t, b in missing:
        lo, hi = BAND_MULTS[b]
        p = PULL_PRICE[t]
        hi_v = "inf" if hi == float("inf") else f"${hi*p:.0f}"
        print(f"  Tier {t} band{b+1}: {COUNTS[b]} rows need a ${lo*p:.0f}-{hi_v} bottle.")

if FILL_GAP_PLACEHOLDERS and placeholder_rows:
    ph = spark.createDataFrame(
        placeholder_rows,
        "name string, distillery string, description string, rarity string, "
        "retail_value double, image_url string, tier int, band_idx int")
    floor_pre = composed.unionByName(ph)
else:
    floor_pre = composed

# COMMAND ----------

# MAGIC %md ### 4. Verify composition against targets

# COMMAND ----------

agg = {(r["tier"], r["band_idx"]): r["c"]
       for r in floor_pre.groupBy("tier", "band_idx")
                         .agg(F.count(F.lit(1)).alias("c")).collect()}

ok_all = True
for t in TIERS:
    total = sum(agg.get((t, b), 0) for b in COUNTS)
    print(f"\nTier {t} (${PULL_PRICE[t]}), {total} rows")
    for b in COUNTS:
        got = agg.get((t, b), 0)
        pct = got / total * 100 if total else 0
        flag = "OK" if got == COUNTS[b] else "!!"
        if got != COUNTS[b]:
            ok_all = False
        print(f"  band{b+1}: target {TARGET_PROBS[b]*100:5.1f}%  "
              f"actual {pct:5.1f}%  ({got:>3}/{COUNTS[b]:<3}) {flag}")
print("\nComposition matches targets." if ok_all else "\n!! Composition mismatch.")

# COMMAND ----------

# MAGIC %md ### 5. Shape to `app.agave` schema and write

# COMMAND ----------

floor = floor_pre.select(
    F.expr("uuid()").alias("id"),
    "name", "distillery", "description", "rarity",
    F.round("retail_value").cast("bigint").alias("retail_value"),
    "image_url",
    F.current_timestamp().alias("created_at"),
    F.col("tier").cast("bigint").alias("tier"),
)

n = floor.count()
if DRY_RUN:
    print(f"DRY RUN: composed {n} rows for {TARGET_TABLE} (nothing written). "
          "Set DRY_RUN=False to write.")
    floor.show(5, truncate=40)
else:
    (floor.write.mode(WRITE_MODE).saveAsTable(TARGET_TABLE))
    print(f"Wrote {n} rows to {TARGET_TABLE} (mode={WRITE_MODE}).")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prod_celr.app.agave

# COMMAND ----------

