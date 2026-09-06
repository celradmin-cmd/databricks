# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 · Build weighted `app.bourbons` floor (wide, weight-driven)
# MAGIC
# MAGIC ## Why this replaces `04_build_bourbon_app_floor`
# MAGIC The original plan was 4,000 physical bottles — 1,000 dedicated units per tier,
# MAGIC no cross-tier sharing needed. `04` achieved odds by **row-count replication**:
# MAGIC explode each bottle into every `(tier, band)` it's eligible for, then duplicate
# MAGIC rows until each cell hits its exact target count. A uniform `ORDER BY random()`
# MAGIC pull then reproduces the curve because the counts *are* the odds.
# MAGIC
# MAGIC The bourbon order shrank to **1,000 physical bottles total**, shared across all
# MAGIC 4 tiers. The same bottle can be eligible for up to 4 tiers at once (e.g. a
# MAGIC $600 bottle sits in a band at every tier from $50 to $1000 — the curve spans
# MAGIC 0.5x-16x of pull price, so $500-$800 clears all four floors simultaneously).
# MAGIC That makes row-count replication actively wrong here: if the same physical
# MAGIC bottle were stored as separate rows per tier, retiring it after one tier's
# MAGIC pull (`DELETE ... WHERE id = :bottleId`, see `celr/src/lib/bourbons.functions.ts`)
# MAGIC would leave phantom copies still listed as available in its other tiers.
# MAGIC
# MAGIC So this build is **one row per physical bottle** (wide), not one row per
# MAGIC `(bottle, tier)` placement (tall). Each row carries up to 4 tier placements —
# MAGIC `primary/secondary/tertiary/quaternary_{tier,band,weight}` — ordered
# MAGIC closest-to-par first (that's already how `eligible_tiers` is sorted upstream
# MAGIC in `00_celr_odds_config.eligible_tiers`). Deleting one row now correctly
# MAGIC removes the bottle from every tier it was eligible for, in one statement.
# MAGIC
# MAGIC ## How odds stay correct with far fewer rows
# MAGIC Odds no longer come from row counts — they come from the **weight** column
# MAGIC (`weight_for_band` in `00_celr_odds_config`, already defined but unused by the
# MAGIC count-based approach). For a `(tier, band)` cell with target probability `p`
# MAGIC and `n` bottles landing in it (via ANY of their 4 slots), each bottle gets
# MAGIC weight `round(p * WEIGHT_SCALE / n)`. A band with only one real bottle simply
# MAGIC puts the entire band's weight on that one row — no replication required. This
# MAGIC is also why total row count lands around ~3,500-4,000 *placements* even though
# MAGIC there are only 1,000 physical rows: it's the natural result of the average
# MAGIC bottle being eligible for ~3.5 tiers, not a target to hit.
# MAGIC
# MAGIC **The app's pull query MUST change to use this column before this table is
# MAGIC live** (out of scope for this script — see the note at the bottom). Today
# MAGIC `performRip` does `SELECT * WHERE tier = :t` then `Math.random() * length`,
# MAGIC ignoring `weight` entirely. Against this schema that reproduces the OLD
# MAGIC count-based odds only if every cell still has equal-weight rows, which will no
# MAGIC longer be true.
# MAGIC
# MAGIC `DRY_RUN=True` reports composition without writing. Flip to `False` to write.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ============================== CONFIG =======================================
SOURCE_TABLE = "prod_celr.gold.bourbon_catalog"   # one row per physical bottle in this shipment
TARGET_TABLE = "prod_celr.app.bourbons_weighted"  # new table; cut the app over once verified
BOTTLES_EXPECTED = 1000          # sanity check only — the real count comes from SOURCE_TABLE
DRY_RUN = True                    # report only; set False to write
WRITE_MODE = "overwrite"          # overwrite = full floor rebuild (replaces all rows)
FILL_GAP_PLACEHOLDERS = True      # inject a single flagged row for any empty (tier, band) cell

# Odds targets, band index -> probability. Must match ODDS_CURVE in 00_celr_odds_config.
TARGET_PROBS = [p for (_lo, _hi, p) in ODDS_CURVE]
TIERS = sorted(TIER_PRICE)
BANDS = list(range(len(TARGET_PROBS)))
SLOT_NAMES = ["primary", "secondary", "tertiary", "quaternary"]   # closest-to-par first
assert len(SLOT_NAMES) >= len(TIERS), "need >= 1 slot per possible tier (there are only 4 tiers)"
# =============================================================================

cat = spark.table(SOURCE_TABLE)
n_bottles = cat.count()
print(f"Source catalog: {n_bottles} physical bottles (expected ~{BOTTLES_EXPECTED}).")
if abs(n_bottles - BOTTLES_EXPECTED) > 0.1 * BOTTLES_EXPECTED:
    print(f"!! Catalog count is off from BOTTLES_EXPECTED by more than 10% — "
          f"confirm {SOURCE_TABLE} reflects the current shipment before proceeding.")

# Precondition: catalog must already carry eligible_tiers (from 00_celr_odds_config
# via 02_classify_and_enrich), sorted closest-to-par first.
if cat.limit(1).count() == 0 or "eligible_tiers" not in cat.columns:
    raise ValueError(
        f"{SOURCE_TABLE} is missing eligible_tiers. Run 02_classify_and_enrich first."
    )

# COMMAND ----------

# MAGIC %md ### 1. Explode each bottle's eligible tiers, keeping slot position
# MAGIC `posexplode` preserves the closest-to-par ordering already computed by
# MAGIC `eligible_tiers()` — position 0 is primary, 1 secondary, 2 tertiary, 3 quaternary.

# COMMAND ----------

placements = (cat
    .select("bottle_serial", F.posexplode("eligible_tiers").alias("slot_idx", "et"))
    .select(
        "bottle_serial", "slot_idx",
        F.col("et.tier").cast("int").alias("tier"),
        F.col("et.band_idx").cast("int").alias("band_idx"),
    )
    .where(F.col("slot_idx") < len(SLOT_NAMES)))

# COMMAND ----------

# MAGIC %md ### 2. Weight per (tier, band) cell
# MAGIC `n` = how many bottles land in this cell via ANY slot. Weight is split evenly
# MAGIC across them so the cell's total weight always equals `target_prob * WEIGHT_SCALE`,
# MAGIC regardless of how few real bottles fill it.

# COMMAND ----------

@F.udf(returnType=IntegerType())
def udf_weight(band_idx, n):
    return weight_for_band(band_idx, n)

cell_counts = placements.groupBy("tier", "band_idx").agg(F.count(F.lit(1)).alias("n"))
placements_w = (placements
    .join(cell_counts, ["tier", "band_idx"])
    .withColumn("weight", udf_weight("band_idx", "n")))

# COMMAND ----------

# MAGIC %md ### 3. Pivot placements back to wide columns, one row per bottle

# COMMAND ----------

wide = cat.select("bottle_serial", "name", "distillery", "description", "rarity",
                   F.col("retail_value").cast("double").alias("retail_value"), "image_url")

for idx, label in enumerate(SLOT_NAMES):
    slot = (placements_w
        .where(F.col("slot_idx") == idx)
        .select(
            "bottle_serial",
            F.col("tier").alias(f"{label}_tier"),
            F.col("band_idx").alias(f"{label}_band"),
            F.col("weight").alias(f"{label}_weight"),
        ))
    wide = wide.join(slot, "bottle_serial", "left")

real_rows = wide.where(F.col("primary_tier").isNotNull())

# COMMAND ----------

# MAGIC %md ### 4. Placeholder rows for any (tier, band) cell with zero real bottles
# MAGIC Same rationale as `04`: don't silently skew a tier's odds when a gap exists —
# MAGIC fill it with one clearly-flagged reserve row carrying the cell's full weight,
# MAGIC so the structure is correct the moment a real bottle in that range is added.

# COMMAND ----------

expected_cells = {(t, b) for t in TIERS for b in BANDS}
present_cells = {(r["tier"], r["band_idx"]) for r in cell_counts.select("tier", "band_idx").collect()}
missing_cells = sorted(expected_cells - present_cells)

placeholder_rows = []
for t, b in missing_cells:
    lo, hi, p = ODDS_CURVE[b]
    price = TIER_PRICE[t]
    label = f"[TIER{t} BAND{b + 1} PLACEHOLDER - ADD {band_label(t, b)} BOTTLE]"
    mid_value = round((lo + hi) / 2 * price) if hi != float("inf") else round(lo * price * 1.5)
    w = weight_for_band(b, 1)
    placeholder_rows.append((
        f"ph-{t}-{b}", label, "PLACEHOLDER",
        "Reserve gap. Do not open this tier to real pulls until replaced.",
        "placeholder", float(mid_value), "",
        t, b, w,      # primary_*
        None, None, None,  # secondary_*
        None, None, None,  # tertiary_*
        None, None, None,  # quaternary_*
    ))

if missing_cells:
    print("RESERVE GAPS (single placeholder row each — was N replicated rows under the old count-based build):")
    for t, b in missing_cells:
        print(f"  Tier {t} band{b + 1} ({band_label(t, b)}): no eligible bottle in this shipment.")

if FILL_GAP_PLACEHOLDERS and placeholder_rows:
    ph_schema = ("bottle_serial string, name string, distillery string, description string, "
                 "rarity string, retail_value double, image_url string, "
                 "primary_tier int, primary_band int, primary_weight int, "
                 "secondary_tier int, secondary_band int, secondary_weight int, "
                 "tertiary_tier int, tertiary_band int, tertiary_weight int, "
                 "quaternary_tier int, quaternary_band int, quaternary_weight int")
    ph = spark.createDataFrame(placeholder_rows, ph_schema)
    floor_pre = real_rows.unionByName(ph)
else:
    floor_pre = real_rows

# COMMAND ----------

# MAGIC %md ### 5. Verify composition against targets (weight-based, not row-count-based)
# MAGIC For each tier, actual% = this band's total weight / the tier's total weight
# MAGIC across every row that carries that tier in ANY slot.

# COMMAND ----------

slot_union = None
for label in SLOT_NAMES:
    s = floor_pre.select(
        F.col(f"{label}_tier").alias("tier"),
        F.col(f"{label}_band").alias("band_idx"),
        F.col(f"{label}_weight").alias("weight"),
    ).where(F.col("tier").isNotNull())
    slot_union = s if slot_union is None else slot_union.unionByName(s)

band_totals = {(r["tier"], r["band_idx"]): (r["w"], r["n"])
               for r in slot_union.groupBy("tier", "band_idx")
                                  .agg(F.sum("weight").alias("w"), F.count(F.lit(1)).alias("n"))
                                  .collect()}
tier_totals = {t: sum(w for (tt, _b), (w, _n) in band_totals.items() if tt == t) for t in TIERS}

ok_all = True
for t in TIERS:
    total_rows_in_tier = sum(n for (tt, _b), (_w, n) in band_totals.items() if tt == t)
    print(f"\nTier {t} (${TIER_PRICE[t]}), {total_rows_in_tier} rows across {len(BANDS)} bands, "
          f"total weight {tier_totals[t]}")
    for b in BANDS:
        w, n = band_totals.get((t, b), (0, 0))
        pct = (w / tier_totals[t] * 100) if tier_totals[t] else 0
        target = TARGET_PROBS[b] * 100
        flag = "OK" if abs(pct - target) < 0.05 else "!!"
        if flag == "!!":
            ok_all = False
        print(f"  band{b + 1}: target {target:5.1f}%  actual {pct:5.1f}%  "
              f"(weight {w:>4} across {n:>3} row{'s' if n != 1 else ' '}) {flag}")
print("\nComposition matches targets." if ok_all else "\n!! Composition mismatch — check rounding or missing cells.")

n_real = real_rows.count()
n_placeholder = len(placeholder_rows)
n_placements = sum(n for (_t, _b), (_w, n) in band_totals.items())
print(f"\n{n_real} real bottle rows + {n_placeholder} placeholder rows = {n_real + n_placeholder} floor rows.")
print(f"{n_placements} total (bottle, tier) placements across those rows "
      f"(avg {n_placements / (n_real + n_placeholder):.2f} tiers/bottle) — "
      f"this is the '~3,500-4,000' figure, not a row count.")

# COMMAND ----------

# MAGIC %md ### 6. Shape to the wide schema and write
# MAGIC Same base columns as `app.bourbons` today, plus 12 new placement columns.

# COMMAND ----------

floor = floor_pre.select(
    F.expr("uuid()").alias("id"),
    "name", "distillery", "description", "rarity",
    F.round("retail_value").cast("bigint").alias("retail_value"),
    "image_url",
    F.current_timestamp().alias("created_at"),
    F.col("primary_tier").cast("int").alias("primary_tier"),
    F.col("primary_band").cast("int").alias("primary_band"),
    F.col("primary_weight").cast("int").alias("primary_weight"),
    F.col("secondary_tier").cast("int").alias("secondary_tier"),
    F.col("secondary_band").cast("int").alias("secondary_band"),
    F.col("secondary_weight").cast("int").alias("secondary_weight"),
    F.col("tertiary_tier").cast("int").alias("tertiary_tier"),
    F.col("tertiary_band").cast("int").alias("tertiary_band"),
    F.col("tertiary_weight").cast("int").alias("tertiary_weight"),
    F.col("quaternary_tier").cast("int").alias("quaternary_tier"),
    F.col("quaternary_band").cast("int").alias("quaternary_band"),
    F.col("quaternary_weight").cast("int").alias("quaternary_weight"),
)

n = floor.count()
if DRY_RUN:
    print(f"\nDRY RUN: composed {n} rows for {TARGET_TABLE} (nothing written). "
          "Set DRY_RUN=False to write.")
    floor.orderBy(F.desc("primary_weight")).show(10, truncate=40)
else:
    (floor.write.mode(WRITE_MODE).saveAsTable(TARGET_TABLE))
    print(f"Wrote {n} rows to {TARGET_TABLE} (mode={WRITE_MODE}).")

# COMMAND ----------

# MAGIC %md ### 7. NOT done here — the app-side cutover
# MAGIC This script only builds the Databricks table. Before pointing the live app at
# MAGIC it, `celr/src/lib/payments.functions.ts` (`performRip`) needs to:
# MAGIC 1. Select rows where `:tier IN (primary_tier, secondary_tier, tertiary_tier, quaternary_tier)`.
# MAGIC 2. Pick with a WEIGHTED random draw using whichever `*_weight` column matches
# MAGIC    the requested tier (currently it does `SELECT * WHERE tier = :t` then
# MAGIC    `Math.random() * bourbons.length` — a uniform pick that ignores weight).
# MAGIC 3. `retireBottle()`'s single `DELETE ... WHERE id = :bottleId` already works
# MAGIC    correctly against this shape (one row = one physical bottle) — no change
# MAGIC    needed there.
# MAGIC That migration touches the live checkout path and was intentionally left out
# MAGIC of this script — do it as a reviewed follow-up.