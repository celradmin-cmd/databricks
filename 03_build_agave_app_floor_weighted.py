# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03 · Build weighted `app.agave` floor (wide, weight-driven)
# MAGIC
# MAGIC Agave counterpart of `02_build_bourbon_app_floor_weighted.py` — same design,
# MAGIC same odds config, different spirit. See that file for the full rationale;
# MAGIC summarized below.
# MAGIC
# MAGIC ## Why wide, one row per physical bottle
# MAGIC The agave order is **1,000 physical bottles total, shared across all 5 tiers**,
# MAGIC not 1,000 dedicated units per tier. The same bottle can be eligible for up to 5
# MAGIC tiers at once (e.g. a $600 bottle sits in a band at every tier from $50 to
# MAGIC $1000 — the curve spans 0.5x-16x of pull price, so $500-$800 clears all five
# MAGIC floors simultaneously). Storing that bottle as separate rows per tier would mean
# MAGIC retiring it after one tier's pull (`DELETE ... WHERE id = :bottleId`, see
# MAGIC `celr/src/lib/bourbons.functions.ts`) leaves phantom copies still listed as
# MAGIC available in its other tiers.
# MAGIC
# MAGIC So this build is **one row per physical bottle** (wide), not one row per
# MAGIC `(bottle, tier)` placement (tall). Each row carries up to 5 tier placements —
# MAGIC `primary/secondary/tertiary/quaternary/quinary_{tier,band,weight}` — ordered
# MAGIC closest-to-par first (that's already how `eligible_tiers` is sorted upstream
# MAGIC in `00_celr_odds_config.eligible_tiers`). Deleting one row correctly removes the
# MAGIC bottle from every tier it was eligible for, in one statement.
# MAGIC
# MAGIC ## How odds stay correct with far fewer rows
# MAGIC Odds come from the **weight** column (`cell_weights` in
# MAGIC `00_celr_odds_config`), not row counts. For a `(tier, band)` cell with target
# MAGIC probability `p` and `n` bottles landing in it (via ANY of their 5 slots), the
# MAGIC bottles' weights sum to `p * WEIGHT_SCALE`, split toward the cheaper bottles
# MAGIC in the cell so it pays out at its band target. A band with only one real
# MAGIC bottle simply puts the entire band's weight on that one row — no replication
# MAGIC required. `WEIGHT_SCALE` must stay large (currently 1,000,000) — a small scale
# MAGIC rounds a cell's per-bottle weight to 0 the moment more than a few dozen bottles
# MAGIC land in it via incidental multi-tier overlap, which silently zeroes that band's
# MAGIC odds. See `00_celr_odds_config.py` for the full explanation.
# MAGIC
# MAGIC **The app's pull query MUST change to use this column before this table is
# MAGIC live** (out of scope for this script — see the note at the bottom).
# MAGIC
# MAGIC `DRY_RUN=True` reports composition without writing. Flip to `False` to write.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

import uuid

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ============================== CONFIG =======================================
SOURCE_TABLE = "prod_celr.gold.agave_catalog"   # one row per physical bottle in this shipment
TARGET_TABLE = "prod_celr.app.agave_weighted"   # new table; cut the app over once verified
BOTTLES_EXPECTED = 1000          # sanity check only — the real count comes from SOURCE_TABLE
DRY_RUN = True                    # report only; set False to write
WRITE_MODE = "overwrite"          # overwrite = full floor rebuild (replaces all rows)
FILL_GAP_PLACEHOLDERS = True      # inject a single flagged row for any empty (tier, band) cell
# True  = bottles currently on the floor stay eligible, so a rebuild keeps them and
#         only reaches into the reserve pool for the gaps. Use this to re-shape an
#         existing live floor after an odds change.
# False = build a completely fresh floor out of never-placed stock only. Use this
#         for a brand-new shipment, never against a live floor.
KEEP_BOTTLES_ALREADY_ON_FLOOR = True

# Odds targets, band index -> probability. Must match ODDS_CURVE in 00_celr_odds_config.
TARGET_PROBS = [p for (_lo, _hi, p) in ODDS_CURVE]
# Only tiers the app actually sells get placements — see SELLABLE_TIERS in
# 00_celr_odds_config. All five tiers (including tier 5, $250) are sellable today.
TIERS = sorted(SELLABLE_TIERS)
BANDS = list(range(len(TARGET_PROBS)))
SLOT_NAMES = ["primary", "secondary", "tertiary", "quaternary", "quinary"]   # closest-to-par first

# How many slot columns `celr/src/lib/weighted-sync.server.ts` actually SELECTs
# when it pulls this table into Supabase. Anything beyond this is written here and
# then silently discarded on sync — the bottle keeps its warehouse placement but
# loses the odds contribution in the live app. Slots are ordered closest-to-par,
# NOT by tier number, so an overflowing slot is a random real tier, not a specific one.
APP_SYNC_SLOTS = 5
assert len(SLOT_NAMES) >= len(TIERS), f"need >= 1 slot per possible tier (there are {len(TIERS)} tiers)"
assert len(TIERS) <= APP_SYNC_SLOTS, (
    f"{len(TIERS)} sellable tiers but the app sync only reads {APP_SYNC_SLOTS} slot columns — "
    f"placements past slot {APP_SYNC_SLOTS} would be dropped and the live odds would "
    f"diverge from this build. Add the extra slot to bourbons_weighted/agave_weighted in "
    f"Supabase, merge_weighted_catalog, weighted_tier_catalog and weighted-sync.server.ts "
    f"before widening SELLABLE_TIERS."
)
# =============================================================================

cat = spark.table(SOURCE_TABLE)

# Never re-use a bottle. Two filters, both written by
# `sync_collection_from_unicorn.py` Part 2:
#
#   app_status == 'retired'  -> shipped or stored; physically gone from the building.
#   ever_placed == true      -> has been on an app floor at least once, ever.
#
# The second is the one that matters for a rebuild. `app_status` is recomputed from
# live state on every run, so a bottle that left the floor without being ripped
# reads as 'available' again and would be re-placed. `ever_placed` is sticky and
# never cleared, so the reserve pool only ever shrinks.
#
# Bottles currently on the floor are `ever_placed` too, so a full rebuild with this
# filter produces an entirely NEW floor from the reserve pool rather than keeping
# the existing one. That is intentional for a fresh shipment, but it is not how you
# top up a floor day to day — use `04_replenish_floor.py`, which preserves the
# existing rows and only fills what left.
if "app_status" in cat.columns:
    n_before = cat.count()
    cat = cat.where((F.col("app_status").isNull()) | (F.col("app_status") != "retired"))
    n_retired = n_before - cat.count()
    if n_retired:
        print(f"Excluded {n_retired} retired bottles (shipped/stored) from this floor build.")
else:
    print("[info] app_status column not present on the source catalog yet — "
          "run sync_collection_from_unicorn.py's Part 2 to enable retired-bottle exclusion.")

if "ever_placed" in cat.columns:
    n_before = cat.count()
    if KEEP_BOTTLES_ALREADY_ON_FLOOR:
        cat = cat.where(~F.coalesce(F.col("ever_placed"), F.lit(False))
                        | (F.col("app_status") == "in_app"))
    else:
        cat = cat.where(~F.coalesce(F.col("ever_placed"), F.lit(False)))
    n_used = n_before - cat.count()
    if n_used:
        print(f"Excluded {n_used} already-used bottles (ever_placed) — these have been on "
              f"the app before and must not be shown again.")
    print(f"Reserve pool available to this build: {cat.count()} never-placed bottles.")
else:
    print("[warn] ever_placed column not present on the source catalog — this build CANNOT "
          "guarantee it won't re-use a bottle that has already appeared on the app. "
          "Run sync_collection_from_unicorn.py's Part 2 first.")

n_bottles = cat.count()
print(f"Source catalog: {n_bottles} physical bottles (expected ~{BOTTLES_EXPECTED}).")
if abs(n_bottles - BOTTLES_EXPECTED) > 0.1 * BOTTLES_EXPECTED:
    print(f"!! Catalog count is off from BOTTLES_EXPECTED by more than 10% — "
          f"confirm {SOURCE_TABLE} reflects the current shipment before proceeding.")

# Precondition: catalog must already carry eligible_tiers (from 00_celr_odds_config
# via 01_classify_and_enrich), sorted closest-to-par first.
if cat.limit(1).count() == 0 or "eligible_tiers" not in cat.columns:
    raise ValueError(
        f"{SOURCE_TABLE} is missing eligible_tiers. Run 01_classify_and_enrich first."
    )

# COMMAND ----------

# MAGIC %md ### 1. Explode each bottle's eligible tiers, keeping slot position
# MAGIC `posexplode` preserves the closest-to-par ordering already computed by
# MAGIC `eligible_tiers()` — position 0 is primary, 1 secondary, 2 tertiary, 3 quaternary, 4 quinary.

# COMMAND ----------

placements = (cat
    .select("bottle_serial", F.posexplode("eligible_tiers").alias("slot_idx", "et"))
    .select(
        "bottle_serial", "slot_idx",
        F.col("et.tier").cast("int").alias("tier"),
        F.col("et.band_idx").cast("int").alias("band_idx"),
    )
    .where(F.col("slot_idx") < len(SLOT_NAMES)))

# Guard the sync boundary. `eligible_tiers()` only returns SELLABLE_TIERS, so with
# four of them nothing can reach slot 4 (quinary) — but assert it rather than trust
# it, because a silent overflow here shows up as live odds that quietly disagree
# with this build's verification output and nothing else.
n_overflow = placements.where(F.col("slot_idx") >= APP_SYNC_SLOTS).count()
assert n_overflow == 0, (
    f"{n_overflow} placements landed in slot {APP_SYNC_SLOTS} or beyond. "
    f"weighted-sync.server.ts reads only the first {APP_SYNC_SLOTS} slots, so these "
    f"would be dropped on sync and the affected tiers would under-fill. "
    f"Check SELLABLE_TIERS in 00_celr_odds_config."
)

# COMMAND ----------

# MAGIC %md ### 2. Weight per (tier, band) cell
# MAGIC The cell's total weight always equals `target_prob * WEIGHT_SCALE`, however
# MAGIC many bottles fill it. Inside the cell it is split by value — `cell_weights()`
# MAGIC in `00_celr_odds_config` draws cheaper bottles more often, so a win band pays
# MAGIC out near its low end rather than at whatever the stocked bottles average.
# MAGIC Computed on the driver (a few thousand placements), not in a Python UDF.

# COMMAND ----------

cell_counts = placements.groupBy("tier", "band_idx").agg(F.count(F.lit(1)).alias("n"))

_valued = (placements
    .join(cat.select("bottle_serial", F.col("retail_value").cast("double").alias("retail_value")),
          "bottle_serial")
    .select("bottle_serial", "slot_idx", "tier", "band_idx", "retail_value")
    .collect())
_cells = {}
for r in _valued:
    _cells.setdefault((int(r["tier"]), int(r["band_idx"])), []).append(r)
_weight_rows = []
for (t, b), rows in _cells.items():
    for r, w in zip(rows, cell_weights(b, [r["retail_value"] / TIER_PRICE[t] for r in rows])):
        _weight_rows.append((r["bottle_serial"], int(r["slot_idx"]), w))
_weights = spark.createDataFrame(_weight_rows, "bottle_serial string, slot_idx int, weight int")

placements_w = placements.join(_weights, ["bottle_serial", "slot_idx"])

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
# MAGIC Don't silently skew a tier's odds when a gap exists — fill it with one
# MAGIC clearly-flagged reserve row carrying the cell's full weight, so the structure
# MAGIC is correct the moment a real bottle in that range is added.

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
        # Deterministic uuid5 rather than "ph-1-2": the Supabase column is `uuid`,
        # so a non-uuid literal fails the sync insert. Deterministic so the same
        # gap keeps the same id across rebuilds.
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"celr:floor-placeholder:{t}:{b}")), label, "PLACEHOLDER",
        "Reserve gap. Do not open this tier to real pulls until replaced.",
        "placeholder", float(mid_value), "",
        t, b, w,      # primary_*
        None, None, None,  # secondary_*
        None, None, None,  # tertiary_*
        None, None, None,  # quaternary_*
        None, None, None,  # quinary_*
    ))

if missing_cells:
    print("RESERVE GAPS (single placeholder row each):")
    for t, b in missing_cells:
        print(f"  Tier {t} band{b + 1} ({band_label(t, b)}): no eligible bottle in this shipment.")

if FILL_GAP_PLACEHOLDERS and placeholder_rows:
    ph_schema = ("bottle_serial string, name string, distillery string, description string, "
                 "rarity string, retail_value double, image_url string, "
                 "primary_tier int, primary_band int, primary_weight int, "
                 "secondary_tier int, secondary_band int, secondary_weight int, "
                 "tertiary_tier int, tertiary_band int, tertiary_weight int, "
                 "quaternary_tier int, quaternary_band int, quaternary_weight int, "
                 "quinary_tier int, quinary_band int, quinary_weight int")
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
        F.col("retail_value").cast("double").alias("retail_value"),
    ).where(F.col("tier").isNotNull())
    slot_union = s if slot_union is None else slot_union.unionByName(s)

# The payout a player actually faces per tier (weight x value), next to the design
# target. Band shares can all read OK while this is far off — that is how a 0.89x
# design ran at ~1.0x on the Oct 2026 floor.
payout_by_tier = {}
for r in slot_union.collect():
    payout_by_tier.setdefault(int(r["tier"]), []).append(
        (int(r["weight"] or 0), (r["retail_value"] or 0.0) / TIER_PRICE[int(r["tier"])]))

band_totals = {(r["tier"], r["band_idx"]): (r["w"], r["n"])
               for r in slot_union.groupBy("tier", "band_idx")
                                  .agg(F.sum("weight").alias("w"), F.count(F.lit(1)).alias("n"))
                                  .collect()}
tier_totals = {t: sum(w for (tt, _b), (w, _n) in band_totals.items() if tt == t) for t in TIERS}

ok_all = True
for t in TIERS:
    total_rows_in_tier = sum(n for (tt, _b), (_w, n) in band_totals.items() if tt == t)
    payout = floor_payout(payout_by_tier.get(t, []))
    payout_ok = payout is not None and payout <= TARGET_PAYOUT_MULTIPLE + 0.02
    if not payout_ok:
        ok_all = False
    print(f"\nTier {t} (${TIER_PRICE[t]}), {total_rows_in_tier} rows across {len(BANDS)} bands, "
          f"total weight {tier_totals[t]}, floor payout "
          f"{f'{payout:.3f}x' if payout is not None else 'n/a'} vs design "
          f"{TARGET_PAYOUT_MULTIPLE:.3f}x {'OK' if payout_ok else '!!'}")
    for b in BANDS:
        w, n = band_totals.get((t, b), (0, 0))
        pct = (w / tier_totals[t] * 100) if tier_totals[t] else 0
        target = TARGET_PROBS[b] * 100
        flag = "OK" if abs(pct - target) < 0.05 else "!!"
        if flag == "!!":
            ok_all = False
        print(f"  band{b + 1}: target {target:5.1f}%  actual {pct:5.1f}%  "
              f"(weight {w:>4} across {n:>3} row{'s' if n != 1 else ' '}) {flag}")
print("\nComposition and payout match targets." if ok_all else
      "\n!! Composition or payout mismatch — check rounding or missing cells; a payout over "
      "design means cells are stocked above their band targets (buy toward band_target_value).")

# COMMAND ----------

n_real = real_rows.count()
n_placeholder = len(placeholder_rows)
n_placements = sum(n for (_t, _b), (_w, n) in band_totals.items())
print(f"\n{n_real} real bottle rows + {n_placeholder} placeholder rows = {n_real + n_placeholder} floor rows.")
print(f"{n_placements} total (bottle, tier) placements across those rows "
      f"(avg {n_placements / (n_real + n_placeholder):.2f} tiers/bottle) — "
      f"this is the '~3,500-4,000' figure, not a row count.")

# COMMAND ----------

# MAGIC %md ### 6. Shape to the wide schema and write
# MAGIC Same base columns as `app.agave` today, plus 12 new placement columns.

# COMMAND ----------

floor = floor_pre.select(
    # The floor row's id IS the physical bottle's gold bottle_serial — not a fresh
    # uuid. Everything that tracks a bottle across systems keys on this: `rips.
    # bourbon_id` / `rips.agave_id` store it, `retired_bottles.bottle_id` stores it,
    # and `sync_collection_from_unicorn.py` joins it back to
    # `gold.*_catalog.bottle_serial` to decide app_status. Generating a new uuid here
    # broke every one of those joins — app_status could never resolve to 'in_app' or
    # 'retired', so the never-re-use ledger silently matched nothing.
    # bottle_serial is already a uuid4 (see mock_data_generator), which satisfies the
    # `id uuid` column type on the Supabase side.
    F.col("bottle_serial").cast("string").alias("id"),
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
    F.col("quinary_tier").cast("int").alias("quinary_tier"),
    F.col("quinary_band").cast("int").alias("quinary_band"),
    F.col("quinary_weight").cast("int").alias("quinary_weight"),
)

n = floor.count()
if DRY_RUN:
    print(f"\nDRY RUN: composed {n} rows for {TARGET_TABLE} (nothing written). "
          "Set DRY_RUN=False to write.")
    floor.orderBy(F.desc("primary_weight")).show(10, truncate=40)
else:
    (floor.write.mode(WRITE_MODE).saveAsTable(TARGET_TABLE))
    print(f"Wrote {n} rows to {TARGET_TABLE} (mode={WRITE_MODE}).")

    # Spend the bottles in the ledger immediately. Waiting for
    # `sync_collection_from_unicorn.py` to notice them leaves a window in which
    # another build — or `04_replenish_floor.py` — could hand the same physical
    # bottle to a second floor. `ever_placed` is OR-ed, never cleared.
    from delta.tables import DeltaTable
    placed_serials = real_rows.select("bottle_serial").distinct()
    try:
        (DeltaTable.forName(spark, SOURCE_TABLE).alias("t")
            .merge(placed_serials.alias("s"), "t.bottle_serial = s.bottle_serial")
            .whenMatchedUpdate(set={
                "app_status": "'in_app'",
                "ever_placed": "true",
                "first_placed_at": "COALESCE(t.first_placed_at, current_timestamp())",
            })
            .execute())
        print(f"Marked {placed_serials.count()} bottles as in_app / ever_placed in {SOURCE_TABLE}.")
    except Exception as e:
        print(f"!! Could not mark bottles as placed in {SOURCE_TABLE}: {e}\n"
              f"   Run sync_collection_from_unicorn.py Part 2 to repair the ledger "
              f"BEFORE any other floor build or replenishment run, or bottles may be re-used.")

# COMMAND ----------

# MAGIC %md ### 7. NOT done here — the app-side cutover
# MAGIC This script only builds the Databricks table. Before pointing the live app at
# MAGIC it, `celr/src/lib/payments.functions.ts` (`performRip`, agave branch) needs to:
# MAGIC 1. Select rows where `:tier IN (primary_tier, secondary_tier, tertiary_tier, quaternary_tier, quinary_tier)`.
# MAGIC 2. Pick with a WEIGHTED random draw using whichever `*_weight` column matches
# MAGIC    the requested tier (currently it does `SELECT * WHERE tier = :t` then
# MAGIC    `Math.random() * agave.length` — a uniform pick that ignores weight).
# MAGIC 3. `retireBottle()`'s single `DELETE ... WHERE id = :bottleId` already works
# MAGIC    correctly against this shape (one row = one physical bottle) — no change
# MAGIC    needed there.
# MAGIC That migration touches the live checkout path and was intentionally left out
# MAGIC of this script — do it as a reviewed follow-up, ideally alongside the bourbon
# MAGIC cutover so `performRip` only needs to change once for both spirits.
