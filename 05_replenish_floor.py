# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 · Replenish the app floor
# MAGIC
# MAGIC When a player ships a bottle home (or vaults it), the app deletes that row
# MAGIC from the floor on both sides — `retireBottle()` in
# MAGIC `celr/src/lib/bourbons.functions.ts` removes it from Supabase and fires
# MAGIC `deleteBottleFromDatabricks()` against `app.*_weighted`. Nothing puts a bottle
# MAGIC back. This notebook is what puts a bottle back.
# MAGIC
# MAGIC ## The two things a departure breaks
# MAGIC
# MAGIC **1. Depth.** The cell the bottle occupied is one bottle shallower. Enough
# MAGIC departures and a cell empties, which takes its band's probability to zero.
# MAGIC
# MAGIC **2. Odds — and this one is live today.** A bottle's `weight` is
# MAGIC `target_prob * WEIGHT_SCALE / n`, computed against the `n` bottles that were
# MAGIC in its cell *at build time*. Delete one row and the survivors keep their old
# MAGIC weight, so the cell's **total** weight falls while every other cell's stays
# MAGIC put. A band built with two bottles that loses one has its probability halved.
# MAGIC The app has been deleting rows without recomputing weights, so the live curve
# MAGIC has been drifting away from `ODDS_CURVE` with every shipment.
# MAGIC
# MAGIC So replenishment is two passes, and the second matters even on a run that
# MAGIC finds nothing to replace:
# MAGIC
# MAGIC | Pass | What it does |
# MAGIC |---|---|
# MAGIC | A · Swap | For each departed bottle, place a never-used reserve bottle of equivalent value |
# MAGIC | B · Reweight | Recompute `weight` for **every** cell on the floor from its current count |
# MAGIC | C · Report | List cells still below target, for `06_inventory_reorder_alert.py` |
# MAGIC
# MAGIC ## How a replacement is chosen
# MAGIC A bottle's band membership is decided entirely by `retail_value / tier_price`.
# MAGIC Two bottles of the same retail value therefore occupy the **same cells in the
# MAGIC same tiers** — swap one for the other and the floor's composition is bit-for-bit
# MAGIC unchanged. So the match is ranked:
# MAGIC
# MAGIC 1. **exact** — the reserve bottle's full `(tier, band)` set is identical to the
# MAGIC    departed bottle's. Zero odds distortion. Almost always available, because
# MAGIC    the set only changes when a value crosses a band edge.
# MAGIC 2. **primary** — same `(tier, band)` for the departed bottle's home tier, but
# MAGIC    it differs in some other tier. Its home cell is preserved; a secondary cell
# MAGIC    shifts by one bottle, which pass B then reweights away.
# MAGIC 3. **nearest** — closest retail value available. Used only when the reserve
# MAGIC    pool has nothing better; logged so you can see the quality degrading.
# MAGIC
# MAGIC Each reserve bottle is claimed at most once per run and marked `ever_placed`
# MAGIC the moment it is used, so it can never be handed out twice.
# MAGIC
# MAGIC ## Idempotency
# MAGIC Every replacement is written to `app.replenishment_log`, keyed on the `rip_id`
# MAGIC that caused it. A rip already in the log is never replaced again, so re-running
# MAGIC this notebook — or running it twice concurrently — cannot double-stock the
# MAGIC floor. Pass B is idempotent by construction.
# MAGIC
# MAGIC ## Where it runs
# MAGIC As a scheduled Databricks job, after `sync-rips-to-databricks` has landed the
# MAGIC day's rips. It deliberately does **not** run inline on the checkout path:
# MAGIC `decideRip` already does a wallet credit, two table writes and two emails, and
# MAGIC a warehouse round-trip there would put floor replenishment in the blast radius
# MAGIC of a user-facing request. The floor can be a few minutes stale; a failed
# MAGIC shipment decision cannot.
# MAGIC
# MAGIC After this runs, hit the app's `sync-bourbons-weighted-from-databricks` /
# MAGIC `sync-agave-weighted-from-databricks` hook to pull the new floor into Supabase.
# MAGIC
# MAGIC `dry_run=true` reports every decision without writing anything.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("gold_schema", "gold", "Gold schema")
dbutils.widgets.text("app_schema", "app", "App schema")
dbutils.widgets.dropdown("spirit", "both", ["both", "bourbon", "agave"], "Spirit to replenish")
dbutils.widgets.text("floor_depth_per_tier", str(FLOOR_DEPTH_PER_TIER), "Target placements per tier")
dbutils.widgets.text("max_pool_rows", "50000", "Safety cap on rows pulled to the driver")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (decide + report, no writes)")

CATALOG      = dbutils.widgets.get("catalog")
GOLD         = dbutils.widgets.get("gold_schema")
APP          = dbutils.widgets.get("app_schema")
SPIRIT_ARG   = dbutils.widgets.get("spirit")
FLOOR_DEPTH  = int(dbutils.widgets.get("floor_depth_per_tier"))
MAX_POOL     = int(dbutils.widgets.get("max_pool_rows"))
DRY_RUN      = dbutils.widgets.get("dry_run") == "true"

RIPS_TBL = f"{CATALOG}.{APP}.rips"
LOG_TBL  = f"{CATALOG}.{APP}.replenishment_log"
GAP_TBL  = f"{CATALOG}.{GOLD}.floor_gaps"   # read by 06_inventory_reorder_alert.py

SPIRITS = {
    "bourbon": {
        "gold":     f"{CATALOG}.{GOLD}.bourbon_catalog",
        "floor":    f"{CATALOG}.{APP}.bourbons_weighted",
        "rip_col":  "bourbon_id",
    },
    "agave": {
        "gold":     f"{CATALOG}.{GOLD}.agave_catalog",
        "floor":    f"{CATALOG}.{APP}.agave_weighted",
        "rip_col":  "agave_id",
    },
}
TARGETS = list(SPIRITS) if SPIRIT_ARG == "both" else [SPIRIT_ARG]

SLOT_NAMES = ["primary", "secondary", "tertiary", "quaternary", "quinary"]
APP_SYNC_SLOTS = 4   # see 02_build_bourbon_app_floor_weighted.py
TIERS = sorted(SELLABLE_TIERS)
BANDS = list(range(len(ODDS_CURVE)))

print(f"catalog={CATALOG} spirits={TARGETS} floor_depth_per_tier={FLOOR_DEPTH} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md ## Setup — the replenishment ledger
# MAGIC One row per replacement. `rip_id` is the idempotency key: a rip that appears
# MAGIC here has already been compensated for and is skipped on every later run.

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {LOG_TBL} (
        rip_id STRING,
        spirit STRING,
        departed_bottle_id STRING,
        departed_retail_value DOUBLE,
        replacement_bottle_serial STRING,
        replacement_retail_value DOUBLE,
        match_quality STRING,
        replaced_at TIMESTAMP,
        note STRING
    ) USING DELTA
""")

# COMMAND ----------

# MAGIC %md ## Shared helpers

# COMMAND ----------

def cell_set(retail_value):
    """The full set of (tier, band) cells a bottle of this value occupies.
    This IS the bottle's identity as far as the odds curve is concerned."""
    return frozenset((t, b) for (t, _m, b) in eligible_tiers(retail_value))


def home_cell(retail_value):
    """The bottle's closest-to-par (tier, band) — its primary placement."""
    et = eligible_tiers(retail_value)
    return (et[0][0], et[0][2]) if et else None


def placement_columns(retail_value):
    """Wide slot columns for a bottle, in the same closest-to-par order the floor
    builder uses. Weights are left at 0 — pass B sets every weight on the floor
    from the real per-cell counts, so anything written here would be overwritten."""
    et = eligible_tiers(retail_value)
    assert len(et) <= APP_SYNC_SLOTS, (
        f"a ${retail_value} bottle needs {len(et)} slots but the app sync reads "
        f"{APP_SYNC_SLOTS} — check SELLABLE_TIERS"
    )
    out = {}
    for i, name in enumerate(SLOT_NAMES):
        if i < len(et):
            tier, _mult, band = et[i]
            out[f"{name}_tier"], out[f"{name}_band"], out[f"{name}_weight"] = tier, band, 0
        else:
            out[f"{name}_tier"], out[f"{name}_band"], out[f"{name}_weight"] = None, None, None
    return out


def floor_placements(floor_df):
    """Explode a wide floor into one row per (bottle, tier, band) placement."""
    parts = []
    for name in SLOT_NAMES:
        parts.append(floor_df.select(
            F.col("id"),
            F.col(f"{name}_tier").alias("tier"),
            F.col(f"{name}_band").alias("band_idx"),
        ).where(F.col(f"{name}_tier").isNotNull()))
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out


def census(floor_df):
    """{(tier, band): n bottles currently in that cell}."""
    rows = (floor_placements(floor_df)
            .groupBy("tier", "band_idx").count().collect())
    return {(int(r["tier"]), int(r["band_idx"])): int(r["count"]) for r in rows}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pass A · Swap in a replacement for every departed bottle
# MAGIC Runs on the driver. The reserve pool is a few thousand rows at most and the
# MAGIC "claim each bottle once" constraint is inherently sequential, so a Spark join
# MAGIC would be both slower and harder to read. `max_pool_rows` guards the assumption.

# COMMAND ----------

def pick_replacement(target_value, pool):
    """Best never-used reserve bottle for a departed bottle of `target_value`.

    `pool` is a list of dicts, mutated: the chosen bottle is removed so it cannot
    be handed out twice within this run. Returns (bottle, match_quality) or
    (None, 'none') when the pool is empty."""
    if not pool:
        return None, "none"

    want_cells = cell_set(target_value)
    want_home = home_cell(target_value)

    def rank(candidate):
        cells = candidate["_cells"]
        if cells == want_cells:
            quality = 0                                   # exact
        elif want_home is not None and want_home in cells:
            quality = 1                                   # primary preserved
        else:
            quality = 2                                   # nearest only
        # Within a quality tier, closest retail value wins.
        return (quality, abs(candidate["retail_value"] - target_value))

    best = min(pool, key=rank)
    quality = ["exact", "primary", "nearest"][rank(best)[0]]
    pool.remove(best)
    return best, quality


def load_reserve_pool(gold_table):
    """Never-placed, priced, placeable bottles — the stock this notebook can draw on.

    `ever_placed` is the sticky ledger from `04_sync_collection_from_unicorn.py`:
    once a bottle has been on the app it is out of the pool forever, which is the
    whole point of that column."""
    g = spark.table(gold_table)
    missing = [c for c in ("ever_placed", "app_status") if c not in g.columns]
    if missing:
        raise ValueError(
            f"{gold_table} is missing {missing}. Run 04_sync_collection_from_unicorn.py "
            f"Part 2 first — without the ledger this notebook cannot tell a fresh bottle "
            f"from one that has already been on the app, and would re-use stock."
        )

    pool_df = (g
        .where(~F.coalesce(F.col("ever_placed"), F.lit(False)))
        .where(F.col("app_status") != "retired")
        .where(F.col("retail_value").isNotNull())
        .select("bottle_serial", "name", "distillery", "description", "rarity",
                F.col("retail_value").cast("double").alias("retail_value"), "image_url"))

    n = pool_df.count()
    if n > MAX_POOL:
        raise ValueError(
            f"reserve pool is {n} rows, above the max_pool_rows guard of {MAX_POOL}. "
            f"Raise the widget deliberately, or this notebook will pull all of it to the driver."
        )

    pool = [r.asDict() for r in pool_df.collect()]
    # Bottles outside the curve entirely (too cheap / too dear for any sellable
    # tier) can never be placed; drop them here rather than failing mid-loop.
    usable, unplaceable = [], 0
    for b in pool:
        cells = cell_set(b["retail_value"])
        if not cells:
            unplaceable += 1
            continue
        b["_cells"] = cells
        usable.append(b)
    if unplaceable:
        print(f"  {unplaceable} reserve bottles fit no sellable tier's curve "
              f"(retail outside ${CURVE_LO * TIER_PRICE[min(TIERS)]:.0f}"
              f"–${CURVE_HI * TIER_PRICE[max(TIERS)]:.0f}); ignored.")
    return usable


def find_departures(spirit):
    """Rips that removed a bottle from the floor and have not yet been compensated.

    'sold' is excluded on purpose: `decideRip`'s sell branch never calls
    retireBottle(), so a sold-back bottle is still physically on the floor and needs
    no replacement."""
    rip_col = SPIRITS[spirit]["rip_col"]
    gone = (spark.table(RIPS_TBL)
        .where(f"{rip_col} IS NOT NULL AND status IN ('shipped', 'stored')")
        .select(
            F.col("id").alias("rip_id"),
            F.col(rip_col).alias("departed_bottle_id"),
            F.col("bourbon_retail_value").cast("double").alias("departed_retail_value"),
            F.col("decided_at"),
        ))
    already = (spark.table(LOG_TBL)
        .where(F.col("spirit") == spirit)
        .select("rip_id").distinct())
    return (gone.join(already, "rip_id", "left_anti")
                .orderBy(F.col("decided_at").asc_nulls_last())
                .collect())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pass B · Recompute every weight on the floor
# MAGIC Not limited to the cells this run touched. Weights drift whenever the app
# MAGIC deletes a row, which happens continuously and outside this notebook, so the
# MAGIC only safe move is to recompute the whole floor from its current counts each
# MAGIC time. Cheap — a few thousand rows — and it makes the run self-healing: the
# MAGIC floor is correct after this pass regardless of what state it was in before.

# COMMAND ----------

@F.udf(returnType=IntegerType())
def udf_weight(band_idx, n):
    return weight_for_band(band_idx, n)


def reweight_floor(floor_table):
    """Rewrite every `*_weight` on the floor so each cell's total weight equals
    `target_prob * WEIGHT_SCALE` again. Returns the verification report."""
    floor = spark.table(floor_table)
    counts = (floor_placements(floor)
              .groupBy("tier", "band_idx").agg(F.count(F.lit(1)).alias("n")))

    out = floor
    for name in SLOT_NAMES:
        c = counts.select(
            F.col("tier").alias(f"_{name}_t"),
            F.col("band_idx").alias(f"_{name}_b"),
            F.col("n").alias(f"_{name}_n"),
        )
        out = (out.join(
                    c,
                    (out[f"{name}_tier"] == c[f"_{name}_t"]) &
                    (out[f"{name}_band"] == c[f"_{name}_b"]),
                    "left")
                  .withColumn(
                      f"{name}_weight",
                      F.when(F.col(f"{name}_tier").isNull(), F.lit(None).cast("int"))
                       .otherwise(udf_weight(F.col(f"{name}_band"), F.col(f"_{name}_n"))))
                  .drop(f"_{name}_t", f"_{name}_b", f"_{name}_n"))

    if DRY_RUN:
        print(f"  DRY RUN: would reweight {out.count()} rows in {floor_table}.")
        return verify(out)

    # Materialize before overwriting the table we are reading from.
    out.cache()
    out.count()
    out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(floor_table)
    out.unpersist()
    print(f"  Reweighted {spark.table(floor_table).count()} rows in {floor_table}.")
    return verify(spark.table(floor_table))


def verify(floor_df):
    """Actual vs. target probability per (tier, band), from the weights on the floor.

    Carries `weight` as well as the cell, which `floor_placements` does not, so it
    builds its own union rather than reusing it."""
    parts = []
    for name in SLOT_NAMES:
        parts.append(floor_df.select(
            F.col(f"{name}_tier").alias("tier"),
            F.col(f"{name}_band").alias("band_idx"),
            F.col(f"{name}_weight").alias("weight"),
        ).where(F.col(f"{name}_tier").isNotNull()))
    slots = parts[0]
    for p in parts[1:]:
        slots = slots.unionByName(p)

    band_totals = {(int(r["tier"]), int(r["band_idx"])): (int(r["w"] or 0), int(r["n"]))
                   for r in slots.groupBy("tier", "band_idx")
                                 .agg(F.sum("weight").alias("w"), F.count(F.lit(1)).alias("n"))
                                 .collect()}
    ok = True
    for t in TIERS:
        tier_total = sum(w for (tt, _b), (w, _n) in band_totals.items() if tt == t)
        print(f"\n  Tier {t} (${TIER_PRICE[t]}) — total weight {tier_total}")
        for b in BANDS:
            w, n = band_totals.get((t, b), (0, 0))
            pct = (w / tier_total * 100) if tier_total else 0.0
            tgt = target_prob(b) * 100
            flag = "OK" if abs(pct - tgt) < 0.05 else "!!"
            if flag == "!!":
                ok = False
            want = target_cell_count(b, FLOOR_DEPTH)
            depth = "" if n >= want else f"  SHORT {want - n}"
            print(f"    band{b}: target {tgt:5.2f}%  actual {pct:5.2f}%  "
                  f"({n:>3} bottles / want {want:>3}){depth} {flag}")
    print("\n  Curve matches ODDS_CURVE." if ok else
          "\n  !! Curve does NOT match ODDS_CURVE — investigate before the next sync.")
    return band_totals

# COMMAND ----------

# MAGIC %md ## Run

# COMMAND ----------

import datetime

LOG_SCHEMA = ("rip_id string, spirit string, departed_bottle_id string, "
              "departed_retail_value double, replacement_bottle_serial string, "
              "replacement_retail_value double, match_quality string, "
              "replaced_at timestamp, note string")

FLOOR_BASE_COLS = ["id", "name", "distillery", "description", "rarity",
                   "retail_value", "image_url", "created_at"]

all_gaps = []

for spirit in TARGETS:
    cfg = SPIRITS[spirit]
    print(f"\n{'=' * 72}\n{spirit.upper()}\n{'=' * 72}")

    floor_before = spark.table(cfg["floor"])
    print(f"Floor: {floor_before.count()} bottles on {cfg['floor']}")

    pool = load_reserve_pool(cfg["gold"])
    print(f"Reserve pool: {len(pool)} never-placed bottles.")

    departures = find_departures(spirit)
    print(f"Departures needing replacement: {len(departures)}")

    # ---- Pass A ---------------------------------------------------------------
    new_rows, log_rows = [], []
    quality_counts = {"exact": 0, "primary": 0, "nearest": 0, "none": 0}
    now = datetime.datetime.now()

    for dep in departures:
        value = dep["departed_retail_value"]
        if value is None:
            # No recorded value means no band, so no like-for-like target. Leave it
            # for the gap-fill report rather than guessing and distorting a cell.
            log_rows.append((dep["rip_id"], spirit, dep["departed_bottle_id"], None,
                             None, None, "none", now,
                             "departed bottle had no retail_value on the rip; "
                             "not replaced — covered by the gap report instead"))
            quality_counts["none"] += 1
            continue

        pick, quality = pick_replacement(float(value), pool)
        quality_counts[quality] += 1
        if pick is None:
            log_rows.append((dep["rip_id"], spirit, dep["departed_bottle_id"], float(value),
                             None, None, "none", now,
                             "reserve pool exhausted — buy more stock"))
            continue

        row = {
            "id": pick["bottle_serial"],
            "name": pick["name"],
            "distillery": pick["distillery"],
            "description": pick["description"],
            "rarity": pick["rarity"],
            "retail_value": int(round(pick["retail_value"])),
            "image_url": pick["image_url"],
            "created_at": now,
        }
        row.update(placement_columns(pick["retail_value"]))
        new_rows.append(row)
        log_rows.append((dep["rip_id"], spirit, dep["departed_bottle_id"], float(value),
                         pick["bottle_serial"], float(pick["retail_value"]), quality, now, None))

    print("Match quality: " + ", ".join(f"{k}={v}" for k, v in quality_counts.items() if v))
    if quality_counts["nearest"]:
        print(f"  [warn] {quality_counts['nearest']} replacements were value-nearest only — "
              f"the reserve pool no longer covers the bands being drawn from. "
              f"Pass C below says what to buy.")
    if quality_counts["none"]:
        print(f"  [warn] {quality_counts['none']} departures could NOT be replaced. "
              f"The floor is shrinking.")

    # ---- Write pass A ---------------------------------------------------------
    if new_rows and not DRY_RUN:
        slot_cols = [f"{n}_{s}" for n in SLOT_NAMES for s in ("tier", "band", "weight")]
        ordered = FLOOR_BASE_COLS + slot_cols
        add_df = spark.createDataFrame(
            [tuple(r.get(c) for c in ordered) for r in new_rows],
            ", ".join(
                [f"{c} string" for c in ("id", "name", "distillery", "description", "rarity")]
                + ["retail_value bigint", "image_url string", "created_at timestamp"]
                + [f"{c} int" for c in slot_cols]
            ),
        )
        add_df.write.mode("append").saveAsTable(cfg["floor"])
        print(f"Appended {len(new_rows)} replacement bottles to {cfg['floor']}.")

        # Spend them in the ledger in the same run. If this fails the bottles are on
        # a floor but still look fresh in gold, which is the one state that lets a
        # later run place them a second time — so it is loud, not swallowed.
        serials = spark.createDataFrame([(r["id"],) for r in new_rows], "bottle_serial string")
        (DeltaTable.forName(spark, cfg["gold"]).alias("t")
            .merge(serials.alias("s"), "t.bottle_serial = s.bottle_serial")
            .whenMatchedUpdate(set={
                "app_status": "'in_app'",
                "ever_placed": "true",
                "first_placed_at": "COALESCE(t.first_placed_at, current_timestamp())",
            })
            .execute())
        print(f"Marked {len(new_rows)} bottles as in_app / ever_placed in {cfg['gold']}.")
    elif new_rows:
        print(f"DRY RUN: would append {len(new_rows)} bottles to {cfg['floor']} "
              f"and mark them spent in {cfg['gold']}.")
        for r in new_rows[:5]:
            print(f"    {r['retail_value']:>6}  {r['name'][:48]:<48} "
                  f"-> tier {r['primary_tier']} band {r['primary_band']}")

    if log_rows and not DRY_RUN:
        (spark.createDataFrame(log_rows, LOG_SCHEMA)
             .write.mode("append").saveAsTable(LOG_TBL))
        print(f"Logged {len(log_rows)} replacement decisions to {LOG_TBL}.")

    # ---- Pass B ---------------------------------------------------------------
    print(f"\nReweighting {cfg['floor']} against ODDS_CURVE:")
    band_totals = reweight_floor(cfg["floor"])

    # ---- Pass C ---------------------------------------------------------------
    # What the floor is still short of, cell by cell. This is the buy list.
    for t in TIERS:
        for b in BANDS:
            _w, n = band_totals.get((t, b), (0, 0))
            want = target_cell_count(b, FLOOR_DEPTH)
            if n < want:
                all_gaps.append((
                    spirit, t, TIER_PRICE[t], b, band_label(t, b),
                    float(band_mid_value(t, b)), n, want, want - n,
                    float(target_prob(b)), now,
                ))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Publish the gap table
# MAGIC `06_inventory_reorder_alert.py` reads this rather than recomputing it, so the
# MAGIC numbers in the reorder email are exactly the numbers this run acted on.

# COMMAND ----------

GAP_SCHEMA = ("spirit string, tier int, pull_price int, band_idx int, value_range string, "
              "target_retail_value double, bottles_on_floor int, target_bottles int, "
              "shortfall int, band_probability double, computed_at timestamp")

if DRY_RUN:
    print(f"DRY RUN: {len(all_gaps)} short cells; {GAP_TBL} not written.")
    for g in sorted(all_gaps, key=lambda r: -r[8])[:15]:
        print(f"  {g[0]:<8} tier {g[1]} band{g[3]} {g[4]:>14}  "
              f"on floor {g[6]:>3} / want {g[7]:>3}  short {g[8]}")
else:
    (spark.createDataFrame(all_gaps, GAP_SCHEMA)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GAP_TBL))
    print(f"Wrote {len(all_gaps)} short cells to {GAP_TBL}.")

print("\nNext: call the app's sync-bourbons-weighted-from-databricks / "
      "sync-agave-weighted-from-databricks hook to pull this floor into Supabase.")
