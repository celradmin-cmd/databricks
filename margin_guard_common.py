# Databricks notebook source
# MAGIC %md
# MAGIC # Margin guard — shared helpers
# MAGIC `%run` by `07_margin_guard.py` and `08_apply_odds_proposal.py`, **after**
# MAGIC `00_celr_odds_config`. Holds the table definitions and the one piece of math
# MAGIC both need: what a floor pays out, from its weights and bottle values.
# MAGIC
# MAGIC Expects `CATALOG` and `APP` to be set by the caller.

# COMMAND ----------

from pyspark.sql import functions as F

SLOT_NAMES = ["primary", "secondary", "tertiary", "quaternary", "quinary"]
FLOOR_TABLES = {"bourbon": "bourbons_weighted", "agave": "agave_weighted"}
SYNC_HOOKS = {"bourbon": "sync-bourbons-weighted-from-databricks",
              "agave": "sync-agave-weighted-from-databricks"}

SNAPSHOT_TBL = f"{CATALOG}.{APP}.margin_guard_snapshots"
ALERT_TBL    = f"{CATALOG}.{APP}.margin_guard_alerts"
PROPOSAL_TBL = f"{CATALOG}.{APP}.odds_proposals"
CURVE_TBL    = ODDS_CURVE_TABLE   # from 00 — the table it loads approved curves from
RIPS_TBL     = f"{CATALOG}.{APP}.rips"

assert CURVE_TBL.startswith(f"{CATALOG}.{APP}."), (
    f"00_celr_odds_config reads approved curves from {CURVE_TBL}, but this run writes to "
    f"catalog {CATALOG}.{APP} — an approved curve would never reach the floor builders")


def ensure_tables():
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {SNAPSHOT_TBL} (
        run_at TIMESTAMP, spirit STRING, tier INT, window STRING, pulls BIGINT,
        revenue DOUBLE, cost DOUBLE, realized_margin DOUBLE, expected_margin DOUBLE,
        z_score DOUBLE, floor_rows INT, win_rate DOUBLE
    ) COMMENT 'Hourly house-margin readings from 07_margin_guard.py'""")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {ALERT_TBL} (
        alert_key STRING, kind STRING, spirit STRING, tier INT, message STRING,
        first_seen TIMESTAMP, last_seen TIMESTAMP, last_notified TIMESTAMP, resolved_at TIMESTAMP
    ) COMMENT 'Open margin alerts, so the guard notifies once a day per problem, not every hour'""")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {PROPOSAL_TBL} (
        proposal_id STRING, spirit STRING, created_at TIMESTAMP, expires_at TIMESTAMP,
        status STRING, reason STRING, tiers_json STRING, ai_summary STRING, model STRING,
        decided_by STRING, decided_at TIMESTAMP, decision_note STRING,
        applied_at TIMESTAMP, apply_result STRING
    ) COMMENT 'Odds changes awaiting human approval on /admin/odds. status: pending, approved, applying, applied, rejected, expired, failed'""")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {CURVE_TBL} (
        spirit STRING, tier INT, band_probs ARRAY<DOUBLE>, status STRING,
        proposal_id STRING, activated_at TIMESTAMP, superseded_at TIMESTAMP
    ) COMMENT 'Approved per-tier band probabilities. 00_celr_odds_config loads status = active'""")


def load_floor(spirit):
    """{tier: [(band_idx, weight, multiple), ...]} for every placed slot on the live
    floor — placeholder rows included, since the app draws them too."""
    df = spark.table(f"{CATALOG}.{APP}.{FLOOR_TABLES[spirit]}")
    parts = [df.select(F.col(f"{n}_tier").alias("tier"), F.col(f"{n}_band").alias("band"),
                       F.col(f"{n}_weight").alias("weight"),
                       F.col("retail_value").cast("double").alias("retail_value"))
               .where(F.col(f"{n}_tier").isNotNull() & (F.coalesce(F.col(f"{n}_weight"), F.lit(0)) > 0))
             for n in SLOT_NAMES]
    slots = parts[0]
    for p in parts[1:]:
        slots = slots.unionByName(p)
    out = {}
    for r in slots.collect():
        t = int(r["tier"])
        if t not in TIER_PRICE:
            continue
        out.setdefault(t, []).append((int(r["band"]), int(r["weight"]), (r["retail_value"] or 0.0) / TIER_PRICE[t]))
    return out


def payout_stats(placements):
    """(expected payout multiple, per-pull variance of the multiple, win rate) for a
    list of (band, weight, multiple). The variance is what tells luck from a real
    problem: one 16x grail moves an hour's numbers a long way."""
    tot = sum(w for _b, w, _m in placements)
    if not tot:
        return None, None, None
    ev = sum(w * m for _b, w, m in placements) / tot
    var = sum(w * m * m for _b, w, m in placements) / tot - ev * ev
    wins = sum(w for _b, w, m in placements if m >= 1.0) / tot
    return ev, max(var, 0.0), wins


def reweighted(placements, probs):
    """The same bottles re-weighted with `probs` exactly as 02/03/04 would weight
    them (cell_weights, per cell). Returns a new placement list."""
    cells = {}
    for b, _w, m in placements:
        cells.setdefault(b, []).append(m)
    out = []
    for b, mults in cells.items():
        for m, w in zip(mults, cell_weights(b, mults, prob=probs[b])):
            out.append((b, w, m))
    return out
