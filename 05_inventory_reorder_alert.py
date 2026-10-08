# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 06 · Low-inventory reorder alert
# MAGIC
# MAGIC `04_replenish_floor.py` refills the app floor out of the reserve pool — the
# MAGIC never-placed bottles sitting in gold. That pool only ever shrinks: a bottle
# MAGIC leaves it when it goes on the floor and never comes back. This notebook
# MAGIC watches it, and tells you **what to buy and what you can pay** before it runs
# MAGIC out.
# MAGIC
# MAGIC ## Two different kinds of "low"
# MAGIC
# MAGIC **Floor gaps** — cells on the live app that are below their target depth right
# MAGIC now. Read from `gold.floor_gaps`, written by `05`. Urgent: a cell at zero takes
# MAGIC its band's probability to zero and reshapes the curve.
# MAGIC
# MAGIC **Reserve coverage** — for each cell, how many never-placed bottles exist that
# MAGIC could fill it. This is the forward-looking one. A cell can be fully stocked on
# MAGIC the floor and still be an emergency if there is nothing behind it: the next few
# MAGIC shipments empty it and `05` has nothing to swap in. Coverage is measured in
# MAGIC **refills**: reserve bottles for that cell divided by its target depth.
# MAGIC
# MAGIC ## What you can pay
# MAGIC There is no cost or purchase-price column anywhere in bronze, gold or the app,
# MAGIC so the alert cannot report what stock actually cost. It reports the ceiling
# MAGIC instead: the most you can pay for a bottle destined for a cell and still clear
# MAGIC your target margin.
# MAGIC
# MAGIC ```
# MAGIC max buy price = band midpoint retail x (1 - gross_margin)
# MAGIC ```
# MAGIC
# MAGIC Margin is taken **against retail**, so at 35% a bottle that needs to be worth
# MAGIC ~$56 to fill its cell can be bought for at most ~$36. That is the number to
# MAGIC negotiate with. Set `gross_margin` to whatever the business actually runs at.
# MAGIC
# MAGIC Note what this does *not* cover: the pull itself carries a house edge by
# MAGIC design (`expected_multiple()` in `00_celr_odds_config` is ~0.84x — the floor
# MAGIC gives back about 84 cents of retail for every dollar pulled, provided cells are
# MAGIC stocked around `band_target_value`, which is what this alert asks you to buy). Buying below
# MAGIC retail and the sellback discount add margin on top of that.
# MAGIC Buy above these ceilings for long and the business does not make money, no
# MAGIC matter what the odds curve says.
# MAGIC
# MAGIC ## Delivery
# MAGIC Plain-text summary written to the notebook output and to
# MAGIC `gold.reorder_recommendations`, then raised as a job-level failure/alert when
# MAGIC anything is below threshold. Wire the recipients in the Databricks **job**
# MAGIC definition (Job → Notifications → on failure / on success), not here — no
# MAGIC addresses in source control.
# MAGIC
# MAGIC Set `fail_on_alert=true` (the default) so a Databricks job email fires when
# MAGIC there is something to buy, and nothing arrives on a quiet day.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("gold_schema", "gold", "Gold schema")
dbutils.widgets.text("app_schema", "app", "App schema")
dbutils.widgets.dropdown("spirit", "both", ["both", "bourbon", "agave"], "Spirit to check")
dbutils.widgets.text("gross_margin", "0.35", "Target gross margin against retail (0.35 = 35%)")
dbutils.widgets.text("pulls_per_tier_per_day", str(PULLS_PER_TIER_PER_DAY), "Expected pulls per tier per day (sizes cell depth)")
dbutils.widgets.text("min_refills", "1.0", "Alert when reserve covers fewer than this many refills of a cell")
dbutils.widgets.dropdown("fail_on_alert", "true", ["true", "false"], "Fail the job when stock is low (drives the job email)")

CATALOG      = dbutils.widgets.get("catalog")
GOLD         = dbutils.widgets.get("gold_schema")
APP          = dbutils.widgets.get("app_schema")
SPIRIT_ARG   = dbutils.widgets.get("spirit")
GROSS_MARGIN = float(dbutils.widgets.get("gross_margin"))
PULLS_PER_DAY = int(dbutils.widgets.get("pulls_per_tier_per_day"))
MIN_REFILLS  = float(dbutils.widgets.get("min_refills"))
FAIL_ON_ALERT = dbutils.widgets.get("fail_on_alert") == "true"

if not 0.0 <= GROSS_MARGIN < 1.0:
    raise ValueError(f"gross_margin must be between 0 and 1, got {GROSS_MARGIN}")

GAP_TBL  = f"{CATALOG}.{GOLD}.floor_gaps"
REC_TBL  = f"{CATALOG}.{GOLD}.reorder_recommendations"

SPIRITS = {
    "bourbon": f"{CATALOG}.{GOLD}.bourbon_catalog",
    "agave":   f"{CATALOG}.{GOLD}.agave_catalog",
}
TARGETS = list(SPIRITS) if SPIRIT_ARG == "both" else [SPIRIT_ARG]

TIERS = sorted(SELLABLE_TIERS)
BANDS = list(range(len(ODDS_CURVE)))

print(f"catalog={CATALOG} spirits={TARGETS} gross_margin={GROSS_MARGIN:.0%} "
      f"min_refills={MIN_REFILLS} fail_on_alert={FAIL_ON_ALERT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Reserve coverage per cell
# MAGIC A reserve bottle counts toward a cell if its retail value puts it in that
# MAGIC cell — the same `eligible_tiers` rule the floor builder uses. One bottle
# MAGIC therefore covers several cells at once, which is why coverage is reported per
# MAGIC cell and the buy list is deduplicated at the end.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField


@F.udf(returnType=ArrayType(StructType([
    StructField("tier", IntegerType()),
    StructField("band_idx", IntegerType()),
])))
def udf_cells(retail_value):
    return [(int(t), int(b)) for (t, _m, b) in eligible_tiers(retail_value)]


def reserve_coverage(gold_table):
    """{(tier, band): how many never-placed bottles could fill this cell}."""
    g = spark.table(gold_table)
    missing = [c for c in ("ever_placed", "app_status") if c not in g.columns]
    if missing:
        raise ValueError(
            f"{gold_table} is missing {missing}. Run 04_sync_collection_from_unicorn.py "
            f"Part 2 first — without the ledger every bottle looks like fresh stock and "
            f"this alert would under-report."
        )
    reserve = (g
        .where(~F.coalesce(F.col("ever_placed"), F.lit(False)))
        .where(F.col("app_status") != "retired")
        .where(F.col("retail_value").isNotNull()))

    n_total = reserve.count()
    rows = (reserve
        .select(F.explode(udf_cells("retail_value")).alias("c"))
        .select(F.col("c.tier").alias("tier"), F.col("c.band_idx").alias("band_idx"))
        .groupBy("tier", "band_idx").count()
        .collect())
    return n_total, {(int(r["tier"]), int(r["band_idx"])): int(r["count"]) for r in rows}


def floor_gaps(spirit):
    """{(tier, band): (on_floor, target, shortfall)} from what 05 last published."""
    try:
        rows = (spark.table(GAP_TBL).where(F.col("spirit") == spirit).collect())
    except Exception as e:
        print(f"  [info] {GAP_TBL} not readable ({e}) — run 05_replenish_floor.py first. "
              f"Reporting reserve coverage only.")
        return {}
    return {(int(r["tier"]), int(r["band_idx"])):
            (int(r["bottles_on_floor"]), int(r["target_bottles"]), int(r["shortfall"]))
            for r in rows}

# COMMAND ----------

# MAGIC %md ## 2. Build the recommendation table

# COMMAND ----------

import datetime

now = datetime.datetime.now()
recommendations = []
pool_totals = {}

for spirit in TARGETS:
    n_reserve, coverage = reserve_coverage(SPIRITS[spirit])
    gaps = floor_gaps(spirit)
    pool_totals[spirit] = n_reserve

    for t in TIERS:
        for b in BANDS:
            target = target_cell_count(b, PULLS_PER_DAY)
            in_reserve = coverage.get((t, b), 0)
            on_floor, _tgt, shortfall = gaps.get((t, b), (None, target, 0))
            refills = in_reserve / target if target else 0.0

            # Buy enough to close the floor gap AND restore a MIN_REFILLS-deep
            # bench behind it, minus what the reserve already covers. The target is
            # already RESTOCK_DAYS of departures, so one refill of bench is one more
            # restock period of cover.
            want_in_reserve = int(round(MIN_REFILLS * target))
            to_buy = max(0, shortfall + want_in_reserve - in_reserve)

            low_floor = shortfall > 0
            low_reserve = refills < MIN_REFILLS
            if not (low_floor or low_reserve):
                continue

            recommendations.append({
                "spirit": spirit,
                "tier": t,
                "pull_price": TIER_PRICE[t],
                "band_idx": b,
                "value_range": band_label(t, b),
                "band_probability": float(target_prob(b)),
                "is_win_band": bool(is_win_band(b)),
                "target_retail_value": float(band_target_value(t, b)),
                "max_buy_price": float(max_buy_price(t, b, GROSS_MARGIN)),
                "bottles_on_floor": on_floor,
                "target_bottles": target,
                "floor_shortfall": shortfall,
                "bottles_in_reserve": in_reserve,
                "refills_of_cover": float(refills),
                "bottles_to_buy": to_buy,
                "line_budget": float(max_buy_price(t, b, GROSS_MARGIN) * to_buy),
                # Empty cell on a live floor is a different kind of problem from a
                # thin bench — it is actively distorting the odds right now.
                "urgency": ("critical" if on_floor == 0 else
                            "high" if low_floor else
                            "watch"),
                "gross_margin": GROSS_MARGIN,
                "computed_at": now,
            })

# MAGIC ### One bottle fills several cells
# MAGIC A $105 bottle is a tier 2 band 2 win *and* a tier 1 band 4 big win. Summing
# MAGIC each cell's shortfall independently would buy it twice. So the list is walked
# MAGIC most-urgent first, and every bottle bought for a cell is credited to all the
# MAGIC other cells a bottle at that cell's target value lands in.

# COMMAND ----------

URGENCY_ORDER = {"critical": 0, "high": 1, "watch": 2}


def dedupe_shared_bottles(recs):
    remaining = {(r["spirit"], r["tier"], r["band_idx"]): r["bottles_to_buy"] for r in recs}
    for r in sorted(recs, key=lambda r: (URGENCY_ORDER[r["urgency"]], -r["band_probability"])):
        key = (r["spirit"], r["tier"], r["band_idx"])
        buy = max(0, remaining[key])
        r["covered_by_other_cells"] = buy == 0 and r["bottles_to_buy"] > 0
        r["bottles_to_buy"] = buy
        r["line_budget"] = float(r["max_buy_price"] * buy)
        for (t2, _m, b2) in eligible_tiers(r["target_retail_value"]):
            other = (r["spirit"], t2, b2)
            if other != key and other in remaining:
                remaining[other] -= buy
        remaining[key] = 0
    return recs


recommendations = dedupe_shared_bottles(recommendations)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The alert body
# MAGIC Plain text, ordered by urgency then spend. Written to the notebook output so
# MAGIC it lands in the Databricks job-run email verbatim.

# COMMAND ----------

recommendations.sort(key=lambda r: (URGENCY_ORDER[r["urgency"]], -r["line_budget"]))

lines = []
lines.append("=" * 74)
lines.append("CELR INVENTORY REORDER ALERT")
lines.append(f"{now:%Y-%m-%d %H:%M}   target gross margin {GROSS_MARGIN:.0%} against retail")
lines.append("=" * 74)

for spirit in TARGETS:
    lines.append(f"\n{spirit.upper()}: {pool_totals.get(spirit, 0)} never-placed bottles "
                 f"left in the reserve pool.")

if not recommendations:
    lines.append("\nNothing to buy. Every cell is stocked on the floor and has at least "
                 f"{MIN_REFILLS:.1f} refills of cover behind it.")
else:
    total = sum(r["line_budget"] for r in recommendations)
    n_units = sum(r["bottles_to_buy"] for r in recommendations)
    crit = [r for r in recommendations if r["urgency"] == "critical"]
    lines.append(f"\n{len(recommendations)} cells need stock. "
                 f"{n_units} bottles, up to ${total:,.0f} at the ceiling prices below.")
    if crit:
        lines.append(f"\n!! {len(crit)} cells are EMPTY on the live floor. Their bands are "
                     f"contributing zero probability right now, which means the odds players "
                     f"see do not match ODDS_CURVE until these are filled.")

    current_spirit = None
    for r in recommendations:
        if r["spirit"] != current_spirit:
            current_spirit = r["spirit"]
            lines.append(f"\n{'-' * 74}\n{current_spirit.upper()}\n{'-' * 74}")
        floor_txt = "unknown" if r["bottles_on_floor"] is None else str(r["bottles_on_floor"])
        lines.append(
            f"\n[{r['urgency'].upper()}] Tier {r['tier']} (${r['pull_price']} pull) "
            f"band {r['band_idx']} — retail {r['value_range']}"
            f"{'  [WIN BAND]' if r['is_win_band'] else ''}")
        lines.append(f"    drawn {r['band_probability'] * 100:.2f}% of the time in this tier")
        lines.append(f"    on floor:  {floor_txt} / {r['target_bottles']} target"
                     f"{'   SHORT ' + str(r['floor_shortfall']) if r['floor_shortfall'] else ''}")
        lines.append(f"    in reserve:{r['bottles_in_reserve']:>4}  "
                     f"({r['refills_of_cover']:.1f} refills of cover)")
        if r["covered_by_other_cells"]:
            lines.append("    BUY 0 — the bottles bought for another cell above also land here")
            continue
        lines.append(f"    BUY {r['bottles_to_buy']} bottles worth ~${r['target_retail_value']:,.0f} retail")
        lines.append(f"    MAX PRICE ${r['max_buy_price']:,.0f} each   "
                     f"(line budget ${r['line_budget']:,.0f})")

    lines.append(f"\n{'=' * 74}")
    lines.append(f"TOTAL: {n_units} bottles, up to ${total:,.0f}")
    lines.append("Prices are ceilings. Anything above them does not clear "
                 f"{GROSS_MARGIN:.0%} margin on retail.")
    lines.append("=" * 74)

body = "\n".join(lines)
print(body)

# COMMAND ----------

# MAGIC %md ## 4. Persist, then raise the alert

# COMMAND ----------

REC_SCHEMA = (
    "spirit string, tier int, pull_price int, band_idx int, value_range string, "
    "band_probability double, is_win_band boolean, target_retail_value double, "
    "max_buy_price double, bottles_on_floor int, target_bottles int, floor_shortfall int, "
    "bottles_in_reserve int, refills_of_cover double, bottles_to_buy int, "
    "line_budget double, urgency string, gross_margin double, computed_at timestamp"
)
REC_COLS = [c.strip().split(" ")[0] for c in REC_SCHEMA.split(",")]

if recommendations:
    (spark.createDataFrame([tuple(r[c] for c in REC_COLS) for r in recommendations], REC_SCHEMA)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(REC_TBL))
else:
    (spark.createDataFrame([], REC_SCHEMA)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(REC_TBL))
print(f"Wrote {len(recommendations)} recommendations to {REC_TBL}.")

# The job's notification settings decide who gets told. Failing the run is what
# triggers the "on failure" email, and it carries this notebook's output with it —
# so a quiet day sends nothing and a low-stock day sends the buy list.
if recommendations and FAIL_ON_ALERT:
    n_units = sum(r["bottles_to_buy"] for r in recommendations)
    total = sum(r["line_budget"] for r in recommendations)
    raise Exception(
        f"CELR REORDER: buy {n_units} bottles, up to ${total:,.0f}. "
        f"{len([r for r in recommendations if r['urgency'] == 'critical'])} cells empty on the "
        f"live floor. Full buy list in the run output and in {REC_TBL}."
    )
