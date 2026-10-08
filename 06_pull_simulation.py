# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 · Pull simulation
# MAGIC
# MAGIC Simulates a person pulling bottles in the app, tier by tier, and writes the
# MAGIC results to four Delta tables that `dashboards/pull_simulation.lvdash.json`
# MAGIC turns into a Lakeview dashboard. Built to answer three questions a coworker
# MAGIC will actually ask when looking at the curve in `00_celr_odds_config.py`:
# MAGIC
# MAGIC 1. **How much would they win or lose?** — real dollars, not just percentages.
# MAGIC 2. **How often do they win more than they paid?** — the simulated win rate,
# MAGIC    checked against the 24.6% target `00` asserts.
# MAGIC 3. **How many pulls land between wins?** — e.g. "3 losses then a win" — the
# MAGIC    streak distribution, which a probability alone doesn't make tangible.
# MAGIC
# MAGIC ## Two sources of truth, pick one
# MAGIC - **`live_floor`** (default) — draws from `app.bourbons_weighted` /
# MAGIC   `app.agave_weighted` exactly the way `draw_weighted_bottle()` does in
# MAGIC   Supabase: explode the four slot columns for the requested tier, weight by
# MAGIC   `*_weight`. This is "like a person might in the app" literally — it reflects
# MAGIC   today's real inventory, including any gap placeholders or thin cells.
# MAGIC - **`theoretical`** — ignores inventory and samples `ODDS_CURVE` directly, with
# MAGIC   a bottle's retail value drawn uniformly inside its band's dollar range. Use
# MAGIC   this to demo the *designed* game to coworkers independent of today's stock,
# MAGIC   or as a sanity check when the live floor is thin or not yet built.
# MAGIC
# MAGIC Both modes draw **with replacement** across pulls, which matches reality for a
# MAGIC single session: a pull doesn't remove a row from the floor, a *decision*
# MAGIC (`decideRip` → `retireBottle()`) does, and `04_replenish_floor.py` restocks
# MAGIC behind it — so treating the floor as static across one person's 100 pulls is
# MAGIC the same assumption `weighted_tier_catalog()`/`draw_weighted_bottle()` already make.
# MAGIC
# MAGIC ## Why many sessions, not just one
# MAGIC "100 pulls" is one person's night. A dashboard built from a single session
# MAGIC would show whatever streak that one run happened to hit, which is exactly the
# MAGIC kind of noise coworkers misread as "the game." `N_SESSIONS` independent
# MAGIC 100-pull sessions give the streak distribution and win-rate convergence
# MAGIC enough data to be a real answer — and session `EXAMPLE_SESSION_ID` is always
# MAGIC kept in full, pull-by-pull, as the concrete walkthrough ("3 losses, then a
# MAGIC $62 win on a $50 pull") for the dashboard's narrative table.
# MAGIC
# MAGIC `DRY_RUN=True` reports without writing.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("gold_schema", "gold", "Gold schema (where simulation tables are written)")
dbutils.widgets.text("app_schema", "app", "App schema (where the live floor is read from)")
dbutils.widgets.dropdown("spirit", "both", ["both", "bourbon", "agave"], "Spirit to simulate")
dbutils.widgets.dropdown("source", "live_floor", ["live_floor", "theoretical"],
                          "Draw from the real floor, or sample ODDS_CURVE directly")
dbutils.widgets.text("pulls_per_session", "100", "Pulls in one simulated session (one person's night)")
dbutils.widgets.text("n_sessions", "1000", "Independent sessions per tier (statistical depth for the dashboard)")
dbutils.widgets.text("example_session_id", "0", "Which session (0-indexed) is kept pull-by-pull for the walkthrough table")
dbutils.widgets.text("random_seed", "", "Seed — blank = fresh random run; set one (e.g. a past run's seed) to replay it exactly")
dbutils.widgets.dropdown("dry_run", "false", ["true", "false"], "Dry run (report only, no writes)")

import random

CATALOG     = dbutils.widgets.get("catalog")
GOLD        = dbutils.widgets.get("gold_schema")
APP         = dbutils.widgets.get("app_schema")
SPIRIT_ARG  = dbutils.widgets.get("spirit")
SOURCE      = dbutils.widgets.get("source")
PULLS_PER_SESSION = int(dbutils.widgets.get("pulls_per_session"))
N_SESSIONS  = int(dbutils.widgets.get("n_sessions"))
EXAMPLE_SESSION_ID = int(dbutils.widgets.get("example_session_id"))
# Blank seed = a new one every run. A fixed default (it used to be 42) made every
# run against the same floor produce identical numbers. The seed actually used is
# written to the KPI table, so any run can still be replayed exactly.
_seed_arg   = dbutils.widgets.get("random_seed").strip()
SEED        = int(_seed_arg) if _seed_arg else random.SystemRandom().randrange(2**31)
DRY_RUN     = dbutils.widgets.get("dry_run") == "true"

if not (0 <= EXAMPLE_SESSION_ID < N_SESSIONS):
    raise ValueError(f"example_session_id must be in [0, {N_SESSIONS}), got {EXAMPLE_SESSION_ID}")

SPIRITS = {
    "bourbon": f"{CATALOG}.{APP}.bourbons_weighted",
    "agave":   f"{CATALOG}.{APP}.agave_weighted",
}
TARGETS = list(SPIRITS) if SPIRIT_ARG == "both" else [SPIRIT_ARG]
TIERS = sorted(SELLABLE_TIERS)
SLOT_NAMES = ["primary", "secondary", "tertiary", "quaternary", "quinary"]  # all 5 tiers are sellable now

PULLS_TBL     = f"{CATALOG}.{GOLD}.pull_simulation_pulls"
SESSIONS_TBL  = f"{CATALOG}.{GOLD}.pull_simulation_sessions"
STREAKS_TBL   = f"{CATALOG}.{GOLD}.pull_simulation_streaks"
TIER_KPI_TBL  = f"{CATALOG}.{GOLD}.pull_simulation_tier_kpi"

print(f"catalog={CATALOG} spirits={TARGETS} source={SOURCE} "
      f"pulls_per_session={PULLS_PER_SESSION} n_sessions={N_SESSIONS} seed={SEED} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md ## 1. Build the draw pool for one tier
# MAGIC Same shape regardless of source: a list of candidate bottles, each with the
# MAGIC `(retail_value, band_idx, weight)` that decides both its draw probability and
# MAGIC its win/loss outcome. Keeping the two sources behind one interface is what lets
# MAGIC everything below — the sampler, the streak math, the tables — stay source-agnostic.

# COMMAND ----------

from pyspark.sql import functions as F

rng = random.Random(SEED)


def live_floor_pool(spirit, tier):
    """Mirrors `weighted_tier_catalog()` in Supabase: explode the four slot columns,
    keep the ones placed in this tier with positive weight. Flags placeholder rows
    (`rarity = 'placeholder'`) rather than hiding them — a placeholder in the draw
    pool means a real gap in today's inventory, which is itself worth showing."""
    floor = spark.table(SPIRITS[spirit])
    parts = []
    for name in SLOT_NAMES:
        parts.append(floor.select(
            F.col("id"), F.col("name"), F.col("rarity"),
            F.col("retail_value").cast("double").alias("retail_value"),
            F.col(f"{name}_tier").alias("tier"),
            F.col(f"{name}_band").alias("band_idx"),
            F.col(f"{name}_weight").alias("weight"),
        ).where((F.col(f"{name}_tier") == tier) & (F.coalesce(F.col(f"{name}_weight"), F.lit(0)) > 0)))
    slots = parts[0]
    for p in parts[1:]:
        slots = slots.unionByName(p)

    rows = slots.collect()
    pool = [{
        "label": r["name"],
        "retail_value": float(r["retail_value"]),
        "band_idx": int(r["band_idx"]),
        "weight": float(r["weight"]),
        "is_placeholder": (r["rarity"] or "") == "placeholder",
    } for r in rows]
    return pool


def theoretical_pool(tier):
    """One synthetic 'bottle' per band, weighted exactly as ODDS_CURVE specifies.
    Retail value is resampled uniformly within the band's dollar range on every
    draw (see `draw_from_pool` below) rather than fixed once, so repeated draws of
    the same band don't all report the exact same dollar amount."""
    price = TIER_PRICE[tier]
    pool = []
    for b, (lo, hi, prob) in enumerate(ODDS_CURVE):
        pool.append({
            "label": f"band {b} ({band_label(tier, b)})",
            "band_idx": b,
            "weight": prob,            # probabilities sum to 1.0 — a valid weight set as-is
            "is_placeholder": False,
            "_lo_dollars": lo * price,
            "_hi_dollars": hi * price,
        })
    return pool


def load_pool(spirit, tier):
    if SOURCE == "theoretical":
        return theoretical_pool(tier)
    pool = live_floor_pool(spirit, tier)
    if not pool:
        raise ValueError(
            f"{SPIRITS[spirit]} has no bottles placed in tier {tier} — nothing to simulate. "
            f"Run 02/03 (build) or 05 (replenish) first, or switch source to 'theoretical' "
            f"to simulate the designed curve independent of today's inventory."
        )
    return pool

# COMMAND ----------

# MAGIC %md ## 2. Draw — weighted sample with replacement, same algorithm as the app
# MAGIC `draw_weighted_bottle()` in Supabase does a cumulative-weight walk against
# MAGIC `random() * total`; `random.choices` is the same algorithm, just vectorized by
# MAGIC the standard library instead of hand-rolled SQL. Using a single `Random`
# MAGIC instance seeded once (not reseeded per session) is what makes the whole
# MAGIC `N_SESSIONS` run reproducible end to end from `random_seed` alone.

# COMMAND ----------

def draw_one(pool):
    """One pull: returns (label, retail_value, band_idx, is_placeholder)."""
    choice = rng.choices(pool, weights=[p["weight"] for p in pool], k=1)[0]
    if SOURCE == "theoretical":
        # Fresh dollar value every draw so a band isn't one repeated number.
        rv = rng.uniform(choice["_lo_dollars"], choice["_hi_dollars"])
    else:
        rv = choice["retail_value"]
    return choice["label"], rv, choice["band_idx"], choice["is_placeholder"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run every session, tier by tier
# MAGIC Streak bookkeeping happens here, inline, because it's inherently sequential —
# MAGIC "pulls since the last win" only means something walked in order. For each win,
# MAGIC `losses_before` is exactly the "3 losing then a winning" count the dashboard
# MAGIC narrates; a session's trailing losses after its last win are also recorded
# MAGIC (`trailing_losses` on the session row) so a streak in progress isn't lost.

# COMMAND ----------

pull_rows = []
session_rows = []
streak_rows = []     # (spirit, tier, losses_before_win) — one row per win, long-form for the histogram

for spirit in TARGETS:
    for tier in TIERS:
        price = TIER_PRICE[tier]
        pool = load_pool(spirit, tier)
        placeholder_share = (
            sum(p["weight"] for p in pool if p.get("is_placeholder"))
            / sum(p["weight"] for p in pool)
        ) if SOURCE == "live_floor" else 0.0
        if placeholder_share > 0:
            print(f"[warn] {spirit} tier {tier}: {placeholder_share:.1%} of draw weight is "
                  f"placeholder (gap) rows — this simulation will show pulls that don't "
                  f"correspond to a real bottle yet.")

        for session_id in range(N_SESSIONS):
            is_example = session_id == EXAMPLE_SESSION_ID
            losses_in_a_row = 0
            cumulative_net = 0.0
            total_spent = 0.0
            total_retail_won = 0.0
            win_count = 0
            biggest_win = 0.0
            biggest_win_label = None

            for pull_number in range(1, PULLS_PER_SESSION + 1):
                label, retail_value, band_idx, is_placeholder = draw_one(pool)
                win = is_win_band(band_idx)
                net = retail_value - price
                cumulative_net += net
                total_spent += price
                total_retail_won += retail_value

                if win:
                    win_count += 1
                    streak_rows.append((spirit, tier, losses_in_a_row))
                    if net > biggest_win:
                        biggest_win, biggest_win_label = net, label
                    losses_in_a_row = 0
                else:
                    losses_in_a_row += 1

                if is_example:
                    pull_rows.append((
                        spirit, tier, session_id, pull_number, label,
                        float(retail_value), float(price), float(net), bool(win),
                        bool(is_placeholder), float(cumulative_net),
                        losses_in_a_row if not win else 0,
                    ))

            session_rows.append((
                spirit, tier, session_id, PULLS_PER_SESSION,
                float(total_spent), float(total_retail_won), float(cumulative_net),
                win_count, float(win_count / PULLS_PER_SESSION),
                losses_in_a_row,               # trailing losses, streak still open at session end
                float(biggest_win), biggest_win_label, is_example,
            ))

print(f"Simulated {len(session_rows)} sessions "
      f"({len(TARGETS)} spirit(s) x {len(TIERS)} tiers x {N_SESSIONS} sessions), "
      f"{len(pull_rows)} pull-level rows kept for the example session(s), "
      f"{len(streak_rows)} total wins recorded across all sessions.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-tier KPIs — simulated vs. the config's own targets
# MAGIC Cross-checks the simulation against `00_celr_odds_config`'s `win_rate()` and
# MAGIC `expected_multiple()` rather than just trusting it. On `source=live_floor`
# MAGIC these can legitimately differ from the target (real inventory has gaps,
# MAGIC placeholders, finite cell depth); on `source=theoretical` they should converge
# MAGIC to the target as `n_sessions` grows, and a persistent gap there would mean a
# MAGIC bug in this notebook, not in the odds.

# COMMAND ----------

import datetime
from collections import defaultdict

RUN_AT = datetime.datetime.now()

by_tier = defaultdict(list)
for row in session_rows:
    spirit, tier = row[0], row[1]
    by_tier[(spirit, tier)].append(row)

kpi_rows = []
for (spirit, tier), rows in by_tier.items():
    n = len(rows)
    total_pulls = sum(r[3] for r in rows)
    total_spent = sum(r[4] for r in rows)
    total_won = sum(r[5] for r in rows)
    total_net = sum(r[6] for r in rows)
    total_wins = sum(r[7] for r in rows)
    win_rates = [r[8] for r in rows]
    nets = [r[6] for r in rows]
    losses_before = sorted(ls for (sp, t, ls) in streak_rows if sp == spirit and t == tier)

    def pct(vals, p):
        if not vals:
            return None
        idx = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
        return float(sorted(vals)[idx])

    kpi_rows.append((
        spirit, tier, TIER_PRICE[tier], SOURCE,
        n, total_pulls, float(total_spent), float(total_won), float(total_net),
        float(total_net / total_pulls) if total_pulls else None,   # avg net $ per pull
        float(total_wins / total_pulls) if total_pulls else None,  # simulated win rate
        float(win_rate()),                                         # 00_celr_odds_config target
        float(total_won / total_spent) if total_spent else None,   # simulated payout multiple
        float(expected_multiple()),                                # 00_celr_odds_config target
        float(sum(losses_before) / len(losses_before)) if losses_before else None,
        pct(losses_before, 0.5), pct(losses_before, 0.9),
        max(losses_before) if losses_before else None,
        pct(nets, 0.5), min(nets) if nets else None, max(nets) if nets else None,
        RUN_AT, SEED,
    ))

KPI_COLS = [
    "spirit", "tier", "pull_price", "source",
    "n_sessions", "total_pulls", "total_spent", "total_retail_won", "net_profit_loss",
    "avg_net_dollars_per_pull", "simulated_win_rate", "target_win_rate",
    "simulated_payout_multiple", "target_payout_multiple",
    "avg_losses_before_win", "median_losses_before_win", "p90_losses_before_win",
    "max_losses_before_win",
    "median_session_net", "worst_session_net", "best_session_net",
    "computed_at", "seed",
]

print(f"\n{'spirit':<8}{'tier':<6}{'win% sim':<10}{'win% tgt':<10}{'payout sim':<12}{'payout tgt':<12}{'avg losses->win':<16}")
for r in sorted(kpi_rows, key=lambda r: (r[0], r[1])):
    d = dict(zip(KPI_COLS, r))
    print(f"{d['spirit']:<8}{d['tier']:<6}"
          f"{d['simulated_win_rate']*100:>7.1f}%  {d['target_win_rate']*100:>7.1f}%  "
          f"{d['simulated_payout_multiple']:>9.4f}x  {d['target_payout_multiple']:>9.4f}x  "
          f"{d['avg_losses_before_win']:>14.2f}")

# COMMAND ----------

# MAGIC %md ## 5. Write

# COMMAND ----------

SESSION_COLS = [
    "spirit", "tier", "session_id", "pulls",
    "total_spent", "total_retail_won", "net_profit_loss",
    "win_count", "win_rate", "trailing_losses",
    "biggest_win_dollars", "biggest_win_label", "is_example_session",
]
PULL_COLS = [
    "spirit", "tier", "session_id", "pull_number", "bottle_label",
    "retail_value", "pull_price", "net_dollars", "is_win",
    "is_placeholder", "cumulative_net", "losses_before_this_pull",
]

STREAK_HIST_MAX = 10  # bucket anything beyond this into a single "10+" row, like the band tail

streak_hist = defaultdict(int)
for spirit, tier, losses in streak_rows:
    bucket = min(losses, STREAK_HIST_MAX)
    streak_hist[(spirit, tier, bucket)] += 1

streak_hist_rows = []
for (spirit, tier), rows in by_tier.items():
    total_wins_this_tier = sum(c for (sp, t, _b), c in streak_hist.items() if sp == spirit and t == tier)
    for bucket in range(STREAK_HIST_MAX + 1):
        count = streak_hist.get((spirit, tier, bucket), 0)
        label = f"{bucket}+" if bucket == STREAK_HIST_MAX else str(bucket)
        streak_hist_rows.append((
            spirit, tier, bucket, label, count,
            float(count / total_wins_this_tier) if total_wins_this_tier else 0.0,
        ))

STREAK_COLS = ["spirit", "tier", "losses_before_win", "losses_before_win_label", "occurrences", "pct_of_wins"]

if DRY_RUN:
    print(f"\nDRY RUN — would write:")
    print(f"  {len(pull_rows)} rows -> {PULLS_TBL}")
    print(f"  {len(session_rows)} rows -> {SESSIONS_TBL}")
    print(f"  {len(streak_hist_rows)} rows -> {STREAKS_TBL}")
    print(f"  {len(kpi_rows)} rows -> {TIER_KPI_TBL}")
else:
    (spark.createDataFrame(pull_rows, PULL_COLS)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PULLS_TBL))
    (spark.createDataFrame(session_rows, SESSION_COLS)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SESSIONS_TBL))
    (spark.createDataFrame(streak_hist_rows, STREAK_COLS)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(STREAKS_TBL))
    (spark.createDataFrame(kpi_rows, KPI_COLS)
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TIER_KPI_TBL))
    print(f"Wrote {len(pull_rows)} / {len(session_rows)} / {len(streak_hist_rows)} / {len(kpi_rows)} rows to "
          f"{PULLS_TBL}, {SESSIONS_TBL}, {STREAKS_TBL}, {TIER_KPI_TBL}.")

print("\nNext: import dashboards/pull_simulation.lvdash.json as a Lakeview dashboard "
      "(Dashboards -> Create dashboard -> Import dashboard file) and point it at these "
      "four tables — see databricks/README.md.")
