# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Apply approved odds proposals
# MAGIC
# MAGIC Picks up every proposal an admin approved on `/admin/odds` and puts it on the live
# MAGIC floor. The app starts this job the moment someone clicks Approve (when
# MAGIC `DATABRICKS_ODDS_APPLY_JOB_ID` is set in the app's environment); schedule it every
# MAGIC 15 minutes as well, so an approval still lands if that trigger fails. A run with
# MAGIC nothing approved exits in seconds.
# MAGIC
# MAGIC ## What applying means
# MAGIC 1. **Claim** the proposal (`approved → applying`), so two runs can't apply it twice.
# MAGIC 2. **Activate the curve** for each `curve_shift` tier in `app.odds_curve_versions`,
# MAGIC    superseding that tier's previous row. `reweight` tiers keep their curve.
# MAGIC 3. **Re-weight the floor** by running `04_replenish_floor.py` for that spirit. It
# MAGIC    `%run`s `00`, which now loads the approved curve, so the weights it writes are
# MAGIC    the approved ones. The proposal approved the *curve*, not a list of weights:
# MAGIC    bottles ship between proposal and approval, and the weights are worked out
# MAGIC    against the floor as it is now.
# MAGIC 4. **Sync to the app** — POST the `sync-*-weighted-from-databricks` hook, so
# MAGIC    Supabase (where draws happen) gets the new weights immediately.
# MAGIC 5. Record before/after margins in `apply_result` and mark it `applied`.
# MAGIC
# MAGIC If step 2 or 3 fails, the curve rows are rolled back and the proposal is marked
# MAGIC `failed`. A failed sync (step 4) leaves it `applied` — the floor in Databricks is
# MAGIC right, the app just hasn't pulled it yet — and fails the run so you hear about it.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("app_schema", "app", "App schema")
dbutils.widgets.text("app_base_url", "https://thecelr.com", "App URL (for the sync hooks)")
dbutils.widgets.text("approval_max_age_hours", "24", "Ignore approvals older than this")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (show what would apply)")

CATALOG   = dbutils.widgets.get("catalog")
APP       = dbutils.widgets.get("app_schema")
APP_URL   = dbutils.widgets.get("app_base_url").rstrip("/")
MAX_AGE_H = int(dbutils.widgets.get("approval_max_age_hours"))
DRY_RUN   = dbutils.widgets.get("dry_run") == "true"

# COMMAND ----------

# MAGIC %run ./margin_guard_common

# COMMAND ----------

import datetime as dt, json
import requests

def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"

def sql_str(v):
    return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"

def floor_margins(spirit, tiers):
    fl = load_floor(spirit)
    out = {}
    for t in tiers:
        ev, _v, wr = payout_stats(fl.get(t, []))
        out[t] = {"margin": None if ev is None else 1 - ev, "win_rate": wr}
    return out

ensure_tables()
if not DRY_RUN:
    spark.sql(f"""UPDATE {PROPOSAL_TBL} SET status = 'expired', apply_result = 'approval older than {MAX_AGE_H}h'
                  WHERE status = 'approved' AND decided_at < current_timestamp() - INTERVAL {MAX_AGE_H} HOURS""")
approved = spark.sql(f"SELECT * FROM {PROPOSAL_TBL} WHERE status = 'approved' ORDER BY decided_at").collect()
print(f"{len(approved)} approved proposal(s) to apply.")

# COMMAND ----------

problems = []
for prop in approved:
    pid, spirit = prop["proposal_id"], prop["spirit"]
    tiers = json.loads(prop["tiers_json"])
    shifts = [p for p in tiers if p["kind"] == "curve_shift"]
    print(f"\n=== {spirit} proposal {pid} (approved by {prop['decided_by']}) ===")
    for p in tiers:
        print(f"  tier {p['tier']}: {p['kind']}" + (f", shift {p['shift']:+.4f}" if p["kind"] == "curve_shift" else ""))
    if DRY_RUN:
        continue

    claimed = spark.sql(f"UPDATE {PROPOSAL_TBL} SET status = 'applying' "
                        f"WHERE proposal_id = {sql_str(pid)} AND status = 'approved'").collect()[0][0]
    if not claimed:
        print("  already claimed by another run — skipping")
        continue

    tier_list = sorted({int(p["tier"]) for p in tiers})
    before = floor_margins(spirit, tier_list)
    stamp = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(sep=" ", timespec="microseconds")
    try:
        # The curve the proposal was computed from must still be the one in force;
        # otherwise its shift would land on a different base than the one approved.
        for p in shifts:
            now_probs = tier_probs(int(p["tier"]), spirit)
            if max(abs(a - b) for a, b in zip(now_probs, p["probs_now"])) > 1e-9:
                raise RuntimeError(f"tier {p['tier']}'s curve changed since this proposal was made")
            _validate_tier_probs(spirit, int(p["tier"]), [float(x) for x in p["probs_after"]])

        if shifts:
            tier_sql = ", ".join(str(int(p["tier"])) for p in shifts)
            spark.sql(f"""UPDATE {CURVE_TBL} SET status = 'superseded', superseded_at = TIMESTAMP '{stamp}'
                          WHERE spirit = {sql_str(spirit)} AND tier IN ({tier_sql}) AND status = 'active'""")
            values = ", ".join(
                f"({sql_str(spirit)}, {int(p['tier'])}, ARRAY({', '.join(repr(float(x)) for x in p['probs_after'])}), "
                f"'active', {sql_str(pid)}, TIMESTAMP '{stamp}', NULL)" for p in shifts)
            spark.sql(f"INSERT INTO {CURVE_TBL} VALUES {values}")
            APPROVED_TIER_PROBS.clear()
            APPROVED_TIER_PROBS.update(_load_approved_tier_probs())   # validates what we just wrote

        dbutils.notebook.run("./04_replenish_floor", 3600,
                             {"catalog": CATALOG, "app_schema": APP, "spirit": spirit, "dry_run": "false"})
    except Exception as e:
        if shifts:
            spark.sql(f"DELETE FROM {CURVE_TBL} WHERE proposal_id = {sql_str(pid)}")
            spark.sql(f"""UPDATE {CURVE_TBL} SET status = 'active', superseded_at = NULL
                          WHERE spirit = {sql_str(spirit)} AND superseded_at = TIMESTAMP '{stamp}'""")
        spark.sql(f"UPDATE {PROPOSAL_TBL} SET status = 'failed', apply_result = {sql_str(str(e)[:2000])} "
                  f"WHERE proposal_id = {sql_str(pid)}")
        problems.append(f"[{spirit}] proposal {pid} FAILED and was rolled back: {e}")
        print(f"  FAILED, curve rolled back: {e}")
        continue

    after = floor_margins(spirit, tier_list)
    sync = {"ok": False}
    try:
        r = requests.post(f"{APP_URL}/api/public/hooks/{SYNC_HOOKS[spirit]}", timeout=300)
        sync = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {"ok": r.ok}
        sync["http_status"] = r.status_code
    except Exception as e:
        sync = {"ok": False, "error": str(e)}
    if not sync.get("ok"):
        problems.append(f"[{spirit}] proposal {pid} applied in Databricks, but the app sync failed: "
                        f"{sync.get('error') or sync}. POST {APP_URL}/api/public/hooks/{SYNC_HOOKS[spirit]} "
                        f"to finish.")

    result = {"before": before, "after": after, "app_sync": sync}
    spark.sql(f"UPDATE {PROPOSAL_TBL} SET status = 'applied', applied_at = current_timestamp(), "
              f"apply_result = {sql_str(json.dumps(result, default=str))} WHERE proposal_id = {sql_str(pid)}")
    for t in tier_list:
        print(f"  tier {t}: expected margin {pct(before[t]['margin'])} -> {pct(after[t]['margin'])}, "
              f"win rate {pct(before[t]['win_rate'])} -> {pct(after[t]['win_rate'])}")
    print(f"  applied; app sync {'ok' if sync.get('ok') else 'FAILED'}")

if DRY_RUN:
    print("\nDRY RUN — nothing applied.")
if problems:
    raise Exception("CELR ODDS APPLY:\n" + "\n".join(problems))
