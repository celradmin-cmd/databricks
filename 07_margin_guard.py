# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Margin guard (hourly)
# MAGIC
# MAGIC Watches the house margin and keeps it inside **15–20%** of pull revenue. Runs every
# MAGIC hour. It never changes the live odds itself: when the floor needs re-weighting it
# MAGIC writes a **proposal** that an admin approves or rejects on `/admin/odds`, and
# MAGIC `08_apply_odds_proposal.py` applies the approved ones.
# MAGIC
# MAGIC ## Two margins, two different responses
# MAGIC
# MAGIC **Expected margin** — what the live floor (`app.*_weighted`) pays out on average,
# MAGIC from its weights and bottle values: `1 − Σ(weight × multiple) / Σ weight`, per tier.
# MAGIC Exact, no luck involved. This is the number the guard **fixes**. Outside 15–20%:
# MAGIC 1. Re-weight the current floor with the curve in force (the same `cell_weights()`
# MAGIC    `04` uses). Shipped bottles leave cells lopsided; often that alone is enough.
# MAGIC 2. If not, shift probability between the shallow-loss band (0.70–1.00x) and the
# MAGIC    win band just above it (1.00–1.25x) until the tier lands on 17.5%. At most
# MAGIC    `max_shift` per proposal, never more than `MAX_DRIFT_PER_BAND` from the base
# MAGIC    curve, never a win rate outside `MIN_WIN_RATE`–`MAX_WIN_RATE` (all in `00`).
# MAGIC    When the guardrails stop it short, the proposal says so: the real fix is then
# MAGIC    stocking cheaper bottles in the cells that run high.
# MAGIC
# MAGIC **Realized margin** — what actually happened in `app.rips` over the last 24 h / 7 d /
# MAGIC 30 d. Revenue is `price_paid`; cost is the sellback `payout` for sold rips and the
# MAGIC bottle's retail value for shipped / vaulted / undecided ones. Below 15% it
# MAGIC **notifies, and never changes odds**: retuning in reaction to players winning chases
# MAGIC luck and changes published odds because people won. Each alert carries a z-score
# MAGIC against what the floor should have paid, so you can see whether it was luck
# MAGIC (|z| < 2) or something to investigate — wrong retail values, a sync bug, abuse.
# MAGIC
# MAGIC ## Notifications
# MAGIC Same pattern as `05`: the run **fails** when there is something to tell you, and the
# MAGIC job's "on failure" email carries the message. Each problem notifies once, then again
# MAGIC every 24 h while it lasts (`app.margin_guard_alerts`), so an hourly job doesn't send
# MAGIC 24 emails about one thing. A new proposal always notifies.
# MAGIC
# MAGIC ## Tables (all in `app`)
# MAGIC | Table | What |
# MAGIC |---|---|
# MAGIC | `margin_guard_snapshots` | Every reading, every hour — for a dashboard |
# MAGIC | `margin_guard_alerts` | Open problems and when they were last notified |
# MAGIC | `odds_proposals` | Proposed changes, their AI summary, and the admin's decision |
# MAGIC | `odds_curve_versions` | Approved curves; `00` loads the active ones |

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("app_schema", "app", "App schema")
dbutils.widgets.text("windows_hours", "24,168,720", "Realized-margin windows, hours")
dbutils.widgets.text("min_pulls", "30", "Pulls needed before a window's realized margin can alert")
dbutils.widgets.text("z_threshold", "2.0", "z-score above which a payout is beyond luck")
dbutils.widgets.text("max_shift", "0.02", "Most probability one proposal may move per tier")
dbutils.widgets.text("proposal_ttl_hours", "24", "Hours before an undecided proposal expires")
dbutils.widgets.text("renotify_hours", "24", "Re-send an unresolved alert after this many hours")
dbutils.widgets.text("reject_cooldown_hours", "24", "Hours after a rejection before proposing again")
dbutils.widgets.text("llm_endpoint", "databricks-llama-4-maverick", "Model serving endpoint for the summary")
dbutils.widgets.text("app_base_url", "https://thecelr.com", "App URL (for the approval link)")
dbutils.widgets.dropdown("fail_on_alert", "true", ["true", "false"], "Fail the run to send the alert email")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (read + decide, no writes, no alert)")

CATALOG       = dbutils.widgets.get("catalog")
APP           = dbutils.widgets.get("app_schema")
WINDOWS       = [int(h) for h in dbutils.widgets.get("windows_hours").split(",") if h.strip()]
MIN_PULLS     = int(dbutils.widgets.get("min_pulls"))
Z_THRESHOLD   = float(dbutils.widgets.get("z_threshold"))
MAX_SHIFT     = float(dbutils.widgets.get("max_shift"))
TTL_HOURS     = int(dbutils.widgets.get("proposal_ttl_hours"))
RENOTIFY_H    = int(dbutils.widgets.get("renotify_hours"))
REJECT_COOLDOWN_H = int(dbutils.widgets.get("reject_cooldown_hours"))
LLM_ENDPOINT  = dbutils.widgets.get("llm_endpoint").strip()
APP_URL       = dbutils.widgets.get("app_base_url").rstrip("/")
FAIL_ON_ALERT = dbutils.widgets.get("fail_on_alert") == "true"
DRY_RUN       = dbutils.widgets.get("dry_run") == "true"

# COMMAND ----------

# MAGIC %run ./margin_guard_common

# COMMAND ----------

import datetime as dt, json, math, uuid

NOW = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
if not DRY_RUN:
    ensure_tables()

def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"

def zfmt(z):
    return "n/a" if z is None else f"{z:+.1f}"

# COMMAND ----------

# MAGIC %md ## 1. Expected margin — the live floor

# COMMAND ----------

floor, expected = {}, {}   # expected[(spirit, tier)] = dict
for spirit in FLOOR_TABLES:
    floor[spirit] = load_floor(spirit)
    for t in sorted(SELLABLE_TIERS):
        pl = floor[spirit].get(t, [])
        ev, var, wr = payout_stats(pl)
        expected[(spirit, t)] = {"ev": ev, "var": var, "win_rate": wr, "rows": len(pl),
                                 "margin": None if ev is None else 1 - ev}
        print(f"{spirit:<8} tier {t} (${TIER_PRICE[t]:>5}): {len(pl):>5} placements  "
              f"expected margin {pct(expected[(spirit, t)]['margin']):>7}  win rate {pct(wr)}")

# COMMAND ----------

# MAGIC %md ## 2. Realized margin — `app.rips`

# COMMAND ----------

realized = []   # one dict per (spirit, tier or 0 = all tiers, window)
for hours in WINDOWS:
    label = f"{hours}h" if hours < 48 else f"{hours // 24}d"
    rows = spark.sql(f"""
        SELECT COALESCE(spirit, 'bourbon') AS spirit, tier,
               COUNT(*) AS pulls,
               SUM(price_paid) AS revenue,
               SUM(CASE WHEN status = 'sold' AND payout IS NOT NULL THEN payout
                        ELSE bourbon_retail_value END) AS cost,
               SUM(CASE WHEN bourbon_retail_value IS NULL THEN 1 ELSE 0 END) AS missing_value
        FROM {RIPS_TBL}
        WHERE ripped_at >= current_timestamp() - INTERVAL {hours} HOURS
          AND price_paid > 0
        GROUP BY 1, 2
    """).collect()
    by_spirit = {}
    for r in rows:
        spirit, t = r["spirit"], int(r["tier"])
        e = expected.get((spirit, t), {})
        n, rev, cost = int(r["pulls"]), float(r["revenue"] or 0), float(r["cost"] or 0)
        exp_cost = rev * e["ev"] if e.get("ev") is not None else None
        se = TIER_PRICE.get(t, 0) * math.sqrt(n * e["var"]) if e.get("var") else None
        rec = {"spirit": spirit, "tier": t, "window": label, "pulls": n, "revenue": rev, "cost": cost,
               "realized_margin": 1 - cost / rev if rev else None, "missing_value": int(r["missing_value"]),
               "exp_cost": exp_cost, "se": se,
               "z": (cost - exp_cost) / se if se else None}
        realized.append(rec)
        by_spirit.setdefault(spirit, []).append(rec)
    for spirit, recs in by_spirit.items():   # spirit-wide roll-up, tier 0
        rev, cost = sum(x["revenue"] for x in recs), sum(x["cost"] for x in recs)
        have = [x for x in recs if x["exp_cost"] is not None and x["se"]]
        se = math.sqrt(sum(x["se"] ** 2 for x in have)) if have else None
        exp_cost = sum(x["exp_cost"] for x in have) if have else None
        realized.append({"spirit": spirit, "tier": 0, "window": label, "pulls": sum(x["pulls"] for x in recs),
                         "revenue": rev, "cost": cost, "realized_margin": 1 - cost / rev if rev else None,
                         "missing_value": sum(x["missing_value"] for x in recs),
                         "exp_cost": exp_cost, "se": se,
                         "z": (cost - exp_cost) / se if se and exp_cost is not None else None})

for r in sorted(realized, key=lambda r: (r["spirit"], r["tier"], r["window"])):
    print(f"{r['spirit']:<8} {'all' if r['tier'] == 0 else 'tier ' + str(r['tier']):<7} {r['window']:>4}: "
          f"{r['pulls']:>6} pulls  ${r['revenue']:>10,.0f} in  ${r['cost']:>10,.0f} out  "
          f"margin {pct(r['realized_margin']):>7}  z {zfmt(r['z'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Proposals — fix the expected margin
# MAGIC Only one open proposal per spirit: while one is waiting for a decision (or being
# MAGIC applied), the guard keeps measuring but doesn't stack a second on top of it.

# COMMAND ----------

TARGET_EV = 1 - HOUSE_MARGIN_TARGET
LOSS_BAND, WIN_BAND = WIN_BAND_MIN - 1, WIN_BAND_MIN   # 0.70–1.00x and 1.00–1.25x
BASE_PROBS = [p for (_lo, _hi, p) in ODDS_CURVE]

def in_band(margin):
    return margin is not None and HOUSE_MARGIN_MIN <= margin <= HOUSE_MARGIN_MAX

def shifted(probs, delta):
    """Move `delta` of probability from the win band into the loss band below it
    (negative delta moves it back). Positive = the house keeps more."""
    p = list(probs)
    p[LOSS_BAND] += delta
    p[WIN_BAND] -= delta
    return p

def shift_bounds(probs):
    """The delta range the guardrails in 00 allow from here."""
    wr = sum(probs[WIN_BAND_MIN:])
    hi = min(MAX_SHIFT, probs[WIN_BAND] - 0.001,
             probs[WIN_BAND] - (BASE_PROBS[WIN_BAND] - MAX_DRIFT_PER_BAND),
             (BASE_PROBS[LOSS_BAND] + MAX_DRIFT_PER_BAND) - probs[LOSS_BAND],
             wr - MIN_WIN_RATE)
    lo = max(-MAX_SHIFT, -(probs[LOSS_BAND] - 0.001),
             probs[WIN_BAND] - (BASE_PROBS[WIN_BAND] + MAX_DRIFT_PER_BAND),
             (BASE_PROBS[LOSS_BAND] - MAX_DRIFT_PER_BAND) - probs[LOSS_BAND],
             wr - MAX_WIN_RATE)
    return min(lo, 0.0), max(hi, 0.0)

def plan_tier(spirit, t):
    """What to propose for one tier whose expected margin is outside the band, or None."""
    pl = floor[spirit].get(t, [])
    e = expected[(spirit, t)]
    probs = tier_probs(t, spirit)
    base = {"tier": t, "pull_price": TIER_PRICE[t], "floor_rows": len(pl),
            "margin_now": e["margin"], "win_rate_now": e["win_rate"], "probs_now": probs}

    ev_rw, _v, wr_rw = payout_stats(reweighted(pl, probs))
    if in_band(1 - ev_rw):
        return {**base, "kind": "reweight", "shift": 0.0, "capped": False, "probs_after": probs,
                "margin_after": 1 - ev_rw, "win_rate_after": wr_rw,
                "note": "Weights have drifted as bottles shipped; re-weighting with the current curve fixes it."}

    ev_at = lambda d: payout_stats(reweighted(pl, shifted(probs, d)))[0]
    lo, hi = shift_bounds(probs)
    # Payout falls as delta rises (the win band's bottles are worth more than the
    # loss band's), so the target sits on one side of 0 and bisection finds it.
    limit = hi if ev_rw > TARGET_EV else lo
    if limit == 0.0:
        delta = 0.0
    elif (ev_at(limit) > TARGET_EV) == (ev_rw > TARGET_EV):
        delta = limit                                  # target is beyond the guardrails
    else:
        a, b = sorted((0.0, limit))
        for _ in range(40):
            mid = (a + b) / 2
            if ev_at(mid) > TARGET_EV:
                a = mid
            else:
                b = mid
        delta = (a + b) / 2
    delta = round(delta, 4)
    after = [round(p, 6) for p in shifted(probs, delta)]
    after[LOSS_BAND] += round(1.0 - sum(after), 6)    # keep the sum exactly 1
    ev_new, _v, wr_new = payout_stats(reweighted(pl, after))
    capped = not in_band(1 - ev_new)
    if delta == 0.0:
        return {**base, "kind": "blocked", "shift": 0.0, "capped": True, "probs_after": probs,
                "margin_after": 1 - ev_rw, "win_rate_after": wr_rw,
                "note": "Guardrails allow no further shift. The cells in this tier are stocked too "
                        "high in their bands — stock cheaper bottles (see 04's HIGH cells)."}
    return {**base, "kind": "curve_shift", "shift": delta, "capped": capped, "probs_after": after,
            "margin_after": 1 - ev_new, "win_rate_after": wr_new,
            "note": ("Moves {:.2f} points of probability {} the 1.00–1.25x win band.{}".format(
                abs(delta) * 100, "out of" if delta > 0 else "into",
                " Guardrails stop it short of 17.5%; another step will be proposed after this "
                "one is applied, and restocking cheaper bottles is the lasting fix." if capped else ""))}

open_props = {}
if spark.catalog.tableExists(PROPOSAL_TBL):
    if not DRY_RUN:
        spark.sql(f"UPDATE {PROPOSAL_TBL} SET status = 'expired' "
                  f"WHERE status = 'pending' AND expires_at < current_timestamp()")
    # A rejection holds for REJECT_COOLDOWN_H: re-proposing the same change every
    # hour after someone said no is noise. The expected-margin alert keeps running.
    open_props = {r["spirit"]: r["proposal_id"] for r in spark.sql(
        f"SELECT spirit, proposal_id FROM {PROPOSAL_TBL} "
        f"WHERE status IN ('pending', 'approved', 'applying') "
        f"   OR (status = 'rejected' AND decided_at > current_timestamp() - INTERVAL {REJECT_COOLDOWN_H} HOURS)"
    ).collect()}

plans = {}
for (spirit, t), e in expected.items():
    if e["rows"] and not in_band(e["margin"]):
        plans.setdefault(spirit, []).append(plan_tier(spirit, t))

for spirit, tiers in plans.items():
    for p in tiers:
        print(f"{spirit:<8} tier {p['tier']}: {p['kind']:<11} margin {pct(p['margin_now'])} -> "
              f"{pct(p['margin_after'])}  win rate {pct(p['win_rate_now'])} -> {pct(p['win_rate_after'])}"
              f"{'  (capped)' if p['capped'] else ''}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. AI summary
# MAGIC The model only **explains** — every number it sees was computed above, and it is
# MAGIC told not to introduce others. If the endpoint fails, a plain summary is used, so a
# MAGIC model outage never blocks a proposal.

# COMMAND ----------

SYSTEM = (
    "You are the margin analyst for CËLR, a mystery-bottle app where players pay a fixed price "
    "per pull and receive a bottle whose retail value is drawn from weighted odds. The house "
    "targets a 15–20% margin on pull revenue. Write a short briefing for the owner, who must "
    "approve or reject a proposed odds change. Use ONLY the numbers in the data provided; never "
    "invent or extrapolate figures. Plain language, no hype. Markdown with three short sections: "
    "**What's happening**, **What this change does**, **What it can't fix**. Under 200 words."
)

def plain_summary(spirit, tiers, recent):
    lines = [f"**What's happening** — {spirit} floor margin is outside 15–20% in "
             f"{len(tiers)} tier(s)."]
    for p in tiers:
        lines.append(f"- Tier {p['tier']} (${p['pull_price']}): {pct(p['margin_now'])} → "
                     f"{pct(p['margin_after'])}, win rate {pct(p['win_rate_now'])} → "
                     f"{pct(p['win_rate_after'])}. {p['note']}")
    if recent:
        lines.append(f"Realized, last 24h: {pct(recent['realized_margin'])} on "
                     f"{recent['pulls']} pulls.")
    return "\n".join(lines)

def ai_summary(spirit, tiers, recent):
    payload = {"spirit": spirit, "margin_band": [HOUSE_MARGIN_MIN, HOUSE_MARGIN_MAX],
               "target_margin": HOUSE_MARGIN_TARGET, "tiers": tiers, "realized_last_24h": recent}
    try:
        import mlflow.deployments
        resp = mlflow.deployments.get_deploy_client("databricks").predict(
            endpoint=LLM_ENDPOINT, inputs={
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": json.dumps(payload, default=float)}],
                "max_tokens": 500, "temperature": 0.2})
        text = resp["choices"][0]["message"]["content"].strip()
        return (text, LLM_ENDPOINT) if text else (plain_summary(spirit, tiers, recent), "fallback")
    except Exception as e:
        print(f"[warn] summary model failed ({e}) — using the plain summary")
        return plain_summary(spirit, tiers, recent), "fallback"

# COMMAND ----------

# MAGIC %md ## 5. Write proposals

# COMMAND ----------

PROPOSAL_SCHEMA = ("proposal_id string, spirit string, created_at timestamp, expires_at timestamp, "
                   "status string, reason string, tiers_json string, ai_summary string, model string, "
                   "decided_by string, decided_at timestamp, decision_note string, "
                   "applied_at timestamp, apply_result string")
notify = []        # lines for the alert email
new_proposals = []
for spirit, tiers in plans.items():
    actionable = [p for p in tiers if p["kind"] in ("reweight", "curve_shift")]
    for p in tiers:
        if p["kind"] == "blocked":
            notify.append(f"[{spirit} tier {p['tier']}] expected margin {pct(p['margin_now'])} and the "
                          f"guard can't move it further — restock needed. {p['note']}")
    if not actionable:
        continue
    if spirit in open_props:
        print(f"{spirit}: proposal {open_props[spirit]} still open (or recently rejected) — not adding another.")
        continue
    recent = next((r for r in realized if r["spirit"] == spirit and r["tier"] == 0
                   and r["window"] == "24h"), None)
    recent_slim = recent and {k: recent[k] for k in ("pulls", "revenue", "cost", "realized_margin", "z")}
    summary, model = ai_summary(spirit, actionable, recent_slim)
    pid = str(uuid.uuid4())
    worst = min(p["margin_now"] for p in actionable)
    reason = (f"Expected margin {pct(worst)} in " +
              ", ".join(f"tier {p['tier']} ({pct(p['margin_now'])})" for p in actionable))
    new_proposals.append((pid, spirit, NOW, NOW + dt.timedelta(hours=TTL_HOURS), "pending", reason,
                          json.dumps(actionable, default=float), summary, model,
                          None, None, None, None, None))
    notify.append(f"[{spirit}] NEW ODDS PROPOSAL awaiting approval — {reason}. "
                  f"Review: {APP_URL}/admin/odds")
    print(f"\n--- {spirit} proposal {pid} ---\n{summary}\n")

if new_proposals and not DRY_RUN:
    spark.createDataFrame(new_proposals, PROPOSAL_SCHEMA).write.mode("append").saveAsTable(PROPOSAL_TBL)
    print(f"Wrote {len(new_proposals)} proposal(s) to {PROPOSAL_TBL}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alerts, snapshots, and the email
# MAGIC Realized-margin alerts never touch the odds. Expected-margin alerts are covered by
# MAGIC the proposal above (or the "restock needed" line when the guard is blocked).

# COMMAND ----------

current = {}   # alert_key -> (kind, spirit, tier, message)
for r in realized:
    if r["pulls"] >= MIN_PULLS and r["realized_margin"] is not None and r["realized_margin"] < HOUSE_MARGIN_MIN:
        beyond = r["z"] is not None and r["z"] > Z_THRESHOLD
        scope = "all tiers" if r["tier"] == 0 else f"tier {r['tier']}"
        kind = "realized_low_beyond_luck" if beyond else "realized_low_luck"
        msg = (f"[{r['spirit']} {scope}, last {r['window']}] realized margin {pct(r['realized_margin'])} "
               f"on {r['pulls']} pulls (${r['revenue']:,.0f} in, ${r['cost']:,.0f} out). "
               + (f"z = {r['z']:+.1f}: more than luck explains — check retail values, sync and sellbacks."
                  if beyond else
                  f"z = {zfmt(r['z'])}: within normal luck; odds unchanged."))
        current[f"{kind}:{r['spirit']}:{r['tier']}:{r['window']}"] = (kind, r["spirit"], r["tier"], msg)
    if r["missing_value"] and r["window"] == f"{WINDOWS[0]}h":
        current[f"data:{r['spirit']}:{r['tier']}"] = (
            "data", r["spirit"], r["tier"],
            f"[{r['spirit']} tier {r['tier']}] {r['missing_value']} rips in the last {r['window']} have no "
            f"bourbon_retail_value — they count as $0 cost, so realized margin reads too high.")
for (spirit, t), e in expected.items():
    if e["margin"] is not None and e["margin"] < 0:
        current[f"expected_negative:{spirit}:{t}"] = (
            "expected_negative", spirit, t,
            f"[{spirit} tier {t}] CRITICAL: the floor is designed to LOSE money — expected margin "
            f"{pct(e['margin'])}. Approve the open proposal or pull this tier from sale.")

open_alerts = {}
if spark.catalog.tableExists(ALERT_TBL):
    open_alerts = {r["alert_key"]: r["last_notified"] for r in
                   spark.sql(f"SELECT alert_key, last_notified FROM {ALERT_TBL} WHERE resolved_at IS NULL").collect()}
due = [k for k in current
       if k not in open_alerts or open_alerts[k] is None
       or open_alerts[k] < NOW - dt.timedelta(hours=RENOTIFY_H)]
notify += [current[k][3] for k in sorted(due)]
resolved = [k for k in open_alerts if k not in current]

snap_rows = [(NOW, r["spirit"], r["tier"], r["window"], r["pulls"], r["revenue"], r["cost"],
              r["realized_margin"],
              None if r["tier"] == 0 else expected.get((r["spirit"], r["tier"]), {}).get("margin"),
              r["z"], None, None) for r in realized]
snap_rows += [(NOW, s, t, "floor", None, None, None, None, e["margin"], None, e["rows"], e["win_rate"])
              for (s, t), e in expected.items()]
SNAP_SCHEMA = ("run_at timestamp, spirit string, tier int, window string, pulls bigint, revenue double, "
               "cost double, realized_margin double, expected_margin double, z_score double, "
               "floor_rows int, win_rate double")

if DRY_RUN:
    print(f"DRY RUN — {len(snap_rows)} snapshot rows, {len(current)} open problems "
          f"({len(due)} due to notify), {len(resolved)} resolved. Nothing written.")
else:
    spark.createDataFrame(snap_rows, SNAP_SCHEMA).write.mode("append").saveAsTable(SNAPSHOT_TBL)
    alert_rows = [(k, v[0], v[1], int(v[2]), v[3]) for k, v in current.items()]
    if alert_rows:
        spark.createDataFrame(alert_rows, "alert_key string, kind string, spirit string, tier int, message string") \
             .createOrReplaceTempView("_guard_alerts")
        due_sql = ", ".join(f"'{k}'" for k in due) or "''"
        spark.sql(f"""
            MERGE INTO {ALERT_TBL} t USING _guard_alerts s
            ON t.alert_key = s.alert_key AND t.resolved_at IS NULL
            WHEN MATCHED THEN UPDATE SET t.message = s.message, t.last_seen = current_timestamp(),
                 t.last_notified = CASE WHEN s.alert_key IN ({due_sql}) THEN current_timestamp() ELSE t.last_notified END
            WHEN NOT MATCHED THEN INSERT (alert_key, kind, spirit, tier, message, first_seen, last_seen, last_notified, resolved_at)
                 VALUES (s.alert_key, s.kind, s.spirit, s.tier, s.message, current_timestamp(), current_timestamp(), current_timestamp(), NULL)
        """)
    if resolved:
        keys = ", ".join(f"'{k}'" for k in resolved)
        spark.sql(f"UPDATE {ALERT_TBL} SET resolved_at = current_timestamp() "
                  f"WHERE resolved_at IS NULL AND alert_key IN ({keys})")
    print(f"Snapshot written. {len(current)} open problems, {len(due)} notified, {len(resolved)} resolved.")

if notify:
    print("\n".join(["", "NOTIFY:"] + notify))
if notify and FAIL_ON_ALERT and not DRY_RUN:
    raise Exception(f"CELR MARGIN GUARD — {len(notify)} item(s):\n" + "\n".join(notify))
