# databricks

All Things Celr databricks.

## Notebooks, in run order

| Notebook | What it does |
|---|---|
| `00_celr_odds_config.py` | The odds curve. `%run` by everything else. Self-tests on every run. |
| `mock_data_generator/generate_*_bottles.py` | Creates bottle inventory in bronze. |
| `01_classify_and_enrich.py` | bronze → gold: assigns tiers/bands, LLM rarity + description. |
| `02_build_bourbon_app_floor_weighted.py` / `03_..._agave_...` | gold → `app.*_weighted`: builds a floor from scratch. |
| `04_sync_collection_from_unicorn.py` | Pulls the real collection; **Part 2 maintains the never-re-use ledger.** |
| `05_replenish_floor.py` | Refills the floor after shipments and reweights it. **Schedule this.** |
| `06_inventory_reorder_alert.py` | Emails what to buy and what to pay. **Schedule this.** |
| `07_pull_simulation.py` | Simulates pulls tier by tier; feeds the dashboard below. Run ad hoc. |

## The odds curve

Tuned to two targets, both asserted in `00_celr_odds_config.py`:

- **40% win rate** — a bottle worth at least the pull price roughly every 2.5
  pulls, with 79.5% of those wins in band 2 (1.00–1.25x par), the band immediately
  above the loss line.
- **0.9975x payout** — the floor gives back the same retail value per dollar
  pulled as the previous six-band curve did at a 24.4% win rate.

Band 2 is split at 1.25x specifically to make that pair possible; a single wide
1.00–1.60x win band costs ~8% more payout at the same win rate.

Edit `ODDS_CURVE` and the self-test tells you immediately if you have moved either
target. Nothing else in the repo hardcodes a probability.

## Two invariants that are easy to break

**1. Odds live in `weight`, not in row counts.** A bottle's weight is
`target_prob * WEIGHT_SCALE / n`, where `n` is how many bottles shared its cell
when the weight was written. Add or remove a floor row without recomputing that
cell and its band's probability moves. The app deletes rows continuously
(`retireBottle()`), so `05_replenish_floor.py` recomputes **every** weight on
**every** run — including runs that replace nothing.

**2. A physical bottle is shown on the app once, ever.** Enforced by the sticky
`ever_placed` / `first_placed_at` columns, written to gold *and* bronze by
`04_sync_collection_from_unicorn.py` Part 2 and set immediately by `02`/`03`/`05`
as they place bottles. `app_status` is *not* sufficient on its own: it is
recomputed from live state each run, so a bottle that leaves the floor without
being ripped reads as `available` again.

Bronze carries the flags too, because `01_classify_and_enrich.py` rebuilds gold
with `mode("overwrite")` — flags that lived only in gold would be destroyed on
every enrichment run and the whole spent pool would look fresh.

## Scheduling

```
sync-rips-to-databricks (app hook)
  └─> 05_replenish_floor.py          daily
        └─> sync-*-weighted-from-databricks (app hook)
  └─> 06_inventory_reorder_alert.py  daily, after 05
```

`06` raises an exception when there is stock to buy, so the **job's** "on failure"
notification carries the buy list. Recipients go in the Databricks job definition,
never in source. A quiet day sends nothing.

To act on the buy list, re-run a generator with
`restock_from_reorder=true, write_mode=append` — it takes its demand straight from
`gold.reorder_recommendations`. Restock refuses to run in `overwrite` mode, since
that would delete the bronze rows holding the `ever_placed` ledger.

## Tier 5

`TIER_PRICE` prices a $250 tier 5, but `SELLABLE_TIERS` excludes it because the app
does not sell it (`TIER_PRICE_CENTS` in `celr/src/lib/payments.functions.ts` has
1/2/3/4 only). This is load-bearing, not cosmetic: placements are stored in wide
slot columns ordered *closest-to-par*, and the app-side sync reads exactly four of
them, so a fifth placement pushes a real sellable tier out of range and it is
dropped on sync. Four sellable tiers means at most four placements.

Re-enabling tier 5 means adding a quinary slot in four places at once —
`bourbons_weighted`/`agave_weighted`, `merge_weighted_catalog`,
`weighted_tier_catalog`, and the SELECT in `weighted-sync.server.ts`. The builders
assert this and the sync refuses to import a floor it cannot represent.

## Pull simulation + dashboard

`07_pull_simulation.py` plays the app for you: it draws `pulls_per_session` (100,
by default — one person's night) bottles per tier, `n_sessions` times (1,000 by
default, for statistical depth), and writes four tables:

| Table | Grain | Answers |
|---|---|---|
| `gold.pull_simulation_tier_kpi` | spirit, tier | Simulated win rate & payout vs. the config's own targets; avg/median/p90/worst losing streak; best/worst/median session net. |
| `gold.pull_simulation_streaks` | spirit, tier, losses-before-a-win | "How many pulls between wins" as a histogram — "3 losses then a win" is one row. |
| `gold.pull_simulation_sessions` | spirit, tier, session | One row per simulated 100-pull session: total spent, total won, net. |
| `gold.pull_simulation_pulls` | spirit, tier, pull # | Full pull-by-pull detail, but **only for the designated example session** — this is the concrete walkthrough, not all 1,000 sessions' worth of rows. |

Two draw modes, set by the `source` widget:
- `live_floor` (default) — draws from `app.bourbons_weighted`/`agave_weighted`
  exactly the way `draw_weighted_bottle()` does: today's real inventory, gaps and
  placeholder rows included (flagged via `is_placeholder`, not hidden).
- `theoretical` — samples `ODDS_CURVE` directly, independent of current stock. Use
  this to demo the *designed* game, or as a check when the floor is thin: on this
  setting the simulated win rate/payout should converge to `00`'s targets as
  `n_sessions` grows, run after run, with the same `random_seed`.

### Dashboard

`dashboards/pull_simulation.lvdash.json` is a ready-to-import Lakeview dashboard
over those four tables — two pages, "Overview" (KPI table, win-rate-vs-target bar,
streak-distribution bar, session-net variance) and "Walkthrough" (one table + one
running-total line chart per tier, pull by pull, for a single bourbon session —
literally "3 losses, then a $62 win on a $50 pull").

**Import it:** Databricks UI → **Dashboards** → **Create dashboard** → **︙** menu →
**Import dashboard file** → pick `pull_simulation.lvdash.json` → set its SQL
warehouse → open it. Re-running `07` with `dry_run=false` refreshes the tables in
place; the dashboard's cache/schedule picks up the new numbers without touching
the dashboard file again.

If you ever need to change the tier set (`SELLABLE_TIERS`/`TIER_PRICE` in `00`),
regenerate the dashboard rather than hand-editing the JSON:
`python3 databricks/dashboards/build_pull_simulation_dashboard.py` — it rebuilds
the per-tier walkthrough page to match whatever `00` currently says, then
re-import the file.

**A thing to know about the Lakeview JSON format**, confirmed by round-tripping a
dashboard through a live workspace: the `queryLines` array is concatenated by the
server with **no separator** between elements — `["...computed_at", "FROM t"]`
becomes the single invalid token `computed_atFROM t`. The generator script joins
each dataset's SQL with explicit `\n` before writing a single array element, which
the server accepts and later re-splits back into one element per line on its own.
Don't hand-edit `queryLines` in the generated file as a list of logical lines
without newlines — it will silently produce broken SQL that only fails when the
dashboard tries to run the query.

**Not deployed from here.** None of the Databricks CLI profiles on this machine
reach `prod_celr` — the one that authenticates is a different workspace (catalogs
`devramp`/`samples`/`system`, not `prod_celr`), and the other profiles need an
interactive `databricks auth login` this session can't do non-interactively. Run
`07` in the real workspace yourself (or re-auth a profile here — e.g.
`! databricks auth login --profile <name>` — and I can run it and import the
dashboard for you).
