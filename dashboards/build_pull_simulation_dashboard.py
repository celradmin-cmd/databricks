"""Generates pull_simulation.lvdash.json from the four tables `07_pull_simulation.py`
writes. A script rather than a hand-edited JSON blob because the walkthrough page
repeats the same table+chart pair once per sellable tier — easier to keep that
in sync with TIER_PRICE by generating it than by hand-editing four copies.

Run this (plain `python3 build_pull_simulation_dashboard.py`, not a Databricks
notebook — it has no Spark/dbutils dependency) whenever TIER_PRICE or SELLABLE_TIERS
in `00_celr_odds_config.py` changes, then re-import the regenerated .lvdash.json.

Validated shape: the dataset/page/widget JSON this script emits was round-tripped
through a live workspace's `databricks lakeview create` / `get` (counter, bar,
line and table widget specs each came back byte-for-byte identical) before this
generator was written. What could NOT be verified here is live data from
prod_celr, since this machine's Databricks CLI profiles don't reach that
workspace — import it and open the dashboard once to confirm the queries run.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# Reuse the real tier/price table rather than hardcoding it a second time here.
_cfg_src = (pathlib.Path(__file__).resolve().parent.parent / "00_celr_odds_config.py").read_text()
_cfg_ns: dict = {}
exec("\n".join(l for l in _cfg_src.split("\n") if not l.startswith(("# MAGIC", "# COMMAND"))), _cfg_ns)
TIER_PRICE = _cfg_ns["TIER_PRICE"]
SELLABLE_TIERS = sorted(_cfg_ns["SELLABLE_TIERS"])

CATALOG = "prod_celr"
GOLD = "gold"
PULLS_TBL = f"{CATALOG}.{GOLD}.pull_simulation_pulls"
SESSIONS_TBL = f"{CATALOG}.{GOLD}.pull_simulation_sessions"
STREAKS_TBL = f"{CATALOG}.{GOLD}.pull_simulation_streaks"
KPI_TBL = f"{CATALOG}.{GOLD}.pull_simulation_tier_kpi"

WALKTHROUGH_SPIRIT = "bourbon"  # the detailed per-tier page; see note at the bottom of the page


def dataset(name, display_name, sql_lines):
    # Lakeview's API does NOT join queryLines elements with a newline — it
    # concatenates them with nothing, so "...computed_at" + "FROM ..." becomes the
    # single invalid token "computed_atFROM ...". Confirmed by round-tripping a
    # multi-line dataset through a live workspace's `lakeview create`/`get`: the
    # server re-splits on embedded "\n" characters and re-wraps each resulting
    # line, so the newlines have to already be IN the text, not implied by the
    # list boundaries. Joining here and storing as one element is what the server
    # itself settles on after that round trip.
    return {"name": name, "displayName": display_name, "queryLines": ["\n".join(sql_lines)]}


def widget(name, title, widget_type, dataset_name, fields, encodings, query_name="main_query",
           disaggregated=True):
    return {
        "widget": {
            "name": name,
            "queries": [{
                "name": query_name,
                "query": {
                    "datasetName": dataset_name,
                    "fields": [{"name": f, "expression": f"`{f}`"} for f in fields],
                    "disaggregated": disaggregated,
                },
            }],
            "spec": {
                "version": 3,
                "widgetType": widget_type,
                "encodings": encodings,
                "frame": {"title": title, "showTitle": True},
            },
        },
    }


def counter(name, title, dataset_name, field, fmt=None):
    enc = {"value": {"fieldName": field, "displayName": title}}
    if fmt:
        enc["value"]["format"] = fmt
    w = widget(name, title, "counter", dataset_name, [field], enc, disaggregated=False)
    return w


def laid_out(w, x, y, width, height):
    return {**w, "position": {"x": x, "y": y, "width": width, "height": height}}


PCT_FMT = {"type": "number-percent", "decimalPlaces": {"type": "max", "places": 1}}
USD_FMT = {"type": "number-currency", "currencyCode": "USD", "decimalPlaces": {"type": "max", "places": 0}}
MULT_FMT = {"type": "number-plain", "decimalPlaces": {"type": "max", "places": 3}}

# ============================================================================
# Datasets
# ============================================================================

ds_kpi = dataset("ds_kpi", "Tier KPIs", [
    "SELECT spirit, tier, pull_price, source, n_sessions, total_pulls,",
    "       total_spent, total_retail_won, net_profit_loss, avg_net_dollars_per_pull,",
    "       simulated_win_rate, target_win_rate,",
    "       simulated_payout_multiple, target_payout_multiple,",
    "       avg_losses_before_win, median_losses_before_win, p90_losses_before_win,",
    "       max_losses_before_win, median_session_net, worst_session_net, best_session_net,",
    "       computed_at",
    f"FROM {KPI_TBL}",
    "ORDER BY spirit, tier",
])

ds_kpi_totals = dataset("ds_kpi_totals", "Run totals", [
    "SELECT SUM(total_pulls) AS total_pulls_simulated, SUM(n_sessions) AS total_sessions,",
    "       MAX(computed_at) AS last_computed, MAX(source) AS source",
    f"FROM {KPI_TBL}",
])

ds_winrate_long = dataset("ds_winrate_long", "Win rate — simulated vs. target (long form)", [
    "SELECT concat(spirit, ' t', tier) AS series, 'Simulated' AS metric, simulated_win_rate AS value",
    f"FROM {KPI_TBL}",
    "UNION ALL",
    "SELECT concat(spirit, ' t', tier), 'Target (00_celr_odds_config)', target_win_rate",
    f"FROM {KPI_TBL}",
    "ORDER BY 1, 2",
])

ds_streaks = dataset("ds_streaks", "Losses-before-a-win distribution", [
    "SELECT concat(spirit, ' t', tier) AS series, losses_before_win, losses_before_win_label,",
    "       occurrences, pct_of_wins",
    f"FROM {STREAKS_TBL}",
    "ORDER BY spirit, tier, losses_before_win",
])

ds_session_net = dataset("ds_session_net", "Net profit/loss per session (variance)", [
    "SELECT concat(spirit, ' t', tier) AS series, net_profit_loss",
    f"FROM {SESSIONS_TBL}",
])

datasets = [ds_kpi, ds_kpi_totals, ds_winrate_long, ds_streaks, ds_session_net]

# Per-tier walkthrough datasets (bourbon only — see page note).
walkthrough_datasets = {}
for tier in SELLABLE_TIERS:
    name = f"ds_walk_t{tier}"
    walkthrough_datasets[tier] = dataset(
        name, f"Tier {tier} (${TIER_PRICE[tier]} pull) — example session, pull by pull", [
            "SELECT pull_number, bottle_label, retail_value, pull_price, net_dollars,",
            "       is_win, is_placeholder, cumulative_net, losses_before_this_pull",
            f"FROM {PULLS_TBL}",
            f"WHERE spirit = '{WALKTHROUGH_SPIRIT}' AND tier = {tier}",
            "ORDER BY pull_number",
        ])
    datasets.append(walkthrough_datasets[tier])

# ============================================================================
# Page 1 — Overview
# ============================================================================

overview_layout = [
    laid_out(counter("c_total_pulls", "Total pulls simulated", "ds_kpi_totals", "total_pulls_simulated"),
             0, 0, 2, 3),
    laid_out(counter("c_total_sessions", "Total sessions simulated", "ds_kpi_totals", "total_sessions"),
             2, 0, 2, 3),
    laid_out(counter("c_last_run", "Simulation last run", "ds_kpi_totals", "last_computed",
                      fmt={"type": "date-time"}),
             4, 0, 2, 3),

    laid_out(widget(
        "t_kpi", "Win / lose by tier — the core table", "table", "ds_kpi",
        ["spirit", "tier", "pull_price", "n_sessions", "total_pulls",
         "simulated_win_rate", "target_win_rate",
         "simulated_payout_multiple", "target_payout_multiple",
         "avg_losses_before_win", "median_losses_before_win", "p90_losses_before_win",
         "max_losses_before_win", "avg_net_dollars_per_pull",
         "median_session_net", "worst_session_net", "best_session_net",
         "net_profit_loss", "total_spent", "total_retail_won"],
        {"columns": [
            {"fieldName": "spirit", "displayName": "Spirit"},
            {"fieldName": "tier", "displayName": "Tier"},
            {"fieldName": "pull_price", "displayName": "Pull price", "type": "number-currency",
             "format": USD_FMT},
            {"fieldName": "n_sessions", "displayName": "Sessions sim."},
            {"fieldName": "total_pulls", "displayName": "Total pulls"},
            {"fieldName": "simulated_win_rate", "displayName": "Win rate (sim)", "format": PCT_FMT},
            {"fieldName": "target_win_rate", "displayName": "Win rate (target)", "format": PCT_FMT},
            {"fieldName": "simulated_payout_multiple", "displayName": "Payout x (sim)", "format": MULT_FMT},
            {"fieldName": "target_payout_multiple", "displayName": "Payout x (target)", "format": MULT_FMT},
            {"fieldName": "avg_losses_before_win", "displayName": "Avg losses before a win"},
            {"fieldName": "median_losses_before_win", "displayName": "Median losses before a win"},
            {"fieldName": "p90_losses_before_win", "displayName": "P90 losses before a win"},
            {"fieldName": "max_losses_before_win", "displayName": "Worst losing streak"},
            {"fieldName": "avg_net_dollars_per_pull", "displayName": "Avg $ net / pull", "format": USD_FMT},
            {"fieldName": "median_session_net", "displayName": "Median session net ($)", "format": USD_FMT},
            {"fieldName": "worst_session_net", "displayName": "Worst 100-pull session ($)", "format": USD_FMT},
            {"fieldName": "best_session_net", "displayName": "Best 100-pull session ($)", "format": USD_FMT},
            {"fieldName": "net_profit_loss", "displayName": "Net across all sessions ($)", "format": USD_FMT},
            {"fieldName": "total_spent", "displayName": "Total spent ($)", "format": USD_FMT},
            {"fieldName": "total_retail_won", "displayName": "Total retail won ($)", "format": USD_FMT},
        ]},
    ), 0, 3, 12, 6),

    laid_out(widget(
        "b_winrate", "Win rate: simulated vs. the configured target", "bar", "ds_winrate_long",
        ["series", "metric", "value"],
        {
            "x": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
            "y": {"fieldName": "value", "scale": {"type": "quantitative"}, "displayName": "Win rate",
                  "format": PCT_FMT},
            "color": {"fieldName": "metric", "scale": {"type": "categorical"}, "displayName": "Metric"},
        },
    ), 0, 9, 6, 6),

    laid_out(widget(
        "b_session_net", "Net profit/loss variance across simulated sessions", "bar", "ds_session_net",
        ["series", "net_profit_loss"],
        {
            "x": {"fieldName": "net_profit_loss", "scale": {"type": "quantitative"}, "displayName": "Session net $"},
            "y": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
        },
    ), 6, 9, 6, 6),

    laid_out(widget(
        "b_streaks", "How many losses before a win, per tier", "bar", "ds_streaks",
        ["series", "losses_before_win_label", "occurrences"],
        {
            "x": {"fieldName": "losses_before_win_label", "scale": {"type": "categorical"},
                  "displayName": "Losses before the win"},
            "y": {"fieldName": "occurrences", "scale": {"type": "quantitative"}, "displayName": "Occurrences"},
            "color": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
        },
    ), 0, 15, 12, 7),
]

page_overview = {
    "name": "page_overview",
    "displayName": "Overview — win/lose by tier",
    "pageType": "PAGE_TYPE_CANVAS",
    "layoutVersion": "GRID_V1",
    "layout": overview_layout,
}

# ============================================================================
# Page 2 — Walkthrough: one concrete 100-pull session per tier
# ============================================================================

walkthrough_layout = []
row_height = 7
for i, tier in enumerate(SELLABLE_TIERS):
    y = i * row_height
    ds_name = walkthrough_datasets[tier]["name"]
    walkthrough_layout.append(laid_out(widget(
        f"t_walk_t{tier}", f"Tier {tier} (${TIER_PRICE[tier]} pull) — pull by pull", "table", ds_name,
        ["pull_number", "bottle_label", "retail_value", "pull_price", "net_dollars",
         "is_win", "is_placeholder", "cumulative_net", "losses_before_this_pull"],
        {"columns": [
            {"fieldName": "pull_number", "displayName": "Pull #"},
            {"fieldName": "bottle_label", "displayName": "Bottle"},
            {"fieldName": "retail_value", "displayName": "Retail value", "format": USD_FMT},
            {"fieldName": "pull_price", "displayName": "Paid", "format": USD_FMT},
            {"fieldName": "net_dollars", "displayName": "Net $", "format": USD_FMT},
            {"fieldName": "is_win", "displayName": "Won?"},
            {"fieldName": "is_placeholder", "displayName": "Gap placeholder?"},
            {"fieldName": "cumulative_net", "displayName": "Running total $", "format": USD_FMT},
            {"fieldName": "losses_before_this_pull", "displayName": "Losses right before this win"},
        ]},
    ), 0, y, 6, row_height))
    walkthrough_layout.append(laid_out(widget(
        f"l_walk_t{tier}", f"Tier {tier} — running total across the session", "line", ds_name,
        ["pull_number", "cumulative_net"],
        {
            "x": {"fieldName": "pull_number", "scale": {"type": "quantitative"}, "displayName": "Pull #"},
            "y": {"fieldName": "cumulative_net", "scale": {"type": "quantitative"},
                  "displayName": "Running total $", "format": USD_FMT},
        },
    ), 6, y, 6, row_height))

page_walkthrough = {
    "name": "page_walkthrough",
    "displayName": f"Walkthrough — one {WALKTHROUGH_SPIRIT} session",
    "pageType": "PAGE_TYPE_CANVAS",
    "layoutVersion": "GRID_V1",
    "layout": walkthrough_layout,
}

# ============================================================================
# Assemble
# ============================================================================

dashboard = {
    "datasets": datasets,
    "pages": [page_overview, page_walkthrough],
    "uiSettings": {"theme": {"widgetHeaderAlignment": "ALIGNMENT_UNSPECIFIED"}, "applyModeEnabled": False},
}

out_path = pathlib.Path(__file__).resolve().parent / "pull_simulation.lvdash.json"
out_path.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"Wrote {out_path} "
      f"({len(datasets)} datasets, {len(overview_layout)} overview widgets, "
      f"{len(walkthrough_layout)} walkthrough widgets across {len(SELLABLE_TIERS)} tiers)")
