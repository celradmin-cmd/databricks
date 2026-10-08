"""Generates pull_simulation.lvdash.json from the four tables `06_pull_simulation.py`
writes. A script rather than a hand-edited JSON blob because the walkthrough page
repeats the same table+chart pair once per sellable tier — easier to keep that
in sync with TIER_PRICE by generating it than by hand-editing four copies.

Run this (plain `python3 build_pull_simulation_dashboard.py`, or as a Databricks
notebook from its folder in the repo — it has no Spark/dbutils dependency) whenever TIER_PRICE or SELLABLE_TIERS
in `00_celr_odds_config.py` changes, then re-import the regenerated .lvdash.json.

Spec versions are per widget type, and getting one wrong costs you the widget:
the import keeps the dashboard but replaces the offending widget with "Invalid
widget definition is imported." The first import of this file lost all three
counters (emitted at v3 — counters are v2) and every table (tables are v1,
with long-form columns; see table_column). See SPEC_VERSION.

This machine's Databricks CLI profiles don't reach prod_celr, so neither the
widget schema nor the queries can be checked from here — import the file and
look at the canvas: every widget rendering, with data, is the only real
confirmation.
"""
import json
import pathlib
import sys

# A Databricks notebook has no __file__, but its working directory is the
# notebook's own folder, so cwd stands in for it there.
HERE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(HERE.parent))
# Reuse the real tier/price table rather than hardcoding it a second time here.
_cfg_src = (HERE.parent / "00_celr_odds_config.py").read_text()
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


# Lakeview validates a widget's `encodings` against the schema for its spec
# version, and the versions are per widget *type*, not one number for the whole
# dashboard. Charts (bar/line) are v3, counters v2, tables v1 — a table at v3 is
# rejected with "spec/version must be equal to constant", a counter at v3 is
# likewise dropped with "Invalid widget definition is imported."
SPEC_VERSION = {"counter": 2, "table": 1}
DEFAULT_SPEC_VERSION = 3


def widget(name, title, widget_type, dataset_name, fields, encodings, query_name="main_query",
           disaggregated=True, mark=None):
    spec_extra = {"mark": mark} if mark else {}
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
                "version": SPEC_VERSION.get(widget_type, DEFAULT_SPEC_VERSION),
                "widgetType": widget_type,
                "encodings": encodings,
                "frame": {"title": title, "showTitle": True},
                **spec_extra,
            },
        },
    }


def counter(name, title, dataset_name, field, fmt=None):
    enc = {"value": {"fieldName": field, "displayName": title}}
    if fmt:
        enc["value"]["format"] = fmt
    w = widget(name, title, "counter", dataset_name, [field], enc, disaggregated=False)
    return w


# A v1 table column is the long-form object the table editor itself exports;
# formatting goes in `numberFormat` (a numeral.js pattern), not a v3-style
# `format` object, which the import rejects with 'unknown property "type"'.
COL_KINDS = {
    "string": ("string", "string", None),
    "integer": ("integer", "number", "0,0"),
    "float": ("float", "number", "0,0.0"),
    "usd": ("float", "number", "$0,0"),
    # The table's "%" pattern only appends the sign — it does NOT multiply by 100
    # the way numeral.js does — so a pct column's dataset must already be 0-100.
    "pct": ("float", "number", "0.0%"),
    "mult": ("float", "number", "0.000"),
    "boolean": ("boolean", "boolean", None),
}


def table_column(field, title, kind, order):
    col_type, display_as, number_format = COL_KINDS[kind]
    col = {
        "fieldName": field,
        "booleanValues": ["false", "true"],
        "imageUrlTemplate": "{{ @ }}",
        "imageTitleTemplate": "{{ @ }}",
        "imageWidth": "",
        "imageHeight": "",
        "linkUrlTemplate": "{{ @ }}",
        "linkTextTemplate": "{{ @ }}",
        "linkTitleTemplate": "{{ @ }}",
        "linkOpenInNewTab": True,
        "type": col_type,
        "displayAs": display_as,
        "visible": True,
        "order": 100000 + order,
        "title": title,
        "allowSearch": False,
        "alignContent": "left" if display_as == "string" else "right",
        "allowHTML": False,
        "highlightLinks": False,
        "useMonospaceFont": False,
        "preserveWhitespace": False,
        "displayName": title,
    }
    if number_format:
        col["numberFormat"] = number_format
    return col


def table(name, title, dataset_name, columns):
    """columns: [(field, title, kind)] with kind a COL_KINDS key."""
    w = widget(name, title, "table", dataset_name, [c[0] for c in columns],
               {"columns": [table_column(f, t, k, i) for i, (f, t, k) in enumerate(columns)]})
    w["widget"]["spec"].update({
        "invisibleColumns": [], "allowHTMLByDefault": False, "itemsPerPage": 25,
        "paginationSize": "default", "condensed": True, "withRowNumber": False,
    })
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
    "       100 * simulated_win_rate AS simulated_win_rate,",
    "       100 * target_win_rate AS target_win_rate,  -- 0-100 for t_kpi's pct columns",
    "       simulated_payout_multiple, target_payout_multiple,",
    "       avg_losses_before_win, median_losses_before_win, p90_losses_before_win,",
    "       max_losses_before_win, median_session_net, worst_session_net, best_session_net,",
    "       computed_at",
    f"FROM {KPI_TBL}",
    "ORDER BY spirit, tier",
])

ds_kpi_totals = dataset("ds_kpi_totals", "Run totals", [
    "SELECT SUM(total_pulls) AS total_pulls_simulated, SUM(n_sessions) AS total_sessions,",
    "       MAX(computed_at) AS last_computed, MAX(source) AS source, MAX(seed) AS seed",
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
            # CELR's side, like the Profits page: the pulls table stores the
            # player's net (retail - price), so both dollar columns are negated.
            "SELECT pull_number, bottle_label, retail_value, pull_price,",
            "       -net_dollars AS celr_profit, is_win, is_placeholder,",
            "       -cumulative_net AS celr_running_total, losses_before_this_pull",
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
    # Changes every run unless 06 was given a fixed random_seed — if this and the
    # numbers stay put across runs, the seed widget is pinned.
    laid_out(counter("c_seed", "Random seed (replay with this)", "ds_kpi_totals", "seed"),
             6, 0, 2, 3),

    laid_out(table("t_kpi", "Win / lose by tier — the core table", "ds_kpi", [
        ("spirit", "Spirit", "string"),
        ("tier", "Tier", "integer"),
        ("pull_price", "Pull price", "usd"),
        ("n_sessions", "Sessions sim.", "integer"),
        ("total_pulls", "Total pulls", "integer"),
        ("simulated_win_rate", "Win rate (sim)", "pct"),
        ("target_win_rate", "Win rate (target)", "pct"),
        ("simulated_payout_multiple", "Payout x (sim)", "mult"),
        ("target_payout_multiple", "Payout x (target)", "mult"),
        ("avg_losses_before_win", "Avg losses before a win", "float"),
        ("median_losses_before_win", "Median losses before a win", "float"),
        ("p90_losses_before_win", "P90 losses before a win", "float"),
        ("max_losses_before_win", "Worst losing streak", "integer"),
        ("avg_net_dollars_per_pull", "Player avg net / pull ($)", "usd"),
        ("median_session_net", "Player median session net ($)", "usd"),
        ("worst_session_net", "Player worst 100-pull session ($)", "usd"),
        ("best_session_net", "Player best 100-pull session ($)", "usd"),
        ("net_profit_loss", "Player net, all sessions ($)", "usd"),
        ("total_spent", "Total spent ($)", "usd"),
        ("total_retail_won", "Total retail won ($)", "usd"),
    ]), 0, 3, 12, 6),

    laid_out(widget(
        "b_winrate", "Win rate: simulated vs. the configured target", "bar", "ds_winrate_long",
        ["series", "metric", "value"],
        {
            "x": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
            "y": {"fieldName": "value", "scale": {"type": "quantitative"}, "displayName": "Win rate",
                  "format": PCT_FMT},
            "color": {"fieldName": "metric", "scale": {"type": "categorical"}, "displayName": "Metric"},
        },
        mark={"layout": "group"},  # side by side; the default stacks sim on top of target
    ), 0, 9, 6, 6),

    laid_out(widget(
        "b_session_net", "Player net profit/loss across simulated sessions", "bar", "ds_session_net",
        ["series", "net_profit_loss"],
        {
            "x": {"fieldName": "net_profit_loss", "scale": {"type": "quantitative"}, "displayName": "Player session net $"},
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
    walkthrough_layout.append(laid_out(table(
        f"t_walk_t{tier}", f"Tier {tier} (${TIER_PRICE[tier]} pull) — pull by pull", ds_name, [
            ("pull_number", "Pull #", "integer"),
            ("bottle_label", "Bottle", "string"),
            ("retail_value", "Retail value", "usd"),
            ("pull_price", "Paid", "usd"),
            ("celr_profit", "CELR profit $", "usd"),
            ("is_win", "Player won?", "boolean"),
            ("is_placeholder", "Gap placeholder?", "boolean"),
            ("celr_running_total", "CELR running total $", "usd"),
            ("losses_before_this_pull", "Losses right before this win", "integer"),
        ]), 0, y, 6, row_height))
    walkthrough_layout.append(laid_out(widget(
        f"l_walk_t{tier}", f"Tier {tier} — CELR running profit across the session", "line", ds_name,
        ["pull_number", "celr_running_total"],
        {
            "x": {"fieldName": "pull_number", "scale": {"type": "quantitative"}, "displayName": "Pull #"},
            "y": {"fieldName": "celr_running_total", "scale": {"type": "quantitative"},
                  "displayName": "CELR running total $", "format": USD_FMT},
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
# Page 3 — Profits: the same simulation from the house's side of the counter
# ============================================================================
# Everything above is the player's view (net = retail won - spent). Here it's
# flipped: revenue is what players paid for pulls, "paid out" is the catalog
# retail_value (app.*_weighted, i.e. gold catalog) of every bottle those pulls
# landed on, and house profit is the difference. That's exact for the app's two
# outcomes: ~90-95% of bottles are bought back at 100% of retail as wallet credit
# the player can withdraw as cash (the bottle stays in inventory), and the
# ~5-10% shipped/vaulted cost the house the bottle itself, valued here at retail. Payout x = retail given
# out / revenue — the same number 00_celr_odds_config designs to (target ~0.89x);
# anything at or above 1.0x means the house gives away more retail than it takes in.

ds_profit_tier = dataset("ds_profit_tier", "House profit by tier", [
    "SELECT k.spirit, k.tier, concat(k.spirit, ' t', k.tier) AS series, k.pull_price,",
    "       k.total_pulls, k.total_spent AS revenue, k.total_retail_won AS retail_given_out,",
    "       k.total_retail_won / k.total_pulls AS avg_bottle_retail,",
    "       k.simulated_payout_multiple AS payout_multiple,",
    "       k.target_payout_multiple,",
    "       k.total_spent - k.total_retail_won AS house_profit,",
    "       (k.total_spent - k.total_retail_won) / k.total_spent AS house_margin,",
    "       100 * (k.total_spent - k.total_retail_won) / k.total_spent AS house_margin_pct,",
    "       (k.total_spent - k.total_retail_won) / k.total_pulls AS house_profit_per_pull,",
    "       -k.median_session_net AS median_session_profit,",
    "       -k.best_session_net AS worst_session_profit,",
    "       s.pct_sessions_house_lost",
    f"FROM {KPI_TBL} k",
    "LEFT JOIN (",
    "  SELECT spirit, tier,",
    "         100 * AVG(CASE WHEN net_profit_loss > 0 THEN 1.0 ELSE 0.0 END) AS pct_sessions_house_lost",
    f"  FROM {SESSIONS_TBL} GROUP BY spirit, tier",
    ") s ON s.spirit = k.spirit AND s.tier = k.tier",
    "ORDER BY k.spirit, k.tier",
])

# Counters stay unformatted (only the v2 counters without a `format` are known
# to import cleanly), so the rounding is done here.
ds_profit_totals = dataset("ds_profit_totals", "House profit totals", [
    "SELECT ROUND(SUM(total_spent), 0) AS total_revenue,",
    "       ROUND(SUM(total_spent - total_retail_won), 0) AS total_house_profit,",
    "       ROUND(100 * SUM(total_spent - total_retail_won) / SUM(total_spent), 1) AS house_margin_pct",
    f"FROM {KPI_TBL}",
])

ds_profit_long = dataset("ds_profit_long", "Revenue vs. retail given out (long form)", [
    "SELECT concat(spirit, ' t', tier) AS series, 'Revenue (pulls sold)' AS metric, total_spent AS dollars",
    f"FROM {KPI_TBL}",
    "UNION ALL",
    "SELECT concat(spirit, ' t', tier), 'Paid out (buybacks + shipped bottles)', total_retail_won",
    f"FROM {KPI_TBL}",
    "ORDER BY 1, 2",
])

ds_session_profit = dataset("ds_session_profit", "House profit per session", [
    "SELECT concat(spirit, ' t', tier) AS series, -net_profit_loss AS house_profit",
    f"FROM {SESSIONS_TBL}",
])

datasets += [ds_profit_tier, ds_profit_totals, ds_profit_long, ds_session_profit]

profits_layout = [
    laid_out(counter("c_revenue", "Total pull revenue ($)", "ds_profit_totals", "total_revenue"),
             0, 0, 2, 3),
    laid_out(counter("c_house_profit", "Total house profit ($)", "ds_profit_totals", "total_house_profit"),
             2, 0, 2, 3),
    laid_out(counter("c_house_margin", "House margin (%)", "ds_profit_totals", "house_margin_pct"),
             4, 0, 2, 3),

    laid_out(table("t_profit", "House profit by tier", "ds_profit_tier", [
        ("spirit", "Spirit", "string"),
        ("tier", "Tier", "integer"),
        ("pull_price", "Pull price", "usd"),
        ("total_pulls", "Pulls sold", "integer"),
        ("revenue", "Revenue: pulls sold ($)", "usd"),
        ("retail_given_out", "Paid out: buybacks + shipped, at retail ($)", "usd"),
        ("avg_bottle_retail", "Avg bottle retail / pull ($)", "usd"),
        ("payout_multiple", "Payout x (sim)", "mult"),
        ("target_payout_multiple", "Payout x (target)", "mult"),
        ("house_profit", "House profit ($)", "usd"),
        ("house_margin_pct", "Margin", "pct"),
        ("house_profit_per_pull", "Profit / pull ($)", "usd"),
        ("median_session_profit", "Median session profit ($)", "usd"),
        ("worst_session_profit", "Worst session for the house ($)", "usd"),
        ("pct_sessions_house_lost", "Sessions the house lost", "pct"),
    ]), 0, 3, 12, 6),

    laid_out(widget(
        "b_profit_per_pull", "House profit per pull, by tier", "bar", "ds_profit_tier",
        ["series", "house_profit_per_pull"],
        {
            "x": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
            "y": {"fieldName": "house_profit_per_pull", "scale": {"type": "quantitative"},
                  "displayName": "Profit / pull", "format": USD_FMT},
        },
    ), 0, 9, 6, 6),

    laid_out(widget(
        "b_margin", "House margin, by tier", "bar", "ds_profit_tier",
        ["series", "house_margin"],
        {
            "x": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
            "y": {"fieldName": "house_margin", "scale": {"type": "quantitative"},
                  "displayName": "Margin", "format": PCT_FMT},
        },
    ), 6, 9, 6, 6),

    laid_out(widget(
        "b_rev_vs_cost", "Revenue vs. paid out (buybacks + shipped), by tier", "bar", "ds_profit_long",
        ["series", "metric", "dollars"],
        {
            "x": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
            "y": {"fieldName": "dollars", "scale": {"type": "quantitative"}, "displayName": "Dollars",
                  "format": USD_FMT},
            "color": {"fieldName": "metric", "scale": {"type": "categorical"}, "displayName": "Metric"},
        },
    ), 0, 15, 6, 7),

    laid_out(widget(
        "b_session_profit", "House profit variance across simulated sessions", "bar", "ds_session_profit",
        ["series", "house_profit"],
        {
            "x": {"fieldName": "house_profit", "scale": {"type": "quantitative"},
                  "displayName": "Session house profit $"},
            "y": {"fieldName": "series", "scale": {"type": "categorical"}, "displayName": "Spirit / tier"},
        },
    ), 6, 15, 6, 7),
]

page_profits = {
    "name": "page_profits",
    "displayName": "Profits — the house view",
    "pageType": "PAGE_TYPE_CANVAS",
    "layoutVersion": "GRID_V1",
    "layout": profits_layout,
}

# ============================================================================
# Assemble
# ============================================================================

dashboard = {
    "datasets": datasets,
    "pages": [page_overview, page_profits, page_walkthrough],
    "uiSettings": {"theme": {"widgetHeaderAlignment": "ALIGNMENT_UNSPECIFIED"}, "applyModeEnabled": False},
}

out_path = HERE / "pull_simulation.lvdash.json"
out_path.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"Wrote {out_path} "
      f"({len(datasets)} datasets, {len(overview_layout)} overview widgets, "
      f"{len(profits_layout)} profits widgets, "
      f"{len(walkthrough_layout)} walkthrough widgets across {len(SELLABLE_TIERS)} tiers)")
