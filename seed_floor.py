"""
seed_floor.py  --  initial bulk upload of the CELR floor.

Builds BOTTLES_PER_TIER bottles per tier in the live `bourbons` table, split
across the six bands so a uniform-random pull yields 37/39/20/3/0.8/0.2.

Odds live in floor COMPOSITION, so the app's pull is:
    select * from bourbons where tier = :t order by random() limit 1;

If any tier/band lacks enough reserve stock to hit its target count, nothing is
uploaded and the script reports exactly how many more bottles each cell needs.

Reserve pool assumed: table `bourbon_reserve`, one row per physical bottle,
columns incl. name, distillery, description, rarity, retail_value, image_url,
tier, status ('available' | 'on_floor'). Adjust names in CONFIG.
"""
import os, sys, math, random
from collections import defaultdict
from supabase import create_client

# ============ ODDS CONFIG (keep in sync with 00_celr_odds_config) ============
PULL_PRICE = {1: 50, 2: 100, 3: 500, 4: 1000}
# (low_mult, high_mult, target_prob); outer bands open-ended
BANDS = [
    (0.00, 0.75, 0.370),
    (0.75, 1.00, 0.390),
    (1.00, 1.50, 0.200),
    (1.50, 3.00, 0.030),
    (3.00, 6.00, 0.008),
    (6.00, float("inf"), 0.002),
]

def band_of(retail_value, tier):
    m = retail_value / PULL_PRICE[tier]
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= m < hi:
            return i
    return None  # below 0.0x is impossible; open top band catches the rest

def band_label(tier, i):
    lo, hi, _ = BANDS[i]
    p = PULL_PRICE[tier]
    hi_s = "inf" if hi == float("inf") else f"${hi*p:,.0f}"
    return f"band{i+1} ({lo:g}-{'inf' if hi==float('inf') else f'{hi:g}'}x, ${lo*p:,.0f}-{hi_s})"
# =============================================================================

# ==================================== CONFIG =================================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]   # service role, bypasses RLS
RESERVE_TABLE = "bourbon_reserve"
FLOOR_TABLE = "bourbons"
BOTTLES_PER_TIER = 500
INSERT_BATCH = 500
FLOOR_COLS = ["name", "distillery", "description", "rarity",
              "retail_value", "image_url", "tier"]
# =============================================================================


def target_counts(n):
    """Largest-remainder rounding so band counts always sum to n."""
    raw = [(i, n * p) for i, (_, _, p) in enumerate(BANDS)]
    floors = {i: math.floor(x) for i, x in raw}
    leftover = n - sum(floors.values())
    for i, _ in sorted(raw, key=lambda t: t[1] - math.floor(t[1]), reverse=True)[:leftover]:
        floors[i] += 1
    return floors


def load_reserve(client):
    """All available reserve bottles, grouped by (tier, band)."""
    grouped = defaultdict(list)
    start = 0
    while True:
        rows = (client.table(RESERVE_TABLE)
                .select("*").eq("status", "available")
                .range(start, start + 999).execute().data)
        if not rows:
            break
        for r in rows:
            t = int(r["tier"])
            b = band_of(float(r["retail_value"]), t)
            if b is not None:
                grouped[(t, b)].append(r)
        if len(rows) < 1000:
            break
        start += 1000
    return grouped


def plan(grouped):
    """Return (selection, shortfalls). selection: (t,b) -> list of reserve rows."""
    selection, shortfalls = {}, []
    for t in sorted(PULL_PRICE):
        want = target_counts(BOTTLES_PER_TIER)
        for b in range(len(BANDS)):
            avail = grouped.get((t, b), [])
            need = want[b]
            if len(avail) < need:
                shortfalls.append((t, b, need, len(avail), need - len(avail)))
            else:
                selection[(t, b)] = random.sample(avail, need)
    return selection, shortfalls


def main():
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    grouped = load_reserve(client)
    selection, shortfalls = plan(grouped)

    if shortfalls:
        print("SEED ABORTED. Not enough reserve stock to build the floor.\n")
        total = 0
        for t, b, need, have, short in shortfalls:
            total += short
            print(f"  Tier {t} {band_label(t, b)}: need {need}, have {have}, "
                  f"SHORT {short}")
        print(f"\nAdd {total} more bottles total, distributed as above, then rerun.")
        sys.exit(1)

    # build and insert floor rows
    floor_rows, used_ids = [], []
    for (t, b), rows in selection.items():
        for r in rows:
            floor_rows.append({c: r.get(c) for c in FLOOR_COLS})
            used_ids.append(r["id"])

    for i in range(0, len(floor_rows), INSERT_BATCH):
        client.table(FLOOR_TABLE).insert(floor_rows[i:i + INSERT_BATCH]).execute()
    for i in range(0, len(used_ids), INSERT_BATCH):
        (client.table(RESERVE_TABLE).update({"status": "on_floor"})
         .in_("id", used_ids[i:i + INSERT_BATCH]).execute())

    print(f"Uploaded {len(floor_rows)} bottles "
          f"({BOTTLES_PER_TIER} x {len(PULL_PRICE)} tiers) to {FLOOR_TABLE}.")


if __name__ == "__main__":
    main()
