# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · CËLR Odds Config (shared)
# MAGIC
# MAGIC The single source of truth for the **house odds curve** and tier/price mapping.
# MAGIC `%run` this notebook from every other notebook in this folder so they all place
# MAGIC bottles into the *same* bands and assign the *same* weights.
# MAGIC
# MAGIC ## The curve
# MAGIC A bottle's band is decided by `retail_value / tier_price` — its **multiple of
# MAGIC par**. Below 1.0x the player got back less retail than they paid: that is a
# MAGIC loss. At or above 1.0x it is a win.
# MAGIC
# MAGIC The curve is tuned to two targets, both asserted by the self-test at the bottom:
# MAGIC
# MAGIC 1. **Expected payout ~0.84x of pull price** — the house keeps ~16% of GMV
# MAGIC    from the curve alone (target range 10–20% of GMV). Sellback at 100% of
# MAGIC    retail inside 2 minutes passes this payout straight through to the player,
# MAGIC    so this number *is* the house edge for instant sellers.
# MAGIC 2. **Win rate 25%** — a win every 4 pulls, with 86% of those wins in band 2
# MAGIC    (1.00–1.25x), the band immediately above the loss line — and, inside that
# MAGIC    band, weighted toward its cheap end (see "Where a bottle sits inside its
# MAGIC    band"). A typical win on a $50 pull is a $50–$55 bottle: it reads as a win
# MAGIC    without costing the house much. Big wins (1.6x+) are 1% of pulls.
# MAGIC
# MAGIC ## Where a bottle sits inside its band
# MAGIC Band probabilities alone don't fix the payout: a band is a dollar *range*, and
# MAGIC if every bottle in it is drawn equally often, the band pays out at whatever the
# MAGIC stocked bottles happen to average. Floors stocked toward the top of their bands
# MAGIC ran ~1.0x payout against a 0.89x design (Oct 2026 simulation), because the old
# MAGIC design math assumed every band paid out at its midpoint and nothing checked.
# MAGIC
# MAGIC So `cell_weights()` sets each bottle's weight from its value, not just the
# MAGIC cell's bottle count: a cell's total weight is still exactly its band
# MAGIC probability, but inside the cell cheaper bottles are drawn more often, until the
# MAGIC cell's average multiple is down at `BAND_TARGET_POSITION` (20% of the way up a
# MAGIC win band, the middle of a loss band). Every builder prints the floor's real,
# MAGIC weight-and-value payout per tier (`floor_payout()`) next to this design target.
# MAGIC
# MAGIC ## Where the shape comes from
# MAGIC The probabilities follow a market-standard outcomes table (36/39/22/2/0.5/0.1
# MAGIC over the same multiples of par), which on its own runs ~7.9% house edge. Its
# MAGIC 1.00–1.60x band is split here at 1.25x (18.5% / 3.5%) so most wins land at
# MAGIC the cheap end of that range; that split is what lifts the edge to ~10.7%.
# MAGIC The source table sums to 99.6%; the missing 0.4% sits in the deep-loss band.
# MAGIC
# MAGIC ## History
# MAGIC Oct 2026: 36.4/39/18.5/3.5/2/0.5/0.1 (24.6% win rate, 0.8935x on midpoints)
# MAGIC simulated at ~1.0x on the live floor — players were winning big bottles too
# MAGIC often. Moved probability from the big-win bands into band 2 and added the
# MAGIC in-band value weighting above.
# MAGIC
# MAGIC Before that, a curve of 29.2/30.8/31.8/5/2/0.9/0.3 ran a 40% win rate at 0.9975x
# MAGIC payout: no house edge. A 30 x $1k session finished ahead 34% of the time; on
# MAGIC this curve it finishes ahead ~12% of the time, median net about -$4k.
# MAGIC
# MAGIC Everything below is the *only* place you tune odds. Change a number here and
# MAGIC every notebook follows. Re-run the self-test after any edit.

# COMMAND ----------

# Pull price for each tier (the "dollar amount you spent on the pull").
TIER_PRICE = {1: 50, 2: 100, 5: 250, 3: 500, 4: 1000}

# Tiers the app actually sells. ONLY these get placements on a floor.
#
# Tier 5 ($250) went live alongside tiers 1-4 — `celr/src/lib/tiers.ts`,
# `payments.functions.ts` and `bourbons.functions.ts` all sell it. That made the
# fifth (quinary) wide-slot column load-bearing rather than unused: a bottle
# eligible for all five tiers now genuinely needs all five placements carried
# through, in `bourbons_weighted`/`agave_weighted`, `merge_weighted_catalog`,
# `weighted_tier_catalog`, and the SELECT in `weighted-sync.server.ts`. If you
# ever need to pull a tier from sale again, remove it here first — the builders'
# `len(TIERS) <= APP_SYNC_SLOTS` assert is what would catch a slot mismatch if
# a 6th tier were added without widening the schema the same way.
SELLABLE_TIERS = (1, 2, 3, 4, 5)

# The house curve, as multiplier bands of the pull price.
# Each tuple: (lo_multiple_inclusive, hi_multiple_exclusive, target_probability).
# Bands must be contiguous, ascending, and sum to 1.0 — all asserted below.
ODDS_CURVE = [
    (0.50, 0.70, 0.3300),  # deep loss        -> $25–$35 on a $50 pull
    (0.70, 1.00, 0.4200),  # shallow loss     -> $35–$50
    # ---------------------------------------------------------------- win line
    (1.00, 1.25, 0.2150),  # the win          -> $50–$62   (86% of all wins)
    (1.25, 1.60, 0.0250),  # good win         -> $62–$80
    (1.60, 3.00, 0.0080),  # nice win         -> $80–$150
    (3.00, 8.00, 0.0015),  # big win          -> $150–$400
    (8.00, 16.00, 0.0005), # grail            -> $400–$800
]

# Where inside each band the cell's average bottle should sit, as a fraction of
# the way from the band's low edge to its high edge (0 = cheapest, 1 = priciest).
# cell_weights() tilts weight toward cheaper bottles until the cell averages here.
# Loss bands sit mid-band; win bands sit low, so a win is usually a bottle just
# over what the player paid rather than one near the top of the range.
BAND_TARGET_POSITION = [0.50, 0.50, 0.20, 0.20, 0.20, 0.20, 0.20]

# Weights are integers so the pull selector stays simple. SCALE = the total weight
# budget a single tier's curve is divided into. With a shared ~1,000-bottle pool, a
# single (tier, band) cell can hold hundreds of real bottles at once (a bottle's
# price can make it eligible for several tiers' cells simultaneously) —
# weight_for_band() splits the band's fixed budget (target_prob * WEIGHT_SCALE)
# evenly across all of them, and with too small a scale that per-bottle share rounds
# down to 0, silently zeroing the band's odds. 1,000,000 keeps a nonzero weight (>=1)
# up to ~thousands of bottles per cell even for the 0.3% grail band — comfortably
# above anything a 1,000–4,000-row floor will ever produce.
WEIGHT_SCALE = 1_000_000

# The lowest and highest multiple the curve covers. A bottle whose retail value falls
# outside [CURVE_LO * price, CURVE_HI * price] does NOT belong in that tier.
CURVE_LO = ODDS_CURVE[0][0]    # 0.50
CURVE_HI = ODDS_CURVE[-1][1]   # 16.0

# First band index at or above par — the boundary between "lost" and "won".
# Derived, not hardcoded, so re-cutting the bands can't leave it stale.
WIN_BAND_MIN = next(i for i, (lo, _hi, _p) in enumerate(ODDS_CURVE) if lo >= 1.0)

# COMMAND ----------

# MAGIC %md ### Curve integrity — these run on every `%run` of this notebook

# COMMAND ----------

_probs = [p for (_lo, _hi, p) in ODDS_CURVE]
assert abs(sum(_probs) - 1.0) < 1e-9, f"ODDS_CURVE probabilities sum to {sum(_probs)}, not 1.0"
assert all(p > 0 for p in _probs), "every band needs a positive probability"
for _i in range(len(ODDS_CURVE) - 1):
    assert ODDS_CURVE[_i][1] == ODDS_CURVE[_i + 1][0], (
        f"band {_i} ends at {ODDS_CURVE[_i][1]} but band {_i + 1} starts at "
        f"{ODDS_CURVE[_i + 1][0]} — bands must be contiguous or bottles fall through the gap"
    )
assert any(lo == 1.0 for (lo, _hi, _p) in ODDS_CURVE), (
    "a band must start exactly at 1.0x, or the win/loss line falls mid-band and "
    "win_rate() becomes meaningless"
)
assert set(SELLABLE_TIERS) <= set(TIER_PRICE), "SELLABLE_TIERS references a tier with no price"
assert len(BAND_TARGET_POSITION) == len(ODDS_CURVE), "one BAND_TARGET_POSITION per band"
assert all(0.0 <= f <= 1.0 for f in BAND_TARGET_POSITION), "BAND_TARGET_POSITION is a 0-1 fraction"

# COMMAND ----------

def band_index(retail_value, tier):
    """Return the 0-based band index for a bottle's retail value within a tier, or None
    if the bottle's value is outside the curve for that tier (too cheap or too pricey)."""
    price = TIER_PRICE.get(tier)
    if price is None or retail_value is None or price <= 0:
        return None
    m = float(retail_value) / float(price)
    for i, (lo, hi, _p) in enumerate(ODDS_CURVE):
        # inclusive low, exclusive high; final band is inclusive on the top edge
        if (lo <= m < hi) or (i == len(ODDS_CURVE) - 1 and m == hi):
            return i
    return None


def eligible_tiers(retail_value):
    """All SELLABLE tiers whose curve can host this retail value, with the multiplier
    for each. Returns list of (tier, multiple, band_index) sorted by closeness to par.

    Only sellable tiers are returned — see the SELLABLE_TIERS comment above. Because
    there are four of them, this list is never longer than the four wide slot columns
    downstream, so a placement can never be silently dropped."""
    out = []
    for tier in SELLABLE_TIERS:
        bi = band_index(retail_value, tier)
        if bi is not None:
            out.append((tier, float(retail_value) / TIER_PRICE[tier], bi))
    # closest-to-par first -> the bottle's most "natural" home tier
    out.sort(key=lambda t: abs(t[1] - 1.0))
    return out


def primary_tier(retail_value):
    """The single best home tier for a bottle: where it sits closest to par. None if it
    fits no sellable tier (retail < $25 fits nothing; retail > $16k fits nothing)."""
    et = eligible_tiers(retail_value)
    return et[0][0] if et else None


def band_label(tier, band_idx):
    """Human-readable value range for a band, e.g. '$50–$62'. Drives the UI breakdown
    and the reorder alert's 'buy bottles in this range' lines."""
    price = TIER_PRICE[tier]
    lo, hi, _ = ODDS_CURVE[band_idx]
    return f"${int(round(lo * price))}–${int(round(hi * price))}"


def band_mid_value(tier, band_idx):
    """Arithmetic midpoint retail value of a band, in dollars. Used as the 'what a
    bottle in this cell should be worth' target when sourcing replacements and when
    quoting a max buy price. An approximation: real bottles are not uniformly
    distributed inside a band, but for the narrow low bands the error is small and
    for the wide tail bands the cell needs only a handful of bottles anyway."""
    price = TIER_PRICE[tier]
    lo, hi, _ = ODDS_CURVE[band_idx]
    return (lo + hi) / 2.0 * price


def band_target_multiple(band_idx):
    """The multiple of par a cell in this band should average once weighted —
    BAND_TARGET_POSITION of the way up the band."""
    lo, hi, _p = ODDS_CURVE[band_idx]
    return lo + BAND_TARGET_POSITION[band_idx] * (hi - lo)


def band_target_value(tier, band_idx):
    """band_target_multiple in dollars for a tier: what the average bottle drawn
    from this cell should be worth. The sourcing target for the cell — buy bottles
    around here, not at the band midpoint."""
    return band_target_multiple(band_idx) * TIER_PRICE[tier]


def cell_weights(band_idx, multiples):
    """Integer weights for the bottles in ONE (tier, band) cell, given each
    bottle's multiple of par (retail_value / tier price), in the same order.

    The weights sum to the band's budget (target_prob * WEIGHT_SCALE) — so the
    band's probability is exactly what ODDS_CURVE says, the same guarantee
    weight_for_band() gives. What changes is the split inside the cell:
    exponential tilting, w_i ~ exp(-theta * position_i), with theta found by
    bisection so the cell's weighted-average multiple comes down to
    band_target_multiple(). Cheaper bottles get more weight, pricier ones less.

    If the cell already averages at or below the target, the split stays even
    (theta = 0) — it never tilts toward expensive bottles. If every bottle sits
    above the target, theta tops out and the cheapest bottles carry most of the
    weight. Every bottle keeps a weight of at least 1, so nothing on the floor is
    undrawable."""
    import math
    n = len(multiples)
    if n == 0:
        return []
    lo, hi, p = ODDS_CURVE[band_idx]
    budget = p * WEIGHT_SCALE
    target = band_target_multiple(band_idx)
    span = (hi - lo) or 1.0
    pos = [min(max((m - lo) / span, 0.0), 1.0) for m in multiples]

    def shares(theta):
        raw = [math.exp(-theta * x) for x in pos]
        tot = sum(raw)
        return [r / tot for r in raw]

    def mean(theta):
        return sum(s * m for s, m in zip(shares(theta), multiples))

    theta = 0.0
    if mean(0.0) > target:
        lo_t, hi_t = 0.0, 60.0    # exp(-60) ~ 1e-26: past here the split stops changing
        if mean(hi_t) > target:
            theta = hi_t
        else:
            for _ in range(60):
                mid = (lo_t + hi_t) / 2
                if mean(mid) > target:
                    lo_t = mid
                else:
                    hi_t = mid
            theta = hi_t
    return [max(1, int(round(budget * s))) for s in shares(theta)]


def floor_payout(placements):
    """A tier's real expected payout multiple from the weights on a floor:
    sum(weight * multiple) / sum(weight), over (weight, multiple) pairs for every
    bottle placed in that tier. This — not expected_multiple() — is what a player
    actually faces; compare the two after every build."""
    tot = sum(w for w, _m in placements)
    return sum(w * m for w, m in placements) / tot if tot else None


def weight_for_band(band_idx, n_bottles_in_band):
    """Weight for ONE bottle so the band's total weight hits its target probability,
    split evenly across however many bottles currently sit in that band.

    band_total = target_prob * WEIGHT_SCALE ; per-bottle = band_total / n.

    This is what keeps tier odds stable no matter how many bottles fill a band. It
    also means the weights are only correct for the `n` they were computed against:
    remove a bottle from a cell without recomputing and that cell's total weight —
    and therefore its odds — drops in proportion. Any process that adds or removes
    a floor row MUST recompute the affected cells (see `04_replenish_floor.py`)."""
    if n_bottles_in_band <= 0:
        return 0
    target_prob = ODDS_CURVE[band_idx][2]
    return int(round(target_prob * WEIGHT_SCALE / n_bottles_in_band))


def target_prob(band_idx):
    return ODDS_CURVE[band_idx][2]

# COMMAND ----------

# MAGIC %md
# MAGIC ### How deep each cell should be stocked
# MAGIC Odds come from `weight`, not from counts, so a cell with one bottle has exactly
# MAGIC the same probability as a cell with fifty. Depth buys two other things:
# MAGIC **variety** (players in the 39% band shouldn't keep seeing the same bottle)
# MAGIC and **headroom** (a cell that empties takes its band's probability to zero and
# MAGIC silently reshapes the whole curve).
# MAGIC
# MAGIC So target depth is proportional to how often a cell is drawn from, with a hard
# MAGIC minimum so even the 0.3% grail band can't run dry. This is the single
# MAGIC definition of "stocked" used by both `04_replenish_floor.py` (what to top up)
# MAGIC and `05_inventory_reorder_alert.py` (what to buy).

# COMMAND ----------

# Target placements per tier. Four sellable tiers at 250 = 1,000 placements; at an
# average of ~2.5 tiers per bottle that is roughly 400 physical bottles on the floor.
FLOOR_DEPTH_PER_TIER = 250

# No cell ever goes below this, however rare its band. Two so that a single ship
# order can never empty one outright.
MIN_BOTTLES_PER_CELL = 2

# Below this fraction of target, a cell is "low" and the reorder alert fires.
REORDER_THRESHOLD = 0.50


def target_cell_count(band_idx, depth_per_tier=None):
    """How many bottles cell (any tier, this band) should hold when fully stocked."""
    depth = FLOOR_DEPTH_PER_TIER if depth_per_tier is None else depth_per_tier
    return max(MIN_BOTTLES_PER_CELL, int(round(target_prob(band_idx) * depth)))


def max_buy_price(tier, band_idx, gross_margin):
    """The most you can pay for a bottle destined for this cell and still clear
    `gross_margin` against its retail value.

    Margin is taken against retail, not against the pull price: a bottle bought at
    $36 and placed as a $56-retail win is a 35% gross margin on retail. That is the
    number the reorder alert quotes, because it is the number you negotiate with."""
    if not 0.0 <= gross_margin < 1.0:
        raise ValueError(f"gross_margin must be in [0, 1), got {gross_margin}")
    return band_target_value(tier, band_idx) * (1.0 - gross_margin)


def is_win_band(band_idx):
    """True if a bottle in this band is worth at least what the pull cost."""
    return band_idx is not None and band_idx >= WIN_BAND_MIN


def win_rate():
    """Probability that any single pull returns a bottle worth >= the pull price."""
    return sum(p for (_lo, _hi, p) in ODDS_CURVE[WIN_BAND_MIN:])


def expected_multiple():
    """Designed retail value returned per dollar of pull price: each band's
    probability times band_target_multiple() — where cell_weights() puts the
    cell's average, not the band midpoint.

    This is the house's payout ratio; 1 minus it is the house edge on GMV. At 1.0 the
    floor gives back exactly the pull price in retail value and the curve earns
    nothing. It is a design number: floor_payout() is what the stocked floor
    actually pays, and only matches this when every cell can reach its target."""
    return sum(p * band_target_multiple(i) for i, (_lo, _hi, p) in enumerate(ODDS_CURVE))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Self-test — asserts the two design targets and prints the tier-1 breakdown
# MAGIC Fails loudly if an edit to `ODDS_CURVE` breaks the 25% win rate or moves the
# MAGIC payout ratio off the ~0.84x it was tuned to. Widen the tolerances here only
# MAGIC deliberately: they are the contract, not a formality.

# COMMAND ----------

TARGET_WIN_RATE = 0.25        # a win every 4 pulls
TARGET_PAYOUT_MULTIPLE = 0.840    # house keeps ~16% of GMV (target 10–20%)

_wr = win_rate()
_ev = expected_multiple()

print(f"Win rate:       {_wr * 100:.1f}%   (1 win per {1 / _wr:.1f} pulls)")
print(f"Payout ratio:   {_ev:.4f}x of pull price")
print(f"Win line:       band {WIN_BAND_MIN} and above ({ODDS_CURVE[WIN_BAND_MIN][0]:.2f}x+)")
print(f"Share of wins in band {WIN_BAND_MIN}: {ODDS_CURVE[WIN_BAND_MIN][2] / _wr * 100:.1f}%")
print(f"Big winners (1.6x+): {sum(p for (lo, _hi, p) in ODDS_CURVE if lo >= 1.6) * 100:.2f}%")
print(f"Grail (8x+):         {sum(p for (lo, _hi, p) in ODDS_CURVE if lo >= 8.0) * 100:.2f}%")

assert abs(_wr - TARGET_WIN_RATE) < 0.02, (
    f"win rate is {_wr:.3f}, target {TARGET_WIN_RATE} — a pull should win about every 4 tries"
)
assert abs(_ev - TARGET_PAYOUT_MULTIPLE) < 0.01, (
    f"payout ratio is {_ev:.4f}, target {TARGET_PAYOUT_MULTIPLE} — this curve gives away "
    f"{(_ev - TARGET_PAYOUT_MULTIPLE) * 100:+.1f}% more retail value per pull than intended"
)

print(f"\nTier 1 (${TIER_PRICE[1]} pull) bands:")
for _i, (_lo, _hi, _p) in enumerate(ODDS_CURVE):
    _mark = "WIN " if is_win_band(_i) else "loss"
    print(f"  band {_i} [{_mark}]: {band_label(1, _i):>10}  ->  {_p * 100:>5.1f}%   "
          f"(weight if 1 bottle: {weight_for_band(_i, 1)})")

print("\nA $300 bottle is eligible for tiers:", eligible_tiers(300))
print("Its primary (closest-to-par) tier:", primary_tier(300))
