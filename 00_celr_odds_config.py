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
# MAGIC 1. **Win rate 40%** — a win roughly every 2.5 pulls, and the overwhelming
# MAGIC    majority of those wins land in band 2 (1.00–1.25x), the band immediately
# MAGIC    above the loss line. A band-2 win on a $50 pull is a $50–$62 bottle: it
# MAGIC    reads as a win without costing the house much.
# MAGIC 2. **Expected payout 0.9975x of pull price** — identical to the previous
# MAGIC    six-band curve (41/35/20/3/1/0.5). The win rate went from 24.4% to 40.0%
# MAGIC    for free, in EV terms.
# MAGIC
# MAGIC ## Why band 2 had to be split
# MAGIC The old curve's first win band was a single wide 1.00–1.60x. Pushing enough
# MAGIC probability into it to hit a 40% win rate costs ~8% more payout, because the
# MAGIC average win in that band is 1.3x. Splitting it at 1.25x lets the bulk of the
# MAGIC probability sit on 1.00–1.25x (average 1.125x) while 1.25–1.60x stays a thin
# MAGIC 5%. That is what buys the win rate back to EV-neutral.
# MAGIC
# MAGIC ## What paid for the rest
# MAGIC The loss side. To win 40% of the time at flat EV the losses have to be deeper:
# MAGIC the deep-loss band (0.50–0.70x) goes from 35% to 29.2%, but the shallow-loss
# MAGIC band (0.70–1.00x) goes from 41% to 30.8% — so the *share of losses* that are
# MAGIC deep rose from 46% to 49%. Big winners also got rarer, deliberately: 1.6x-plus
# MAGIC went from 4.5% to 3.2%, and the 8x-plus grail from 0.50% to 0.30%.
# MAGIC
# MAGIC Everything below is the *only* place you tune odds. Change a number here and
# MAGIC every notebook follows. Re-run the self-test after any edit.

# COMMAND ----------

# Pull price for each tier (the "dollar amount you spent on the pull").
# Tier 5 ($250) exists in the warehouse model but is NOT purchasable in the app —
# `TIER_PRICE_CENTS` in `celr/src/lib/payments.functions.ts` only sells 1/2/3/4.
# It is kept here so re-enabling it is a one-line change to SELLABLE_TIERS, but it
# is excluded from every floor build; see SELLABLE_TIERS below for why that matters.
TIER_PRICE = {1: 50, 2: 100, 5: 250, 3: 500, 4: 1000}

# Tiers the app actually sells. ONLY these get placements on a floor.
#
# This is not cosmetic. A bottle's placements are stored in fixed wide slot columns
# (primary/secondary/tertiary/quaternary), ordered closest-to-par, and the app-side
# sync in `celr/src/lib/weighted-sync.server.ts` reads exactly four of them. If
# tier 5 were placed, a bottle eligible for all five tiers would push one real,
# sellable tier into a fifth slot that the sync silently drops — losing that
# bottle's odds contribution in a tier players can actually buy. Four sellable
# tiers means at most four placements, so nothing can overflow.
#
# If you re-enable tier 5: add a quinary slot to `bourbons_weighted`/`agave_weighted`
# in Supabase, to `merge_weighted_catalog`, to `weighted_tier_catalog`, and to the
# SELECT in `weighted-sync.server.ts` — all four, or the odds go quietly wrong.
SELLABLE_TIERS = (1, 2, 3, 4)

# The house curve, as multiplier bands of the pull price.
# Each tuple: (lo_multiple_inclusive, hi_multiple_exclusive, target_probability).
# Bands must be contiguous, ascending, and sum to 1.0 — all asserted below.
ODDS_CURVE = [
    (0.50, 0.70, 0.292),   # deep loss        -> $25–$35 on a $50 pull
    (0.70, 1.00, 0.308),   # shallow loss     -> $35–$50
    # ---------------------------------------------------------------- win line
    (1.00, 1.25, 0.318),   # the win          -> $50–$62   (79.5% of all wins)
    (1.25, 1.60, 0.050),   # good win         -> $62–$80
    (1.60, 3.00, 0.020),   # nice win         -> $80–$150
    (3.00, 8.00, 0.009),   # big win          -> $150–$400
    (8.00, 16.00, 0.003),  # grail            -> $400–$800
]

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


def weight_for_band(band_idx, n_bottles_in_band):
    """Weight for ONE bottle so the band's total weight hits its target probability,
    split evenly across however many bottles currently sit in that band.

    band_total = target_prob * WEIGHT_SCALE ; per-bottle = band_total / n.

    This is what keeps tier odds stable no matter how many bottles fill a band. It
    also means the weights are only correct for the `n` they were computed against:
    remove a bottle from a cell without recomputing and that cell's total weight —
    and therefore its odds — drops in proportion. Any process that adds or removes
    a floor row MUST recompute the affected cells (see `05_replenish_floor.py`)."""
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
# MAGIC **variety** (players in the 31.8% band shouldn't keep seeing the same bottle)
# MAGIC and **headroom** (a cell that empties takes its band's probability to zero and
# MAGIC silently reshapes the whole curve).
# MAGIC
# MAGIC So target depth is proportional to how often a cell is drawn from, with a hard
# MAGIC minimum so even the 0.3% grail band can't run dry. This is the single
# MAGIC definition of "stocked" used by both `05_replenish_floor.py` (what to top up)
# MAGIC and `06_inventory_reorder_alert.py` (what to buy).

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
    return band_mid_value(tier, band_idx) * (1.0 - gross_margin)


def is_win_band(band_idx):
    """True if a bottle in this band is worth at least what the pull cost."""
    return band_idx is not None and band_idx >= WIN_BAND_MIN


def win_rate():
    """Probability that any single pull returns a bottle worth >= the pull price."""
    return sum(p for (_lo, _hi, p) in ODDS_CURVE[WIN_BAND_MIN:])


def expected_multiple():
    """Expected retail value returned per dollar of pull price, using band midpoints.

    This is the house's payout ratio. At 1.0 the floor gives back exactly the pull
    price in retail value, and the business's margin comes from buying bottles below
    retail plus the sellback discount — not from the curve. Keep an eye on it: every
    probability edit moves it."""
    return sum(p * (lo + hi) / 2.0 for (lo, hi, p) in ODDS_CURVE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Self-test — asserts the two design targets and prints the tier-1 breakdown
# MAGIC Fails loudly if an edit to `ODDS_CURVE` breaks the 40% win rate or moves the
# MAGIC payout ratio off the 0.9975x it was tuned to. Widen the tolerances here only
# MAGIC deliberately: they are the contract, not a formality.

# COMMAND ----------

TARGET_WIN_RATE = 0.40        # a win roughly every 2.5 pulls
TARGET_PAYOUT_MULTIPLE = 0.9975   # matches the previous six-band curve exactly

_wr = win_rate()
_ev = expected_multiple()

print(f"Win rate:       {_wr * 100:.1f}%   (1 win per {1 / _wr:.1f} pulls)")
print(f"Payout ratio:   {_ev:.4f}x of pull price")
print(f"Win line:       band {WIN_BAND_MIN} and above ({ODDS_CURVE[WIN_BAND_MIN][0]:.2f}x+)")
print(f"Share of wins in band {WIN_BAND_MIN}: {ODDS_CURVE[WIN_BAND_MIN][2] / _wr * 100:.1f}%")
print(f"Big winners (1.6x+): {sum(p for (lo, _hi, p) in ODDS_CURVE if lo >= 1.6) * 100:.2f}%")
print(f"Grail (8x+):         {sum(p for (lo, _hi, p) in ODDS_CURVE if lo >= 8.0) * 100:.2f}%")

assert abs(_wr - TARGET_WIN_RATE) < 0.02, (
    f"win rate is {_wr:.3f}, target {TARGET_WIN_RATE} — a pull should win every 2–3 tries"
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
