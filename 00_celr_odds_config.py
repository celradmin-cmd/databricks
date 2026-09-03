# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · CËLR Odds Config (shared)
# MAGIC
# MAGIC The single source of truth for the **house odds curve** and tier/price mapping.
# MAGIC `%run` this notebook from `01_classify_and_image` and `02_replacement_engine`
# MAGIC so both notebooks place bottles into the *same* bands and assign the *same* weights.
# MAGIC
# MAGIC ## Where the curve comes from
# MAGIC Reverse-engineered from your existing `bourbons.weight` column. Tier-1 ($50 pull)
# MAGIC weights were 410 / 350 / 200 / 30 / 10 / 5 (sum ≈ 1000), which is exactly the
# MAGIC screenshot: 41% / 35% / 20% / 3% / 1% / 0.5%. Expressed as **multiples of the
# MAGIC pull price** the band edges are clean: 0.5x · 0.7x · 1.0x · 1.6x · 3.0x · 8.0x · 16.0x.
# MAGIC
# MAGIC Everything below is the *only* place you tune odds. Change a number here and both
# MAGIC notebooks follow.

# COMMAND ----------

# Pull price for each tier (the "dollar amount you spent on the pull").
TIER_PRICE = {1: 50, 2: 100, 3: 500, 4: 1000}

# The house curve, as multiplier bands of the pull price.
# Each tuple: (lo_multiple_inclusive, hi_multiple_exclusive, target_probability)
# Probability DECREASES as a bottle's retail value moves away from what you paid (1.0x).
# Edit edges/probabilities here only.
ODDS_CURVE = [
    (0.50, 0.70, 0.350),   # well below par   -> $25–$35 on a $50 pull
    (0.70, 1.00, 0.410),   # just below par   -> $35–$50   (the peak / house margin)
    (1.00, 1.60, 0.200),   # slight upside    -> $50–$80
    (1.60, 3.00, 0.030),   # nice win         -> $80–$150
    (3.00, 8.00, 0.010),   # big win          -> $150–$400
    (8.00, 16.00, 0.005),  # grail            -> $400–$800
]

# Weights are integers so the pull selector stays simple. SCALE = sum of all weights in
# a tier; with SCALE=1000 a 0.5% band carries weight 5 (matches your seed data exactly).
WEIGHT_SCALE = 1000

# The lowest and highest multiple the curve covers. A bottle whose retail value falls
# outside [floor_lo * price, ceil_hi * price] does NOT belong in that tier.
CURVE_LO = ODDS_CURVE[0][0]    # 0.50
CURVE_HI = ODDS_CURVE[-1][1]   # 16.0

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
    """All tiers whose curve can host this retail value, with the multiplier for each.
    Returns list of (tier, multiple, band_index) sorted by closeness to par (1.0x)."""
    out = []
    for tier, price in TIER_PRICE.items():
        bi = band_index(retail_value, tier)
        if bi is not None:
            out.append((tier, float(retail_value) / price, bi))
    # closest-to-par first -> the bottle's most "natural" home tier
    out.sort(key=lambda t: abs(t[1] - 1.0))
    return out


def primary_tier(retail_value):
    """The single best home tier for a bottle: where it sits closest to par. None if it
    fits no tier (e.g. retail < $25 fits nothing; retail > $16k fits nothing)."""
    et = eligible_tiers(retail_value)
    return et[0][0] if et else None


def band_label(tier, band_idx):
    """Human-readable value range for a band, e.g. '$35–$50'. Drives the UI breakdown."""
    price = TIER_PRICE[tier]
    lo, hi, _ = ODDS_CURVE[band_idx]
    return f"${int(round(lo * price))}\u2013${int(round(hi * price))}"


def weight_for_band(band_idx, n_bottles_in_band):
    """Weight for ONE bottle so the band's total weight hits its target probability,
    split evenly across however many bottles currently sit in that band.

    band_total = target_prob * WEIGHT_SCALE ; per-bottle = band_total / n.
    This is what keeps tier odds stable no matter how many bottles fill a band:
    add/remove a bottle from the band and you recompute only that band."""
    if n_bottles_in_band <= 0:
        return 0
    target_prob = ODDS_CURVE[band_idx][2]
    return int(round(target_prob * WEIGHT_SCALE / n_bottles_in_band))


def target_prob(band_idx):
    return ODDS_CURVE[band_idx][2]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick self-test — should reproduce the "$50 lands" screenshot
# MAGIC print("Tier-1 ($50) bands:")
# MAGIC for i, (lo, hi, p) in enumerate(ODDS_CURVE):
# MAGIC     print(f"  band {i}: {band_label(1, i):>10}  ->  {p*100:>5.1f}%   (weight if 1 bottle: {weight_for_band(i, 1)})")
# MAGIC print("\nA $300 bottle is eligible for tiers:", eligible_tiers(300))
# MAGIC print("Its primary (closest-to-par) tier:", primary_tier(300))