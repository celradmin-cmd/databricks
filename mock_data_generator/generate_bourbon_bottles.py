# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Generate Common Bottles (Bourbon)
# MAGIC
# MAGIC Generates a fixed **total** of `total_bottles` (default 1,000) synthetic bourbon
# MAGIC records and writes them into the **bronze** table `bourbon_inventory` — raw
# MAGIC inventory only: `bottle_serial, name, category, description, rarity,
# MAGIC retail_value, distillery`. No `primary_tier` / `eligible_tiers` / `image_url` /
# MAGIC `cataloged_at` here — those are gold-layer enrichments computed downstream by
# MAGIC `archive/01_classify_and_enrich.py` (point its source table at `bourbon_inventory`
# MAGIC and re-run it before `02_build_bourbon_app_floor_weighted.py`).
# MAGIC
# MAGIC ## Why a fixed total instead of a per-tier target
# MAGIC The order changed from 4,000 bottles (1,000 dedicated per tier, no sharing
# MAGIC needed) to **1,000 bottles total, shared across all 4 tiers**. This script still
# MAGIC prioritizes cells the same way the old per-tier version did — greedily filling
# MAGIC whichever `(tier, band)` cell is currently neediest, crediting a new bottle to
# MAGIC every tier it happens to qualify for (a $600 bottle can land in a real band at
# MAGIC all 4 tiers at once) — but now stops at a **hard cap of `total_bottles`** rather
# MAGIC than running until every tier independently hits its own 1,000-bottle target.
# MAGIC Because the loop always attacks the single neediest cell first, the
# MAGIC high-probability common bands (37/39/20%) get covered early and the rare bands
# MAGIC (3/0.8/0.2%) — which need only a handful of bottles each — get covered almost
# MAGIC immediately and then stop competing for budget. Any cell still short when the
# MAGIC budget runs out is reported so the weighted floor builder's placeholder
# MAGIC mechanism (`02_build_bourbon_app_floor_weighted.py`) can cover the gap.
# MAGIC
# MAGIC Odds truth (`BANDS`, `PROBS`, `TIER_PRICE`) comes only from `00_celr_odds_config`.

# COMMAND ----------

# MAGIC %run ./00_celr_odds_config

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Environment (prod/dev)")
dbutils.widgets.text("total_bottles", "1000", "Total bottles to generate (hard cap, shared across all tiers)")
dbutils.widgets.text("per_tier_shape", "1000", "Per-tier demand basis used only to prioritize which cell is neediest")
dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("bronze_schema", "bronze", "Bronze schema")
dbutils.widgets.text("inventory_table", "bourbon_inventory", "Bronze inventory table")
dbutils.widgets.text("category", "WHISKEY", "Category value for synthetic rows")
dbutils.widgets.dropdown("naming_mode", "real_commodity", ["real_commodity", "fictional_house"], "Naming mode")
dbutils.widgets.text("llm_endpoint", "databricks-llama-4-maverick", "Foundation Model endpoint")
dbutils.widgets.text("gen_batch", "20", "Records per LLM call")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "append"], "Write mode (overwrite = fresh 1,000-bottle inventory)")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (no write)")

ENV            = dbutils.widgets.get("environment")
TOTAL_BOTTLES  = int(dbutils.widgets.get("total_bottles"))
PER_TIER_SHAPE = int(dbutils.widgets.get("per_tier_shape"))
CATALOG        = dbutils.widgets.get("catalog")
BRONZE         = dbutils.widgets.get("bronze_schema")
BRONZE_TABLE   = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('inventory_table')}"
CATEGORY       = dbutils.widgets.get("category")
NAMING         = dbutils.widgets.get("naming_mode")
LLM            = dbutils.widgets.get("llm_endpoint")
GEN_BATCH      = int(dbutils.widgets.get("gen_batch"))
WRITE_MODE     = dbutils.widgets.get("write_mode")
DRY_RUN        = dbutils.widgets.get("dry_run") == "true"

print(f"env={ENV} total_bottles={TOTAL_BOTTLES} table={BRONZE_TABLE} naming={NAMING} "
      f"write_mode={WRITE_MODE} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Classification (eligibility only — not persisted to bronze)
# MAGIC `classify()` is used purely to steer generation toward the neediest cells and
# MAGIC to credit multi-tier coverage. The bronze table never stores `eligible_tiers`
# MAGIC or `primary_tier` — that's `02_classify_and_enrich`'s job downstream.

# COMMAND ----------

BANDS = [(lo, hi) for (lo, hi, _p) in ODDS_CURVE]
PROBS = [p for (_lo, _hi, p) in ODDS_CURVE]

def band_index(mult):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= mult < hi:
            return i
    return None

def classify(retail_value):
    """Return (eligible_tiers, primary_tier, primary_band) for a retail value.
    eligible_tiers = list of (tier, multiple, band_idx)."""
    elig = []
    for t, price in TIER_PRICE.items():
        mult = retail_value / price
        b = band_index(mult)
        if b is not None:
            elig.append((int(t), round(float(mult), 4), int(b)))
    if not elig:
        return [], None, None
    primary = min(elig, key=lambda e: abs(e[1] - 1.0))
    return elig, primary[0], primary[2]

def target_count(band):
    return round(PROBS[band] * PER_TIER_SHAPE)

targets = {(t, b): target_count(b) for t in TIER_PRICE for b in range(len(BANDS))}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Existing coverage per cell
# MAGIC Recomputed from whatever `retail_value`s already sit in the bronze table (if
# MAGIC any, e.g. a prior `append` run) — bronze has no stored `eligible_tiers` to
# MAGIC read directly, so this reclassifies each existing row.

# COMMAND ----------

from pyspark.sql import functions as F

have = {}
try:
    existing = spark.table(BRONZE_TABLE).select("retail_value").collect()
    for r in existing:
        elig, _p, _b = classify(float(r["retail_value"]))
        for (t, _m, b) in elig:
            have[(t, b)] = have.get((t, b), 0) + 1
    print(f"existing bottles ({len(existing)}) contribute coverage to {len(have)} cells")
except Exception as e:
    print(f"(table not readable / empty: {e}); assuming zero coverage")

need = {}
total = 0
print("\ntier band  target  have  need")
for t in TIER_PRICE:
    for b in range(len(BANDS)):
        tg = targets[(t, b)]
        hv = have.get((t, b), 0)
        gap = max(tg - hv, 0)
        need[(t, b)] = gap
        total += gap
        print(f"  {t}   {b}   {tg:>5}  {hv:>4}  {gap:>4}")
print(f"\nremaining cell-coverage to fill: {total} (across a {TOTAL_BOTTLES}-bottle budget)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. LLM text generation (names + descriptions only; we assign price)

# COMMAND ----------

import json, uuid, random, re
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
RARITY_BY_BAND = {0: "common", 1: "common", 2: "common", 3: "rare", 4: "rare", 5: "legendary"}

def text_prompt(n, lo_d, hi_d):
    if NAMING == "real_commodity":
        rule = (f"Use REAL, actual American whiskey bottles that genuinely retail around "
                f"${lo_d}-${hi_d} and are widely available at national retailers. Use their "
                "true product and distillery names. Repeats are fine if few options exist "
                "at this price.")
    else:
        rule = ("Invent plausible but ENTIRELY FICTIONAL American whiskey and distillery "
                "names (no real brands). Craft/small-batch styling. No duplicate names.")
    return (f"List {n} American whiskey bottles. {rule} "
            "Return ONLY a JSON array, no prose, no markdown. Each element: "
            '{"name": string, "distillery": string, '
            '"description": one sentence under 30 words, tasting-note tone}. '
            "Inside string values use plain ASCII only: no double quotes, no "
            "apostrophes that would break JSON, no newlines."
            '"distillery": What distillery is this bottle from? If unknown, say so')

def call_llm_text(n, lo_d, hi_d):
    resp = client.predict(endpoint=LLM, inputs={
        "messages": [
            {"role": "system", "content": "You output strictly valid JSON and nothing else."},
            {"role": "user", "content": text_prompt(n, lo_d, hi_d)},
        ],
        "temperature": 0.9, "max_tokens": 4000,
    })
    text = resp["choices"][0]["message"]["content"]
    return parse_bottles(text)

def parse_bottles(text):
    """Tolerant parse: try the whole array, else salvage object-by-object so one
    malformed or truncated element doesn't discard the batch. Returns list[dict]."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e > s:
        try:
            data = json.loads(text[s:e + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):   # objects are flat
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict):
                out.append(o)
        except json.JSONDecodeError:
            continue
    return out

def clean_desc(d):
    d = " ".join(str(d).split())
    return " ".join(d.split()[:30]) if d else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fill loop — two phases, both driven off the same TOTAL_BOTTLES budget
# MAGIC **Phase 1** targets the neediest cell first, same as before, crediting every
# MAGIC tier a bottle happens to qualify for. Because one bottle's price can satisfy
# MAGIC several tiers' cells at once, total demand across all 24 cells is often
# MAGIC satisfied well under the budget — that's the point, not a bug. But stopping
# MAGIC there means the run silently generates fewer bottles than asked for.
# MAGIC
# MAGIC **Phase 2** spends whatever budget is left over once demand is satisfied,
# MAGIC picking bands weighted by the odds curve (so extras still land mostly in the
# MAGIC common bands) purely for variety. This is safe with a properly-sized
# MAGIC `WEIGHT_SCALE`: extra bottles in an already-satisfied band just mean more
# MAGIC distinct SKUs sharing that band's weight, not skewed odds.

# COMMAND ----------

def retail_for_cell(tier, band):
    lo, hi = BANDS[band]
    price = TIER_PRICE[tier]
    lo_d, hi_d = int(round(lo * price)), int(round(hi * price))
    return random.randint(lo_d, max(lo_d, hi_d - 1))

rows = []
used_names = set()   # only enforced for fictional naming
MAX_ITERS = TOTAL_BOTTLES * 3 + 1000
iters = 0

def accept_records(recs, tier, band, credit_need):
    """Classify + dedupe a batch of LLM records into `rows`. Returns count added."""
    global total
    added = 0
    for rec in recs:
        if len(rows) >= TOTAL_BOTTLES:
            break
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        key = name.lower()
        if not name:
            continue
        if NAMING == "fictional_house" and key in used_names:
            continue  # real names may repeat across physical bottles
        rv = float(retail_for_cell(tier, band))
        elig, _ptier, pband = classify(rv)
        if not elig:
            continue
        used_names.add(key)
        rows.append({
            "bottle_serial": str(uuid.uuid4()),
            "name": name,
            "category": CATEGORY,
            "description": clean_desc(rec.get("description")),
            "distillery": rec.get("distillery"),
            "rarity": RARITY_BY_BAND[pband],
            "retail_value": rv,
        })
        added += 1
        if credit_need:
            # credit every needy cell this bottle covers
            for (et, _em, eb) in elig:
                if need.get((et, eb), 0) > 0:
                    need[(et, eb)] -= 1
                    total -= 1
    return added

while len(rows) < TOTAL_BOTTLES and total > 0 and iters < MAX_ITERS:
    iters += 1
    (t, b), gap = max(need.items(), key=lambda kv: kv[1])
    if gap <= 0:
        break
    want = min(GEN_BATCH, gap, TOTAL_BOTTLES - len(rows))
    lo, hi = BANDS[b]
    lo_d, hi_d = int(round(lo * TIER_PRICE[t])), int(round(hi * TIER_PRICE[t]))
    try:
        recs = call_llm_text(want, lo_d, hi_d)
    except Exception as e:
        print(f"[warn] LLM call failed ({e}); retrying")
        continue
    accept_records(recs, t, b, credit_need=True)

phase1_count = len(rows)
print(f"phase 1: generated {phase1_count} bottles over {iters} iterations; remaining need={total}")
leftover = {k: v for k, v in need.items() if v > 0}
if leftover:
    print(f"[info] cells still short after phase 1 (weighted floor builder will placeholder these): {leftover}")

# Phase 2: demand satisfied (or LLM/iteration budget hit) before the bottle budget did —
# spend what's left on odds-curve-weighted variety instead of stopping short.
while len(rows) < TOTAL_BOTTLES and iters < MAX_ITERS:
    iters += 1
    b = random.choices(range(len(BANDS)), weights=PROBS, k=1)[0]
    t = random.choice(list(TIER_PRICE))
    want = min(GEN_BATCH, TOTAL_BOTTLES - len(rows))
    try:
        recs = call_llm_text(want, *[int(round(m * TIER_PRICE[t])) for m in BANDS[b]])
    except Exception as e:
        print(f"[warn] LLM call failed ({e}); retrying")
        continue
    accept_records(recs, t, b, credit_need=False)

if len(rows) > phase1_count:
    print(f"phase 2: added {len(rows) - phase1_count} bonus bottles to reach the {TOTAL_BOTTLES} budget")

print(f"generated {len(rows)} bottles total over {iters} iterations (budget {TOTAL_BOTTLES})")
if len(rows) < TOTAL_BOTTLES:
    print(f"[warn] stopped {TOTAL_BOTTLES - len(rows)} short of budget after {MAX_ITERS} iterations "
          "— check LLM call failures above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write to bronze (raw inventory columns only)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("bottle_serial", StringType()),
    StructField("name", StringType()),
    StructField("category", StringType()),
    StructField("description", StringType()),
    StructField("rarity", StringType()),
    StructField("retail_value", DoubleType()),
    StructField("distillery", StringType()),
])

def to_tuple(r):
    return (r["bottle_serial"], r["name"], r["category"], r["description"],
            r["rarity"], r["retail_value"], r["distillery"])

if not rows:
    print("nothing to write")
elif DRY_RUN:
    print("DRY RUN - sample of what would be written:")
    for r in rows[:5]:
        print(json.dumps(r, indent=2))
    print(f"... {len(rows)} rows total (set dry_run=false to write)")
else:
    df = spark.createDataFrame([to_tuple(r) for r in rows], schema)
    (df.write.mode(WRITE_MODE).option("mergeSchema", "true").saveAsTable(BRONZE_TABLE))
    print(f"{WRITE_MODE} {len(rows)} rows to {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Next steps (not run here)
# MAGIC 1. Point `archive/01_classify_and_enrich.py`'s `SRC_TABLE` at
# MAGIC    `{catalog}.bronze.bourbon_inventory` (its column names — `name`, `distillery`,
# MAGIC    `description`, `rarity`, `retail_value` — already match; drop its
# MAGIC    `source_sku`/`quantity`/`avg_price`/`median_price` handling, no longer present)
# MAGIC    and re-run it to compute `primary_tier`/`primary_band`/`eligible_tiers` into
# MAGIC    `gold.bourbon_catalog`.
# MAGIC 2. Then run `02_build_bourbon_app_floor_weighted.py` to build the wide,
# MAGIC    weight-driven app floor from that gold catalog.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prod_celr.bronze.bourbon_inventory
