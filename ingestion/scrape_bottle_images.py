# Databricks notebook source
# MAGIC %md
# MAGIC # Scrape bottle images
# MAGIC
# MAGIC For every bottle in `bronze.kindred_bottles_raw` (written by
# MAGIC `load_kindred_bottles_to_bronze.py`), finds the product photo on the brand's own site
# MAGIC (`source_url`), downloads it into a UC volume, and records what it found in
# MAGIC `bronze.bottle_image_manifest`.
# MAGIC
# MAGIC ## How an image is found, in order
# MAGIC 1. **Shopify `/products.json`** — most of the brands (Black Emerald, Shāng, Centinela,
# MAGIC    Kinsale US, Heaven's Door) run Shopify, which publishes the whole catalog as JSON.
# MAGIC    Each product is matched to the CSV row by title + image filename.
# MAGIC 2. **Single-product page** (`/product/x`, `/product-page/x` — Copper & Kings) — the page
# MAGIC    exists for that one bottle, so its gallery image (or `og:image`) is the bottle.
# MAGIC 3. **Listing page, then the homepage** (Devil Road, Rare Hare's Wix site) — every `<img>`
# MAGIC    is scored by its alt text + filename.
# MAGIC
# MAGIC Matching rewards the words that make an expression unique (*Ascension*, *Hazmat*, *1953*)
# MAGIC and barely counts brand/generic words (*whiskey*, *Kentucky*, the brand name) — otherwise
# MAGIC the brand's own logo outscores the bottle. Bundles, gift sets and 50 mL minis are
# MAGIC penalised.
# MAGIC
# MAGIC ## Statuses in the manifest
# MAGIC | status | image downloaded? | meaning |
# MAGIC |---|---|---|
# MAGIC | `matched` | yes | confident match — `approved = true` |
# MAGIC | `manual` | yes | taken from `bronze.bottle_image_overrides` — `approved = true` |
# MAGIC | `low_confidence` | yes | best guess, **look at it** before using |
# MAGIC | `no_match` / `no_candidates` | no | nothing on the page resembled this bottle |
# MAGIC | `no_source_url` | no | CSV row has no URL (Pinhook, Stranahan's, Bootleg 1–4, ...) |
# MAGIC | `robots_disallowed` / `download_failed` | no | see `error` |
# MAGIC
# MAGIC **Fixing a wrong or missing image:** insert into `bronze.bottle_image_overrides`
# MAGIC (`bottle_serial, image_url, note`) and re-run. An override always wins over scraping.
# MAGIC To approve a `low_confidence` guess as-is, override it with the `image_url` the manifest
# MAGIC already shows.
# MAGIC
# MAGIC **Politeness:** checks each site's `robots.txt`, sends an honest User-Agent, and pauses
# MAGIC `delay_seconds` between requests to the same site. Last checked 2026-10-21: every site in
# MAGIC the CSV allowed this.
# MAGIC
# MAGIC **S3:** the manifest also carries `bottle_id, side, cdn_url, s3_key`, so
# MAGIC `archive/fetch_images_to_s3.py` can mirror these images with
# MAGIC `manifest_table=bottle_image_manifest`.

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("bronze_schema", "bronze", "Bronze schema")
dbutils.widgets.text("source_table", "kindred_bottles_raw", "Bronze bottle list (needs source_url)")
dbutils.widgets.text("manifest_table", "bottle_image_manifest", "Bronze image manifest table")
dbutils.widgets.text("overrides_table", "bottle_image_overrides", "Bronze manual override table")
dbutils.widgets.text("images_volume", "bottle_images", "UC volume for image files")
dbutils.widgets.text("volume_prefix", "kindred", "Folder inside the volume")
dbutils.widgets.text("min_score", "0.5", "Match score needed for 'matched' (else low_confidence)")
dbutils.widgets.text("delay_seconds", "1.0", "Pause between requests to the same site")
dbutils.widgets.dropdown("overwrite_images", "false", ["true", "false"], "Re-download images already in the volume?")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (match only — no download, no table write)")

CATALOG       = dbutils.widgets.get("catalog")
BRONZE        = dbutils.widgets.get("bronze_schema")
SOURCE_TABLE  = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('source_table')}"
MANIFEST      = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('manifest_table')}"
OVERRIDES     = f"{CATALOG}.{BRONZE}.{dbutils.widgets.get('overrides_table')}"
VOLUME        = dbutils.widgets.get("images_volume")
VOLUME_PREFIX = dbutils.widgets.get("volume_prefix").strip("/")
VOLUME_DIR    = f"/Volumes/{CATALOG}/{BRONZE}/{VOLUME}/{VOLUME_PREFIX}"
MIN_SCORE     = float(dbutils.widgets.get("min_score"))
DELAY         = float(dbutils.widgets.get("delay_seconds"))
OVERWRITE     = dbutils.widgets.get("overwrite_images") == "true"
DRY_RUN       = dbutils.widgets.get("dry_run") == "true"

print(f"source={SOURCE_TABLE}\nmanifest={MANIFEST}\noverrides={OVERRIDES}\nimages -> {VOLUME_DIR}\n"
      f"min_score={MIN_SCORE} delay={DELAY}s overwrite_images={OVERWRITE} dry_run={DRY_RUN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Volume + override table
# MAGIC Created on first run so the override table is there to `INSERT` into.

# COMMAND ----------

if not DRY_RUN:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE}.{VOLUME}")
    import os
    os.makedirs(VOLUME_DIR, exist_ok=True)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {OVERRIDES} (
        bottle_serial STRING NOT NULL,
        image_url     STRING NOT NULL,
        note          STRING,
        added_at      TIMESTAMP
    ) COMMENT 'Manual bottle image URLs. Always beat scrape_bottle_images.py matching.'
""")

bottles = [r.asDict() for r in spark.table(SOURCE_TABLE)
           .select("bottle_serial", "brand", "product", "source_url").collect()]
overrides = {r["bottle_serial"]: r["image_url"] for r in spark.table(OVERRIDES).collect()}
print(f"{len(bottles)} bottles, {len(overrides)} manual overrides")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Matcher

# COMMAND ----------

import html, re, time, unicodedata
import urllib.robotparser
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

UA = "celr-image-scraper/1.0"

# Words every bottle on these sites shares — matching on them proves nothing, so they
# only count a little. The distinctive words (Ascension, Hazmat, 1953) carry the match.
GENERIC = {"whiskey", "whisky", "bourbon", "irish", "american", "single", "straight",
           "kentucky", "tennessee", "tequila", "blend", "blended", "brandy", "spirit",
           "spirits", "the", "of", "in", "and", "a", "finished", "aged", "year", "old",
           "ml", "cl", "750", "700", "75", "70", "bottle", "edition"}
# Signs a candidate is NOT the 750 mL bottle itself.
PENALTY = {"bundle", "special", "gift", "set", "trilogy", "collection", "stones", "50ml",
           "5cl", "miniature", "mini", "logo", "wordmark", "lockup", "banner", "hero", "favicon", "icon"}
FULL_SIZE = {"750ml", "700ml", "75cl", "70cl"}

PRODUCT_PAGE = re.compile(r"/(product|products|product-page|shop)/[^/]+/?$", re.I)
PRODUCT_PAGE_SCORE = 0.9   # a page that exists for exactly this bottle; its gallery image is the bottle
CONTAINERS = {"div", "section", "main", "figure", "picture", "ul", "li", "a", "article", "span"}


def tokens(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "")                          # "V.S.O.P." -> "vsop"
    s = re.sub(r"(\d+)\s*-?\s*(yr|yrs|years?)\b", r"\1 year", s)   # "7YR" / "7-Year" -> "7 year"
    s = re.sub(r"(\d+)\s*(ml|cl)\b", r"\1\2", s)                    # "50 ml" -> "50ml"
    return set(re.findall(r"[a-z0-9]+", s))


def score(brand, product, candidate):
    """How well `candidate` text (title / alt / filename) names this bottle. Words unique
    to the expression carry the match; brand and generic words only nudge it, since
    every image on a brand's own site shares them (its logo included)."""
    b, c = tokens(brand), tokens(candidate)
    t = tokens(product) - b
    distinct, weak = t - GENERIC, (t & GENERIC) | b
    if not c or not (distinct or weak):
        return 0.0
    hits = len(distinct & c)
    if distinct and not hits:
        return 0.0
    s = (hits + 0.25 * len(weak & c)) / (len(distinct) + 0.25 * len(weak))
    bad = (c & PENALTY) - t
    if c & FULL_SIZE:
        bad -= {"50ml", "5cl"}   # "750ml + includes free 5cl" is still the full bottle
    return round(s - 0.5 * len(bad), 3)


class _Imgs(HTMLParser):
    """Collects <img>s with their alt text and whether they sit inside a product gallery
    (an ancestor whose class/id mentions product/gallery/slide)."""
    def __init__(self):
        super().__init__()
        self.imgs, self.og, self.stack = [], None, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in CONTAINERS:
            self.stack.append((tag, f"{a.get('class') or ''} {a.get('id') or ''}".lower()))
        if tag == "meta" and a.get("property") == "og:image" and a.get("content"):
            self.og = self.og or a["content"]
        if tag == "img":
            src = a.get("data-src") or a.get("src") or ""
            srcset = a.get("srcset") or a.get("data-srcset") or ""
            if srcset:   # largest width in srcset
                best = max((p.strip().rsplit(" ", 1) for p in srcset.split(",") if " " in p.strip()),
                           key=lambda x: int(re.sub(r"\D", "", x[1]) or 0), default=None)
                if best:
                    src = best[0]
            in_gallery = any(re.search(r"product|gallery|slide", c) for _, c in self.stack)
            if src and not src.startswith("data:"):
                self.imgs.append((src, a.get("alt") or "", a.get("title") or "", in_gallery))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


class ImageResolver:
    """Finds the best product image for one bottle. Caches per site, so a brand with
    ten bottles costs one /products.json call, not ten."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self._robots, self._last_hit, self._shop, self._page = {}, {}, {}, {}

    # -- polite fetching ------------------------------------------------------------
    def allowed(self, url):
        u = urlparse(url)
        origin = f"{u.scheme}://{u.netloc}"
        if origin not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.session.get(origin + "/robots.txt", timeout=20)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])   # unreachable robots.txt = no rules
            self._robots[origin] = rp
        return self._robots[origin].can_fetch(UA, url)

    def get(self, url, **kw):
        host = urlparse(url).netloc
        wait = DELAY - (time.time() - self._last_hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.time()
        return self.session.get(url, timeout=20, **kw)

    # -- candidate sources ----------------------------------------------------------
    def _shopify(self, origin):
        """Every product on a Shopify store via /products.json; None if not Shopify."""
        if origin in self._shop:
            return self._shop[origin]
        out = None
        if self.allowed(origin + "/products.json"):
            for page in range(1, 11):
                try:
                    r = self.get(f"{origin}/products.json", params={"limit": 250, "page": page})
                except requests.RequestException:
                    break
                if r.status_code != 200 or "json" not in r.headers.get("Content-Type", ""):
                    break
                products = r.json().get("products", [])
                out = out or []
                if not products:
                    break
                for p in products:
                    if p.get("images"):
                        img = p["images"][0]
                        fname = urlparse(img["src"]).path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                        # Title + filename only: Shopify alt text is often a paragraph of
                        # marketing copy naming every bottle in a bundle.
                        out.append({"label": f'{p["title"]} {fname.replace("_", " ")}',
                                    "image_url": img["src"],
                                    "page_url": f"{origin}/products/{p['handle']}"})
        self._shop[origin] = out
        return out

    def _html(self, url):
        """(<img> candidates, og:image) for one page; ([], None) if unreachable."""
        if url in self._page:
            return self._page[url]
        result = ([], None)
        if self.allowed(url):
            try:
                r = self.get(url)
                r.raise_for_status()
                p = _Imgs()
                p.feed(r.text)
                cands = []
                for src, alt, title, in_gallery in p.imgs:
                    full = urljoin(r.url, html.unescape(src))
                    path = urlparse(full).path
                    if path.lower().endswith(".svg") or "." not in path.rsplit("/", 1)[-1]:
                        continue   # logos/icons and tracking pixels, never a bottle photo
                    fname = re.sub(r"\.(png|jpe?g|webp|gif|avif)$", "", path.rsplit("/", 1)[-1], flags=re.I)
                    fname = re.sub(r"[-_]\d+x\d+$", "", fname)   # WordPress thumbnail suffix
                    cands.append({"label": f"{alt} {title} {fname.replace('_', ' ').replace('-', ' ')}",
                                  "image_url": full, "page_url": url, "in_gallery": in_gallery})
                result = (cands, urljoin(r.url, p.og) if p.og else None)
            except requests.RequestException:
                pass
        self._page[url] = result
        return result

    # -- matching -------------------------------------------------------------------
    def resolve(self, brand, product, source_url):
        if not source_url:
            return {"status": "no_source_url"}
        u = urlparse(source_url)
        origin = f"{u.scheme}://{u.netloc}"
        if not self.allowed(source_url):
            return {"status": "robots_disallowed", "error": f"robots.txt disallows {source_url}"}
        sc = lambda c: score(brand, product, c["label"])

        # 1. Shopify store: the whole catalog as JSON.
        shop = self._shopify(origin)
        if shop is not None:
            return self._pick(shop, sc, "shopify_json")

        # 2. A page for this one bottle: its gallery image, or og:image, is the bottle.
        imgs, og = self._html(source_url)
        if PRODUCT_PAGE.search(u.path):
            gallery = [c for c in imgs if c["in_gallery"]]
            if gallery or og:
                best = (max(gallery, key=sc) if gallery and max(map(sc, gallery)) > 0
                        else (gallery[0] if gallery else None))
                return {"status": "matched", "match_method": "product_page",
                        "match_score": PRODUCT_PAGE_SCORE,
                        "matched_label": (best["label"].strip() if best else "og:image")[:200],
                        "image_url": best["image_url"] if best else og, "page_url": source_url}

        # 3. A listing page: score every <img>; fall back to the homepage.
        cands = list(imgs)
        if not cands or max(map(sc, cands)) < MIN_SCORE:
            cands += self._html(origin + "/")[0]
        return self._pick(cands, sc, "html_scan")

    @staticmethod
    def _pick(cands, sc, method):
        if not cands:
            return {"status": "no_candidates", "match_method": method}
        best = max(cands, key=sc)
        s = sc(best)
        if s <= 0:
            return {"status": "no_match", "match_method": method, "match_score": s}
        return {"status": "matched" if s >= MIN_SCORE else "low_confidence", "match_method": method,
                "match_score": s, "matched_label": best["label"].strip()[:200],
                "image_url": best["image_url"], "page_url": best["page_url"]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Match every bottle

# COMMAND ----------

resolver = ImageResolver()
results = []
for b in sorted(bottles, key=lambda b: (b["brand"], b["product"])):
    if b["bottle_serial"] in overrides:
        res = {"status": "manual", "match_method": "override", "match_score": 1.0,
               "image_url": overrides[b["bottle_serial"]]}
    else:
        try:
            res = resolver.resolve(b["brand"], b["product"], (b["source_url"] or "").strip())
        except Exception as e:   # one broken site must not sink the whole run
            res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    results.append({**b, **res})
    print(f"{res['status']:17} {str(res.get('match_score', '')):6} {b['brand'][:20]:20} "
          f"{b['product'][:45]:45} <- {(res.get('matched_label') or '')[:50]}")

from collections import Counter
print("\n", dict(Counter(r["status"] for r in results)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download into the volume
# MAGIC `{volume}/{prefix}/{bottle_serial}.{ext}` — keyed by serial, so a re-run finds the file it
# MAGIC already has and skips it (unless `overwrite_images=true`, or the source image changed
# MAGIC because an override was added).

# COMMAND ----------

import os

DOWNLOAD = {"matched", "manual", "low_confidence"}
EXT_BY_TYPE = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp",
               "image/avif": "avif", "image/gif": "gif"}

def existing_file(serial):
    if not os.path.isdir(VOLUME_DIR):
        return None
    hits = [f for f in os.listdir(VOLUME_DIR) if f.rsplit(".", 1)[0] == serial]
    return os.path.join(VOLUME_DIR, hits[0]) if hits else None

# What the previous run downloaded, so a changed image_url (e.g. a new override) re-downloads.
previous = {}
if spark.catalog.tableExists(MANIFEST):
    previous = {r["bottle_serial"]: r["image_url"]
                for r in spark.table(MANIFEST).select("bottle_serial", "image_url").collect()}

for r in results:
    r.setdefault("volume_path", None)
    if r["status"] not in DOWNLOAD or DRY_RUN:
        continue
    have = existing_file(r["bottle_serial"])
    if have and not OVERWRITE and previous.get(r["bottle_serial"]) == r["image_url"]:
        r["volume_path"] = have
        continue
    try:
        resp = resolver.get(r["image_url"])
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not ctype.startswith("image/"):
            raise ValueError(f"not an image: Content-Type {ctype or 'missing'}")
        ext = EXT_BY_TYPE.get(ctype) or os.path.splitext(urlparse(r["image_url"]).path)[1].lstrip(".") or "img"
        if have:
            os.remove(have)
        path = os.path.join(VOLUME_DIR, f"{r['bottle_serial']}.{ext}")
        with open(path, "wb") as f:
            f.write(resp.content)
        r["volume_path"] = path
        print(f"[saved] {r['product'][:45]:45} -> {path} ({len(resp.content) // 1024} KB)")
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        r["status"] = "download_failed"
        print(f"[FAIL]  {r['product'][:45]:45} {r['image_url']} -> {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write the manifest
# MAGIC Full overwrite — the manifest is the latest scrape. Manual fixes live in the override
# MAGIC table, which this notebook only reads.

# COMMAND ----------

from pyspark.sql import functions as F

COLS = ["bottle_serial", "brand", "product", "source_url", "status", "match_method", "match_score",
        "matched_label", "image_url", "page_url", "volume_path", "error"]
SCHEMA = ", ".join(f"{c} {'double' if c == 'match_score' else 'string'}" for c in COLS)
rows = [tuple(float(r[c]) if c == "match_score" and r.get(c) is not None else r.get(c) for c in COLS)
        for r in results]

manifest = (spark.createDataFrame(rows, schema=SCHEMA)
    .withColumn("approved", F.col("status").isin("matched", "manual"))
    # Same columns archive/fetch_images_to_s3.py reads, so it can mirror these to S3.
    .withColumn("bottle_id", F.col("bottle_serial"))
    .withColumn("side", F.lit("FRONT"))
    .withColumn("cdn_url", F.col("image_url"))
    .withColumn("s3_key", F.when(F.col("volume_path").isNotNull(),
                                 F.concat(F.lit(f"{VOLUME_PREFIX}/"), F.element_at(F.split("volume_path", "/"), -1))))
    .withColumn("scraped_at", F.current_timestamp()))

if DRY_RUN:
    print(f"DRY RUN - would overwrite {MANIFEST} with {manifest.count()} rows (nothing downloaded)")
else:
    manifest.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(MANIFEST)
    print(f"overwrote {MANIFEST} with {manifest.count()} rows")

display(manifest.orderBy(F.col("approved"), "status", "brand", "product"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Needs a human
# MAGIC Everything not `approved`. For each one, look at `image_url` (`low_confidence`) or find the
# MAGIC photo yourself, then:
# MAGIC ```sql
# MAGIC INSERT INTO prod_celr.bronze.bottle_image_overrides
# MAGIC VALUES ('<bottle_serial>', 'https://...', 'why', current_timestamp());
# MAGIC ```
# MAGIC and re-run this notebook.

# COMMAND ----------

display(manifest.filter(~F.col("approved"))
        .select("bottle_serial", "brand", "product", "status", "match_score", "matched_label",
                "image_url", "source_url", "error")
        .orderBy("status", "brand", "product"))
