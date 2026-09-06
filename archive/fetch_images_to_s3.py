# Databricks notebook source
# MAGIC %md
# MAGIC # Fetch images to S3
# MAGIC
# MAGIC Reads the image manifest from `prod_celr.bronze.image_manifest`, downloads each
# MAGIC CDN image, and mirrors it into S3. Idempotent: skips objects that already exist
# MAGIC (single `head_object` per key, same pattern as `01_classify_and_image`).
# MAGIC
# MAGIC Manifest columns expected: `bottle_id, side, cdn_url, s3_key`.
# MAGIC
# MAGIC **Auth:** the cluster's instance profile / IAM role needs `s3:GetObject` and
# MAGIC `s3:PutObject` on the target bucket, plus outbound internet to reach CloudFront.
# MAGIC Nothing here touches Supabase.

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_celr", "Unity Catalog")
dbutils.widgets.text("bronze_schema", "bronze", "Bronze schema")
dbutils.widgets.text("manifest_table", "image_manifest", "Manifest table")
dbutils.widgets.text("bucket", "celr-prod-images", "Target S3 bucket")
dbutils.widgets.text("prefix", "catalog/unicorn/", "S3 key prefix")
dbutils.widgets.text("workers", "8", "Parallel download/upload workers")
dbutils.widgets.dropdown("overwrite", "false", ["true", "false"], "Overwrite existing objects?")
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (no upload)")
dbutils.widgets.text("aws_region", "us-east-1", "AWS region")
dbutils.widgets.text("secret_scope", "", "Databricks secret scope (blank = use instance profile)")
dbutils.widgets.text("ak_key", "aws_access_key_id", "Secret key name: access key id")
dbutils.widgets.text("sk_key", "aws_secret_access_key", "Secret key name: secret access key")

CATALOG   = dbutils.widgets.get("catalog")
SCHEMA    = dbutils.widgets.get("bronze_schema")
TABLE     = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('manifest_table')}"
BUCKET    = dbutils.widgets.get("bucket")
PREFIX    = dbutils.widgets.get("prefix")
WORKERS   = int(dbutils.widgets.get("workers"))
OVERWRITE = dbutils.widgets.get("overwrite") == "true"
DRY_RUN   = dbutils.widgets.get("dry_run") == "true"
REGION    = dbutils.widgets.get("aws_region")
SCOPE     = dbutils.widgets.get("secret_scope").strip()
AK_KEY    = dbutils.widgets.get("ak_key")
SK_KEY    = dbutils.widgets.get("sk_key")
if PREFIX and not PREFIX.endswith("/"):
    PREFIX += "/"

print(f"manifest={TABLE} bucket={BUCKET} prefix={PREFIX} overwrite={OVERWRITE} "
      f"dry_run={DRY_RUN} creds={'secret_scope' if SCOPE else 'instance_profile'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the manifest

# COMMAND ----------

from pyspark.sql import functions as F
rows = [r.asDict() for r in
        spark.table(TABLE).filter(F.upper('side') == "FRONT").select("bottle_id", "side", "cdn_url", "s3_key").collect()]
print(f"{len(rows)} images in manifest")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download from CDN, upload to S3 (skip what already exists)

# COMMAND ----------

import time
import boto3, requests
from botocore.exceptions import ClientError, NoCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed

def make_s3():
    if SCOPE:
        ak = dbutils.secrets.get(scope=SCOPE, key=AK_KEY)
        sk = dbutils.secrets.get(scope=SCOPE, key=SK_KEY)
        return boto3.client("s3", region_name=REGION,
                            aws_access_key_id=ak, aws_secret_access_key=sk)
    return boto3.client("s3", region_name=REGION)  # instance-profile / default chain

s3 = make_s3()

# Preflight: fail fast with a clear message instead of 244 identical errors.
try:
    s3.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
    print("credentials OK, bucket reachable")
except NoCredentialsError:
    raise SystemExit(
        "No AWS credentials found. Either attach an instance profile with S3 access "
        "to this cluster, or set the secret_scope widget to a Databricks scope holding "
        f"'{AK_KEY}' and '{SK_KEY}'.")
except ClientError as e:
    raise SystemExit(f"Bucket check failed ({e.response['Error']['Code']}): "
                     "verify bucket name, region, and that the identity has "
                     "s3:ListBucket / s3:GetObject / s3:PutObject.")

def object_exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound", "403"):
            return False
        raise

def handle_one(row, session):
    key = PREFIX + row["s3_key"]
    if not OVERWRITE and object_exists(key):
        return ("skip", key)
    if DRY_RUN:
        return ("would_upload", key)
    r = session.get(row["cdn_url"], timeout=30)
    r.raise_for_status()
    s3.put_object(
        Bucket=BUCKET, Key=key, Body=r.content,
        ContentType=r.headers.get("Content-Type", "image/jpeg"),
        CacheControl="public, max-age=31536000, immutable",
    )
    return ("upload", key)

session = requests.Session()
session.headers.update({"User-Agent": "celr-image-sync/1.0"})

uploaded = skipped = would = failed = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(handle_one, r, session): r for r in rows}
    for fut in as_completed(futs):
        r = futs[fut]
        try:
            action, key = fut.result()
            if action == "upload":
                uploaded += 1; print(f"[up]   s3://{BUCKET}/{key}")
            elif action == "would_upload":
                would += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {r['cdn_url']} -> {e}")

print(f"\nDone in {time.time()-t0:.1f}s  "
      f"uploaded={uploaded} would_upload={would} skipped={skipped} failed={failed}")
if DRY_RUN:
    print("DRY RUN - set dry_run=false to actually upload")

# COMMAND ----------

