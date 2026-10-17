# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Sales Reasoning Agent POC — from star schema to "why did sales drop?"
# MAGIC
# MAGIC **Companion data:** `databricks_reasoning_agent_poc.xlsx` (sheets `dim_*`, `fact_*`, `business_events`, plus `Test_Questions` answer key).
# MAGIC
# MAGIC ```
# MAGIC  Delta tables (dim_*, fact_*, business_events)          §1  land data (kept light)
# MAGIC        │
# MAGIC  Views  vw_dim_*, vw_fact_sales_enriched                 §2  semantic layer
# MAGIC        │
# MAGIC  Metric view  mv_sales  (governed KPIs, YAML)            §3
# MAGIC        │
# MAGIC  ┌─────┴──────────────────────────┐
# MAGIC  Genie space  ("WHAT": chat)      UC functions ("WHY" tools)   §4 / §5
# MAGIC  └─────┬──────────────────────────┘
# MAGIC  Reasoning agent  (ResponsesAgent + managed MCP + LLM)  §6–§7
# MAGIC        │
# MAGIC  Model Serving + Review App / Playground / Databricks App §8–§10,  MLflow eval §9
# MAGIC ```
# MAGIC
# MAGIC **Why this split works:** Genie is excellent at turning a question into one SQL query (WHAT). A WHY question needs a
# MAGIC *sequence* of analyses — size the change, locate it, test causes, reconcile — so the agent orchestrates purpose-built,
# MAGIC deterministic tools (UC functions) and lets the LLM do the planning and narration, never the arithmetic.
# MAGIC
# MAGIC Runtime: serverless notebook or DBR 17.2+ (metric view YAML 1.1). Unity Catalog required.

# COMMAND ----------

# MAGIC %pip install -U -q openpyxl "mlflow[databricks]>=3.1" databricks-agents databricks-mcp "databricks-sdk[openai]"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
for e in w.serving_endpoints.list():
    if e.name.startswith("databricks-"):
        print(e.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §0 Configuration — edit these widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "sales_poc", "Schema")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-120b", "LLM serving endpoint (tool-calling capable)")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse id (for Genie + deploy resources)")
dbutils.widgets.text("genie_space_id", "", "Genie space id (fill after §4, optional)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id")
FQ = f"{CATALOG}.{SCHEMA}"
XLSX = f"/Volumes/{CATALOG}/{SCHEMA}/raw/databricks_reasoning_agent_poc.xlsx"
fq = lambda name: f"{FQ}.{name}"
print(FQ, "|", LLM_ENDPOINT, "| genie:", GENIE_SPACE_ID or "(not set)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 Land the data (light)
# MAGIC 1. Run the next cell to create the schema and a volume.
# MAGIC 2. Catalog Explorer → `<catalog>.sales_poc.raw` → **Upload to this volume** → pick the .xlsx.
# MAGIC 3. Run the load cell: every data sheet becomes a Delta table; column comments come from the `Data_Dictionary` sheet
# MAGIC    and PK/FK constraints are declared (Genie and the agent use both to understand joins).

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.raw")

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

DATA_SHEETS = ["dim_date", "dim_region", "dim_store", "dim_product", "dim_customer", "dim_promotion",
               "fact_sales", "fact_inventory_monthly", "fact_marketing_spend_monthly",
               "fact_sales_target_monthly", "business_events"]
DATE_COLS = {"dim_date": ["date"], "dim_store": ["open_date"], "dim_promotion": ["start_date", "end_date"],
             "business_events": ["start_date", "end_date"]}

for sheet in DATA_SHEETS:
    pdf = pd.read_excel(XLSX, sheet_name=sheet)
    for c in DATE_COLS.get(sheet, []):
        pdf[c] = pd.to_datetime(pdf[c]).dt.strftime("%Y-%m-%d")
    for c in pdf.columns[pdf.dtypes == object]:
        pdf[c] = pdf[c].where(pdf[c].notna(), None)
    sdf = spark.createDataFrame(pdf)
    for c in DATE_COLS.get(sheet, []):
        sdf = sdf.withColumn(c, F.to_date(c))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(sheet))
    print(f"{sheet:32s} {sdf.count():>8,} rows")

# column comments from the data dictionary
dd = pd.read_excel(XLSX, sheet_name="Data_Dictionary")
for r in dd.itertuples(index=False):
    comment = str(r.Description).replace("'", "\\'")
    spark.sql(f"ALTER TABLE {fq(r.Table)} ALTER COLUMN {r.Column} COMMENT '{comment}'")

# informational PK / FK constraints (not enforced; they teach Genie and the agent the joins)
PKS = {"dim_date": "date_key", "dim_region": "country_code", "dim_store": "store_id", "dim_product": "product_id",
       "dim_customer": "customer_id", "dim_promotion": "promo_id", "fact_sales": "sales_line_id"}
FKS = [("fact_sales", "date_key", "dim_date"), ("fact_sales", "store_id", "dim_store"),
       ("fact_sales", "customer_id", "dim_customer"), ("fact_sales", "product_id", "dim_product"),
       ("fact_sales", "promo_id", "dim_promotion"), ("dim_store", "country_code", "dim_region")]
for t, c in PKS.items():
    try:
        spark.sql(f"ALTER TABLE {fq(t)} ALTER COLUMN {c} SET NOT NULL")
        spark.sql(f"ALTER TABLE {fq(t)} ADD CONSTRAINT pk_{t} PRIMARY KEY ({c})")
    except Exception as e:
        print("skip PK", t, str(e)[:80])
for t, c, ref in FKS:
    try:
        spark.sql(f"ALTER TABLE {fq(t)} ADD CONSTRAINT fk_{t}_{c} FOREIGN KEY ({c}) REFERENCES {fq(ref)}")
    except Exception as e:
        print("skip FK", t, c, str(e)[:80])

# COMMAND ----------

# MAGIC %md
# MAGIC ## §2 Semantic layer — dimension and fact views
# MAGIC Thin, business-friendly views. Put business logic here once (new-store flag, price band, key-account flag)
# MAGIC so Genie, the metric view and the agent all inherit it. `vw_fact_sales_enriched` is the single wide view
# MAGIC the metric view and the reasoning functions sit on.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_dim_store')}
COMMENT 'Stores and sales channels with country. is_new_store = opened in 2026 (not like-for-like).'
AS SELECT s.store_id, s.store_name, s.city, s.channel, s.open_date, r.country_code, r.country, r.sub_region, r.country_manager,
          s.open_date >= DATE'2026-01-01' AS is_new_store
FROM {fq('dim_store')} s JOIN {fq('dim_region')} r ON s.country_code = r.country_code""")

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_dim_product')}
COMMENT 'Products with category, brand and price band.'
AS SELECT product_id, product_name, category, brand, list_price, unit_cost,
          CASE WHEN list_price >= 500 THEN 'Premium' WHEN list_price >= 50 THEN 'Mid' ELSE 'Everyday' END AS price_band
FROM {fq('dim_product')}""")

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_dim_customer')}
COMMENT 'Customers with segment; is_key_account flags the large B2B contracts.'
AS SELECT c.customer_id, c.customer_name, c.segment, c.loyalty_tier, r.country, c.segment = 'Key Account' AS is_key_account
FROM {fq('dim_customer')} c JOIN {fq('dim_region')} r ON c.country_code = r.country_code""")

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_dim_date')} COMMENT 'Calendar with GCC weekend (Fri/Sat) and Ramadan flags.'
AS SELECT * FROM {fq('dim_date')}""")

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_fact_sales_enriched')}
COMMENT 'One row per order line with all dimension attributes. Main analysis source. Country = country of the selling store.'
AS SELECT
  f.sales_line_id, f.order_id,
  d.date AS order_date, d.year, d.quarter, d.year_quarter, d.year_month, d.month_name, d.iso_week, d.is_weekend, d.is_ramadan,
  s.store_id, s.store_name, s.city, s.channel, s.is_new_store, s.country_code, s.country, s.sub_region,
  p.product_id, p.product_name, p.category, p.brand, p.price_band,
  c.customer_id, c.customer_name, c.segment, c.is_key_account,
  f.promo_id, pr.promo_name, f.promo_id IS NOT NULL AS is_promo,
  f.quantity, f.list_price, f.unit_price, f.discount_pct, f.gross_sales, f.discount_amount, f.net_sales, f.cogs,
  f.net_sales - f.cogs AS gross_margin
FROM {fq('fact_sales')} f
JOIN {fq('dim_date')} d      ON f.date_key = d.date_key
JOIN {fq('vw_dim_store')} s  ON f.store_id = s.store_id
JOIN {fq('vw_dim_product')} p ON f.product_id = p.product_id
JOIN {fq('vw_dim_customer')} c ON f.customer_id = c.customer_id
LEFT JOIN {fq('dim_promotion')} pr ON f.promo_id = pr.promo_id""")

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_fact_inventory')} COMMENT 'Monthly stock and stockout days by country and product.'
AS SELECT i.month_key, i.year_month, r.country, p.product_name, p.category, i.units_sold, i.closing_stock_units, i.stockout_days
FROM {fq('fact_inventory_monthly')} i JOIN {fq('dim_region')} r ON i.country_code = r.country_code
JOIN {fq('dim_product')} p ON i.product_id = p.product_id""")

spark.sql(f"""
CREATE OR REPLACE VIEW {fq('vw_business_events')} COMMENT 'Business context log: promotions, price changes, stockouts, competitor moves, churn, store openings, Ramadan.'
AS SELECT e.*, COALESCE(r.country, 'ALL') AS country FROM {fq('business_events')} e
LEFT JOIN {fq('dim_region')} r ON e.country_code = r.country_code""")

display(spark.sql(f"SELECT year_month, ROUND(SUM(net_sales)) AS net_sales FROM {fq('vw_fact_sales_enriched')} GROUP BY ALL ORDER BY 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## §3 Metric view — one governed definition of every KPI
# MAGIC Dimensions = how you slice, measures = how you aggregate. Genie, dashboards and SQL users all get the same
# MAGIC "Net Sales". Query it with `MEASURE()`. Synonyms help Genie map business words ("revenue", "turnover") to measures.
# MAGIC If your runtime only supports YAML 0.1, the cell retries without `comment`/`synonyms` automatically.

# COMMAND ----------

import re

MV_YAML = f"""
version: 1.1
comment: "Governed sales KPIs for the GCC retail POC. Amounts in AED. Country = selling store country."
source: {fq('vw_fact_sales_enriched')}
dimensions:
  - name: order_date
    expr: order_date
  - name: year
    expr: year
  - name: year_quarter
    expr: year_quarter
  - name: year_month
    expr: year_month
  - name: order_month
    expr: DATE_TRUNC('MONTH', order_date)
    comment: "First day of the order month - use for trends"
  - name: is_ramadan
    expr: is_ramadan
  - name: is_weekend
    expr: is_weekend
  - name: country
    expr: country
    synonyms: ["market", "region"]
  - name: sub_region
    expr: sub_region
  - name: channel
    expr: channel
    comment: "Retail, Online or B2B"
  - name: store_name
    expr: store_name
    synonyms: ["store", "branch"]
  - name: is_new_store
    expr: is_new_store
  - name: category
    expr: category
    synonyms: ["product category"]
  - name: product_name
    expr: product_name
    synonyms: ["product", "SKU", "item"]
  - name: brand
    expr: brand
  - name: price_band
    expr: price_band
  - name: segment
    expr: segment
    synonyms: ["customer segment"]
  - name: customer_name
    expr: customer_name
    synonyms: ["customer", "account"]
  - name: promo_name
    expr: COALESCE(promo_name, 'No promotion')
    synonyms: ["promotion", "campaign"]
measures:
  - name: net_sales
    expr: SUM(net_sales)
    comment: "Revenue after discounts (default meaning of sales)"
    synonyms: ["sales", "revenue", "turnover"]
  - name: gross_sales
    expr: SUM(gross_sales)
  - name: discount_amount
    expr: SUM(discount_amount)
  - name: discount_rate
    expr: SUM(discount_amount) / NULLIF(SUM(gross_sales), 0)
  - name: cogs
    expr: SUM(cogs)
  - name: gross_margin
    expr: SUM(net_sales) - SUM(cogs)
    synonyms: ["profit", "margin"]
  - name: gross_margin_pct
    expr: (SUM(net_sales) - SUM(cogs)) / NULLIF(SUM(net_sales), 0)
  - name: units
    expr: SUM(quantity)
    synonyms: ["volume", "quantity"]
  - name: orders
    expr: COUNT(DISTINCT order_id)
  - name: active_customers
    expr: COUNT(DISTINCT customer_id)
  - name: avg_selling_price
    expr: SUM(net_sales) / NULLIF(SUM(quantity), 0)
    synonyms: ["ASP", "average price"]
  - name: avg_order_value
    expr: SUM(net_sales) / NULLIF(COUNT(DISTINCT order_id), 0)
    synonyms: ["AOV", "basket size"]
  - name: promo_sales_share
    expr: SUM(CASE WHEN promo_id IS NOT NULL THEN net_sales END) / NULLIF(SUM(net_sales), 0)
"""

def create_metric_view(yaml_text):
    spark.sql(f"CREATE OR REPLACE VIEW {fq('mv_sales')} WITH METRICS LANGUAGE YAML AS $$\n{yaml_text}\n$$")

try:
    create_metric_view(MV_YAML)
except Exception as e:
    print("YAML 1.1 failed, retrying as 0.1:", str(e)[:200])
    y = re.sub(r"^\s*(comment|synonyms):.*\n", "", MV_YAML, flags=re.M).replace("version: 1.1", "version: 0.1")
    create_metric_view(y)

display(spark.sql(f"""
SELECT country, year_quarter, MEASURE(net_sales) AS net_sales, MEASURE(gross_margin_pct) AS gm_pct, MEASURE(units) AS units
FROM {fq('mv_sales')} WHERE year_quarter IN ('2025-Q3','2026-Q3') GROUP BY ALL ORDER BY country, year_quarter"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## §4 Genie space — the chat layer for WHAT questions (UI, ~10 minutes)
# MAGIC 1. **New → Genie space**. Warehouse: a serverless SQL warehouse.
# MAGIC 2. **Data:** add `mv_sales`, `vw_fact_sales_enriched`, `vw_dim_store`, `vw_dim_product`, `vw_dim_customer`, `vw_dim_date`,
# MAGIC    `vw_fact_inventory`, `vw_business_events`, `fact_sales_target_monthly`, `fact_marketing_spend_monthly`.
# MAGIC 3. **Instructions → General instructions:** paste the text printed by the next cell.
# MAGIC 4. **Instructions → SQL queries (trusted):** add the sample queries printed by the next cell, each with its question as title.
# MAGIC 5. **Sample questions:** paste Test_Questions 1, 2 and 8 from the workbook.
# MAGIC 6. **Benchmarks:** add the same three with their expected SQL, run them, iterate on instructions until they pass.
# MAGIC 7. Copy the space id from the URL (`/genie/rooms/<id>`) into the `genie_space_id` widget — the agent will use Genie as a tool.
# MAGIC
# MAGIC Try in Genie: *"Net sales by country Q3 2026 vs Q3 2025"* (works) and *"Why is KSA down in August?"* — Genie returns
# MAGIC a breakdown but not a reasoned cause. That gap is what §5–§7 close.

# COMMAND ----------

GENIE_INSTRUCTIONS = f"""
- This is a GCC retailer (UAE, KSA, Qatar, Kuwait, Oman). Currency AED. Data: 2025-01-01 to 2026-09-30.
- "Sales" means net sales. Prefer the metric view mv_sales and its measures (net_sales, units, gross_margin_pct, avg_selling_price...).
- Country refers to the country of the selling store.
- "Last year" comparisons = same period one year earlier. "YTD 2026" = 2026-01-01 to 2026-09-30.
- Channels: Retail, Online, B2B. Key accounts are customers with segment = 'Key Account'.
- Like-for-like (LFL) excludes stores where is_new_store = true.
- Targets live in fact_sales_target_monthly at country_code x category x month; join to dim_region for country names.
- For context on why numbers moved, show rows from vw_business_events overlapping the period.
- Always show amounts rounded to whole AED and percentages with one decimal.
"""
SAMPLE_SQL = {
    "Net sales by country, Q3 2026 vs Q3 2025": f"""SELECT country, year_quarter, MEASURE(net_sales) AS net_sales
FROM {fq('mv_sales')} WHERE year_quarter IN ('2025-Q3','2026-Q3') GROUP BY ALL ORDER BY country, year_quarter""",
    "Monthly net sales trend": f"""SELECT order_month, MEASURE(net_sales) AS net_sales FROM {fq('mv_sales')} GROUP BY ALL ORDER BY order_month""",
    "Target attainment by country for a quarter": f"""WITH tgt AS (
  SELECT r.country, SUM(t.target_net_sales) AS target
  FROM agentpoc.sales_poc.fact_sales_target_monthly t
  JOIN agentpoc.sales_poc.dim_region r ON t.country_code = r.country_code
  WHERE t.year_month BETWEEN '2026-07' AND '2026-09'
  GROUP BY r.country
),
act AS (
  SELECT country, SUM(net_sales) AS actual
  FROM agentpoc.sales_poc.vw_fact_sales_enriched
  WHERE year_quarter = '2026-Q3'
  GROUP BY country
)
SELECT tgt.country,
       ROUND(act.actual, 0) AS actual,
       ROUND(tgt.target, 0) AS target,
       ROUND(act.actual / tgt.target * 100, 1) AS attainment_pct
FROM tgt LEFT JOIN act ON tgt.country = act.country
ORDER BY attainment_pct""",
}
print("=== General instructions ===", GENIE_INSTRUCTIONS)
for q, sql in SAMPLE_SQL.items():
    print(f"=== {q} ===\n{sql}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §5 Reasoning tools — Unity Catalog SQL functions
# MAGIC Each function is one analytical step of a root-cause investigation. They are deterministic, governed and
# MAGIC unit-testable; the LLM only chooses which to call and with what arguments. Function and parameter **comments
# MAGIC become the tool descriptions** the model sees, so they are written as instructions.
# MAGIC
# MAGIC | Tool | Reasoning step | Answers |
# MAGIC |---|---|---|
# MAGIC | `fn_compare_periods` | SIZE | How big is the change; is it units, price, orders or customers? |
# MAGIC | `fn_driver_breakdown` | LOCATE | Which country / channel / category / product / customer / store / promo explains it? |
# MAGIC | `fn_price_volume_mix` | EXPLAIN | Did we sell more, charge more, or sell a different mix? |
# MAGIC | `fn_business_events` | EXPLAIN | What happened in that period and place (price change, churn, competitor, stockout, promo, Ramadan)? |
# MAGIC | `fn_stockouts` | EXPLAIN | Were we unable to sell because of supply? |
# MAGIC | `fn_marketing_spend` | EXPLAIN | Did marketing investment change? |
# MAGIC | `fn_target_attainment` | CONTEXT | Are we on / off plan? |

# COMMAND ----------


V = fq("vw_fact_sales_enriched")
FILTERS = """
  AND (p_country = 'ALL' OR country = p_country OR country_code = p_country)
  AND (p_category = 'ALL' OR category = p_category)
  AND (p_channel = 'ALL' OR channel = p_channel)"""
PERIOD_PARAMS = """
  p_a_start DATE COMMENT 'Baseline period A start date, YYYY-MM-DD',
  p_a_end   DATE COMMENT 'Baseline period A end date (inclusive)',
  p_b_start DATE COMMENT 'Comparison period B start date (the period being explained)',
  p_b_end   DATE COMMENT 'Comparison period B end date (inclusive)'"""
FILTER_PARAMS = """
  p_country  STRING DEFAULT 'ALL' COMMENT 'Country name (UAE, KSA, Qatar, Kuwait, Oman) or ALL',
  p_category STRING DEFAULT 'ALL' COMMENT 'Category (Electronics, Home Appliances, Beverages, Snacks, Personal Care, Apparel) or ALL',
  p_channel  STRING DEFAULT 'ALL' COMMENT 'Retail, Online, B2B or ALL'"""
 
# 1) SIZE
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_compare_periods')}({PERIOD_PARAMS},{FILTER_PARAMS})
RETURNS TABLE (metric STRING, period_a DOUBLE, period_b DOUBLE, abs_change DOUBLE, pct_change DOUBLE)
COMMENT 'STEP 1 of a why-analysis. Compares KPIs between baseline period A and period B with optional filters: net_sales, units, orders, active_customers, avg_selling_price, avg_order_value, gross_margin_pct, discount_rate. Use it to size a change and see whether it is volume-, price- or customer-driven.'
RETURN
WITH base AS (
  SELECT IF(order_date BETWEEN p_a_start AND p_a_end, 'A', 'B') AS per, net_sales, quantity, order_id, customer_id,
         gross_margin, discount_amount, gross_sales
  FROM {V}
  WHERE (order_date BETWEEN p_a_start AND p_a_end OR order_date BETWEEN p_b_start AND p_b_end) {FILTERS}
), agg AS (
  SELECT
    CAST(SUM(IF(per='A', net_sales, 0)) AS DOUBLE) AS ns_a,       CAST(SUM(IF(per='B', net_sales, 0)) AS DOUBLE) AS ns_b,
    CAST(SUM(IF(per='A', quantity, 0)) AS DOUBLE) AS q_a,         CAST(SUM(IF(per='B', quantity, 0)) AS DOUBLE) AS q_b,
    CAST(COUNT(DISTINCT IF(per='A', order_id, NULL)) AS DOUBLE) AS o_a,    CAST(COUNT(DISTINCT IF(per='B', order_id, NULL)) AS DOUBLE) AS o_b,
    CAST(COUNT(DISTINCT IF(per='A', customer_id, NULL)) AS DOUBLE) AS c_a, CAST(COUNT(DISTINCT IF(per='B', customer_id, NULL)) AS DOUBLE) AS c_b,
    CAST(SUM(IF(per='A', gross_margin, 0)) AS DOUBLE) AS gm_a,    CAST(SUM(IF(per='B', gross_margin, 0)) AS DOUBLE) AS gm_b,
    CAST(SUM(IF(per='A', discount_amount, 0)) AS DOUBLE) AS d_a,  CAST(SUM(IF(per='B', discount_amount, 0)) AS DOUBLE) AS d_b,
    CAST(SUM(IF(per='A', gross_sales, 0)) AS DOUBLE) AS g_a,      CAST(SUM(IF(per='B', gross_sales, 0)) AS DOUBLE) AS g_b
  FROM base
), s AS (
  SELECT stack(8,
    'net_sales', ns_a, ns_b,
    'units', q_a, q_b,
    'orders', o_a, o_b,
    'active_customers', c_a, c_b,
    'avg_selling_price', ns_a / NULLIF(q_a, 0), ns_b / NULLIF(q_b, 0),
    'avg_order_value', ns_a / NULLIF(o_a, 0), ns_b / NULLIF(o_b, 0),
    'gross_margin_pct', gm_a / NULLIF(ns_a, 0), gm_b / NULLIF(ns_b, 0),
    'discount_rate', d_a / NULLIF(g_a, 0), d_b / NULLIF(g_b, 0)) AS (metric, period_a, period_b)
  FROM agg)
SELECT metric, ROUND(period_a, 4), ROUND(period_b, 4), ROUND(period_b - period_a, 4), ROUND((period_b - period_a) / NULLIF(period_a, 0), 4)
FROM s""")
 
# 2) LOCATE
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_driver_breakdown')}({PERIOD_PARAMS},
  p_dimension STRING COMMENT 'One of: country, sub_region, channel, store, city, category, brand, product, segment, customer, promo',{FILTER_PARAMS},
  p_top INT DEFAULT 10 COMMENT 'How many members to return, ranked by absolute change')
RETURNS TABLE (rnk INT, dimension STRING, member STRING, sales_a DOUBLE, sales_b DOUBLE, abs_change DOUBLE, pct_change DOUBLE, share_of_total_change DOUBLE)
COMMENT 'STEP 2 of a why-analysis. Splits the net sales change between period A and B by one dimension and ranks members by absolute change. share_of_total_change shows how much of the total change each member explains (can exceed 1 when other members move the opposite way). Drill down by calling again with a filter on the top member (e.g. first by country, then category within that country, then product).'
RETURN
WITH base AS (
  SELECT CASE lower(p_dimension)
           WHEN 'country' THEN country WHEN 'sub_region' THEN sub_region WHEN 'channel' THEN channel
           WHEN 'store' THEN store_name WHEN 'city' THEN city WHEN 'category' THEN category WHEN 'brand' THEN brand
           WHEN 'product' THEN product_name WHEN 'segment' THEN segment WHEN 'customer' THEN customer_name
           WHEN 'promo' THEN COALESCE(regexp_replace(promo_name, '[ ]*20[0-9][0-9]$', ''), 'No promotion')
           ELSE 'UNKNOWN DIMENSION - use country, channel, category, product, customer, store, segment, brand, city, promo' END AS member,
         IF(order_date BETWEEN p_a_start AND p_a_end, net_sales, 0) AS s_a,
         IF(order_date BETWEEN p_b_start AND p_b_end, net_sales, 0) AS s_b
  FROM {V}
  WHERE (order_date BETWEEN p_a_start AND p_a_end OR order_date BETWEEN p_b_start AND p_b_end) {FILTERS}
), g AS (
  SELECT member, CAST(SUM(s_a) AS DOUBLE) AS sales_a, CAST(SUM(s_b) AS DOUBLE) AS sales_b FROM base GROUP BY member
), r AS (
  SELECT member, sales_a, sales_b, sales_b - sales_a AS chg,
         (sales_b - sales_a) / NULLIF(sales_a, 0) AS pct,
         (sales_b - sales_a) / NULLIF(SUM(sales_b - sales_a) OVER (), 0) AS share,
         ROW_NUMBER() OVER (ORDER BY ABS(sales_b - sales_a) DESC) AS rn
  FROM g)
SELECT CAST(rn AS INT), p_dimension, member, ROUND(sales_a, 0), ROUND(sales_b, 0), ROUND(chg, 0), ROUND(pct, 4), ROUND(share, 4)
FROM r WHERE rn <= p_top ORDER BY rn""")
 
# 3) EXPLAIN: price / volume / mix
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_price_volume_mix')}({PERIOD_PARAMS},{FILTER_PARAMS})
RETURNS TABLE (effect STRING, amount DOUBLE, note STRING)
COMMENT 'STEP 3 of a why-analysis. Decomposes the net sales change between A and B into price effect (net price per unit, incl. list price changes and discounts), volume effect (more or fewer units), mix effect (shift between cheaper and pricier products), new products and lost products. The effects add up to total_change. Most meaningful when filtered to one category, because units of different categories are not comparable.'
RETURN
WITH base AS (
  SELECT product_id,
    IF(order_date BETWEEN p_a_start AND p_a_end, net_sales, 0) AS ns_a,
    IF(order_date BETWEEN p_b_start AND p_b_end, net_sales, 0) AS ns_b,
    IF(order_date BETWEEN p_a_start AND p_a_end, quantity, 0) AS qty_a,
    IF(order_date BETWEEN p_b_start AND p_b_end, quantity, 0) AS qty_b
  FROM {V}
  WHERE (order_date BETWEEN p_a_start AND p_a_end OR order_date BETWEEN p_b_start AND p_b_end) {FILTERS}
), prod AS (
  SELECT product_id,
    CAST(SUM(ns_a) AS DOUBLE) AS s_a, CAST(SUM(ns_b) AS DOUBLE) AS s_b,
    CAST(SUM(qty_a) AS DOUBLE) AS q_a, CAST(SUM(qty_b) AS DOUBLE) AS q_b
  FROM base GROUP BY product_id
), cont AS (
  SELECT *, s_a / q_a AS pr_a, s_b / q_b AS pr_b FROM prod WHERE q_a > 0 AND q_b > 0
), t AS (
  SELECT SUM(q_a) AS qa, SUM(q_b) AS qb, SUM(s_a) AS sa,
         SUM((pr_b - pr_a) * q_b) AS price_eff, SUM((q_b - q_a) * pr_a) AS vol_mix FROM cont
), n AS (
  SELECT SUM(s_b) - SUM(s_a) AS total, SUM(IF(q_a = 0, s_b, 0)) AS new_s, SUM(IF(q_b = 0, -s_a, 0)) AS lost_s FROM prod
)
SELECT effect, ROUND(amount, 0), note FROM (
  SELECT stack(6,
    'total_change',  total, 'Net sales B minus A',
    'price_effect',  price_eff, 'Change in net price per unit x period-B units',
    'volume_effect', (qb - qa) * sa / qa, 'Change in total units x baseline average price',
    'mix_effect',    vol_mix - (qb - qa) * sa / qa, 'Shift between cheaper and pricier products',
    'new_products',  new_s, 'Sales of products not sold in period A',
    'lost_products', lost_s, 'Period-A sales of products not sold in period B') AS (effect, amount, note)
  FROM t CROSS JOIN n)""")
 
# 4) EXPLAIN: business events
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_business_events')}(
  p_start DATE COMMENT 'Window start (use the start of period A to see what differs between periods)',
  p_end   DATE COMMENT 'Window end (use the end of period B)',
  p_country  STRING DEFAULT 'ALL' COMMENT 'Country name or ALL; events scoped to ALL countries are always returned',
  p_category STRING DEFAULT 'ALL' COMMENT 'Category or ALL')
RETURNS TABLE (event_id STRING, start_date DATE, end_date DATE, event_type STRING, country STRING, category_scope STRING, entity_id STRING, description STRING, expected_impact STRING)
COMMENT 'Evidence for causes. Returns logged business events overlapping a window: promotions, price changes, stockouts, competitor openings, customer churn, store openings, Ramadan timing. Cite event_id in answers. entity_id is a store_id or product_id when the event concerns one.'
RETURN
SELECT e.event_id, e.start_date, e.end_date, e.event_type, COALESCE(r.country, 'ALL'), e.category_scope, e.entity_id, e.description, e.expected_impact
FROM {fq('business_events')} e LEFT JOIN {fq('dim_region')} r ON e.country_code = r.country_code
WHERE e.start_date <= p_end AND COALESCE(e.end_date, DATE'9999-12-31') >= p_start
  AND (p_country = 'ALL' OR e.country_code = 'ALL' OR r.country = p_country OR e.country_code = p_country)
  AND (p_category = 'ALL' OR e.category_scope = 'ALL' OR array_contains(split(e.category_scope, '[|]'), p_category))
ORDER BY e.start_date""")
 
# 5) EXPLAIN: supply
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_stockouts')}(
  p_start DATE COMMENT 'Start date; whole months are used',
  p_end   DATE COMMENT 'End date; whole months are used',
  p_country  STRING DEFAULT 'ALL' COMMENT 'Country name or ALL',
  p_category STRING DEFAULT 'ALL' COMMENT 'Category or ALL',
  p_min_days INT DEFAULT 3 COMMENT 'Only return product-months with at least this many stockout days')
RETURNS TABLE (year_month STRING, country STRING, product_name STRING, category STRING, stockout_days INT, units_sold INT, units_sold_same_month_last_year INT)
COMMENT 'Checks supply as a cause: product x country x month combinations with stockout days, with units sold vs the same month last year. Large stockouts on big products explain lost sales.'
RETURN
SELECT i.year_month, r.country, p.product_name, p.category, CAST(i.stockout_days AS INT), CAST(i.units_sold AS INT), CAST(ly.units_sold AS INT)
FROM {fq('fact_inventory_monthly')} i
JOIN {fq('dim_region')} r ON i.country_code = r.country_code
JOIN {fq('dim_product')} p ON i.product_id = p.product_id
LEFT JOIN {fq('fact_inventory_monthly')} ly ON ly.country_code = i.country_code AND ly.product_id = i.product_id AND ly.month_key = i.month_key - 100
WHERE i.month_key BETWEEN CAST(date_format(p_start, 'yyyyMM') AS INT) AND CAST(date_format(p_end, 'yyyyMM') AS INT)
  AND i.stockout_days >= p_min_days
  AND (p_country = 'ALL' OR r.country = p_country OR r.country_code = p_country)
  AND (p_category = 'ALL' OR p.category = p_category)
ORDER BY i.stockout_days DESC, i.units_sold DESC
LIMIT 50""")
 
# 6) EXPLAIN: marketing
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_marketing_spend')}({PERIOD_PARAMS},
  p_country STRING DEFAULT 'ALL' COMMENT 'Country name or ALL')
RETURNS TABLE (country STRING, channel STRING, spend_a DOUBLE, spend_b DOUBLE, abs_change DOUBLE, pct_change DOUBLE)
COMMENT 'Checks marketing investment as a cause: marketing spend by country and channel (Retail, Online) in period A vs B. Whole months are used.'
RETURN
WITH base AS (
  SELECT r.country, ms.channel,
    IF(ms.month_key BETWEEN CAST(date_format(p_a_start,'yyyyMM') AS INT) AND CAST(date_format(p_a_end,'yyyyMM') AS INT), ms.marketing_spend, 0) AS sp_a,
    IF(ms.month_key BETWEEN CAST(date_format(p_b_start,'yyyyMM') AS INT) AND CAST(date_format(p_b_end,'yyyyMM') AS INT), ms.marketing_spend, 0) AS sp_b
  FROM {fq('fact_marketing_spend_monthly')} ms JOIN {fq('dim_region')} r ON ms.country_code = r.country_code
  WHERE p_country = 'ALL' OR r.country = p_country OR r.country_code = p_country
), m AS (
  SELECT country, channel, CAST(SUM(sp_a) AS DOUBLE) AS a, CAST(SUM(sp_b) AS DOUBLE) AS b
  FROM base GROUP BY country, channel)
SELECT country, channel, a, b, b - a, ROUND((b - a) / NULLIF(a, 0), 4) FROM m ORDER BY ABS(b - a) DESC""")
 
# 7) CONTEXT: targets
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_target_attainment')}(
  p_start DATE COMMENT 'First day of the first month',
  p_end   DATE COMMENT 'Last day of the last month',
  p_country STRING DEFAULT 'ALL' COMMENT 'Country name or ALL',
  p_by STRING DEFAULT 'country' COMMENT 'country = one row per country; category = country x category')
RETURNS TABLE (country STRING, category STRING, actual DOUBLE, target DOUBLE, attainment DOUBLE, gap DOUBLE)
COMMENT 'Actual net sales vs target for whole months between p_start and p_end, lowest attainment first. 2026 targets = 2025 actuals + 8%.'
RETURN
WITH tgt AS (
  SELECT r.country, tm.category, SUM(tm.target_net_sales) AS target
  FROM {fq('fact_sales_target_monthly')} tm JOIN {fq('dim_region')} r ON tm.country_code = r.country_code
  WHERE tm.month_key BETWEEN CAST(date_format(p_start,'yyyyMM') AS INT) AND CAST(date_format(p_end,'yyyyMM') AS INT)
  GROUP BY r.country, tm.category),
act AS (
  SELECT country, category, SUM(net_sales) AS actual FROM {V}
  WHERE order_date BETWEEN p_start AND p_end GROUP BY country, category),
lvl AS (
  SELECT tgt.country, IF(p_by = 'category', tgt.category, 'ALL') AS grp_category,
         COALESCE(act.actual, 0) AS actual, tgt.target
  FROM tgt LEFT JOIN act ON tgt.country = act.country AND tgt.category = act.category
  WHERE p_country = 'ALL' OR tgt.country = p_country),
j AS (
  SELECT country, grp_category AS category, CAST(SUM(actual) AS DOUBLE) AS actual, CAST(SUM(target) AS DOUBLE) AS target
  FROM lvl GROUP BY country, grp_category)
SELECT country, category, ROUND(actual, 0), ROUND(target, 0), ROUND(actual / NULLIF(target, 0), 4), ROUND(actual - target, 0)
FROM j ORDER BY actual / NULLIF(target, 0)""")
 
# 8) LOCATE: scan all dimensions in one call
spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('fn_dimension_scan')}({PERIOD_PARAMS},{FILTER_PARAMS})
RETURNS TABLE (dimension STRING, top_member STRING, top_member_change DOUBLE, total_change DOUBLE, top_member_share DOUBLE, guidance STRING)
COMMENT 'STEP 2 - CALL THIS FIRST to locate a change. For each dimension (country, channel, category, segment, product, customer) returns the single member with the largest change and its share of the total change. A product or customer with share >= 0.5 is the primary driver.'
RETURN
WITH base AS (
  SELECT country, channel, category, segment, product_name, customer_name,
         IF(order_date BETWEEN p_a_start AND p_a_end, net_sales, 0) AS s_a,
         IF(order_date BETWEEN p_b_start AND p_b_end, net_sales, 0) AS s_b
  FROM {V}
  WHERE (order_date BETWEEN p_a_start AND p_a_end OR order_date BETWEEN p_b_start AND p_b_end) {FILTERS}
), lng AS (
  SELECT d.dimension, d.member, b.s_a, b.s_b
  FROM base b
  LATERAL VIEW stack(6,
    'country', b.country, 'channel', b.channel, 'category', b.category,
    'segment', b.segment, 'product', b.product_name, 'customer', b.customer_name) d AS dimension, member
), g AS (
  SELECT dimension, member, SUM(s_b) - SUM(s_a) AS chg FROM lng GROUP BY dimension, member
), tot AS (
  SELECT SUM(s_b) - SUM(s_a) AS total FROM base
), r AS (
  SELECT g.*, ROW_NUMBER() OVER (PARTITION BY dimension ORDER BY ABS(chg) DESC) AS rn FROM g
)
SELECT r.dimension, r.member, ROUND(r.chg, 0), ROUND(t.total, 0), ROUND(r.chg / NULLIF(t.total, 0), 4),
       CASE WHEN r.chg / NULLIF(t.total, 0) >= 0.5 THEN 'Dominant: explain this member (events, stockouts, PVM)'
            ELSE 'Spread: not concentrated in one member of this dimension' END
FROM r CROSS JOIN tot t
WHERE r.rn = 1
ORDER BY ABS(r.chg / NULLIF(t.total, 0)) DESC""")
 
TOOL_FUNCTIONS = ["fn_compare_periods", "fn_driver_breakdown", "fn_price_volume_mix", "fn_business_events",
                  "fn_stockouts", "fn_marketing_spend", "fn_target_attainment", "fn_dimension_scan"]
print("created:", TOOL_FUNCTIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unit-test the tools by hand — this is the investigation the agent should reproduce for Test_Question 3
# MAGIC ("Why are KSA sales down in August 2026 vs August 2025?")

# COMMAND ----------

A, B = ("2025-08-01", "2025-08-31"), ("2026-08-01", "2026-08-31")
args = f"DATE'{A[0]}', DATE'{A[1]}', DATE'{B[0]}', DATE'{B[1]}'"
display(spark.sql(f"SELECT * FROM {fq('fn_compare_periods')}({args}, 'KSA')"))                                # SIZE
display(spark.sql(f"SELECT * FROM {fq('fn_driver_breakdown')}({args}, 'category', 'KSA')"))                   # LOCATE level 1
display(spark.sql(f"SELECT * FROM {fq('fn_driver_breakdown')}({args}, 'product', 'KSA', 'Electronics')"))     # LOCATE level 2
display(spark.sql(f"SELECT * FROM {fq('fn_driver_breakdown')}({args}, 'channel', 'KSA', 'Home Appliances')")) # competitor hits retail?
display(spark.sql(f"SELECT * FROM {fq('fn_stockouts')}(DATE'2026-07-01', DATE'2026-08-31', 'KSA')"))         # supply
display(spark.sql(f"SELECT * FROM {fq('fn_business_events')}(DATE'{A[0]}', DATE'{B[1]}', 'KSA', 'Electronics')"))  # evidence
display(spark.sql(f"SELECT * FROM {fq('fn_dimension_scan')}({args}, 'KSA')"))
display(spark.sql(f"SELECT * FROM {fq('fn_dimension_scan')}(DATE'2025-07-01', DATE'2025-09-30', DATE'2026-07-01', DATE'2026-09-30', 'Qatar')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## §6 The reasoning agent
# MAGIC * **Pattern:** a tool-calling loop (ReAct style) wrapped in MLflow's `ResponsesAgent`, so it deploys to Model Serving,
# MAGIC   works in AI Playground / Review App and is traced end to end.
# MAGIC * **Tools via Databricks-managed MCP servers:** all UC functions in this schema (`/api/2.0/mcp/functions/{catalog}/{schema}`)
# MAGIC   plus the Genie space (`/api/2.0/mcp/genie/{space_id}`) if you set the widget. Add a function to the schema → the agent sees it.
# MAGIC * **The reasoning lives in the system prompt:** FRAME → SIZE → LOCATE → EXPLAIN → RECONCILE → ANSWER, with
# MAGIC   "every number must come from a tool" as the anti-hallucination rule. Read and tune it in `agent.py`.

# COMMAND ----------

import yaml
with open("agent_config.yml", "w") as f:
    yaml.safe_dump({"catalog": CATALOG, "schema": SCHEMA, "llm_endpoint": LLM_ENDPOINT,
                    "genie_space_id": GENIE_SPACE_ID, "max_steps": 20}, f)
print(open("agent_config.yml").read())

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC """Sales reasoning agent: answers WHAT (via Genie / metric view) and WHY (via UC-function reasoning tools).
# MAGIC
# MAGIC Tools are served by Databricks-managed MCP servers:
# MAGIC   - UC functions  : {host}/api/2.0/mcp/functions/{catalog}/{schema}
# MAGIC   - Genie (opt.)  : {host}/api/2.0/mcp/genie/{genie_space_id}
# MAGIC """
# MAGIC import json
# MAGIC import uuid
# MAGIC from concurrent.futures import ThreadPoolExecutor
# MAGIC from typing import Any, Generator
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_mcp import DatabricksMCPClient
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC cfg = mlflow.models.ModelConfig(development_config="agent_config.yml")
# MAGIC CATALOG = cfg.get("catalog")
# MAGIC SCHEMA = cfg.get("schema")
# MAGIC LLM_ENDPOINT = cfg.get("llm_endpoint")
# MAGIC GENIE_SPACE_ID = cfg.get("genie_space_id") or ""
# MAGIC MAX_STEPS = int(cfg.get("max_steps") or 14)
# MAGIC
# MAGIC # DatabricksMCPClient uses asyncio.run() internally. Notebooks (and some servers) already run an
# MAGIC # event loop, which makes asyncio.run() fail. Running each MCP call on a worker thread gives it a
# MAGIC # fresh, loop-free thread, so it works the same in the notebook and on Model Serving.
# MAGIC _MCP_POOL = ThreadPoolExecutor(max_workers=4)
# MAGIC
# MAGIC def _mcp(fn, *args):
# MAGIC     return _MCP_POOL.submit(fn, *args).result()
# MAGIC
# MAGIC def _schema(t):
# MAGIC     """Tool input schema, whatever the MCP library version calls the field."""
# MAGIC     s = None
# MAGIC     for attr in ("inputSchema", "input_schema", "parameters"):
# MAGIC         s = getattr(t, attr, None)
# MAGIC         if s:
# MAGIC             break
# MAGIC     if s is None and hasattr(t, "model_dump"):
# MAGIC         d = t.model_dump()
# MAGIC         s = d.get("inputSchema") or d.get("input_schema") or d.get("parameters")
# MAGIC     if hasattr(s, "model_dump"):
# MAGIC         s = s.model_dump()
# MAGIC     return s or {"type": "object", "properties": {}}
# MAGIC
# MAGIC def _content_text(content):
# MAGIC     """Plain text of an LLM reply. Reasoning models (e.g. gpt-oss) return a list of parts:
# MAGIC     [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}]. Keep only the text parts."""
# MAGIC     if content is None:
# MAGIC         return ""
# MAGIC     if isinstance(content, str):
# MAGIC         return content
# MAGIC     out = []
# MAGIC     for p in content:
# MAGIC         if not isinstance(p, dict):
# MAGIC             p = p.model_dump() if hasattr(p, "model_dump") else {"type": "text", "text": str(p)}
# MAGIC         if p.get("type") in ("text", "output_text") and p.get("text"):
# MAGIC             out.append(p["text"])
# MAGIC     return "\n".join(out)
# MAGIC
# MAGIC SYSTEM_PROMPT = f"""You are a senior sales analyst for a GCC retailer (UAE, KSA, Qatar, Kuwait, Oman).
# MAGIC Data covers 2025-01-01 to 2026-09-30. Currency is AED. "Sales" means NET SALES unless the user says otherwise.
# MAGIC Today is after the data ends, so "this year" = 2026, "last month" = 2026-09, "last quarter" = 2026-Q3.
# MAGIC
# MAGIC You answer two kinds of questions:
# MAGIC 1. WHAT questions (totals, trends, rankings): get the number with one tool call and answer briefly.
# MAGIC    Use the Genie tool if available for ad-hoc slicing; otherwise fn_compare_periods / fn_driver_breakdown.
# MAGIC 2. WHY questions (why is X up/down): follow this method and do NOT skip steps.
# MAGIC    a. FRAME   – turn the question into two explicit periods (A = baseline, B = current) and filters.
# MAGIC                Default baseline for month/quarter questions: same period last year, unless the user says "vs last month".
# MAGIC    b. SIZE    – fn_compare_periods: how big is the change (AED and %), and is it volume, price or customer-count driven?
# MAGIC    c. LOCATE  – fn_dimension_scan first (one call scans all dimensions), then fn_driver_breakdown to confirm
# MAGIC                and drill into the dominant member. Stop drilling when one member explains most of its parent's change.
# MAGIC    d. EXPLAIN – for the top drivers look for causes: fn_business_events (price changes, churn, competitors,
# MAGIC                store openings, promotions, Ramadan timing), fn_stockouts (supply), fn_price_volume_mix
# MAGIC                (price vs volume vs mix), fn_marketing_spend, and promotions via fn_driver_breakdown(dimension='promo').
# MAGIC    e. RECONCILE – check the drivers you name add up to most (≥70%) of the total change. Say what is unexplained.
# MAGIC    f. ANSWER  – format:
# MAGIC         **Headline:** one sentence with the total change.
# MAGIC         **Drivers:** a ranked list, each with AED impact, share of change, and the evidence (event id, stockout days, PVM split).
# MAGIC         **Not the cause:** anything you checked that was flat (builds trust).
# MAGIC         **Confidence:** High / Medium / Low and why.
# MAGIC         **Next question to ask:** one suggestion.
# MAGIC
# MAGIC Rules:
# MAGIC - Every number you state must come from a tool result in this conversation. Never invent figures or events.
# MAGIC - Distinguish evidence ("event EV08 records a stockout") from inference ("this likely explains").
# MAGIC - Country filters accept the country name (UAE, KSA, Qatar, Kuwait, Oman). Dimension names for fn_driver_breakdown:
# MAGIC   country, sub_region, channel, store, city, category, brand, product, segment, customer, promo.
# MAGIC - Dates are 'YYYY-MM-DD'. Keep tool calls purposeful; you have at most {MAX_STEPS} steps.
# MAGIC - When the question names a country, category or channel, ALWAYS pass it (p_country / p_category / p_channel) to EVERY tool call.
# MAGIC   Check the headline total is for that filter before continuing.
# MAGIC - TIMING questions (one period up and the adjacent period down, or "why did X move between months"):
# MAGIC   call fn_business_events for the whole window FIRST and look for Seasonal/timing events (e.g. Ramadan).
# MAGIC   Then compare the COMBINED window (e.g. Feb+Mar 2026 vs Feb+Mar 2025) to test whether demand only shifted.
# MAGIC   Multiple categories: call with p_category='ALL' rather than one call per category.
# MAGIC     This overrides the LOCATE rule below.
# MAGIC - LOCATE: call fn_dimension_scan FIRST. The most specific dimension (product or customer) with top_member_share >= 0.5
# MAGIC   is the primary driver: confirm it with fn_driver_breakdown on that dimension, then explain it with events/stockouts/PVM.
# MAGIC   If no member reaches 0.5, drill with fn_driver_breakdown into the dimension with the highest share.
# MAGIC - If fn_business_events names a specific customer, product or store, quantify that entity directly with
# MAGIC   fn_driver_breakdown on customer/product/store before attributing anything else.
# MAGIC - If fn_business_events returns a stockout, call fn_stockouts to quantify it (stockout days, units vs last year).
# MAGIC - The Drivers list must be non-overlapping: one level only, shares must sum to <=100%. Do not list a parent and its child.
# MAGIC - Confidence = High only if named, evidenced drivers explain >=70% of the change; otherwise Medium/Low.
# MAGIC - PVM effects (price/volume/mix) explain a driver; never list them as separate drivers alongside products.
# MAGIC   After the top product, quantify the remaining change (total minus top product) and attribute it to the next cause.
# MAGIC - Never estimate or approximate an impact. If a cause cannot be quantified with a tool, say "not quantified".
# MAGIC """
# MAGIC
# MAGIC
# MAGIC class SalesReasoningAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self.w = WorkspaceClient()
# MAGIC         self.llm = self.w.serving_endpoints.get_open_ai_client()
# MAGIC         host = self.w.config.host.rstrip("/")
# MAGIC         self.servers = [f"{host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"]
# MAGIC         if GENIE_SPACE_ID:
# MAGIC             self.servers.append(f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}")
# MAGIC         self._tools = None  # lazy: name -> (server_url, openai_spec)
# MAGIC
# MAGIC     # ---------- tools ----------
# MAGIC     def tools(self):
# MAGIC         if self._tools is None:
# MAGIC             self._tools = {}
# MAGIC             for url in self.servers:
# MAGIC                 client = DatabricksMCPClient(server_url=url, workspace_client=self.w)
# MAGIC                 for t in _mcp(client.list_tools):
# MAGIC                     self._tools[t.name] = (url, {
# MAGIC                         "type": "function",
# MAGIC                         "function": {
# MAGIC                             "name": t.name,
# MAGIC                             "description": (t.description or "")[:1000],
# MAGIC                             "parameters": _schema(t),
# MAGIC                         },
# MAGIC                     })
# MAGIC         return self._tools
# MAGIC
# MAGIC     def resolve_tool(self, name: str):
# MAGIC         """Map what the model called (e.g. 'fn_compare_periods') to the MCP tool name
# MAGIC         (e.g. 'agentpoc__sales_poc__fn_compare_periods')."""
# MAGIC         tools = self.tools()
# MAGIC         if name in tools:
# MAGIC             return name
# MAGIC         short = name.split("__")[-1].split(".")[-1]
# MAGIC         for full in tools:
# MAGIC             if full.split("__")[-1] == short:
# MAGIC                 return full
# MAGIC         return None
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def call_tool(self, name: str, args: dict) -> str:
# MAGIC         full = self.resolve_tool(name)
# MAGIC         if full is None:
# MAGIC             short_names = [n.split("__")[-1] for n in self.tools()]
# MAGIC             return f"TOOL ERROR: unknown tool '{name}'. Available tools: {short_names}"
# MAGIC         url, spec = self.tools()[full]
# MAGIC         name = full
# MAGIC         # Reject misspelled / unknown arguments instead of letting the function silently fall back to defaults
# MAGIC         # (e.g. 'country' instead of 'p_country' would otherwise run for ALL countries).
# MAGIC         props = (spec.get("function", {}).get("parameters") or {}).get("properties") or {}
# MAGIC         if props:
# MAGIC             unknown = [k for k in args if k not in props]
# MAGIC             if unknown:
# MAGIC                 return (f"TOOL ERROR: unknown argument(s) {unknown} for {name.split('__')[-1]}. "
# MAGIC                         f"Valid arguments: {list(props)}. Call again with the correct names.")
# MAGIC         client = DatabricksMCPClient(server_url=url, workspace_client=self.w)
# MAGIC         try:
# MAGIC             res = _mcp(client.call_tool, name, args)
# MAGIC             text = "\n".join(getattr(c, "text", str(c)) for c in res.content)
# MAGIC         except Exception as e:  # return errors to the model so it can correct its call
# MAGIC             text = f"TOOL ERROR: {e}"
# MAGIC         return text[:20000]
# MAGIC
# MAGIC     # ---------- message conversion ----------
# MAGIC     @staticmethod
# MAGIC     def _text(content: Any) -> str:
# MAGIC         if isinstance(content, str):
# MAGIC             return content
# MAGIC         parts = []
# MAGIC         for p in content or []:
# MAGIC             p = p if isinstance(p, dict) else p.model_dump()
# MAGIC             if p.get("text"):
# MAGIC                 parts.append(p["text"])
# MAGIC         return "\n".join(parts)
# MAGIC
# MAGIC     def to_chat(self, items) -> list[dict]:
# MAGIC         msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
# MAGIC         for it in items:
# MAGIC             it = it if isinstance(it, dict) else it.model_dump()
# MAGIC             typ = it.get("type", "message")
# MAGIC             if typ == "function_call":
# MAGIC                 msgs.append({"role": "assistant", "content": None, "tool_calls": [{
# MAGIC                     "id": it["call_id"], "type": "function",
# MAGIC                     "function": {"name": it["name"], "arguments": it["arguments"]}}]})
# MAGIC             elif typ == "function_call_output":
# MAGIC                 msgs.append({"role": "tool", "tool_call_id": it["call_id"], "content": it["output"]})
# MAGIC             elif it.get("role") in ("user", "assistant", "system"):
# MAGIC                 msgs.append({"role": it["role"], "content": self._text(it.get("content"))})
# MAGIC         return msgs
# MAGIC
# MAGIC     # ---------- agent loop ----------
# MAGIC     def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         msgs = self.to_chat(request.input)
# MAGIC         specs = [spec for _, spec in self.tools().values()]
# MAGIC         for _ in range(MAX_STEPS):
# MAGIC             resp = self.llm.chat.completions.create(model=LLM_ENDPOINT, messages=msgs, tools=specs, temperature=0)
# MAGIC             m = resp.choices[0].message
# MAGIC             if not m.tool_calls:
# MAGIC                 text = _content_text(m.content)
# MAGIC                 if not text.strip():
# MAGIC                     # Reasoning models sometimes stop after "thinking" without calling a tool or answering.
# MAGIC                     msgs.append({"role": "user", "content": "Continue. Call the next tool you need now, "
# MAGIC                                  "or if you have enough evidence, write the final answer in the required format."})
# MAGIC                     continue
# MAGIC                 yield ResponsesAgentStreamEvent(type="response.output_item.done",
# MAGIC                                                 item=self.create_text_output_item(text=text, id=str(uuid.uuid4())))
# MAGIC                 return
# MAGIC             msgs.append({"role": "assistant", "content": _content_text(m.content) or None, "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
# MAGIC             for tc in m.tool_calls:
# MAGIC                 yield ResponsesAgentStreamEvent(type="response.output_item.done", item=self.create_function_call_item(
# MAGIC                     id=str(uuid.uuid4()), call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments))
# MAGIC                 try:
# MAGIC                     args = json.loads(tc.function.arguments or "{}")
# MAGIC                 except json.JSONDecodeError:
# MAGIC                     args = {}
# MAGIC                 out = self.call_tool(tc.function.name, args)
# MAGIC                 msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
# MAGIC                 yield ResponsesAgentStreamEvent(type="response.output_item.done",
# MAGIC                                                 item=self.create_function_call_output_item(call_id=tc.id, output=out))
# MAGIC         yield ResponsesAgentStreamEvent(type="response.output_item.done", item=self.create_text_output_item(
# MAGIC             text="I hit the step limit before finishing. Try narrowing the question (one country or one period).",
# MAGIC             id=str(uuid.uuid4())))
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [e.item for e in self.predict_stream(request) if e.type == "response.output_item.done"]
# MAGIC         return ResponsesAgentResponse(output=outputs)
# MAGIC
# MAGIC
# MAGIC AGENT = SalesReasoningAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## §7 Test interactively
# MAGIC Run a question, then open the **MLflow trace** below the cell output: you will see the plan unfold —
# MAGIC compare → breakdown by category → breakdown by product → stockouts → events → answer. Compare with the
# MAGIC `Test_Questions` sheet. If a step is skipped, tighten the system prompt, not the tools.

# COMMAND ----------

import importlib, sys
if "agent" in sys.modules:
    importlib.reload(sys.modules["agent"])
from agent import AGENT

def ask(question: str) -> str:
    resp = AGENT.predict({"input": [{"role": "user", "content": question}]})
    last = resp.output[-1]
    last = last.model_dump() if hasattr(last, "model_dump") else last
    return last["content"][0]["text"]

print(ask("What were total net sales for Jan–Sep 2026 and how does that compare with the same period last year?"))

# COMMAND ----------

print(ask("Why are KSA sales down in August 2026 compared with August 2025?"))

# COMMAND ----------

print(ask("Why did Qatar decline in Q3 2026 vs Q3 2025?"))

# COMMAND ----------

print(ask("Snacks and beverages grew strongly in Feb 2026 but fell in March. Why?"))

# COMMAND ----------

print(ask("Beverages revenue is up April–September 2026 vs last year, but are we selling more? Explain."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## §8 Log, register, deploy
# MAGIC `resources` lets Model Serving issue short-lived credentials for every endpoint, function, table, warehouse and
# MAGIC Genie space the agent touches (automatic auth passthrough) — no PATs in code.

# COMMAND ----------

TOOL_FUNCTIONS = ["fn_compare_periods", "fn_driver_breakdown", "fn_price_volume_mix", "fn_business_events",
                  "fn_stockouts", "fn_marketing_spend", "fn_target_attainment", "fn_dimension_scan"]
DATA_SHEETS = ["dim_date", "dim_region", "dim_store", "dim_product", "dim_customer", "dim_promotion",
               "fact_sales", "fact_inventory_monthly", "fact_marketing_spend_monthly",
               "fact_sales_target_monthly", "business_events"]

# COMMAND ----------

import mlflow
from mlflow.models.resources import (DatabricksServingEndpoint, DatabricksFunction, DatabricksTable,
                                     DatabricksSQLWarehouse, DatabricksGenieSpace)

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]
resources += [DatabricksFunction(function_name=fq(fn)) for fn in TOOL_FUNCTIONS]
resources += [DatabricksTable(table_name=fq(t)) for t in DATA_SHEETS + [
    "vw_fact_sales_enriched", "vw_dim_store", "vw_dim_product", "vw_dim_customer", "vw_dim_date",
    "vw_fact_inventory", "vw_business_events", "mv_sales"]]
if WAREHOUSE_ID:
    resources.append(DatabricksSQLWarehouse(warehouse_id=WAREHOUSE_ID))
if GENIE_SPACE_ID:
    resources.append(DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID))

example = {"input": [{"role": "user", "content": "Why did Qatar decline in Q3 2026 vs Q3 2025?"}]}
with mlflow.start_run(run_name="sales_reasoning_agent"):
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        model_config="agent_config.yml",
        input_example=example,
        resources=resources,
        pip_requirements=["mlflow[databricks]>=3.1", "databricks-agents", "databricks-mcp", "databricks-sdk[openai]"],
    )

# pre-deployment check in a fresh environment
mlflow.models.predict(model_uri=info.model_uri, input_data=example, env_manager="uv")

# COMMAND ----------

from databricks import agents
mlflow.set_registry_uri("databricks-uc")
UC_MODEL = fq("sales_reasoning_agent")
mv = mlflow.register_model(info.model_uri, UC_MODEL)
deployment = agents.deploy(UC_MODEL, mv.version, tags={"poc": "sales-reasoning"})
print("Endpoint:", deployment.endpoint_name)
print("Review App:", deployment.review_app_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §9 Evaluate with the answer key
# MAGIC The `Test_Questions` sheet becomes an eval set. `Correctness` checks the answer against the expected answer,
# MAGIC `Guidelines` checks the reasoning format. Re-run after every prompt or tool change and compare runs in MLflow.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

tq = pd.read_excel(XLSX, sheet_name="Test_Questions")
eval_data = [{"inputs": {"question": r["Question (ask the agent exactly this)"]},
              "expectations": {"expected_response": r["Expected answer (answer key)"]}}
             for _, r in tq.iterrows()]

def predict_fn(question: str) -> str:
    return ask(question)

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Guidelines(name="quantified_drivers",
                   guidelines="For why-questions the answer must name the main drivers with an AED amount for each and cite evidence such as an event id, stockout days or a price/volume split."),
        Guidelines(name="no_invented_causes",
                   guidelines="The answer must not claim causes that are not supported by numbers or events shown in the response."),
    ],
)
display(results.tables["eval_results"])

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM agentpoc.sales_poc.fn_driver_breakdown(
# MAGIC   DATE'2025-08-01', DATE'2025-08-31', DATE'2026-08-01', DATE'2026-08-31', 'promo', 'KSA')

# COMMAND ----------

# MAGIC %md
# MAGIC ## §10 Front end and next steps
# MAGIC **Chat UI options**
# MAGIC * **AI Playground / Review App** — zero code; share the Review App link with business testers to collect feedback.
# MAGIC * **Databricks Apps → chatbot template** — point it at the agent endpoint for a branded chat app with SSO.
# MAGIC * **Agent Bricks → Multi-Agent Supervisor** — a no-code alternative: add the Genie space + the UC functions as
# MAGIC   sub-agents/tools and give it the same instructions; useful as a baseline to compare with this custom agent.
# MAGIC
# MAGIC **Hardening for a real pilot**
# MAGIC 1. Replace the Excel load with your gold tables; keep the view/metric-view/function contracts unchanged.
# MAGIC 2. Feed `business_events` from real sources (pricing system, promo calendar, supply alerts, CRM churn flags) — the agent is only as good as its evidence.
# MAGIC 3. Add tools as you add causes: weather, web traffic/conversion, competitor prices, forecast vs actual.
# MAGIC 4. Grant the agent's service principal `USE`/`SELECT`/`EXECUTE` only on this schema; row filters apply through the views.
# MAGIC 5. Track eval scores per release; add every bad real-world answer to the eval set.