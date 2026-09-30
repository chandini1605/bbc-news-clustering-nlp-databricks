# ============================================================
# STEP 1: DATA COLLECTION
# BBC News Clustering - Azure Databricks + NLP + ML
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, split, size
import pandas as pd

spark = SparkSession.builder \
    .appName("NewsClusteringPipeline") \
    .getOrCreate()

print("=" * 60)
print("STEP 1: DATA COLLECTION")
print("BBC News Article Clustering - Azure Databricks")
print("=" * 60)

# ── 1. Load Dataset ───────────────────────────────────────────
# On Azure Databricks:
# df = spark.read.csv("dbfs:/FileStore/bbc_news.csv", header=True)
df_pandas = pd.read_csv("data/bbc_news.csv", encoding='latin-1')
df_pandas.rename(columns={'type': 'category'}, inplace=True)
df = spark.createDataFrame(df_pandas)

print(f"\n✅ Dataset Loaded!")
print(f"   Total Articles : {df.count():,}")
print(f"   Columns        : {df.columns}")

# ── 2. Preview ────────────────────────────────────────────────
print("\n📌 Sample Records:")
df.show(5, truncate=80)

# ── 3. Category Distribution ──────────────────────────────────
print("\n📌 Articles Per Category:")
df.groupBy("category").count().orderBy("count", ascending=False).show()

# ── 4. Word Count Stats ───────────────────────────────────────
df_wc = df.withColumn("word_count", size(split(col("news"), " ")))
print("📌 Avg Word Count Per Category:")
df_wc.groupBy("category").agg({"word_count": "avg"}).orderBy("avg(word_count)", ascending=False).show()

# ── 5. Data Quality Check ─────────────────────────────────────
print("\n📌 Data Quality Check:")
print(f"   Null in news    : {df.filter(col('news').isNull()).count()}")
print(f"   Null in category: {df.filter(col('category').isNull()).count()}")
print(f"   Duplicates      : {df.count() - df.dropDuplicates().count()}")

# ── 6. Save as Delta/Parquet ──────────────────────────────────
# On Databricks: df.write.format("delta").mode("overwrite").save("/mnt/news/raw")
df.write.mode("overwrite").parquet("outputs/raw_news.parquet")
print("\n✅ Saved to outputs/raw_news.parquet (Delta Lake on Databricks)")
spark.stop()
