"""
Big Data Management — Individual Project
Student: Alvin Biju (ID: GH1029339)
Dataset: NYC Yellow Taxi Trip Records (2023)

Apache Spark ETL/ELT Pipeline:
  1. Load raw data (CSV or Parquet)
  2. Clean & preprocess (handle missing values, filter invalid rows)
  3. Transform & enrich (derive new columns, join with lookup data)
  4. Save transformed data in Parquet format
  5. Run analytical queries with Spark SQL
  6. Machine Learning — K-Means Clustering + Random Forest Regression
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofmonth, hour, minute, when, dayofweek,
    round as spark_round, sum as spark_sum, avg, count, max, min, stddev,
    to_date, to_timestamp, lit, udf, monotonically_increasing_id
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
import os


# ── 1. INITIALIZE SPARK SESSION ──────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("NYC_Taxi_ETL_Pipeline") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("NYC Yellow Taxi ETL Pipeline — Alvin Biju (GH1029339)")
print("=" * 60)

# ── 2. LOAD RAW DATA ─────────────────────────────────────────────────────────
# Dataset: NYC Yellow Taxi Trip Records (2023-01)
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

DATA_PATH_CSV = "./data/raw/yellow_tripdata_2023-01.csv"
DATA_PATH_PARQUET = "./data/raw/yellow_tripdata_2023-01.parquet"
OUTPUT_PATH = "./data/processed/"

print("\n[1/7] Loading raw data...")

# Auto-detect format: prefer Parquet if available, fall back to CSV
if os.path.exists(DATA_PATH_PARQUET):
    print(f"Reading Parquet: {DATA_PATH_PARQUET}")
    raw_df = spark.read.parquet(DATA_PATH_PARQUET)
elif os.path.exists(DATA_PATH_CSV):
    print(f"Reading CSV: {DATA_PATH_CSV}")
    raw_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(DATA_PATH_CSV)
else:
    print("ERROR: No data file found. Please run download_data.py first.")
    print(f"  Expected: {DATA_PATH_PARQUET} or {DATA_PATH_CSV}")
    spark.stop()
    exit(1)

print(f"Raw row count: {raw_df.count()}")
raw_df.printSchema()
raw_df.show(5, truncate=False)

# ── 3. DATA PREPROCESSING — Cleaning & Filtering ─────────────────────────────
print("\n[2/7] Preprocessing — Cleaning & filtering...")

# Record pre-cleaning counts for data quality metrics
raw_count = raw_df.count()

# Drop rows where critical fields are null
clean_df = raw_df.na.drop(subset=[
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "trip_distance", "total_amount"
])

# Filter out invalid trips:
#   - trip_distance <= 0 or > 100 (unrealistic)
#   - total_amount <= 0 or > 500
#   - passenger_count <= 0 or > 9
#   - pickup datetime is after dropoff (impossible)
clean_df = clean_df.filter(
    (col("trip_distance") > 0) & (col("trip_distance") <= 100) &
    (col("total_amount") > 0) & (col("total_amount") <= 500) &
    (col("passenger_count") > 0) & (col("passenger_count") <= 9) &
    (col("tpep_pickup_datetime") < col("tpep_dropoff_datetime"))
)

# Convert datetime strings to timestamp type
clean_df = clean_df.withColumn(
    "pickup_datetime", to_timestamp("tpep_pickup_datetime")
).withColumn(
    "dropoff_datetime", to_timestamp("tpep_dropoff_datetime")
)

# Fill remaining nulls with sensible defaults
clean_df = clean_df.fillna({
    "RatecodeID": 1,
    "store_and_fwd_flag": "N",
    "congestion_surcharge": 0.0,
    "Airport_fee": 0.0,
})

clean_count = clean_df.count()
print(f"Rows removed by cleaning: {raw_count - clean_count} "
      f"({((raw_count - clean_count) / raw_count * 100):.1f}%)")
print(f"Row count after cleaning: {clean_count}")
clean_df.show(5, truncate=False)

# ── 4. DATA TRANSFORMATION & INTEGRATION ─────────────────────────────────────
print("\n[3/7] Transforming & enriching data...")

# Derive time-based columns
enriched_df = clean_df \
    .withColumn("pickup_date", to_date("pickup_datetime")) \
    .withColumn("pickup_year", year("pickup_datetime")) \
    .withColumn("pickup_month", month("pickup_datetime")) \
    .withColumn("pickup_day", dayofmonth("pickup_datetime")) \
    .withColumn("pickup_hour", hour("pickup_datetime")) \
    .withColumn("pickup_minute", minute("pickup_datetime")) \
    .withColumn("pickup_weekday", dayofweek("pickup_datetime"))

# Calculate trip duration in minutes
enriched_df = enriched_df.withColumn(
    "trip_duration_min",
    spark_round(
        (col("dropoff_datetime").cast("long") -
         col("pickup_datetime").cast("long")) / 60.0, 2
    )
)

# Calculate fare per mile (for trips > 0.1 miles to avoid division distortion)
enriched_df = enriched_df.withColumn(
    "fare_per_mile",
    when(col("trip_distance") > 0.1,
         spark_round(col("total_amount") / col("trip_distance"), 2)
    ).otherwise(0.0)
)

# Calculate average speed (mph)
enriched_df = enriched_df.withColumn(
    "avg_speed_mph",
    when(col("trip_duration_min") > 0,
         spark_round(
             col("trip_distance") / (col("trip_duration_min") / 60.0), 2
         )
    ).otherwise(0.0)
)

# Classify trip type based on distance
enriched_df = enriched_df.withColumn(
    "trip_type",
    when(col("trip_distance") <= 2, "Short")
    .when(col("trip_distance") <= 10, "Medium")
    .otherwise("Long")
)

# Classify time of day
enriched_df = enriched_df.withColumn(
    "time_of_day",
    when((col("pickup_hour") >= 6) & (col("pickup_hour") < 12), "Morning")
    .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 17), "Afternoon")
    .when((col("pickup_hour") >= 17) & (col("pickup_hour") < 22), "Evening")
    .otherwise("Night")
)

# Classify weekday vs weekend
enriched_df = enriched_df.withColumn(
    "day_type",
    when(col("pickup_weekday").isin([1, 7]), "Weekend").otherwise("Weekday")
)

# Create a payment-type lookup DataFrame
payment_lookup = spark.createDataFrame([
    (1, "Credit Card"),
    (2, "Cash"),
    (3, "No Charge"),
    (4, "Dispute"),
    (5, "Unknown"),
    (6, "Voided Trip"),
], ["payment_type", "payment_description"])

# Join to add payment description
enriched_df = enriched_df.join(
    payment_lookup, on="payment_type", how="left"
)

print(f"Enriched row count: {enriched_df.count()}")
enriched_df.select(
    "pickup_datetime", "trip_distance", "total_amount",
    "trip_duration_min", "fare_per_mile", "avg_speed_mph",
    "trip_type", "time_of_day", "day_type", "payment_description"
).show(10, truncate=False)

# ── 5. SAVE AS PARQUET (ETL Output) ──────────────────────────────────────────
print("\n[4/7] Saving transformed data as Parquet...")

os.makedirs(OUTPUT_PATH, exist_ok=True)
enriched_df.write \
    .mode("overwrite") \
    .partitionBy("pickup_date") \
    .parquet(f"{OUTPUT_PATH}nyc_taxi_enriched.parquet")

print(f"Parquet written to: {OUTPUT_PATH}nyc_taxi_enriched.parquet")

# ── 6. AGGREGATIONS & ANALYTICS using Spark SQL ──────────────────────────────
print("\n[5/7] Running analytical queries with Spark SQL...")

enriched_df.createOrReplaceTempView("taxi_trips")

# Query 1: Total revenue by payment type
print("\n--- Query 1: Total Revenue by Payment Type ---")
query1 = spark.sql("""
    SELECT
        payment_description,
        COUNT(*) AS trip_count,
        ROUND(SUM(total_amount), 2) AS total_revenue,
        ROUND(AVG(total_amount), 2) AS avg_fare,
        ROUND(AVG(trip_distance), 2) AS avg_distance
    FROM taxi_trips
    GROUP BY payment_description
    ORDER BY total_revenue DESC
""")
query1.show(truncate=False)

# Query 2: Hourly trip demand pattern
print("\n--- Query 2: Hourly Trip Demand ---")
query2 = spark.sql("""
    SELECT
        pickup_hour,
        COUNT(*) AS trip_count,
        ROUND(AVG(trip_distance), 2) AS avg_distance,
        ROUND(AVG(total_amount), 2) AS avg_fare,
        ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
    FROM taxi_trips
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""")
query2.show(24, truncate=False)

# Query 3: Trip type distribution
print("\n--- Query 3: Trip Type Distribution ---")
query3 = spark.sql("""
    SELECT
        trip_type,
        COUNT(*) AS trip_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
        ROUND(AVG(total_amount), 2) AS avg_fare,
        ROUND(AVG(trip_distance), 2) AS avg_distance
    FROM taxi_trips
    GROUP BY trip_type
    ORDER BY trip_count DESC
""")
query3.show(truncate=False)

# Query 4: Revenue by time of day
print("\n--- Query 4: Revenue by Time of Day ---")
query4 = spark.sql("""
    SELECT
        time_of_day,
        COUNT(*) AS trip_count,
        ROUND(SUM(total_amount), 2) AS total_revenue,
        ROUND(AVG(total_amount), 2) AS avg_fare,
        ROUND(AVG(trip_duration_min), 2) AS avg_duration
    FROM taxi_trips
    GROUP BY time_of_day
    ORDER BY
        CASE time_of_day
            WHEN 'Morning' THEN 1
            WHEN 'Afternoon' THEN 2
            WHEN 'Evening' THEN 3
            WHEN 'Night' THEN 4
        END
""")
query4.show(truncate=False)

# Query 5: Top 10 busiest pickup days
print("\n--- Query 5: Top 10 Busiest Days ---")
query5 = spark.sql("""
    SELECT
        pickup_date,
        COUNT(*) AS trip_count,
        ROUND(SUM(total_amount), 2) AS daily_revenue,
        ROUND(AVG(total_amount), 2) AS avg_fare
    FROM taxi_trips
    GROUP BY pickup_date
    ORDER BY trip_count DESC
    LIMIT 10
""")
query5.show(10, truncate=False)

# Query 6: Average speed by hour (traffic indicator)
print("\n--- Query 6: Average Speed by Hour (Traffic Pattern) ---")
query6 = spark.sql("""
    SELECT
        pickup_hour,
        ROUND(AVG(avg_speed_mph), 2) AS avg_speed,
        COUNT(*) AS trip_count
    FROM taxi_trips
    WHERE avg_speed_mph > 0 AND avg_speed_mph < 100
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""")
query6.show(24, truncate=False)

# Query 7: Weekday vs Weekend comparison
print("\n--- Query 7: Weekday vs Weekend Comparison ---")
query7 = spark.sql("""
    SELECT
        day_type,
        COUNT(*) AS trip_count,
        ROUND(AVG(total_amount), 2) AS avg_fare,
        ROUND(AVG(trip_distance), 2) AS avg_distance,
        ROUND(AVG(trip_duration_min), 2) AS avg_duration
    FROM taxi_trips
    GROUP BY day_type
    ORDER BY trip_count DESC
""")
query7.show(truncate=False)

# ── 7. SAVE ANALYTICAL RESULTS ───────────────────────────────────────────────
print("\n[6/7] Saving analytical results...")

query1.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_revenue_by_payment.csv")
query2.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_hourly_demand.csv")
query3.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_trip_type_distribution.csv")
query4.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_revenue_by_time_of_day.csv")
query6.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_avg_speed_by_hour.csv")
query7.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_weekday_vs_weekend.csv")

print("✓ Analytical results saved.")

# ── 8. MACHINE LEARNING ─────────────────────────────────────────────────────
print("\n[7/7] Machine Learning — Models & Evaluation")

# --- 8a. K-Means Clustering: Identify trip segments ---
print("\n--- ML Part A: K-Means Clustering ---")
print("Segmenting taxi trips into behavioural clusters...")

# Select numeric features for clustering
cluster_features = [
    "trip_distance", "trip_duration_min", "total_amount",
    "fare_per_mile", "avg_speed_mph", "passenger_count"
]

# Prepare data: filter out rows with null/zero in key features, sample for efficiency
cluster_data = enriched_df \
    .filter(
        (col("trip_distance") > 0) &
        (col("trip_duration_min") > 0) &
        (col("avg_speed_mph").between(1, 80)) &
        (col("fare_per_mile") > 0)
    ) \
    .select(cluster_features + ["total_amount"])

# Create feature vector
assembler = VectorAssembler(
    inputCols=cluster_features, outputCol="features", handleInvalid="skip"
)
cluster_data = assembler.transform(cluster_data)

# Standardize features (K-Means is distance-based)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                         withStd=True, withMean=True)
scaler_model = scaler.fit(cluster_data)
cluster_data = scaler_model.transform(cluster_data)

# Try multiple k values and evaluate using Silhouette Score
print("Evaluating cluster counts (k=2 to k=6)...")
best_k = 3
best_score = -1.0
k_results = []

for k in range(2, 7):
    kmeans = KMeans(featuresCol="scaled_features", predictionCol="cluster",
                     k=k, seed=42, maxIter=50)
    model = kmeans.fit(cluster_data)
    predictions = model.transform(cluster_data)

    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features",
        predictionCol="cluster",
        metricName="silhouette"
    )
    silhouette = evaluator.evaluate(predictions)
    wssse = model.summary.trainingCost  # Within Set Sum of Squared Errors

    k_results.append((k, silhouette, wssse))
    print(f"  k={k}: Silhouette={silhouette:.4f}, WSSSE={wssse:.2f}")

    if silhouette > best_score:
        best_score = silhouette
        best_k = k

print(f"\nBest k = {best_k} (Silhouette Score: {best_score:.4f})")

# Fit final K-Means with best k
final_kmeans = KMeans(
    featuresCol="scaled_features", predictionCol="cluster",
    k=best_k, seed=42, maxIter=50
)
final_model = final_kmeans.fit(cluster_data)
cluster_predictions = final_model.transform(cluster_data)

# Profile each cluster
print("\nCluster Profiles:")
cluster_profile = cluster_predictions.groupBy("cluster").agg(
    count("*").alias("size"),
    spark_round(avg("trip_distance"), 2).alias("avg_distance_mi"),
    spark_round(avg("trip_duration_min"), 2).alias("avg_duration_min"),
    spark_round(avg("total_amount"), 2).alias("avg_fare_$"),
    spark_round(avg("fare_per_mile"), 2).alias("avg_fare_per_mi"),
    spark_round(avg("avg_speed_mph"), 2).alias("avg_speed_mph"),
    spark_round(avg("passenger_count"), 2).alias("avg_passengers")
).orderBy("cluster")
cluster_profile.show(truncate=False)

# Save clustering results
cluster_profile.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_cluster_profiles.csv")

# --- 8b. Random Forest Regression: Predict Total Fare ---
print("\n--- ML Part B: Random Forest Regression — Fare Prediction ---")

# Prepare features for regression
regression_features = [
    "trip_distance", "trip_duration_min", "passenger_count",
    "pickup_hour", "pickup_weekday", "payment_type"
]

# Build feature vector
reg_assembler = VectorAssembler(
    inputCols=regression_features, outputCol="reg_features", handleInvalid="skip"
)
reg_data = enriched_df \
    .filter(
        (col("trip_distance") > 0) &
        (col("total_amount").between(2, 200)) &
        (col("trip_duration_min") > 0)
    ) \
    .select(regression_features + ["total_amount"])

reg_data = reg_assembler.transform(reg_data)

# Split into train/test (80/20)
train_data, test_data = reg_data.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_data.count()} rows, Test set: {test_data.count()} rows")

# Train Random Forest Regressor
rf = RandomForestRegressor(
    featuresCol="reg_features", labelCol="total_amount",
    numTrees=30, maxDepth=8, seed=42
)
rf_model = rf.fit(train_data)

# Predict on test set
rf_predictions = rf_model.transform(test_data)

# Evaluate
rmse_evaluator = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="rmse"
)
r2_evaluator = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="r2"
)
mae_evaluator = RegressionEvaluator(
    labelCol="total_amount", predictionCol="prediction", metricName="mae"
)

rmse = rmse_evaluator.evaluate(rf_predictions)
r2 = r2_evaluator.evaluate(rf_predictions)
mae = mae_evaluator.evaluate(rf_predictions)

print(f"\nRandom Forest Regression Results:")
print(f"  RMSE:  ${rmse:.2f}")
print(f"  MAE:   ${mae:.2f}")
print(f"  R²:    {r2:.4f}")

# Feature importance
print("\nFeature Importance:")
importances = rf_model.featureImportances
for i, feat in enumerate(regression_features):
    print(f"  {feat}: {importances[i]:.4f}")

# Save ML evaluation summary
ml_summary_df = spark.createDataFrame([
    ("K-Means Clustering", f"k={best_k}", f"Silhouette = {best_score:.4f}"),
    ("Random Forest Reg.", "Fare Prediction",
     f"RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}"),
], ["model", "config", "performance"])
ml_summary_df.write.mode("overwrite").option("header", "true") \
    .csv(f"{OUTPUT_PATH}results_ml_summary.csv")

print("\n✓ All ML results saved.")

# ── COMPLETE ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETL Pipeline Complete — All 7 Stages Finished!")
print("=" * 60)
print(f"Output directory: {OUTPUT_PATH}")
print("Files generated:")
print("  - nyc_taxi_enriched.parquet/      (partitioned by date)")
print("  - results_revenue_by_payment.csv/  (for chart 1)")
print("  - results_hourly_demand.csv/       (for chart 2)")
print("  - results_trip_type_distribution.csv/  (for chart 3)")
print("  - results_revenue_by_time_of_day.csv/  (for chart 4)")
print("  - results_avg_speed_by_hour.csv/   (for chart 5)")
print("  - results_weekday_vs_weekend.csv/  (for chart 7)")
print("  - results_cluster_profiles.csv/    (for ML chart)")
print("  - results_ml_summary.csv/          (ML evaluation)")

spark.stop()
