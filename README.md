# NYC Yellow Taxi Trip Data — Big Data Management Project

**Apache Spark ETL Pipeline + Machine Learning Analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.x-orange)](https://spark.apache.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Student:** Alvin Biju  
**Student ID:** GH1029339  
**Dataset:** [NYC Yellow Taxi Trip Records (Jan 2023)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## Project Overview

So it’s basically a all-inclusive big data management system, that leans on **Apache Spark** to process, examine, and even predict on NYC Yellow Taxi trip data. The pipeline brings in the raw trip dataset (around three million rows each month), then it cleans it up, builds features, and stores everything in a tuned Parquet format. After that, it runs analytic questions with Spark SQL, and later on it trains machine learning models to kind of split up the trips into clusters and foresee the fares, sort of anticipates what they might end up being.

### Key Features

The seven-step ETL pipeline goes like load , clean, transform, integrate , store , analyze , and machine learning. Trip economics , payment behavior , demand patterns and traffic analysis are the six SQL analytical queries .  

For the first part: K-Means Clustering, kind of unsupervised trip segmentation , with Silhouette Score analysis to see what “fits” better . There’s also fare prediction, where we use Random Forest Regression and then judge it with RMSE, MAE , and R2 analysis.  

Matplotlib + Seaborn charts make up eight publication quality visualizations. And there’s this **83 percent Storage Reduction** bit , where Parquet plus Snappy compression beats unprocessed CSV in size .
### Technology Stack

| Layer | Technology |
|-------|-----------|
| Distributed Processing | Apache Spark 3.x (PySpark) |
| Storage | Apache Parquet (columnar, Snappy compressed) |
| SQL Analytics | Spark SQL |
| Machine Learning | Spark MLlib (K-Means, Random Forest) |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.8+ |

---

## Quick Start

### Prerequisites

- Python 3.8+
- Java 8 or 11 (required by PySpark)
- 4+ GB RAM recommended

### 1. Clone the Repository

```bash
git clone https://github.com/alvinbiju1010-byte/nyc-taxi-spark-pipeline.git
cd nyc-taxi-spark-pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

```bash
python download_data.py
```

Downloads ~40 MB Parquet file from the NYC TLC official portal.

### 4. Run the ETL Pipeline

```bash
python spark_etl_pipeline.py
```

Executes all 7 stages: load, clean, transform, store, SQL analytics, K-Means clustering, and Random Forest regression. Results are saved to `./data/processed/`.

### 5. Generate Visualizations

```bash
python visualization.py
```

Produces 8 charts in `./data/processed/charts/`.

---

## Project Structure

```
nyc-taxi-spark-pipeline/
├── spark_etl_pipeline.py      # Main ETL pipeline (7 stages + ML)
├── visualization.py            # Chart generation (8 visualizations)
├── download_data.py            # Dataset downloader
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules

```

---

## ETL Pipeline Stages

| Stage | Description | Key Operations |
|-------|-------------|---------------|
| **1. Load** | Read raw data | Parquet/CSV auto-detection, schema inference |
| **2. Clean** | Data quality | Null removal, range validation, temporal checks |
| **3. Transform** | Feature engineering | 10 derived columns, categorical classification |
| **4. Integrate** | Data enrichment | Payment type lookup join |
| **5. Store** | Optimised output | Date-partitioned Parquet with Snappy compression |
| **6. Analyse** | Spark SQL | 6 business intelligence queries |
| **7. ML** | Machine learning | K-Means clustering + Random Forest regression |

---

## Machine Learning Models

K-Means Clustering ###Divide taxi rides into behavioral groupings, basically.
We used six normalized features (distance, time, fare, fare/mile, speed, passengers) and threw them into the model. Evaluation used Silhouette Score plus WSSSE, and we tested from k=2 to k=6. After a bit, it turned out that the clusters were pretty well separated and the best k came out as the optimum, or at least that’s what the metrics suggested.


Regression w Random Forests, the aim is sort of to anticipate the overall cost from trip things. You know, distance, duration, passengers, hour, weekday, and payment type these are the six features used. Then we did a split like 80/20, seed =42 , with training and testing. For checks we looked at RMSE, MAE, and R2, and the final number R2 = 0.85 shows excellent predictive power, so basically it’s working really well.

## Key Findings

1. **Payment:** Credit cards dominate (~65-70% of trips), cash for shorter rides
2. **Demand:** Bimodal daily pattern — morning peak (7-9 AM) + evening peak (4-7 PM)
3. **Trip Types:** Short trips (<2 mi) = 55% volume; long trips (>10 mi) generate highest per-trip revenue
4. **Traffic:** Speeds drop from ~18 mph to ~12 mph during rush hours
5. **Weekend Effect:** 15-20% lower volume but higher average fares on weekends

---

## Performance

- **Processing Time:** ~3 minutes for 3 million rows (single node, consumer hardware)
- **Storage:** 150 MB CSV → 25 MB Parquet (83% reduction)
- **Query Speed:** Sub-second on date-filtered queries (partition pruning)

---


---

## References

- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- Spark MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
