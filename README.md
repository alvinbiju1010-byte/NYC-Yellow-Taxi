# NYC Yellow Taxi Trip Data — Big Data Management Project

**Apache Spark ETL Pipeline + Machine Learning Analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.x-orange)](https://spark.apache.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Student:** Alvin Biju  
**Student ID:** GH1029339  
**Course:** Big Data Management — Individual Project  
**Dataset:** [NYC Yellow Taxi Trip Records (Jan 2023)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## Project Overview

A complete big data management solution for processing, analysing, and modelling NYC Yellow Taxi trip records using **Apache Spark**. The pipeline ingests raw trip data (~3 million rows/month), performs comprehensive cleaning and feature engineering, stores results in optimised Parquet format, executes analytical queries via Spark SQL, and applies machine learning models for trip segmentation and fare prediction.

### Key Features

- **7-stage ETL Pipeline:** Load → Clean → Transform → Integrate → Store → Analyse → ML
- **6 SQL Analytical Queries:** Payment behaviour, demand patterns, trip economics, traffic analysis
- **K-Means Clustering:** Unsupervised trip segmentation with Silhouette Score evaluation
- **Random Forest Regression:** Fare prediction with RMSE, MAE, and R² evaluation
- **8 Publication-Quality Visualizations:** Matplotlib + Seaborn charts
- **83% Storage Reduction:** Parquet + Snappy compression vs raw CSV

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
├── report.md                   # Project report (source)
├── report.pdf                  # Project report (PDF)
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── data/
│   ├── raw/                    # Raw downloaded data
│   └── processed/              # Parquet output + CSV results + charts
│       └── charts/             # PNG chart files
└── LICENSE
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

### K-Means Clustering
- **Purpose:** Segment taxi trips into behavioural clusters
- **Features:** 6 normalised features (distance, duration, fare, fare/mile, speed, passengers)
- **Evaluation:** Silhouette Score + WSSSE, tested k=2 through k=6
- **Result:** Optimal k identified with well-separated clusters

### Random Forest Regression
- **Purpose:** Predict total fare from trip features
- **Features:** 6 features (distance, duration, passengers, hour, weekday, payment type)
- **Train/Test:** 80/20 split (seed=42)
- **Evaluation:** RMSE, MAE, R²
- **Result:** R² ≈ 0.85 — strong predictive power

---

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

## License

MIT — see [LICENSE](LICENSE) file.

---

## References

- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- Spark MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
