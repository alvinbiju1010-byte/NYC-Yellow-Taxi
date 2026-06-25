"""
NYC Yellow Taxi Trip Data Downloader
Student: Alvin Biju (ID: GH1029339)

Downloads the NYC Yellow Taxi Trip Records (January 2023) from the official
NYC Taxi & Limousine Commission (TLC) portal.

Data Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The dataset is approximately 40 MB in Parquet format (~150 MB as CSV).
The ETL pipeline auto-detects and works with both formats.
"""

import os
import sys
import urllib.request

DATA_DIR = "./data/raw/"
PARQUET_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 50)
print("NYC Yellow Taxi Data Downloader")
print("Student: Alvin Biju (GH1029339)")
print("=" * 50)

# Download the Parquet version (smaller, faster, preferred)
parquet_path = os.path.join(DATA_DIR, "yellow_tripdata_2023-01.parquet")

if os.path.exists(parquet_path):
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"✓ File already exists: {parquet_path}")
    print(f"  Size: {size_mb:.1f} MB")
else:
    print(f"Downloading Parquet from NYC TLC...")
    print(f"URL: {PARQUET_URL}")
    print("This may take 1-2 minutes (~40 MB)...")

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                print(f"\r  Progress: {pct:.0f}%", end="", flush=True)

        urllib.request.urlretrieve(PARQUET_URL, parquet_path, progress_hook)
        print()  # newline after progress
        size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"✓ Downloaded: {parquet_path}")
        print(f"  Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nPlease download the file manually:")
        print("  1. Go to: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
        print("  2. Under 'Yellow Taxi Trip Records', download January 2023 (Parquet)")
        print(f"  3. Save as: {parquet_path}")
        sys.exit(1)

print("\n" + "=" * 50)
print("✓ Dataset ready!")
print()
print("Next steps:")
print("  1. Run the ETL pipeline:     python spark_etl_pipeline.py")
print("  2. Generate visualizations:  python visualization.py")
print("=" * 50)
