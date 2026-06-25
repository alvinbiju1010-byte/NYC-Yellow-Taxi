"""
Data Visualization Module
Student: Alvin Biju (ID: GH1029339)

Generates charts from the processed Spark output using Matplotlib & Seaborn.
Includes ML model evaluation visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import glob

# ── Style Configuration ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

OUTPUT_DIR = "./data/processed/charts/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_csv_results(pattern):
    """Read the first CSV file matching a pattern (Spark output dirs)."""
    files = glob.glob(f"./data/processed/{pattern}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV found for pattern: {pattern}")
    return pd.read_csv(files[0])


# ── Chart 1: Total Revenue by Payment Type (Bar Chart) ───────────────────────
def chart_revenue_by_payment():
    """Bar chart showing revenue breakdown by payment method."""
    df = read_csv_results("results_revenue_by_payment")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["payment_description"], df["total_revenue"],
                  color=sns.color_palette("Set2", len(df)))

    for bar, val in zip(bars, df["total_revenue"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_title("Total Revenue by Payment Type", fontweight="bold")
    ax.set_xlabel("Payment Type")
    ax.set_ylabel("Total Revenue ($)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}01_revenue_by_payment.png")
    plt.close()
    print("✓ Chart 1 saved: revenue_by_payment.png")


# ── Chart 2: Hourly Trip Demand (Line Chart) ────────────────────────────────
def chart_hourly_demand():
    """Line chart showing trip demand pattern across hours of the day."""
    df = read_csv_results("results_hourly_demand")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(data=df, x="pickup_hour", y="trip_count", ax=ax1,
                 color="#2196F3", linewidth=2.5, marker="o", markersize=8)
    ax1.set_title("Hourly Trip Demand Pattern (NYC Yellow Taxi)", fontweight="bold")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Number of Trips", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="pickup_hour", y="avg_fare", ax=ax2,
                 color="#FF5722", linewidth=2, marker="s", markersize=6, linestyle="--")
    ax2.set_ylabel("Average Fare ($)", color="#FF5722")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    ax1.set_xticks(range(0, 24))
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}02_hourly_demand.png")
    plt.close()
    print("✓ Chart 2 saved: hourly_demand.png")


# ── Chart 3: Trip Type Distribution (Pie Chart) ──────────────────────────────
def chart_trip_type_distribution():
    """Pie chart showing short / medium / long trip breakdown."""
    df = read_csv_results("results_trip_type_distribution")

    colors = ["#4CAF50", "#2196F3", "#FF5722"]
    explode = (0.02, 0.02, 0.05)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        df["trip_count"], labels=df["trip_type"],
        autopct="%1.1f%%", startangle=140,
        colors=colors, explode=explode,
        textprops={"fontsize": 12}
    )
    ax.set_title("Trip Distribution by Distance Category", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}03_trip_type_distribution.png")
    plt.close()
    print("✓ Chart 3 saved: trip_type_distribution.png")


# ── Chart 4: Revenue by Time of Day (Grouped Bar Chart) ──────────────────────
def chart_revenue_by_time_of_day():
    """Grouped bar chart comparing revenue and trip counts by time period."""
    df = read_csv_results("results_revenue_by_time_of_day")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.35

    ax.bar([i - width/2 for i in x], df["trip_count"],
           width, label="Trip Count", color="#42A5F5")
    ax2 = ax.twinx()
    ax2.bar([i + width/2 for i in x], df["total_revenue"],
            width, label="Total Revenue ($)", color="#EF5350")

    ax.set_xticks(x)
    ax.set_xticklabels(df["time_of_day"])
    ax.set_ylabel("Number of Trips", color="#42A5F5")
    ax2.set_ylabel("Total Revenue ($)", color="#EF5350")
    ax.set_title("Trips & Revenue by Time of Day", fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}04_revenue_by_time_of_day.png")
    plt.close()
    print("✓ Chart 4 saved: revenue_by_time_of_day.png")


# ── Chart 5: Average Speed by Hour (Traffic Analysis) ───────────────────────
def chart_traffic_pattern():
    """Line chart showing average speed across hours — traffic indicator."""
    df = read_csv_results("results_avg_speed_by_hour")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvspan(7, 10, alpha=0.1, color="red", label="Morning Rush")
    ax.axvspan(16, 19, alpha=0.1, color="orange", label="Evening Rush")

    sns.lineplot(data=df, x="pickup_hour", y="avg_speed", ax=ax,
                 color="#4CAF50", linewidth=2.5, marker="D", markersize=7)

    ax.set_title("Average Taxi Speed by Hour — Traffic Pattern", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Speed (mph)")
    ax.set_xticks(range(0, 24))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}05_traffic_pattern.png")
    plt.close()
    print("✓ Chart 5 saved: traffic_pattern.png")


# ── Chart 6: Average Fare by Payment Type ────────────────────────────────────
def chart_avg_fare_by_payment():
    """Horizontal bar chart showing average fare by payment method."""
    df = read_csv_results("results_revenue_by_payment")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["payment_description"], df["avg_fare"],
                   color=sns.color_palette("Set2", len(df)))

    for bar, val in zip(bars, df["avg_fare"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"${val:.2f}", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Average Fare by Payment Type", fontweight="bold")
    ax.set_xlabel("Average Fare ($)")
    ax.set_ylabel("Payment Type")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}06_avg_fare_by_payment.png")
    plt.close()
    print("✓ Chart 6 saved: avg_fare_by_payment.png")


# ── Chart 7: Weekday vs Weekend Comparison (Grouped Bar) ─────────────────────
def chart_weekday_vs_weekend():
    """Compare trip characteristics between weekdays and weekends."""
    df = read_csv_results("results_weekday_vs_weekend")

    metrics = ["trip_count", "avg_fare", "avg_distance", "avg_duration"]
    titles = ["Trip Count", "Avg Fare ($)", "Avg Distance (mi)", "Avg Duration (min)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ["#42A5F5", "#FF7043"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2][i % 2]
        bars = ax.bar(df["day_type"], df[metric], color=colors)
        ax.set_title(title, fontweight="bold")
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:,.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Weekday vs Weekend Trip Comparison", fontweight="bold", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}07_weekday_vs_weekend.png")
    plt.close()
    print("✓ Chart 7 saved: weekday_vs_weekend.png")


# ── Chart 8: ML — Cluster Profiles ──────────────────────────────────────────
def chart_cluster_profiles():
    """Visualize K-Means cluster characteristics."""
    df = read_csv_results("results_cluster_profiles")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    feature_cols = ["avg_distance_mi", "avg_duration_min", "avg_fare_$",
                     "avg_fare_per_mi", "avg_speed_mph", "avg_passengers"]
    feature_labels = ["Avg Distance (mi)", "Avg Duration (min)", "Avg Fare ($)",
                       "Fare per Mile ($)", "Avg Speed (mph)", "Avg Passengers"]
    colors = sns.color_palette("Set2", len(df))

    for i, (col_name, label) in enumerate(zip(feature_cols, feature_labels)):
        ax = axes[i // 3][i % 3]
        bars = ax.bar(df["cluster"].astype(str), df[col_name], color=colors)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Cluster")
        for bar, val in zip(bars, df[col_name]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("K-Means Cluster Profiles — Taxi Trip Segments",
                 fontweight="bold", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}08_cluster_profiles.png")
    plt.close()
    print("✓ Chart 8 saved: cluster_profiles.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Generating Visualizations")
    print("=" * 50)

    chart_revenue_by_payment()
    chart_hourly_demand()
    chart_trip_type_distribution()
    chart_revenue_by_time_of_day()
    chart_traffic_pattern()
    chart_avg_fare_by_payment()
    chart_weekday_vs_weekend()

    # ML visualizations — skip if ML results not yet generated
    try:
        chart_cluster_profiles()
    except FileNotFoundError:
        print("⚠ ML cluster results not found — run spark_etl_pipeline.py first")

    print(f"\n✓ Charts saved to: {OUTPUT_DIR}")
    print("=" * 50)
