import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def simulate_monthly_performance():
    """
    Simulate model performance over time (like production monitoring).
    """

    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

    performance_data = pd.DataFrame({
        "month": months,
        "auc": [0.82, 0.80, 0.78, 0.75, 0.72],
        "precision": [0.65, 0.63, 0.60, 0.57, 0.54],
        "recall": [0.55, 0.52, 0.50, 0.47, 0.43],
        "f1": [0.59, 0.57, 0.55, 0.51, 0.48],
        "high_risk_rate": [0.25, 0.27, 0.30, 0.34, 0.38]
    })

    return performance_data


def plot_performance_trends(df: pd.DataFrame):
    """
    Plot model performance over time.
    """

    plt.figure()

    plt.plot(df["month"], df["auc"], marker="o", label="AUC")
    plt.plot(df["month"], df["precision"], marker="o", label="Precision")
    plt.plot(df["month"], df["recall"], marker="o", label="Recall")
    plt.plot(df["month"], df["f1"], marker="o", label="F1 Score")

    plt.xlabel("Month")
    plt.ylabel("Metric Value")
    plt.title("Model Performance Over Time")
    plt.legend()

    plt.xticks(rotation=45)

    plt.savefig("reports/figures/performance_trends.png")
    plt.show()


def detect_performance_drop(df: pd.DataFrame):
    """
    Identify if performance degradation is happening.
    """

    alerts = []

    # Simple rules for alerts
    if df["auc"].iloc[-1] < df["auc"].iloc[0] - 0.05:
        alerts.append("AUC dropped significantly")

    if df["recall"].iloc[-1] < df["recall"].iloc[0] - 0.10:
        alerts.append("Recall dropped significantly")

    if df["high_risk_rate"].iloc[-1] > 0.35:
        alerts.append("Too many customers flagged as high risk")

    return alerts


def run_performance_monitoring():
    """
    Run full performance monitoring workflow.
    """

    df = simulate_monthly_performance()

    # Save metrics
    df.to_csv("reports/performance_over_time.csv", index=False)

    print("Performance data:")
    print(df)

    # Plot trends
    plot_performance_trends(df)

    # Detect issues
    alerts = detect_performance_drop(df)

    print("\nMonitoring Alerts:")
    if alerts:
        for alert in alerts:
            print("⚠️", alert)
    else:
        print("No major issues detected.")


if __name__ == "__main__":
    run_performance_monitoring()