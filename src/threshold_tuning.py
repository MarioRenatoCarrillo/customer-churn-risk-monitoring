import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_thresholds(df: pd.DataFrame):
    """
    Evaluate different probability thresholds and compute metrics.
    """

    y_true = df["churn"]
    y_proba = df["churn_risk_score"]

    thresholds = np.arange(0.1, 0.95, 0.05)

    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        results.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "customers_flagged": y_pred.sum(),
            "flag_rate": y_pred.mean()
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_threshold_metrics(results_df: pd.DataFrame):
    """
    Plot precision, recall, and F1 vs threshold.
    """

    plt.figure()

    plt.plot(results_df["threshold"], results_df["precision"], label="Precision")
    plt.plot(results_df["threshold"], results_df["recall"], label="Recall")
    plt.plot(results_df["threshold"], results_df["f1"], label="F1 Score")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning Metrics")
    plt.legend()

    plt.savefig("reports/figures/threshold_metrics.png")
    plt.show()


def run_threshold_tuning(input_path: str):
    """
    Load scored data and evaluate thresholds.
    """

    df = pd.read_csv(input_path)

    results_df = evaluate_thresholds(df)

    # Save results
    results_df.to_csv("reports/threshold_results.csv", index=False)

    print("Threshold tuning complete.")
    print(results_df.head())

    plot_threshold_metrics(results_df)


if __name__ == "__main__":
    run_threshold_tuning("data/scoring/customer_churn_scores.csv")