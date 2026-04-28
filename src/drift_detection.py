import pandas as pd
import numpy as np


def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI).

    expected = training data
    actual = new production/scoring data

    PSI helps measure how much a feature distribution has changed.
    """

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)

    return np.sum(psi_values)


def interpret_psi(psi_value):
    """
    Interpret PSI using common monitoring rules.
    """

    if psi_value < 0.10:
        return "No significant drift"
    elif psi_value < 0.25:
        return "Moderate drift"
    else:
        return "Significant drift"


def simulate_new_month_data(training_path, output_path):
    """
    Simulate new monthly customer data with intentional drift.
    """

    df = pd.read_csv(training_path).copy()

    # Simulate changes in customer behavior
    df["monthly_premium"] = df["monthly_premium"] * 1.10
    df["digital_logins_30d"] = np.maximum(df["digital_logins_30d"] - 2, 0)
    df["service_calls_90d"] = df["service_calls_90d"] + np.random.poisson(1, len(df))

    df.to_csv(output_path, index=False)

    print(f"New month data saved to: {output_path}")


def run_drift_detection(training_path, new_data_path):
    """
    Compare training data vs new data and calculate drift.
    """

    train_df = pd.read_csv(training_path)
    new_df = pd.read_csv(new_data_path)

    numeric_features = [
        "age",
        "tenure_months",
        "monthly_premium",
        "num_products",
        "service_calls_90d",
        "late_payments_12m",
        "digital_logins_30d",
        "claims_12m",
        "satisfaction_score",
    ]

    results = []

    for feature in numeric_features:
        psi_value = calculate_psi(train_df[feature], new_df[feature])

        results.append({
            "feature": feature,
            "psi": round(psi_value, 4),
            "drift_status": interpret_psi(psi_value)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/drift_report.csv", index=False)

    print("Drift detection complete.")
    print(results_df)

    return results_df


if __name__ == "__main__":
    training_path = "data/raw/customer_churn_simulated.csv"
    new_data_path = "data/scoring/customer_churn_new_month.csv"

    simulate_new_month_data(training_path, new_data_path)
    run_drift_detection(training_path, new_data_path)