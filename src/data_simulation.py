import numpy as np
import pandas as pd


def simulate_customer_data(n_customers: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulate customer-level data for a churn prediction project.

    Each row represents one customer.
    The target column is 'churn':
        1 = customer churned
        0 = customer stayed
    """

    np.random.seed(random_seed)

    df = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "age": np.random.randint(21, 80, n_customers),
        "tenure_months": np.random.randint(1, 121, n_customers),
        "monthly_premium": np.random.normal(120, 35, n_customers).clip(20, 300),
        "num_products": np.random.choice([1, 2, 3, 4], size=n_customers, p=[0.45, 0.30, 0.18, 0.07]),
        "service_calls_90d": np.random.poisson(1.8, n_customers),
        "late_payments_12m": np.random.poisson(0.8, n_customers),
        "digital_logins_30d": np.random.poisson(8, n_customers),
        "claims_12m": np.random.poisson(0.5, n_customers),
        "satisfaction_score": np.random.randint(1, 11, n_customers),
        "region": np.random.choice(["North", "South", "East", "West"], size=n_customers),
        "channel": np.random.choice(["Agent", "Online", "Call Center"], size=n_customers, p=[0.50, 0.30, 0.20])
    })

    churn_logit = (
        -2.8
        + 0.015 * (df["monthly_premium"] - 120)
        - 0.018 * df["tenure_months"]
        + 0.30 * df["service_calls_90d"]
        + 0.45 * df["late_payments_12m"]
        - 0.08 * df["digital_logins_30d"]
        + 0.50 * (df["num_products"] == 1).astype(int)
        - 0.22 * df["satisfaction_score"]
    )

    churn_probability = 1 / (1 + np.exp(-churn_logit))

    df["churn_probability_true"] = churn_probability
    df["churn"] = np.random.binomial(1, churn_probability)

    return df


if __name__ == "__main__":
    customer_data = simulate_customer_data()

    customer_data.to_csv("data/raw/customer_churn_simulated.csv", index=False)

    print("Customer data created successfully.")
    print(customer_data.head())
    print("\nChurn rate:")
    print(customer_data["churn"].mean())