import joblib
import pandas as pd


def assign_risk_band(risk_score: float) -> str:
    """
    Convert churn probability into a business-friendly risk category.
    """
    if risk_score >= 0.70:
        return "High Risk"
    elif risk_score >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def recommend_action(risk_band: str) -> str:
    """
    Recommend a retention action based on customer risk level.
    """
    if risk_band == "High Risk":
        return "Route to retention specialist"
    elif risk_band == "Medium Risk":
        return "Send personalized retention offer"
    else:
        return "Monitor only"


def score_customers(
    model_path: str,
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Load trained model, score customers, assign risk bands,
    and save results.
    """

    # 1. Load trained model
    model = joblib.load(model_path)

    # 2. Load customer data
    df = pd.read_csv(input_path)

    # 3. Prepare features
    X = df.drop(columns=["customer_id", "churn", "churn_probability_true"])

    # 4. Generate churn risk score
    df["churn_risk_score"] = model.predict_proba(X)[:, 1]

    # 5. Assign risk band
    df["risk_band"] = df["churn_risk_score"].apply(assign_risk_band)

    # 6. Recommend action
    df["recommended_action"] = df["risk_band"].apply(recommend_action)

    # 7. Save scored customers
    df.to_csv(output_path, index=False)

    print("Customer scoring complete.")
    print(f"Scored file saved to: {output_path}")
    print(df[[
        "customer_id",
        "churn_risk_score",
        "risk_band",
        "recommended_action"
    ]].head())

    return df


if __name__ == "__main__":
    score_customers(
        model_path="models/churn_model.pkl",
        input_path="data/raw/customer_churn_simulated.csv",
        output_path="data/scoring/customer_churn_scores.csv"
    )