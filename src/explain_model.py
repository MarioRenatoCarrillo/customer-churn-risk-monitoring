import joblib
import pandas as pd
import numpy as np


def get_feature_names(preprocessor, numeric_features, categorical_features):
    """
    Extract feature names after preprocessing.
    """

    cat_features = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )

    return numeric_features + cat_features


def get_logistic_feature_importance(model, feature_names):
    """
    Extract feature importance from Logistic Regression coefficients.
    """

    coefficients = model.named_steps["classifier"].coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "importance": np.abs(coefficients)
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


def run_explainability():
    """
    Generate feature importance and save results.
    """

    model = joblib.load("models/churn_model.pkl")

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

    categorical_features = ["region", "channel"]

    preprocessor = model.named_steps["preprocessor"]

    feature_names = get_feature_names(
        preprocessor,
        numeric_features,
        categorical_features
    )

    importance_df = get_logistic_feature_importance(model, feature_names)

    importance_df.to_csv("reports/feature_importance.csv", index=False)

    print("Top churn drivers:")
    print(importance_df.head(10))

    return importance_df


if __name__ == "__main__":
    run_explainability()