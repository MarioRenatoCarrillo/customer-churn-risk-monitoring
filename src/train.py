import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def train_churn_model(data_path: str):
    """
    Train a baseline churn prediction model using Logistic Regression.
    """

    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Define target and features
    target = "churn"

    X = df.drop(columns=["customer_id", "churn", "churn_probability_true"])
    y = df[target]

    # 3. Identify numeric and categorical columns
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

    categorical_features = [
        "region",
        "channel",
    ]

    # 4. Preprocessing for numeric columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 5. Preprocessing for categorical columns
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 6. Combine preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 7. Create full model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # 8. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 9. Train model
    model.fit(X_train, y_train)

    # 10. Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # 11. Convert probabilities to labels using default threshold 0.50
    y_pred = (y_proba >= 0.50).astype(int)

    # 12. Calculate metrics
    metrics = {
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "test_rows": len(X_test),
        "train_rows": len(X_train),
        "churn_rate": y.mean()
    }

    # 13. Save model
    joblib.dump(model, "models/churn_model.pkl")

    # 14. Save metrics
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model training complete.")
    print("Model saved to: models/churn_model.pkl")
    print("Metrics saved to: models/metrics.json")
    print(metrics)


if __name__ == "__main__":
    train_churn_model("data/raw/customer_churn_simulated.csv")