import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Churn Risk Monitoring",
    layout="wide"
)

st.title("Customer Churn Risk + Monitoring Dashboard")

st.write(
    "Operational dashboard for churn risk scoring, threshold monitoring, "
    "model performance tracking, and drift detection."
)


@st.cache_data
def load_data():
    scores = pd.read_csv("data/scoring/customer_churn_scores.csv")
    threshold = pd.read_csv("reports/threshold_results.csv")
    drift = pd.read_csv("reports/drift_report.csv")
    performance = pd.read_csv("reports/performance_over_time.csv")
    return scores, threshold, drift, performance


scores, threshold, drift, performance = load_data()


# KPI cards
total_customers = len(scores)
avg_risk = scores["churn_risk_score"].mean()
high_risk_count = (scores["risk_band"] == "High Risk").sum()
high_risk_rate = high_risk_count / total_customers

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers Scored", f"{total_customers:,}")
col2.metric("Average Churn Risk", f"{avg_risk:.2%}")
col3.metric("High-Risk Customers", f"{high_risk_count:,}")
col4.metric("High-Risk Rate", f"{high_risk_rate:.2%}")


st.divider()


# Risk distribution
st.subheader("Customer Risk Distribution")

risk_counts = scores["risk_band"].value_counts()

fig, ax = plt.subplots()
risk_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Risk Band")
ax.set_ylabel("Customer Count")
ax.set_title("Customers by Risk Band")
st.pyplot(fig)


st.divider()


# High risk customers
st.subheader("High-Risk Customer List")

high_risk = scores[scores["risk_band"] == "High Risk"].sort_values(
    by="churn_risk_score",
    ascending=False
)

st.dataframe(
    high_risk[
        [
            "customer_id",
            "churn_risk_score",
            "risk_band",
            "recommended_action",
            "monthly_premium",
            "tenure_months",
            "service_calls_90d",
            "late_payments_12m",
            "satisfaction_score",
        ]
    ],
    use_container_width=True
)


st.divider()


# Threshold tuning
st.subheader("Threshold Tuning Results")

st.write(
    "This section helps compare precision, recall, and F1 score at different "
    "risk thresholds."
)

st.dataframe(threshold, use_container_width=True)

fig, ax = plt.subplots()
ax.plot(threshold["threshold"], threshold["precision"], marker="o", label="Precision")
ax.plot(threshold["threshold"], threshold["recall"], marker="o", label="Recall")
ax.plot(threshold["threshold"], threshold["f1"], marker="o", label="F1")
ax.set_xlabel("Threshold")
ax.set_ylabel("Metric")
ax.set_title("Precision / Recall / F1 by Threshold")
ax.legend()
st.pyplot(fig)


st.divider()


# Drift monitoring
st.subheader("Data Drift Monitoring")

st.write(
    "PSI compares training data against new production data. Higher PSI means "
    "the feature distribution has changed more."
)

st.dataframe(drift, use_container_width=True)

fig, ax = plt.subplots()
ax.bar(drift["feature"], drift["psi"])
ax.set_xlabel("Feature")
ax.set_ylabel("PSI")
ax.set_title("Feature Drift by PSI")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)


st.divider()


# Performance monitoring
st.subheader("Model Performance Over Time")

st.dataframe(performance, use_container_width=True)

fig, ax = plt.subplots()
ax.plot(performance["month"], performance["auc"], marker="o", label="AUC")
ax.plot(performance["month"], performance["precision"], marker="o", label="Precision")
ax.plot(performance["month"], performance["recall"], marker="o", label="Recall")
ax.plot(performance["month"], performance["f1"], marker="o", label="F1")
ax.set_xlabel("Month")
ax.set_ylabel("Metric Value")
ax.set_title("Model Performance Trend")
ax.legend()
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)


st.divider()


# Business summary
st.subheader("Business Interpretation")

st.markdown(
    """
    **Key operational insights:**

    - High-risk customers should be prioritized for retention outreach.
    - Threshold tuning helps balance customer coverage versus false positives.
    - Drift monitoring identifies when production data changes from training data.
    - Performance monitoring shows whether the model remains reliable over time.
    - If AUC, precision, or recall decline, the model may need retraining or threshold adjustment.
    """
)

st.divider()

st.subheader("Top Drivers of Churn")

importance = pd.read_csv("reports/feature_importance.csv")

top_features = importance.head(10)

fig, ax = plt.subplots()
ax.barh(top_features["feature"], top_features["importance"])
ax.set_title("Top Churn Drivers")
ax.invert_yaxis()

st.pyplot(fig)

st.markdown("""
    **CONCLUSIONS:**

    - Customers with more late payments and frequent service issues are significantly more likely to churn, 
    - While longer-tenured and more engaged customers are more likely to stay.
    """)