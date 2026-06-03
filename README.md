# Customer Churn Risk + Monitoring System

Production-oriented machine learning system for predicting customer churn risk, explaining key churn drivers, tuning business thresholds, monitoring model drift, and tracking model performance over time.

## Business Problem

Customer retention is critical in financial services because churn can reduce revenue, customer lifetime value, and operational efficiency. This project predicts which customers are most likely to leave and provides business teams with risk scores, churn drivers, and recommended retention actions.

## Project Highlights

- Built a churn classification model using Python and scikit-learn
- Generated customer-level churn risk scores
- Converted probabilities into Low, Medium, and High Risk bands
- Added recommended retention actions by risk segment
- Tuned model thresholds using precision, recall, and F1 score
- Monitored data drift using Population Stability Index
- Tracked model performance over time using AUC, precision, recall, and F1
- Added explainability through feature importance
- Built a Streamlit dashboard for operational monitoring
- Designed the project for AWS SageMaker production promotion

  ## Quick Start

git clone <repo-url>
cd customer-churn-risk-monitoring
pip install -r requirements.txt
streamlit run dashboard/dashboard.py

## Key Features
- Risk scoring engine with Low/Medium/High risk bands
- Explainable predictions with feature importance
- Threshold optimization (precision/recall/F1 tradeoffs)
- Data drift detection via PSI
- Performance monitoring dashboard
- AWS SageMaker deployment-ready

## Architecture

Raw Customer Data
        ↓
Feature Engineering + Preprocessing
        ↓
Model Training
        ↓
Risk Scoring
        ↓
Threshold Tuning
        ↓
Drift + Performance Monitoring
        ↓
Dashboard + Business Actions
        ↓
SageMaker Production Design

## Project Structure

| File | Purpose |
|------|---------|
| `src/train.py` | Train churn model |
| `src/score_customers.py` | Generate risk scores for customers |
| `src/threshold_tuning.py` | Optimize business thresholds |
| `src/drift_detection.py` | Monitor feature drift |
| `dashboard/dashboard.py` | Streamlit operational dashboard |


customer-churn-risk-monitoring/
├── app/
├── dashboard/
│   └── dashboard.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── scoring/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── sagemaker/
├── src/
│   ├── data_simulation.py
│   ├── train.py
│   ├── score_customers.py
│   ├── threshold_tuning.py
│   ├── drift_detection.py
│   ├── performance_monitoring.py
│   └── explain_model.py
├── requirements.txt
├── config.yaml
└── README.md

## Model Performance
- **Test AUC:** 0.813
- **Test Dataset:** 1,000 customers
- **Training Dataset:** 4,000 customers
- **Churn Rate:** 2.12%

## Data Requirements
Input features: [list key features like tenure_months, late_payments_12m, service_calls_90d, satisfaction_score]

## Running the Dashboard

The Streamlit dashboard displays:
- KPI metrics (total customers, avg risk, high-risk rate)
- Risk distribution charts
- High-risk customer list with recommended actions
- Threshold tuning analysis
- Data drift monitoring
- Model performance trends

## Business Impact
- Identifies high-risk customers for targeted retention campaigns
- Enables threshold tuning to balance coverage vs. false positives
- Monitors model performance decay to trigger retraining
