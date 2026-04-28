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

## Architecture

```text
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
