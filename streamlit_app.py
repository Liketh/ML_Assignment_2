import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier
import time

st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

st.title("💳 Credit Card Default Prediction – ML Assignment")

# Dataset Upload
# ===============================
uploaded_file = st.file_uploader("Upload UCI Credit Card CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())


    # Preprocessing 
    # ===============================
    def preprocessing_test_file(df):
        df.drop(columns=['ID'],inplace=True)
        df.drop(columns=['default.payment.next.month'],inplace=True)
        return df


    # Model Selection
    # ===============================
    model_name = st.selectbox(
        "Select Machine Learning Model",
        [
            'Logistic Regression',
            'Decision Tree',
            'Random Forest',
            'KNN',
            'Naive Bayes',
            'XGBoost'
        ]
    )

    if model_name == 'Logistic Regression':
        model = joblib.load(r"models/lr1.pkl")
    elif model_name == 'Decision Tree':
        model = joblib.load(r"models/dt1.pkl")
    elif model_name == 'Random Forest':
        model = joblib.load(r"models/rf1.pkl")
    elif model_name == 'KNN':
        model = joblib.load(r"models/KNN1.pkl")
    elif model_name == 'Naive Bayes':
        model = joblib.load(r"models/GNB1.pkl")
    elif model_name == 'XGBoost':
        model = joblib.load(r"models/xgb1.pkl")


    # Test Model
    # ===============================

    y_test = df['default.payment.next.month']
    x_test = preprocessing_test_file(df)


    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]


    # Metrics
    # ===============================
    st.subheader("📊 Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_test, y_pred), 4))
    col3.metric("Recall", round(recall_score(y_test, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
    col5.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))
    col6.metric("ROC AUC", round(roc_auc_score(y_test, y_prob), 4))


    # Classification Report
    # ===============================
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload the UCI Credit Card dataset to proceed.")
