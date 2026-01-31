import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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


# ===============================
# Dataset Upload
# ===============================
uploaded_file = st.file_uploader("Upload UCI Credit Card CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # Preprocessing (Same as Notebook)
    # ===============================
    df.drop(columns=['ID'], inplace=True)

    # Outlier removal (IQR method)
    outliers = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    for col in outliers:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    # Dataset balancing
    df_0 = df[df['default.payment.next.month'] == 0]
    df_1 = df[df['default.payment.next.month'] == 1]
    df_0 = df_0.sample(len(df_1), random_state=42)
    df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

    st.subheader("⚖️ Balanced Target Distribution")
    st.bar_chart(df['default.payment.next.month'].value_counts())

    # ===============================
    # Train-Test Split
    # ===============================
    X = df.drop(columns=['default.payment.next.month'])
    y = df['default.payment.next.month']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ===============================
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
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=49, use_label_encoder=False, eval_metric='logloss')

    # ===============================
    # Train Model
    # ===============================
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # ===============================
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

    st.write(f"⏱ Training Time: {exec_time:.4f} seconds")

    # ===============================
    # Classification Report
    # ===============================
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload the UCI Credit Card dataset to proceed.")
