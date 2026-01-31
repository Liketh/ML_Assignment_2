# Generated from: credit card default prediction.ipynb
# Converted at: 2026-01-31T13:59:11.082Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r"C:\Users\datma003\OneDrive\Desktop\ML_Assignment_2\UCI_Credit_Card.csv")

df

df.info()

df.describe()

df.dtypes

df.isna().sum()

df.drop(columns=['ID'],inplace=True)

df.duplicated().sum()

df['default.payment.next.month'].value_counts()

outliers=[]

for i in df.columns:
    print(i,df[i].nunique())

for i in df.columns:
    if df[i].nunique()>11:
        outliers.append(i)

outliers

plt.figure(figsize=(15,5))
for i in range(0,3):
    plt.subplot(1,3,i+1)
    sns.boxplot(df[outliers[i]])
    plt.xlabel(outliers[i])
plt.show()

plt.figure(figsize=(15,5))
for i in range(3,6):
    plt.subplot(1,3,i-2)
    sns.boxplot(df[outliers[i]])
    plt.xlabel(outliers[i])
plt.show()

plt.figure(figsize=(15,5))
for i in range(6,9):
    plt.subplot(1,3,i-5)
    sns.boxplot(df[outliers[i]])
    plt.xlabel(outliers[i])
plt.show()

plt.figure(figsize=(15,5))
for i in range(9,14):
    plt.subplot(1,5,i-8)
    sns.boxplot(df[outliers[i]])
    plt.xlabel(outliers[i])
plt.show()

outliers=['LIMIT_BAL',
 'AGE',
 'BILL_AMT1',
 'BILL_AMT2',
 'BILL_AMT3',
 'BILL_AMT4',
 'BILL_AMT5',
 'BILL_AMT6']

for i in outliers:
    q1=df[i].quantile(0.25)
    q3=df[i].quantile(0.75)
    iqr=q3-q1
    print("For column ",i," Q1 is ",q1," and Q3 is ",q3)
    print(len(df)-len(df[(df[i]>=q1-(1.5*iqr))&(df[i]<=q3+(1.5*iqr))]),f" is the number of outliers that is being removed from {i} column")
    df=df[(df[i]>=q1-(1.5*iqr))&(df[i]<=q3+(1.5*iqr))]

df.shape

df.info()

plt.figure(figsize=(6, 4))
sns.countplot(x='SEX', data=df, palette='viridis')
plt.title('Gender Distribution')
plt.xlabel('Sex (1 = Male, 2 = Female)')
plt.ylabel('Count')
plt.show()

education_counts = df['EDUCATION'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Education Level Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='MARRIAGE', data=df, palette='coolwarm')
plt.title('Marital Status Distribution')
plt.xlabel('Marriage Status (1 = Married, 2 = Single, 3 = Others)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['AGE'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(y='LIMIT_BAL', data=df, color='lightgreen')
plt.title('Distribution of Credit Limit')
plt.ylabel('Credit Limit (LIMIT_BAL)')
plt.show()

plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
df[bill_cols].mean().plot(kind='line', marker='o', color='purple')
plt.title('Average Bill Amounts Over Time')
plt.xlabel('Month')
plt.ylabel('Average Bill Amount')
plt.xticks(range(len(bill_cols)), bill_cols)
plt.show()

pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
pay_counts = df[pay_cols].apply(pd.Series.value_counts).fillna(0)
pay_counts.T.plot(kind='bar', figsize=(10, 6), stacked=True, colormap='viridis')
plt.title('Payment Status Frequency')
plt.xlabel('Months (PAY_0 to PAY_6)')
plt.ylabel('Count')
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='default.payment.next.month', data=df, palette='Set2')
plt.title('Default Payment Status')
plt.xlabel('Default Status (0 = Not Default, 1 = Default)')
plt.ylabel('Count')
plt.show()

df['default.payment.next.month'].value_counts()

# # Dataset Balancing


df_0=df[df['default.payment.next.month']==0]
df_1=df[df['default.payment.next.month']==1]

df_0=df_0.sample(len(df_1))
df_0.shape

df=pd.concat([df_0,df_1])

df['default.payment.next.month'].value_counts()

df.sample(frac=1,random_state=15)

from sklearn.model_selection import train_test_split

X=df.drop(columns=['default.payment.next.month'])
Y=df['default.payment.next.month']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,stratify=Y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import time
from xgboost import XGBClassifier

lr1=LogisticRegression()

start_time=time.time()
lr1.fit(x_train,y_train)
end_time=time.time()
lrt=end_time-start_time

lr1_pred=lr1.predict(x_test)
lr1_prob=lr1.predict_proba(x_test)[:, 1]

lr1_acc = accuracy_score(y_test, lr1_pred)
lr1_pre = precision_score(y_test, lr1_pred)
lr1_rec = recall_score(y_test, lr1_pred)
lr1_f1  = f1_score(y_test, lr1_pred)
lr1_mcc = matthews_corrcoef(y_test, lr1_pred)
lr1_auc = roc_auc_score(y_test, lr1_prob)


print(classification_report(y_test,lr1_pred))
print("Accuracy of Logistic Regression:",lr1_acc)
print("AUC Score of Logistic Regression:",lr1_auc)
print("Precision of Logistic Regression:",lr1_pre)
print("Recall of Logistic Regression:",lr1_rec)
print("F1 Score of Logistic Regression:",lr1_f1)
print("Matthews Correlation Coefficient of Logistic Regression:",lr1_mcc)

dt1=DecisionTreeClassifier(max_depth=5,min_samples_split=5)

start_time=time.time()
dt1.fit(x_train,y_train)
end_time=time.time()
dtt=end_time-start_time

dt1_pred=dt1.predict(x_test)
dt1_prob=dt1.predict_proba(x_test)[:, 1]

dt1_acc = accuracy_score(y_test, dt1_pred)
dt1_pre = precision_score(y_test, dt1_pred)
dt1_rec = recall_score(y_test, dt1_pred)
dt1_f1  = f1_score(y_test, dt1_pred)
dt1_mcc = matthews_corrcoef(y_test, dt1_pred)
dt1_auc = roc_auc_score(y_test, dt1_prob)


print(classification_report(y_test,dt1_pred))
print("Accuracy of Decision Tree:",dt1_acc)
print("AUC Score of Decision Tree:",dt1_auc)
print("Precision of Decision Tree:",dt1_pre)
print("Recall of Decision Tree:",dt1_rec)
print("F1 Score of Decision Tree:",dt1_f1)
print("Matthews Correlation Coefficient of Decision Tree:",dt1_mcc)

rf1=RandomForestClassifier()

start_time=time.time()
rf1.fit(x_train,y_train)
end_time=time.time()
rft=end_time-start_time

rf1_pred=rf1.predict(x_test)
rf1_prob=rf1.predict_proba(x_test)[:, 1]

rf1_acc = accuracy_score(y_test, rf1_pred)
rf1_pre = precision_score(y_test, rf1_pred)
rf1_rec = recall_score(y_test, rf1_pred)
rf1_f1  = f1_score(y_test, rf1_pred)
rf1_mcc = matthews_corrcoef(y_test, rf1_pred)
rf1_auc = roc_auc_score(y_test, rf1_prob)


print(classification_report(y_test,rf1_pred))
print("Accuracy of Random Forest:",rf1_acc)
print("AUC Score of Random Forest:",rf1_auc)
print("Precision of Random Forest:",rf1_pre)
print("Recall of Random Forest:",rf1_rec)
print("F1 Score of Random Forest:",rf1_f1)
print("Matthews Correlation Coefficient of Random Forest:",rf1_mcc)

KNN1=KNeighborsClassifier()

start_time=time.time()
KNN1.fit(x_train,y_train)
end_time=time.time()
KNNt=end_time-start_time

KNN1_pred=KNN1.predict(x_test)
KNN1_prob=KNN1.predict_proba(x_test)[:, 1]

KNN1_acc = accuracy_score(y_test, KNN1_pred)
KNN1_pre = precision_score(y_test, KNN1_pred)
KNN1_rec = recall_score(y_test, KNN1_pred)
KNN1_f1  = f1_score(y_test, KNN1_pred)
KNN1_mcc = matthews_corrcoef(y_test, KNN1_pred)
KNN1_auc = roc_auc_score(y_test, KNN1_prob)


print(classification_report(y_test,KNN1_pred))
print("Accuracy of KNN:",KNN1_acc)
print("AUC Score of KNN:",KNN1_auc)
print("Precision of KNN:",KNN1_pre)
print("Recall of KNN:",KNN1_rec)
print("F1 Score of KNN:",KNN1_f1)
print("Matthews Correlation Coefficient of KNN:",KNN1_mcc)

GNB1=GaussianNB()

start_time=time.time()
GNB1.fit(x_train,y_train)
end_time=time.time()
GNBt=end_time-start_time

GNB1_pred=GNB1.predict(x_test)
GNB1_prob=GNB1.predict_proba(x_test)[:, 1]

GNB1_acc = accuracy_score(y_test, GNB1_pred)
GNB1_pre = precision_score(y_test, GNB1_pred)
GNB1_rec = recall_score(y_test, GNB1_pred)
GNB1_f1  = f1_score(y_test, GNB1_pred)
GNB1_mcc = matthews_corrcoef(y_test, GNB1_pred)
GNB1_auc = roc_auc_score(y_test, GNB1_prob)


print(classification_report(y_test,GNB1_pred))
print("Accuracy of Gaussian Naive Bayes:",GNB1_acc)
print("AUC Score of Gaussian Naive Bayes:",GNB1_auc)
print("Precision of Gaussian Naive Bayes:",GNB1_pre)
print("Recall of Gaussian Naive Bayes:",GNB1_rec)
print("F1 Score of Gaussian Naive Bayes:",GNB1_f1)
print("Matthews Correlation Coefficient of Gaussian Naive Bayes:",GNB1_mcc)

xgb1=XGBClassifier(random_state=49, use_label_encoder=False, eval_metric='logloss')

start_time=time.time()
xgb1.fit(x_train,y_train)
end_time=time.time()
xgbt=end_time-start_time

xgb1_pred=xgb1.predict(x_test)
xgb1_prob=xgb1.predict_proba(x_test)[:, 1]

xgb1_acc = accuracy_score(y_test, xgb1_pred)
xgb1_pre = precision_score(y_test, xgb1_pred)
xgb1_rec = recall_score(y_test, xgb1_pred)
xgb1_f1  = f1_score(y_test, xgb1_pred)
xgb1_mcc = matthews_corrcoef(y_test, xgb1_pred)
xgb1_auc = roc_auc_score(y_test, xgb1_prob)


print(classification_report(y_test,xgb1_pred))
print("Accuracy of XG Boost:",xgb1_acc)
print("AUC Score of XG Boost:",xgb1_auc)
print("Precision of XG Boost:",xgb1_pre)
print("Recall of XG Boost:",xgb1_rec)
print("F1 Score of XG Boost:",xgb1_f1)
print("Matthews Correlation Coefficient of XG Boost:",xgb1_mcc)

results_df={"ML Model Name":['Logistic Regression',"Decision Tree",'kNN','Naive Bayes','Random Forest(Ensemble)','XGBoost(Ensemble)'],
            "Accuracy":[round(lr1_acc,2),round(dt1_acc,2),round(KNN1_acc,2),round(GNB1_acc,2),round(rf1_acc,2),round(xgb1_acc,2)],  
            "AUC":[round(lr1_auc,2),round(dt1_auc,2),round(KNN1_auc,2),round(GNB1_auc,2),round(rf1_auc,2),round(xgb1_auc,2)],
            "Precision":[round(lr1_pre,2),round(dt1_pre,2),round(KNN1_pre,2),round(GNB1_pre,2),round(rf1_pre,2),round(xgb1_pre,2)],
            "Recall":[round(lr1_rec,2),round(dt1_rec,2),round(KNN1_rec,2),round(GNB1_rec,2),round(rf1_rec,2),round(xgb1_rec,2)],
            "F1":[round(lr1_f1,2),round(dt1_f1,2),round(KNN1_f1,2),round(GNB1_f1,2),round(rf1_f1,2),round(xgb1_f1,2)],
            "MCC":[round(lr1_mcc,2),round(dt1_mcc,2),round(KNN1_mcc,2),round(GNB1_mcc,2),round(rf1_mcc,2),round(xgb1_mcc,2)]}   
           
results_df=pd.DataFrame(results_df)
results_df

time_df={"ML Model Name":['Logistic Regression',"Decision Tree",'kNN','Naive Bayes','Random Forest(Ensemble)','XGBoost(Ensemble)'],
         "Execution Time in Seconds":[lrt,dtt,KNNt,GNBt,rft,xgbt]}
time_df=pd.DataFrame(time_df)
time_df

plt.figure(figsize=(15,5))
sns.barplot(data=time_df,x=time_df['ML Model Name'],y=time_df['Execution Time in Seconds'])
plt.xlabel("Algorithms")
plt.ylabel("Execution Time in Seconds")
plt.title("Time Taken For Model Training")

#Testing
random=int(input("ENter a random number"))
testing=pd.read_csv(r"C:\Users\datma003\Desktop\projects1\New\Credit card default prediction\UCI_Credit_Card.csv")
testing.drop(columns=['ID'],inplace=True)
sc=StandardScaler()
testing[num_cols]=sc.fit_transform(testing[num_cols])
print("Actual value is ",testing['default.payment.next.month'].iloc[random])
testing.drop(columns=['default.payment.next.month'],inplace=True)
testing=pd.DataFrame(testing.iloc[random]).T
print("Predicted value is ",rf1.predict(testing))