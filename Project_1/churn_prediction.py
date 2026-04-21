import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap

# 1. DATA CLEANING & PREPROCESSING
df = pd.read_csv('customer_data.csv')

# Handle specific Telco numeric trap
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df.drop('customerID', axis=1, inplace=True)

# Map Target to 0 and 1
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 2. FEATURE ENGINEERING
# Example: Creating a 'TotalServices' count
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
df['TotalServices'] = df[service_cols].sum(axis=1)

# Convert remaining categorical to dummies
df_final = pd.get_dummies(df, drop_first=True)

# Split and Scale
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. TRAIN MODELS
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # 4. EVALUATE
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds))

# 5. EXPLAIN MODEL USING SHAP (Using XGBoost as the champion)
explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
