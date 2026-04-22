import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Load Data (Use your full path here)
df = pd.read_csv(r'E:\PROJECT_ 4\creditcard.csv.zip')

# 2. Balance the data
X = df.drop('Class', axis=1)
y = df['Class']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# --- MODEL 1: Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("Logistic Regression Done!")

# --- MODEL 2: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_depth=10) # Depth kept at 10 for speed
rf.fit(X_train, y_train)
print("Random Forest Done!")

# --- MODEL 3: XGBoost ---
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print("XGBoost Done!")

# 4. Compare Results
print("\n[LOGISTIC REGRESSION REPORT]")
print(classification_report(y_test, lr.predict(X_test)))

print("\n[RANDOM FOREST REPORT]")
print(classification_report(y_test, rf.predict(X_test)))

print("\n[XGBOOST REPORT]")
print(classification_report(y_test, xgb.predict(X_test)))
