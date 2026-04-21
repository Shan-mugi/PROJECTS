import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE # You may need to pip install imbalanced-learn

# 1. Load Data
df = pd.read_csv(r'C:\Users\ELCOT\OneDrive\Desktop\PROJECT 4\creditcard.csv.zip')

# 2. Split Features and Target
X = df.drop('Class', axis=1)
y = df['Class']

# 3. Handle Imbalance with SMOTE
# This creates synthetic fraud cases so the model can learn what fraud looks like
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"Original Fraud count: {sum(y == 1)}")
print(f"SMOTE Fraud count: {sum(y_res == 1)}")

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 5. Model: Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
