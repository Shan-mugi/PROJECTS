import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LOAD DATA
base_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_path, 'house_data.csv'))
    
print("Data Loaded Successfully!")

# 2. HANDLE MISSING VALUES (Requirement #1)
# Fill numeric missing values with the median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with 'None'
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
df[categorical_cols] = df[categorical_cols].fillna('None')

# 3. ENCODE CATEGORICAL VARIABLES (Requirement #2)
df_encoded = pd.get_dummies(df)

# 4. PREPARE DATA FOR TRAINING
X = df_encoded.drop(['SalePrice', 'Id'], axis=1)
y = df_encoded['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. FEATURE SCALING (Requirement #3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. TRAIN MODELS (Requirement #4)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(alpha=100, max_iter=50000),
    "Gradient Boosting": GradientBoostingRegressor()
}

print("--- Model Performance ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{name}: R2 Score = {r2:.4f}, MAE = ${mae:,.2f}")
