import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Load Data
data_path = os.path.join('data', 'Housing.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find Housing.csv at {data_path}. Please place the Kaggle dataset there.")

df = pd.read_csv(data_path)

# Kaggle Housing dataset typically contains: 
# price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus

# 2. Preprocessing
# Convert binary categorical columns to 0/1
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})

# Handle furnishingstatus (One-Hot Encoding or Mapping)
if 'furnishingstatus' in df.columns:
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Save feature names to ensure web app matches column order exactly
feature_names = list(X.columns)
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(# 6. Save Model and Scaler
f"Model Training Complete. R^2 Score: {model.score(scaler.transform(X_test), y_test):.4f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved model.pkl, scaler.pkl, and features.pkl successfully.")