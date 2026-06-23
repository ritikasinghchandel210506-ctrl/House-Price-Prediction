import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("data/housing.csv")

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

# ==================================================
# REMOVE OUTLIERS FROM PRICE
# ==================================================

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["price"] >= lower) &
    (df["price"] <= upper)
]

print("\nShape After Outlier Removal:", df.shape)

# ==================================================
# FEATURE ENGINEERING
# ==================================================

df["area_per_bedroom"] = (
    df["area"] / df["bedrooms"]
)

df["total_rooms"] = (
    df["bedrooms"] +
    df["bathrooms"]
)

df["luxury_score"] = (
    df["area"]
    + df["bathrooms"] * 1000
    + df["parking"] * 500
)

# ==================================================
# FEATURES & TARGET
# ==================================================

X = df.drop("price", axis=1)

# log transform target
y = np.log1p(df["price"])

# ==================================================
# COLUMN TYPES
# ==================================================

categorical_features = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

numerical_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "area_per_bedroom",
    "total_rooms",
    "luxury_score"
]

# ==================================================
# PREPROCESSING
# ==================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================================
# MODELS
# ==================================================

models = {

    "Linear Regression":

        LinearRegression(),

    "Random Forest":

        RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        ),

    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),

    "XGBoost":

        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
}

# ==================================================
# TRAIN & EVALUATE
# ==================================================

best_model = None
best_r2 = -999
best_name = ""

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions_log = pipeline.predict(X_test)

    predictions = np.expm1(
        predictions_log
    )

    actual = np.expm1(
        y_test
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    mse = mean_squared_error(
        actual,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predictions
    )

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    cv_mean = cv_scores.mean()

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))
    print("CV  :", round(cv_mean, 4))

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_name = name

# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    "model/model.pkl",
    "wb"
) as f:

    pickle.dump(
        best_model,
        f
    )

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model :", best_name)
print("R²    :", round(best_r2, 4))

print("\nModel saved successfully!")