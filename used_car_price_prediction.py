import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("vehicles.csv", low_memory=False)

print("Original dataset shape:", df.shape)


# ==================================================
# 2. INITIAL DATA EXPLORATION
# ==================================================

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nPrice Statistics:")
print(df["price"].describe())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==================================================
# 3. DATA CLEANING
# ==================================================

# Remove unnecessary columns
columns_to_drop = [
    "id",
    "url",
    "region_url",
    "VIN",
    "image_url",
    "description",
    "county",
    "size"
]

df = df.drop(columns=columns_to_drop)

# Remove extreme price values
df = df[
    (df["price"] >= 500) &
    (df["price"] <= 100000)
]

# Fill missing categorical values
categorical_columns = [
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color"
]

for column in categorical_columns:
    df[column] = df[column].fillna("unknown")

# Fill missing numeric values
numeric_columns = [
    "year",
    "odometer",
    "lat",
    "long"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Remove rows without posting date
df = df.dropna(subset=["posting_date"])

# Remove unrealistic odometer values
df = df[
    (df["odometer"] > 0) &
    (df["odometer"] <= 500000)
]

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================

# Convert posting date
df["posting_date"] = pd.to_datetime(
    df["posting_date"],
    utc=True
)

# Extract date features
df["posting_year"] = df["posting_date"].dt.year
df["posting_month"] = df["posting_date"].dt.month

# Calculate car age
df["car_age"] = df["posting_year"] - df["year"]

# Remove invalid car ages
df = df[df["car_age"] >= 0]

# Calculate usage per year
df["km_per_year"] = df["odometer"] / (df["car_age"] + 1)

# Safety check for infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["km_per_year"])

print("\nDataset shape after feature engineering:")
print(df.shape)


# ==================================================
# 5. EXPLORATORY DATA ANALYSIS
# ==================================================

# Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Distribution of Used Car Prices")
plt.xlabel("Price")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# Car age vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="car_age",
    y="price",
    data=df,
    alpha=0.3
)
plt.title("Car Age vs Used Car Price")
plt.xlabel("Car Age (Years)")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Odometer vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="odometer",
    y="price",
    data=df,
    alpha=0.3
)
plt.title("Odometer vs Used Car Price")
plt.xlabel("Odometer")
plt.ylabel("Price")
plt.tight_layout()
plt.show()


# ==================================================
# 6. HANDLE HIGH-CARDINALITY MODEL COLUMN
# ==================================================

print("\nTop 20 car models:")
print(df["model"].value_counts().head(20))

top_models = df["model"].value_counts().head(100).index

df["model"] = df["model"].where(
    df["model"].isin(top_models),
    "other"
)

print("\nNumber of model categories after grouping:")
print(df["model"].nunique())


# ==================================================
# 7. SELECT FEATURES AND TARGET
# ==================================================

features = [
    "year",
    "car_age",
    "odometer",
    "km_per_year",
    "posting_month",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state"
]

X = df[features].copy()
y = df["price"].copy()

print("\nFeatures before encoding:")
print(X.shape)


# ==================================================
# 8. ONE-HOT ENCODING
# ==================================================

X = pd.get_dummies(X, drop_first=True)

print("Features after encoding:")
print(X.shape)


# ==================================================
# 9. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==================================================
# 10. MODEL COMPARISON
# ==================================================

# ---------------- Decision Tree ----------------

tree_model = DecisionTreeRegressor(
    max_depth=20,
    random_state=42
)

tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

tree_r2 = r2_score(y_test, tree_pred)
tree_mae = mean_absolute_error(y_test, tree_pred)
tree_rmse = root_mean_squared_error(y_test, tree_pred)


# ---------------- Original Random Forest ----------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = root_mean_squared_error(y_test, rf_pred)


# ---------------- XGBoost ----------------

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = root_mean_squared_error(y_test, xgb_pred)


# Print comparison
print("\n===== MODEL COMPARISON =====")

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest (Depth 20)",
        "XGBoost"
    ],
    "R² Score": [
        tree_r2,
        rf_r2,
        xgb_r2
    ],
    "MAE": [
        tree_mae,
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        tree_rmse,
        rf_rmse,
        xgb_rmse
    ]
})

print(comparison)


# ==================================================
# 11. FINAL RANDOM FOREST MODEL
# ==================================================

final_rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

final_rf_model.fit(X_train, y_train)

print("\nFinal Random Forest model training completed.")


# ==================================================
# 12. FINAL MODEL EVALUATION
# ==================================================

final_rf_pred = final_rf_model.predict(X_test)

final_rf_mse = mean_squared_error(y_test, final_rf_pred)
final_rf_r2 = r2_score(y_test, final_rf_pred)
final_rf_mae = mean_absolute_error(y_test, final_rf_pred)
final_rf_rmse = root_mean_squared_error(y_test, final_rf_pred)

print("\n===== FINAL RANDOM FOREST RESULTS =====")
print("MSE:", final_rf_mse)
print("R² Score:", final_rf_r2)
print("MAE:", final_rf_mae)
print("RMSE:", final_rf_rmse)


# ==================================================
# 13. OVERFITTING CHECK
# ==================================================

final_rf_train_pred = final_rf_model.predict(X_train)

final_rf_train_r2 = r2_score(
    y_train,
    final_rf_train_pred
)

final_rf_test_r2 = r2_score(
    y_test,
    final_rf_pred
)

print("\n===== OVERFITTING CHECK =====")
print("Training R² Score:", final_rf_train_r2)
print("Testing R² Score:", final_rf_test_r2)
print("Difference:", final_rf_train_r2 - final_rf_test_r2)


# ==================================================
# 14. FEATURE IMPORTANCE
# ==================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": final_rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 15 IMPORTANT FEATURES =====")
print(feature_importance.head(15))


# ==================================================
# 15. FEATURE IMPORTANCE GRAPH
# ==================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 15 Most Important Features for Car Price Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ==================================================
# 16. ACTUAL VS PREDICTED PRICES
# ==================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    final_rf_pred,
    alpha=0.3
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()
plt.show()


# ==================================================
# 17. PREDICT PRICE OF A NEW CAR
# ==================================================

new_car = pd.DataFrame({
    "year": [2020],
    "car_age": [6],
    "odometer": [30000],
    "km_per_year": [5000],
    "posting_month": [8],
    "manufacturer": ["toyota"],
    "model": ["camry"],
    "condition": ["good"],
    "cylinders": ["4 cylinders"],
    "fuel": ["gas"],
    "title_status": ["clean"],
    "transmission": ["automatic"],
    "drive": ["fwd"],
    "type": ["sedan"],
    "paint_color": ["white"],
    "state": ["ca"]
})

# Apply the same model grouping used during training
new_car["model"] = new_car["model"].where(
    new_car["model"].isin(top_models),
    "other"
)

# One-hot encode the new car
new_car_encoded = pd.get_dummies(new_car)

# Match the exact training columns
new_car_encoded = new_car_encoded.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Predict price
predicted_price = final_rf_model.predict(
    new_car_encoded
)

print("\n===== NEW CAR PREDICTION =====")
print("Predicted Car Price: $", round(predicted_price[0], 2))
