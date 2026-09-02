# 🚗 DriveValue: AI-Powered Used Car Price Prediction System

An end-to-end Machine Learning project that predicts the price of used cars based on vehicle characteristics such as manufacturing year, car age, mileage, fuel type, transmission, manufacturer, model, and other features.

---

## 📌 Project Overview

The goal of this project is to build a machine learning model that can estimate the price of a used car based on its characteristics.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- One-Hot Encoding
- Multiple machine learning models
- Model comparison
- Hyperparameter experimentation
- Overfitting analysis
- Feature importance analysis
- Data visualization
- Prediction of prices for new cars

---

## 📊 Dataset

The project uses a used vehicle dataset containing information such as:

- Price
- Manufacturing year
- Manufacturer
- Model
- Condition
- Cylinders
- Fuel type
- Odometer reading
- Title status
- Transmission
- Drive type
- Vehicle type
- Paint color
- State
- Posting date

> Note: The original dataset is not included in this repository due to its large size.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns
- Handled missing values
- Removed unrealistic price values
- Removed unrealistic odometer values
- Handled infinite values
- Converted posting dates to datetime format
- Grouped rare car models into an `other` category

---

## 🔧 Feature Engineering

New features were created to improve model performance:

- **Car Age** – calculated using the manufacturing year and posting year
- **Posting Month** – extracted from the posting date
- **Kilometers per Year** – calculated using odometer reading and car age

---

## 🤖 Machine Learning Models

The following regression models were tested:

1. Decision Tree Regressor
2. Random Forest Regressor
3. XGBoost Regressor

The final model selected was:

### 🏆 Random Forest Regressor

Configuration:

- `n_estimators = 100`
- `max_depth = None`
- `random_state = 42`

---

## 📈 Final Model Performance

| Metric | Result |
|---|---:|
| R² Score | **0.8962** |
| MAE | **$2,070.11** |
| RMSE | **$4,628.46** |
| Training R² Score | **0.9854** |
| Testing R² Score | **0.8962** |

The model explains approximately **89.6% of the variation in used car prices** on the test set.

---

## 🔍 Feature Importance

The most important features identified by the Random Forest model include:

1. Car Age
2. Manufacturing Year
3. Odometer Reading
4. Drive Type
5. Fuel Type
6. Kilometers per Year
7. Number of Cylinders

---

## 📊 Visualizations

The project includes:

- Used car price distribution
- Car age vs price
- Odometer vs price
- Top 15 feature importance graph
- Actual vs predicted price graph

---

## 🚗 Predicting a New Car Price

The trained model can predict the price of a new car by providing features such as:

```python
year
odometer
manufacturer
model
condition
fuel
transmission
drive
type
paint_color
state
