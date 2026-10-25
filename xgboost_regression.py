# XGBoost Regression: House Price Prediction
# Install: pip install xgboost scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the small dataset
df = pd.read_csv("xgboost_regression_small.csv")

# Separate features and target
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create the XGBoost regression model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective="reg:squarederror",
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate predictions
print("XGBOOST REGRESSION")
print("Actual prices:   ", y_test.to_list())
print("Predicted prices:", y_pred.round(2).tolist())
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
print("R2:", round(r2_score(y_test, y_pred), 3))
