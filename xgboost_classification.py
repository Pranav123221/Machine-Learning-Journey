# XGBoost Classification: Student Pass/Fail Prediction
# Install: pip install xgboost scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load the small dataset
df = pd.read_csv("xgboost_classification_small.csv")

# Separate features and target
X = df[["StudyHours", "Attendance", "PreviousScore"]]
y = df["Pass"]  # 0 = Fail, 1 = Pass

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create the XGBoost classification model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict classes and probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate predictions
print("XGBOOST CLASSIFICATION")
print("Actual labels:     ", y_test.to_list())
print("Predicted labels:  ", y_pred.tolist())
print("Pass probabilities:", y_prob.round(3).tolist())
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Confusion matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification report:\\n", classification_report(
    y_test, y_pred, zero_division=0
))
