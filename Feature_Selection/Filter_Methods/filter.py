import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 1. Dummy Data Banana (10 features hain, jinme se sirf 3 kaam ke hain)
X, y = make_classification(n_samples=150, n_features=10, n_informative=3, random_state=42)

print("Original Data Shape:", X.shape)


# --- METHOD 1: LASSO (L1) REGULARIZATION ---
# L1 penalty lagane se bekar features ke coefficients zero (0) ho jaate hain
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)

# SelectFromModel un features ko chunta hai jinki importance zero nahi hoti
lasso_selector = SelectFromModel(estimator=lasso_model)
X_lasso = lasso_selector.fit_transform(X, y)

print("\n--- LASSO (L1) SELECTION ---")
print("Selected Feature Indices:", np.where(lasso_selector.get_support())[0])
print("Data Shape after Lasso:", X_lasso.shape)


# --- METHOD 2: TREE-BASED FEATURE IMPORTANCE ---
# Random Forest tree-splitting ke time har feature ki importance calculate karta hai
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

# SelectFromModel 'mean' importance ke threshold par features filter karega
rf_selector = SelectFromModel(estimator=rf_model, threshold="mean")
X_rf = rf_selector.fit_transform(X, y)

print("\n--- RANDOM FOREST SELECTION ---")
print("Selected Feature Indices:", np.where(rf_selector.get_support())[0])
print("Data Shape after Random Forest:", X_rf.shape)
