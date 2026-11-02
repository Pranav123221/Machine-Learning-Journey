import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

# 1. Dummy data banana (10 features hain, jinme se sirf 3 kaam ke hain)
X, y = make_classification(n_samples=100, n_features=10, n_informative=3, random_state=42)

# 2. Base model select karna
model = LogisticRegression()

# 3. Forward Selector setup (direction="forward")
# Yeh khali se shuru karega aur tab tak feature jodega jab tak 3 nahi ho jaate
forward_selector = SequentialFeatureSelector(
    estimator=model, 
    n_features_to_select=3, 
    direction="forward", 
    cv=5
)

# 4. Model ko fit karna
forward_selector.fit(X, y)

# 5. Selected features ke index nikalna (True = Selected)
selected_indices = np.where(forward_selector.get_support())

print("--- FORWARD SELECTION RESULT ---")
print("Selected Feature Indices:", selected_indices)

# 6. Data ko transform karna (Sirf selected features rakhne ke liye)
X_new = forward_selector.transform(X)
print("New Data Shape (Selected 3 features):", X_new.shape)
