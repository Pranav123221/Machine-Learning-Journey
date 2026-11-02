import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold

# 1. Data & Selector Setup
X, _ = make_classification(n_samples=100, n_features=10, random_state=42)
X[:, 9] = 5  # Dead/Constant feature

selector = VarianceThreshold(threshold=0.1).fit(X)

# 2. Short Plotting
colors = ['teal' if m else 'tomato' for m in selector.get_support()]
plt.bar(range(10), selector.variances_, color=colors, edgecolor='black')
plt.axhline(0.1, color='red', linestyle='--', label='Threshold (0.1)')

plt.title("VarianceThreshold Selection")
plt.xticks(range(10), [f"F_{i}" for i in range(10)])
plt.legend()
plt.show()
