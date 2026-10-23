
Gradient Boosting Trees: Comprehensive Guide & Implementation

Gradient Boosting is a powerful machine learning technique for regression and classification tasks that builds a predictive model in the form of an ensemble of weak prediction models, typically decision trees.
It builds the model in a stage-wise fashion, optimizing a differentiable loss function by sequentially correcting the errors (residuals) made by previous trees.


# Gradient Boosting Regression from Scratch

A clean, production-ready implementation of the Gradient Boosting Regressor algorithm built from scratch using NumPy and Scikit-Learn's DecisionTreeRegressor as the base weak learner.

## 🚀 Key Features
- **Custom Loss Support**: Easily plugs into Mean Squared Error (MSE) loss.
- **Pseudo-Residual Learning**: Updates sequential trees based on negative gradients.
- **Shrinkage (Learning Rate)**: Regularization via shrinkage to prevent overfitting.

## 🛠️ Installation
```bash
pip install numpy scikit-learn
```

## 💻 Code Implementation

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressorFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Gradient Boosting Regressor built from scratch.
        
        Parameters:
        -----------
        n_estimators : int
            The number of boosting stages (trees) to perform.
            
        learning_rate : float
            Shrinkage factor to scale the contribution of each tree.
            
        max_depth : int
            Maximum depth of the individual decision tree estimators.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        """
        # Step 1: Initialize the model with a constant value (mean of target values)
        self.initial_prediction = np.mean(y)
        
        # F_0(x) initialized for all samples
        F_m = np.full(shape=y.shape, fill_value=self.initial_prediction, dtype=np.float64)
        
        for m in range(self.n_estimators):
            # Step 2: Compute pseudo-residuals (Negative Gradient of MSE Loss)
            # Loss = 0.5 * (y - F_m)^2 -> Negative Gradient = y - F_m
            residuals = y - F_m
            
            # Step 3: Fit a base learner (Decision Tree) to the pseudo-residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            
            # Step 4: Update the model prediction
            # F_m(x) = F_{m-1}(x) + v * h_m(x)
            F_m += self.learning_rate * tree.predict(X)
            
            # Save the trained tree
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict regression target for X.
        """
        # Start with the initial constant prediction
        predictions = np.full(shape=(X.shape[0],), fill_value=self.initial_prediction, dtype=np.float64)
        
        # Accumulate the predictions from all staged trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions

# --- Example Usage ---
if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * (X.flatten() ** 2) + np.random.normal(0, 5, 100)
    
    # Initialize and train model
    model = GradientBoostingRegressorFromScratch(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    
    # Make a prediction
    test_x = np.array([[5.0]])
    pred = model.predict(test_x)
    print(f"Prediction for X=5.0: {pred[0]:.4f}")
```



1. Step 1: Base Prediction (F₀)

The algorithm starts with a baseline guess. For regression, this is simply the average (mean) value of all targets (y).
• Analogy: If you want to guess everyone's weight in a room without seeing them, your safest first guess is the average weight of a human.

2. Step 2: Calculate Residuals (Mistakes)

The algorithm evaluates how far off the initial guess is for every single data point.
\(\text{Residual}=\text{Actual\ Value}-\text{Predicted\ Value}\)
These errors are technically called Pseudo-residuals (negative gradients of the loss function).

3. Step 3: Train a Tree on Residuals

Instead of trying to predict the final output y, the next decision tree is trained to predict the residuals (errors). It attempts to answer: "How can we fix the mistakes made in the previous step?"

4. Step 4: Combine & Scale (Learning Rate)

The new tree's prediction is multiplied by a small scale factor called the Learning Rate (ν) (usually between 0.01 and 0.1) and added to the previous prediction.
\(\text{New\ Prediction}=\text{Old\ Prediction}+(\text{Learning\ Rate}\times \text{Tree\ Prediction})\)
• Why scale? If you fix the mistakes 100% instantly, the model overfits and fails on new data. Taking small steps ensures stability.

5. Step 5: Repeat

Steps 2 through 4 are repeated sequentially for N times (n_estimators). Each new tree fixes the remaining errors left over by all its predecessors.
