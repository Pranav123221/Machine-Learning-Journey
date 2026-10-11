# Support Vector Machines (SVM): Maximum-Margin Learning, Kernel Methods & Classification

## Overview

This repository documents my learning and practical exploration of **Support Vector Machines (SVM)**, a supervised machine learning algorithm used for classification and regression tasks.

The focus is on understanding how SVM learns an optimal decision boundary by maximizing the margin between classes, how support vectors influence the decision function, and how kernel methods enable nonlinear classification.

## Learning Objectives

- Understand the mathematical foundations of Support Vector Machines.
- Explore maximum-margin classification and the role of support vectors.
- Understand hyperplanes, margins, and decision boundaries.
- Study hard-margin and soft-margin SVM formulations.
- Analyze the role of regularization through the `C` parameter.
- Explore linear and nonlinear classification using kernel functions.
- Understand feature scaling and its importance for SVM.
- Investigate the impact of hyperparameters on model performance.
- Learn how SVM can be applied to classification and regression.

## 1. Mathematical Foundations

### Hyperplane and Decision Function

An SVM learns a decision boundary that separates observations belonging to different classes.

For a linear SVM, the decision function is:

\[
f(x)=w^T x+b
\]

Where:

- \(x\) represents the input feature vector.
- \(w\) represents the weight vector.
- \(b\) represents the bias term.

For binary classification, the predicted class is determined by the sign of the decision function:

\[
\hat{y}=\operatorname{sign}(w^T x+b)
\]

### Maximum-Margin Principle

SVM aims to find a separating hyperplane that maximizes the margin between the decision boundary and the nearest observations from each class.

The margin width for a canonical linear SVM is:

\[
\text{Margin}=\frac{2}{\|w\|}
\]

Maximizing the margin helps the model learn a decision boundary with strong geometric separation between classes.

### Support Vectors

Support vectors are the training observations that determine or constrain the position of the decision boundary. They are typically the points closest to the separating hyperplane.

These observations play a central role in defining the learned SVM model.

## 2. Hard-Margin and Soft-Margin SVM

### Hard-Margin SVM

Hard-margin SVM assumes the training data is linearly separable and requires every observation to be correctly classified with the specified margin.

Its optimization objective is:

\[
\min_{w,b}\frac{1}{2}\|w\|^2
\]

Subject to:

\[
y_i(w^T x_i+b)\geq 1
\]

This formulation is sensitive to overlapping classes and outliers.

### Soft-Margin SVM

Soft-margin SVM allows certain observations to violate the margin or be misclassified by introducing slack variables.

Its primal optimization objective is:

\[
\min_{w,b,\xi}\frac{1}{2}\|w\|^2+C\sum_{i=1}^{n}\xi_i
\]

Subject to:

\[
y_i(w^T x_i+b)\geq 1-\xi_i
\]

\[
\xi_i\geq 0
\]

The parameter \(C\) controls the trade-off between maximizing the margin and penalizing training violations.

- **Higher \(C\):** Places a stronger penalty on training violations, potentially producing a narrower margin.
- **Lower \(C\):** Allows more violations in exchange for stronger regularization and potentially a wider margin.

The best setting depends on the dataset and validation performance.

## 3. Kernel Methods

Kernel methods enable SVM to learn nonlinear decision boundaries without explicitly constructing every feature in a high-dimensional transformed space.

Common kernel functions include:

| Kernel | Purpose |
|---|---|
| `linear` | Models linear decision boundaries |
| `poly` | Captures polynomial relationships |
| `rbf` | Models flexible nonlinear relationships |
| `sigmoid` | Uses a sigmoid-shaped kernel function |

### Linear Kernel

\[
K(x,z)=x^Tz
\]

Suitable when a linear decision boundary is appropriate.

### Polynomial Kernel

\[
K(x,z)=(\gamma x^Tz+r)^d
\]

Where \(d\) is the polynomial degree, \(\gamma\) is a kernel coefficient, and \(r\) is the independent term.

### Radial Basis Function (RBF) Kernel

\[
K(x,z)=\exp(-\gamma\|x-z\|^2)
\]

The RBF kernel measures similarity based on the distance between observations and can produce nonlinear decision boundaries.

- **Higher \(\gamma\):** Each observation has a more localized influence, potentially creating complex boundaries.
- **Lower \(\gamma\):** Each observation has a broader influence, often producing smoother boundaries.

The effects of kernel parameters should be evaluated using appropriate validation methods.

## 4. Important Hyperparameters

| Parameter | Function |
|---|---|
| `C` | Controls the penalty for margin violations |
| `kernel` | Selects the kernel function |
| `gamma` | Controls kernel influence for applicable kernels |
| `degree` | Sets the polynomial degree |
| `coef0` | Controls the independent term in polynomial and sigmoid kernels |
| `tol` | Sets the tolerance for the stopping criterion |
| `max_iter` | Limits the number of optimization iterations |
| `class_weight` | Adjusts class-specific penalties |
| `probability` | Enables probability estimation in `SVC`, with additional computational cost |

Parameter availability and effects depend on the estimator and kernel configuration.

## 5. Feature Scaling

SVM is sensitive to feature scales because distances, dot products, and margin geometry influence its optimization.

A feature with a large numerical range can disproportionately affect the model if features are not scaled appropriately.

Common preprocessing methods include:

- `StandardScaler`
- `MinMaxScaler`

For reliable evaluation, fit preprocessing transformations on the training data only. Use a pipeline to prevent data leakage between training and validation or test sets.

## 6. SVM for Classification and Regression

### Support Vector Classification (SVC)

`sklearn.svm.SVC` supports binary classification and multiclass classification through strategies implemented by scikit-learn.

### LinearSVC

`sklearn.svm.LinearSVC` is designed for linear classification and can be suitable for larger datasets where a linear decision boundary is appropriate.

### Support Vector Regression (SVR)

`sklearn.svm.SVR` applies support vector methods to regression by using an epsilon-insensitive loss formulation.

## 7. Practical Implementation with Scikit-learn

A basic SVM classification pipeline can be implemented as follows:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the pipeline
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

# Train the model
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

This example demonstrates the standard workflow. Actual performance depends on the dataset, preprocessing, and hyperparameter configuration.

## 8. Model Evaluation

Useful classification metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive observations correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect predictions by class.
- **ROC-AUC:** Evaluates ranking performance across classification thresholds when applicable.

For imbalanced datasets, precision, recall, F1-score, and class-specific performance can provide more useful insight than accuracy alone.

## 9. Advantages and Limitations

### Advantages

- Effective in high-dimensional feature spaces.
- Supports linear and nonlinear classification.
- Uses support vectors to define the decision boundary.
- Incorporates regularization through its optimization objective.
- Can work well on small- to medium-sized datasets.

### Limitations

- Training can become computationally expensive on large datasets, particularly with nonlinear kernels.
- Performance can depend heavily on feature scaling and hyperparameter selection.
- Probability estimation may require additional computation.
- Models with nonlinear kernels can be harder to interpret than linear models.
- Performance may decline when classes overlap heavily or the chosen kernel is unsuitable.

## 10. Key Takeaways

This learning exercise helped me understand SVM as a margin-based learning method rather than just another classification algorithm.

Key concepts explored include:

- Maximum-margin optimization.
- Support vectors and separating hyperplanes.
- Hard-margin and soft-margin formulations.
- Regularization and the `C` parameter.
- Kernel functions and nonlinear decision boundaries.
- Feature scaling and model evaluation.
- SVM applications in classification and regression.

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Numerical Computing:** NumPy
- **Data Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook / Kaggle Notebook

## Future Exploration

- Visualizing support vectors and decision boundaries.
- Comparing linear, polynomial, and RBF kernels.
- Evaluating the effect of `C` and `gamma`.
- Performing cross-validation and systematic hyperparameter search.
- Comparing SVM with Logistic Regression, K-Nearest Neighbors, and tree-based classifiers.
- Investigating SVM performance on imbalanced datasets.

## References

- [Scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Scikit-learn: SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Scikit-learn: SVR Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

---

**Learning approach:** Understand the mathematics → implement the algorithm → experiment with model configurations → evaluate performance → interpret the results.
