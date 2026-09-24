# Multiple Linear Regression

A practical implementation of **Multiple Linear Regression** using Python and Scikit-learn to understand how multiple independent features can be used to predict a continuous target variable.

## 📌 Overview

Multiple Linear Regression extends Simple Linear Regression by using **multiple input features** to model the relationship between several predictors and a continuous target.

The project focuses on understanding the complete workflow:

**Data → Feature Preparation → Train/Test Split → Model Training → Prediction → Evaluation → Visualization**

## 🎯 Objectives

* Understand Multiple Linear Regression practically
* Work with multiple independent variables
* Train a regression model using Scikit-learn
* Understand learned coefficients and intercept
* Generate predictions on unseen data
* Evaluate regression performance
* Visualize predictions in multi-dimensional feature space

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**

## ⚙️ Implementation

### 1. Feature and Target Separation

The dataset is divided into independent variables (`X`) and the dependent target variable (`y`).

```python
X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['target']
```

### 2. Train-Test Split

The dataset is divided into training and testing subsets to evaluate the model on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Model Training

A `LinearRegression` model from Scikit-learn is trained using the training data.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)
```

### 4. Prediction

The trained model is used to predict the target values for the test dataset.

```python
y_pred = lr.predict(X_test)
```

### 5. Coefficients and Intercept

The learned coefficients and intercept can be inspected to understand the parameters learned by the model.

```python
print(lr.coef_)
print(lr.intercept_)
```

The coefficients represent the estimated contribution of each feature while the model considers the other features.

## 📊 Visualization

For a two-feature regression example, a feature grid was created using NumPy's `meshgrid`.

```python
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)

xGrid, yGrid = np.meshgrid(y, x)

final = np.vstack((
    xGrid.ravel().reshape(1, 100),
    yGrid.ravel().reshape(1, 100)
)).T

z_final = lr.predict(final).reshape(10, 10)
```

The resulting predictions can be used to visualize a **3D regression surface**, showing how predicted values change across the feature space.

## 🧠 Key Learnings

* Difference between Simple and Multiple Linear Regression
* Working with multiple independent variables
* Model fitting and prediction using Scikit-learn
* Interpretation of regression coefficients
* Train-test splitting for model evaluation
* Generating predictions across a feature grid
* Understanding regression in higher-dimensional feature spaces
* Connecting mathematical concepts with practical ML implementation

## 📁 Project Structure

```text
multiple-linear-regression/
│
├── Multiple_Linear_Regression.ipynb
├── README.md
└── dataset/
    └── dataset.csv
```

## 🚀 Future Improvements

* Add regression metrics such as **MAE, MSE, RMSE and R²**
* Compare predictions with actual values
* Analyze residuals
* Experiment with different datasets
* Compare Multiple Linear Regression with other regression algorithms

## 👨‍💻 Author

**Pranav Sharma**

B.Tech CSE (AI/ML) | Machine Learning Enthusiast

Building practical projects while strengthening my foundations in Machine Learning.
