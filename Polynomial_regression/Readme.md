# 📈 Polynomial Regression

A practical implementation of **Polynomial Regression** as part of my Machine Learning learning journey.

This project focuses on understanding how polynomial features can help regression models capture **non-linear relationships** between input and target variables.

## 🎯 Objective

The main goal of this project was to understand:

* What Polynomial Regression is
* Why Linear Regression may fail on non-linear data
* How polynomial features are created
* How Polynomial Regression works with **Linear Regression**
* Different polynomial degrees and their effect on the model
* Model fitting and prediction
* Underfitting vs. overfitting
* Evaluating regression performance

## 🧠 Concept

Polynomial Regression extends Linear Regression by transforming the input features into polynomial features.

For example, instead of using only:

`y = b₀ + b₁x`

a polynomial model can learn relationships such as:

`y = b₀ + b₁x + b₂x² + b₃x³ + ...`

This allows the model to represent curved, non-linear patterns while still using Linear Regression internally.

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 📚 What I Practiced

### 1. Data Preparation

* Loading and inspecting the dataset
* Selecting features and target variables
* Preparing data for regression

### 2. Polynomial Feature Engineering

Used `PolynomialFeatures` from Scikit-learn to generate polynomial features of different degrees.

### 3. Model Training

Trained a Linear Regression model using the generated polynomial features.

### 4. Prediction

Used the trained model to generate predictions for unseen/input values.

### 5. Model Evaluation

Evaluated the regression model using appropriate regression metrics and visualizations.

### 6. Model Complexity

Experimented with different polynomial degrees to understand how increasing model complexity can lead to:

* Underfitting
* Better fit
* Overfitting

## 📊 Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Polynomial Feature Transformation
   ↓
Linear Regression
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation & Visualization
```

## 🔍 Key Learning

Polynomial Regression is useful when the relationship between variables is **non-linear**.

However, increasing the polynomial degree also increases model complexity. A very high degree can make the model fit the training data extremely closely and potentially perform poorly on unseen data.

This helped me understand the importance of finding an appropriate balance between **model complexity and generalization**.

## 📁 Project Structure

```text
polynomial-regression/
│
├── Polynomial_Regression.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Future Improvements

* Experiment with different polynomial degrees
* Compare Linear Regression vs Polynomial Regression
* Add cross-validation
* Compare multiple evaluation metrics
* Explore Ridge Regression with polynomial features
* Build a small deployment/demo using the trained model

## 📌 Learning Journey

This project is part of my ongoing **Machine Learning learning journey**, where I am building and implementing core ML algorithms from fundamentals to more advanced concepts.

**Next:** Continuing with more regression and machine learning concepts. 🚀
