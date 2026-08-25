# Feature Transformation & Normality Analysis

This notebook explores different **feature transformation techniques** used in Machine Learning to modify the distribution of numerical features and make them more suitable for analysis and modeling.

The focus is on understanding **why transformations are needed, when to use them, and how to evaluate their effect on the data distribution**.

---

## 📌 Topics Covered

### 1. Function Transformer

A **Function Transformer** allows us to apply a mathematical function to one or more features.

It is useful when we want to apply transformations such as:

* Log
* Square root
* Reciprocal
* Other custom mathematical functions

The general idea is:

```text
Original Feature
       ↓
Mathematical Function
       ↓
Transformed Feature
```

In Scikit-learn, `FunctionTransformer` can be used to create a reusable transformation step.

---

### 2. Log Transformation

**Log transformation** applies the logarithm to the feature values.

It is commonly used when a numerical feature is **positively/right skewed**.

### Why use it?

* Reduces right skewness
* Compresses large values
* Can make the distribution more symmetric
* Can reduce the impact of extreme values

Conceptually:

```text
Original Data → Log Transformation → More Balanced Distribution
```

Example:

```python
np.log1p(x)
```

`log1p(x)` is often preferred because it safely handles zero values.

---

### 3. Square Root Transformation

The **square root transformation** is another technique for reducing skewness.

It is generally considered a **milder transformation compared with logarithmic transformation**.

### Why use it?

* Helps reduce moderate right skewness
* Compresses larger values
* Can improve the shape of a distribution

Example:

```python
np.sqrt(x)
```

---

### 4. Reciprocal Transformation

The **reciprocal transformation** transforms a feature using:

```text
1 / x
```

It can strongly change the shape of a distribution and may be useful for certain highly skewed features.

Example:

```python
1 / x
```

### Important

Reciprocal transformation requires care when values are:

* Zero
* Very close to zero
* Negative

Therefore, the nature of the feature should always be checked before applying this transformation.

---

## 5. Q-Q Plot

A **Q-Q (Quantile-Quantile) Plot** is a graphical technique used to assess whether a dataset approximately follows a particular theoretical distribution, commonly the **normal distribution**.

### How to interpret it?

If the points approximately follow a straight diagonal line:

```text
        •
      •
    •
  •
•
──────────────
```

the data is relatively close to the reference distribution.

Large deviations from the line indicate that the data differs from the expected distribution.

### Why use Q-Q plots?

Q-Q plots can help us compare:

* Original distribution
* Log-transformed distribution
* Square-root transformed distribution
* Reciprocal-transformed distribution

This allows us to visually evaluate whether a transformation has made the feature more normally distributed.

---

## 🔄 Transformation Workflow

The general workflow followed in this notebook is:

```text
Numerical Feature
       ↓
Analyze Distribution
       ↓
Identify Skewness
       ↓
Choose Transformation
       ↓
Apply Transformation
       ↓
Compare Distribution
       ↓
Use Q-Q Plot
       ↓
Evaluate the Result
```

---

## 📊 Transformations at a Glance

| Transformation       | Main Purpose                        | Typical Use                       |
| -------------------- | ----------------------------------- | --------------------------------- |
| Function Transformer | Apply mathematical/custom functions | Flexible feature transformation   |
| Log                  | Reduce strong right skewness        | Highly right-skewed positive data |
| Square Root          | Reduce moderate skewness            | Moderately right-skewed data      |
| Reciprocal           | Strongly alter distribution         | Certain highly skewed features    |
| Q-Q Plot             | Evaluate distribution/normality     | Before & after transformation     |

---

## 💡 Key Learnings

The most important takeaway is that **feature transformation should not be applied blindly**.

A good approach is:

> **Understand the data → Analyze its distribution → Identify skewness → Choose an appropriate transformation → Evaluate the transformed data**

Different datasets require different transformations, and the best transformation depends on the characteristics of the feature.

I also learned that a **Q-Q plot is not a transformation technique**. It is a diagnostic tool that helps us visually evaluate how closely the data follows a chosen theoretical distribution.

---

## 🛠️ Tools & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📓 Notebook

The complete practical implementation is available in:

`Feature_Transformation_and_Normality_Analysis.ipynb`

---

## 🎯 Goal

The goal of this notebook is to build a strong foundation in **feature engineering and data preprocessing**, which are important steps in developing effective Machine Learning models.

This is part of my ongoing **Machine Learning learning journey**.
