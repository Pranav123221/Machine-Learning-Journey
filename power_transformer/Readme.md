# PowerTransformer: Box-Cox & Yeo-Johnson Transformations

A practical exploration of **PowerTransformer** in Scikit-learn using the **Concrete Compressive Strength Dataset**.

This project demonstrates how Power Transformations can be used to transform skewed numerical features and compares the results of **Box-Cox** and **Yeo-Johnson** transformations.

## 🎯 Objective

The main objective is to understand:

* Why numerical feature distributions may need transformation
* How PowerTransformer works
* How Box-Cox transformation works
* How Yeo-Johnson transformation works
* The difference between Box-Cox and Yeo-Johnson
* How feature distributions change after transformation
* When each transformation should be used

---

## 📊 Dataset

**Concrete Compressive Strength Dataset**

The dataset contains different properties of concrete mixtures and their corresponding compressive strength.

The numerical features can have different distributions and levels of skewness, making the dataset useful for demonstrating feature transformation techniques.

---

## 🔄 Workflow

The notebook follows this workflow:

```text
Concrete Dataset
       ↓
Explore Numerical Features
       ↓
Analyze Feature Skewness
       ↓
PowerTransformer
       ↓
 ┌───────────────┬──────────────────┐
 ↓               ↓
Box-Cox       Yeo-Johnson
 ↓               ↓
Transform      Transform
 └───────────────┴──────────────────┘
       ↓
Compare Distributions
       ↓
Analyze Skewness
       ↓
Draw Conclusions
```

---

## 🧠 PowerTransformer

`PowerTransformer` is a Scikit-learn preprocessing technique used to transform numerical features so that their distributions become more Gaussian-like.

It supports two transformation methods:

```python
PowerTransformer(method="box-cox")
```

and

```python
PowerTransformer(method="yeo-johnson")
```

---

## 1️⃣ Box-Cox Transformation

Box-Cox is a power transformation that works with **strictly positive values**.

```python
from sklearn.preprocessing import PowerTransformer

pt_boxcox = PowerTransformer(method="box-cox")

X_boxcox = pt_boxcox.fit_transform(X)
```

### Important condition

Box-Cox requires:

```text
X > 0
```

Therefore, it cannot directly be applied to features containing zero or negative values.

### Purpose

Box-Cox can help:

* Reduce skewness
* Make distributions more symmetric
* Make numerical features more Gaussian-like
* Improve the suitability of features for some Machine Learning algorithms

---

## 2️⃣ Yeo-Johnson Transformation

Yeo-Johnson is a more flexible power transformation.

```python
pt_yeojohnson = PowerTransformer(method="yeo-johnson")

X_yeojohnson = pt_yeojohnson.fit_transform(X)
```

Unlike Box-Cox, Yeo-Johnson can handle:

* Positive values
* Zero values
* Negative values

This makes it useful when the feature contains values that cannot be directly used with Box-Cox.

---

## ⚖️ Box-Cox vs Yeo-Johnson

| Property                  | Box-Cox   | Yeo-Johnson   |
| ------------------------- | --------- | ------------- |
| Positive values           | ✅         | ✅             |
| Zero values               | ❌         | ✅             |
| Negative values           | ❌         | ✅             |
| Reduces skewness          | ✅         | ✅             |
| More flexible input range | ❌         | ✅             |
| Scikit-learn method       | `box-cox` | `yeo-johnson` |

### Simple Rule

**Box-Cox → strictly positive data**

**Yeo-Johnson → positive, zero, or negative data**

---

## 📈 Distribution Comparison

The notebook compares the feature distributions:

```text
Original Feature
       ↓
Highly Skewed Distribution
       ↓
Box-Cox Transformation
       ↓
More Symmetric Distribution
```

and:

```text
Original Feature
       ↓
Highly Skewed Distribution
       ↓
Yeo-Johnson Transformation
       ↓
More Symmetric Distribution
```

The before-and-after visualizations help demonstrate how the transformations affect the shape and skewness of the feature.

---

## 📊 Skewness Analysis

Skewness is calculated before and after transformation to quantitatively evaluate the effect.

A lower absolute skewness value generally indicates a more symmetric distribution.

The notebook compares:

* Original skewness
* Skewness after Box-Cox
* Skewness after Yeo-Johnson

This provides both **visual and numerical evidence** of the transformation's effect.

---

## 🔑 Key Learnings

Through this implementation, I learned that:

* PowerTransformer is useful for transforming skewed numerical features.
* Box-Cox requires strictly positive values.
* Yeo-Johnson can handle positive, zero, and negative values.
* Power transformations can significantly change the distribution of a feature.
* Transformation methods should be selected based on the characteristics of the data.
* Reducing skewness can be useful for ML algorithms that are sensitive to feature distributions.

---

## ⚠️ Important ML Practice

When using PowerTransformer as part of an actual Machine Learning workflow, the transformer should be **fitted only on the training data**.

```python
transformer.fit(X_train)

X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)
```

This prevents information from the test set from influencing the transformation and helps avoid **data leakage**.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Kaggle

---

## 📓 Kaggle Notebook

**PowerTransformer: Box-Cox & Yeo-Johnson Transformations**

The complete implementation, visualizations, and analysis are available in the accompanying Kaggle notebook.

---

## 🚀 Next Steps

The next step is to integrate PowerTransformer into complete Machine Learning workflows using:

* `Pipeline`
* `ColumnTransformer`
* Cross-validation
* Model evaluation
* Hyperparameter tuning

This will help move from understanding individual preprocessing techniques toward building complete and reproducible Machine Learning workflows.
