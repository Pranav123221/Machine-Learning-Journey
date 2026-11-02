# Filter Methods for Feature Selection in Machine Learning

Filter methods ka use machine learning mein features ko evaluate aur select karne ke liye kiya jata hai. Yeh methods kisi bhi Machine Learning model (algorithm) par depend **nahi** karte. Yeh puri tarah se data ke statistical properties (jaise correlation, variance, chi-square) par kaam karte hain aur model training se pehle hi bekar features ko **"filter out"** kar dete hain.

---

## 🚀 Key Benefits (Fayde)
* **Extremely Fast:** Yeh bohot tez hote hain kyunki isme koi model train nahi karna padta. Sirf math aur stats calculations hoti hain.
* **No Overfitting Risk:** Yeh model independent hote hain, isliye overfitting ka khatra bohot kam hota hai.
* **Best for High-Dimensional Data:** Jab dataset mein hazaron (thousands) features hon, toh sabse pehle filter method hi lagaya jata hai.

⚠️ **Limitation:** Yeh features ke aapas ka interaction (relationship) check nahi kar paate. Yeh har feature ko akele-akele evaluate karte hain.

---

## 🛠️ Main Techniques & Types

### 1. Variance Threshold (Unsupervised)
* **Kaise kaam karta hai:** Yeh un features ko hata deta hai jinki value lagbhag constant (zero ya low variance) hoti hai. Agar kisi column mein sabhi rows ke liye ek hi value hai, toh wo model ko kuch naya nahi sikhayega.

### 2. Chi-Square Test (Categorical Data)
* **Kaise kaam karta hai:** Yeh target variable aur categorical features ke beech ke dependence ko check karta hai. Higher value matlab feature aur target aapas mein zyada related hain.

### 3. ANOVA / Correlation (Numerical Data)
* **Kaise kaam karta hai:** `SelectKBest` ke sath `f_classif` (ANOVA) ya `f_regression` ka use karke hum numerical features ka target variable ke sath linear relation check karte hain.

---

## 💻 Quick Code Implementation (`scikit-learn`)

Aap `scikit-learn` ke `VarianceThreshold` aur `SelectKBest` ka use karke isko aasaani se implement kar sakte hain:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# 1. Dummy Data (10 features, 3 useful, aur 1 bilkul constant/dead feature)
X, y = make_classification(n_samples=100, n_features=10, n_informative=3, random_state=42)
X[:, 9] = 5  # Feature index 9 ko constant bana diya (zero variance)

print("Original Data Shape:", X.shape)

# --- 1. VARIANCE THRESHOLD (Low variance features hatane ke liye) ---
# Yeh feature 9 (constant) ko automatic delete kar dega
variance_selector = VarianceThreshold(threshold=0.1)
X_variance = variance_selector.fit_transform(X)
print("After Variance Threshold Shape:", X_variance.shape)


# --- 2. SELECT KBEST (Top 3 sabse best statistical features chunne ke liye) ---
# f_classif (ANOVA) ka use karke top 3 features select kiye ja rahe hain
kbest_selector = SelectKBest(score_func=f_classif, k=3)
X_kbest = kbest_selector.fit_transform(X, y)

# Selected features ke index dekhne ke liye
selected_indices = np.where(kbest_selector.get_support())
print("SelectKBest Chosen Features:", selected_indices)
print("Final Filtered Data Shape:", X_kbest.shape)
```

---

## 📊 Summary Table: Filter vs Wrapper

| Feature | Filter Methods | Wrapper Methods |
| :--- | :--- | :--- |
| **Speed** | 🟢 Extremely Fast | 🔴 Slow (High Compute) |
| **Model Dependent**| 🟢 No (Puri tarah Independent) | 🔴 Yes (Model ke sath chalta hai) |
| **Mechanism** | Statistical tests (Correlation, Variance) | Iterative Search (Forward/Backward) |
| **Best For** | Initial Screening (Hazaron features ke liye) | Final Tuning (Chote datasets ke liye) |
