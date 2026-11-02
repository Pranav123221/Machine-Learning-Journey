# Wrapper Methods for Feature Selection in Machine Learning

Wrapper methods ka use machine learning mein sabse best aur relevant features chunne ke liye kiya jata hai. Yeh methods ek specific Machine Learning algorithm ko **"black box"** ki tarah use karte hain aur alag-alag feature combinations ki performance (accuracy/score) ko evaluate karte hain.

---

## 🚀 Key Benefits (Fayde)
* **High Accuracy:** Yeh direct model ki performance ke hisab se features chunta hai, isliye iska result bohot accurate hota hai.
* **Feature Interactions:** Yeh check karta hai ki do ya do se zyada features aapas mein milkar model ko kitna behtar banate hain.

⚠️ **Limitation:** Yeh computationally bohot expensive hote hain kyunki isme model ko baar-baar train karna padta hai. Isliye yeh bade datasets par slow kaam karte hain.

---

## 🛠️ Main Techniques

### 1. Forward Selection
* **Kaise kaam karta hai:** Yeh khali (0 features) se shuru hota hai. Har step mein ek-ek karke sabse best feature ko tab tak joda jata hai jab tak desired performance ya feature count na mil jaye.
* **Kab use karein:** Jab features ki sankhya kam ho aur aap jaldi shuruat karna chahte hon.

### 2. Backward Elimination
* **Kaise kaam karta hai:** Yeh saare features se shuru hota hai. Har step mein sabse bekar (least significant) feature ko ek-ek karke tab tak hataya jata hai jab tak sirf kaam ke features na bachein.
* **Kab use karein:** Jab aapko lagta hai ki lagbhag saare features thode-bohot zaroori hain aur aap unka interaction check karna chahte hain.

### 3. Recursive Feature Elimination (RFE)
* **Kaise kaam karta hai:** Yeh poore features par model train karta hai, unki importance (weights/coefficients) check karta hai, aur sabse low importance wale features ko prune (hata) deta hai. Yeh process loop mein chalta hai.

---

## 💻 Quick Code Implementation (`scikit-learn`)

Aap `scikit-learn` ki `SequentialFeatureSelector` class ka use karke isko aasaani se implement kar sakte hain:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

# 1. Dummy Data (10 features)
X, y = make_classification(n_samples=100, n_features=10, n_informative=3, random_state=42)
model = LogisticRegression()

# 2. Forward Selection Setup
forward_selector = SequentialFeatureSelector(model, n_features_to_select=3, direction="forward", cv=5)
forward_selector.fit(X, y)
print("Forward Selected Features:", np.where(forward_selector.get_support())[0])

# 3. Backward Elimination Setup
backward_selector = SequentialFeatureSelector(model, n_features_to_select=3, direction="backward", cv=5)
backward_selector.fit(X, y)
print("Backward Selected Features:", np.where(backward_selector.get_support())[0])
```

---

## 📊 Summary Table: Wrapper vs Filter vs Embedded

| Feature | Wrapper Methods | Filter Methods | Embedded Methods |
| :--- | :--- | :--- | :--- |
| **Speed** | 🔴 Slow (High Compute) | 🟢 Fast (Statistical) | 🟡 Medium (Inside Model) |
| **Overfitting Risk** | 🔴 High | 🟢 Low | 🟡 Medium |
| **Model Dependency** | Yes (Model-specific) | No (Independent) | Yes (In-built) |
| **Best For** | Chote aur high-accuracy datasets | Bohot bade (High-dimensional) data ke liye | Tree models aur Regularization (Lasso) ke liye |
