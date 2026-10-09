# Naive Bayes Classifier | Probabilistic Machine Learning

## 📌 Overview

Today, I studied the **Naive Bayes Classifier**, a supervised machine learning algorithm based on Bayes’ Theorem. It predicts class labels by estimating posterior probabilities and assumes that features are conditionally independent given the target class.

## 🧠 Key Concepts Learned

* **Bayes’ Theorem:** Mathematical foundation for probabilistic inference.
* **Prior Probability:** Probability of a class before observing the input features.
* **Likelihood:** Probability of observing the features given a particular class.
* **Posterior Probability:** Updated probability of a class given the observed features.
* **Conditional Independence:** The simplifying assumption that features are independent given the class label.
* **Probabilistic Classification:** Assigning a class based on estimated probabilities.

## 📐 Mathematical Foundation

Bayes’ Theorem:

$$
P(C \mid X) = \frac{P(X \mid C)P(C)}{P(X)}
$$

Where:

* \(P(C \mid X)\): Posterior probability
* \(P(X \mid C)\): Likelihood
* \(P(C)\): Prior probability
* \(P(X)\): Evidence

Under the Naive Bayes conditional-independence assumption, the feature likelihood can be factorized:

$$
P(X \mid C) = \prod_{i=1}^{n} P(x_i \mid C)
$$

The predicted class is the one with the highest posterior probability.

## ⚙️ Types of Naive Bayes

1. **Gaussian Naive Bayes:** Suitable for continuous numerical features, assuming a Gaussian distribution within each class.
2. **Multinomial Naive Bayes:** Commonly used for word counts and text classification.
3. **Bernoulli Naive Bayes:** Suitable for binary features, such as word presence or absence.

## 🚀 Applications

* Email spam detection
* Sentiment analysis
* Document and news classification
* Text categorization
* Basic medical and risk classification tasks

## ✅ Advantages

* Computationally efficient for training and inference.
* Effective for many high-dimensional text-classification problems.
* Requires relatively little training data in some settings.
* Supports probabilistic predictions.

## ⚠️ Limitations

* The conditional-independence assumption may not hold in real-world datasets.
* Zero-frequency features can cause zero likelihoods without smoothing.
* Predicted probabilities may require calibration.

## 🛠️ Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib / Seaborn

## 🎯 Learning Outcome

Developed an understanding of probabilistic classification, Bayes’ Theorem, prior and posterior probabilities, likelihood estimation, conditional independence, and the major variants of the Naive Bayes algorithm.


---

**Learning Focus:** Supervised Machine Learning | Probabilistic Inference | Classification Algorithms
