# ROC Curve — Receiver Operating Characteristic

## Overview

The **Receiver Operating Characteristic (ROC) Curve** is a performance evaluation tool used primarily for **binary classification models**.

It evaluates how well a classifier separates positive and negative classes across different **classification thresholds**, rather than evaluating the model at only one fixed threshold such as 0.5.

The ROC Curve plots:

- **True Positive Rate (TPR)** on the Y-axis
- **False Positive Rate (FPR)** on the X-axis

This makes it useful for understanding the trade-off between correctly detecting positive samples and incorrectly classifying negative samples as positive.

---

## Why ROC Curve?

A classification model usually produces a probability rather than directly producing a class.

For example:

```text
0.91
0.76
0.63
0.48
0.21

A threshold can then be used to convert these probabilities into class predictions.

Changing the threshold changes the number of:

True Positives (TP)
False Positives (FP)
True Negatives (TN)
False Negatives (FN)

The ROC Curve evaluates this behavior across multiple thresholds.

Key Metrics
True Positive Rate (TPR)

Also called Sensitivity or Recall.

$$ TPR = \frac{TP}{TP + FN} $$

It measures the proportion of actual positive samples correctly identified by the model.

False Positive Rate (FPR)
$$ FPR = \frac{FP}{FP + TN} $$

It measures the proportion of actual negative samples incorrectly classified as positive.

ROC Curve

The ROC Curve is created by plotting:

$$ X = FPR $$ $$ Y = TPR $$

for different classification thresholds.

The ideal classifier moves toward the top-left corner, where:

TPR → 1
FPR → 0


Understanding the ROC Curve

A model with better class-separation ability generally produces a curve closer to the top-left corner.

The diagonal line represents random classification, where:

$$ TPR \approx FPR $$

Therefore:

ROC-AUC ≈ 1.0 → Excellent discrimination
ROC-AUC ≈ 0.5 → No better than random
ROC-AUC < 0.5 → Worse than random


ROC-AUC

Area Under the ROC Curve (AUROC) summarizes the model's ability to distinguish between positive and negative classes across different thresholds.

A higher AUC generally indicates better class discrimination.

Importantly, ROC-AUC evaluates the ranking/separation ability of the model, rather than relying on a single classification threshold.

Practical Workflow

In this notebook, I explored the ROC Curve practically using a classification model:

Train a binary classification model
Generate predicted probabilities
Calculate TPR and FPR across thresholds
Plot the ROC Curve
Calculate ROC-AUC
Analyze the model's classification performance


Key Learning

The main takeaway from this implementation was that classification performance should not always be evaluated using a single threshold.

The ROC Curve provides a threshold-independent view of how effectively a model separates the two classes.

It helped me connect:

Predicted Probability → Threshold → Confusion Matrix → TPR/FPR → ROC Curve → ROC-AUC



Conclusion

The ROC Curve is a useful evaluation technique for understanding the discriminative performance of binary classification models.

Rather than asking only "How accurate is the model at one threshold?", ROC analysis helps answer:

"How well can the model distinguish between positive and negative classes across different decision thresholds?"

This makes ROC-AUC particularly useful when comparing binary classification models based on their overall class-separation capability.

