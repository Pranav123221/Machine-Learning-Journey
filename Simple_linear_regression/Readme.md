# Simple Linear Regression: Complete Theory & Mathematical Guide

A comprehensive, theoretical reference explaining the inner workings, mathematics, assumptions, optimization methods, and limitations of **Simple Linear Regression (SLR)**.

---

## 1. What is Simple Linear Regression?

**Simple Linear Regression** is a foundational supervised learning algorithm used to model the relationship between two continuous variables:
- **Independent variable ($X$):** Also called predictor, feature, or explanatory variable.
- **Dependent variable ($y$):** Also called response, target, or outcome variable.

The goal is to find the best-fitting straight line that predicts $y$ given a value of $x$.

### The Linear Equation

The theoretical population model is formulated as:

$$y = \beta_0 + \beta_1 x + \epsilon$$

For a sample dataset of $n$ observations, the predicted value $\hat{y}_i$ for each data point $x_i$ is given by:

$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$

Where:
- $\hat{\beta}_0$ (**Intercept**): The expected value of $y$ when $x = 0$.
- $\hat{\beta}_1$ (**Slope**): The expected change in $y$ for a 1-unit increase in $x$.
- $\epsilon_i$ (**Error / Residual**): The deviation between the actual target and the predicted value: $e_i = y_i - \hat{y}_i$.

---

## 2. Mathematical Derivation (Ordinary Least Squares)

The **Ordinary Least Squares (OLS)** method determines parameters $\beta_0$ and $\beta_1$ by minimizing the **Residual Sum of Squares (RSS)**:

$$J(\beta_0, \beta_1) = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n \left(y_i - (\beta_0 + \beta_1 x_i)\right)^2$$

### Step 1: Partial Derivative with respect to $\beta_0$

Set the partial derivative to zero to find the critical point:

$$\frac{\partial J}{\partial \beta_0} = -2 \sum_{i=1}^n \left(y_i - \beta_0 - \beta_1 x_i\right) = 0$$

$$\sum_{i=1}^n y_i - n\beta_0 - \beta_1 \sum_{i=1}^n x_i = 0$$

Divide by $n$:

$$\bar{y} - \beta_0 - \beta_1 \bar{x} = 0$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

*(This proves that the regression line always passes through the mean point $(\bar{x}, \bar{y})$).*

### Step 2: Partial Derivative with respect to $\beta_1$

Substitute $\beta_0 = \bar{y} - \beta_1 \bar{x}$ into the error equation:

$$y_i - \hat{y}_i = (y_i - \bar{y}) - \beta_1 (x_i - \bar{x})$$

Taking the derivative with respect to $\beta_1$ and setting it to zero yields:

$$\frac{\partial J}{\partial \beta_1} = -2 \sum_{i=1}^n (x_i - \bar{x}) \left((y_i - \bar{y}) - \beta_1 (x_i - \bar{x})\right) = 0$$

$$\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) = \beta_1 \sum_{i=1}^n (x_i - \bar{x})^2$$

Solving for $\hat{\beta}_1$:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{\mathrm{Cov}(X, y)}{\mathrm{Var}(X)}$$

Alternatively, in terms of Pearson's correlation coefficient ($r$):

$$\hat{\beta}_1 = r \cdot \frac{s_y}{s_x}$$

Where $s_y$ and $s_x$ are sample standard deviations of $y$ and $x$.

---

## 3. Alternative Optimization: Gradient Descent

While OLS provides an exact closed-form analytical solution, iterative optimization via **Gradient Descent** is widely used when data does not fit in memory or when scaling to multiple dimensions.

### Cost Function (Mean Squared Error)

$$J(\beta_0, \beta_1) = \frac{1}{2n} \sum_{i=1}^n \left((\beta_0 + \beta_1 x_i) - y_i\right)^2$$

### Update Rules

Iteratively update parameters until convergence using learning rate $\alpha$:

$$\beta_0 := \beta_0 - \alpha \frac{\partial J}{\partial \beta_0} = \beta_0 - \alpha \left(\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)\right)$$

$$\beta_1 := \beta_1 - \alpha \frac{\partial J}{\partial \beta_1} = \beta_1 - \alpha \left(\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i) x_i\right)$$

---

## 4. Core Assumptions (Gauss-Markov Theorem)

OLS estimators are the **Best Linear Unbiased Estimators (BLUE)** only if these core conditions hold:

1. **Linearity:** The relationship between the independent variable $X$ and the conditional mean of $y$ is strictly linear.
   - *Diagnostic:* Scatter plot of $X$ vs. $y$ or Residuals vs. Fitted values.
2. **Strict Exogeneity:** The expected value of errors conditional on $X$ is zero:
   $$\mathbb{E}[\epsilon_i \mid X] = 0$$
3. **Homoscedasticity:** Constant error variance across all values of $X$:
   $$\mathrm{Var}(\epsilon_i \mid X) = \sigma^2$$
   - *Diagnostic:* Scatter of residuals vs. $\hat{y}$. If a fan or funnel shape appears, the assumption is violated (heteroscedasticity).
4. **Independence of Errors:** No autocorrelation or serial correlation:
   $$\mathrm{Cov}(\epsilon_i, \epsilon_j) = 0 \quad \forall i \neq j$$
   - *Diagnostic:* Durbin-Watson statistic ($d \approx 2$ indicates zero autocorrelation).
5. **Normality of Errors:** Residuals follow a normal distribution (required for confidence intervals and p-values):
   $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$
   - *Diagnostic:* Q-Q Plot (Quantile-Quantile) or Shapiro-Wilk test.

---

## 5. Model Evaluation Metrics

### 1. Coefficient of Determination ($R^2$)
Measures the proportion of variance in $y$ explained by $x$:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- $R^2 = 1$: Perfect predictions.
- $R^2 = 0$: Model performs no better than predicting the mean $\bar{y}$.
- $R^2 < 0$: Model performs worse than the horizontal mean line.

### 2. Mean Absolute Error (MAE)
Measures the average magnitude of absolute residuals:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

### 3. Mean Squared Error (MSE)
Penalizes larger errors disproportionately due to squaring:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

### 4. Root Mean Squared Error (RMSE)
Retains the unit of the target variable $y$:

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

---

## 6. Limitations & Considerations

- **Outlier Sensitivity:** Because OLS squares the residuals, extreme outliers have significant leverage and pull the regression line away from the genuine trend.
- **Underfitting:** Fails to capture non-linear dynamics, thresholds, or saturation points without polynomial feature engineering.
- **Extrapolation Risk:** Predictions made outside the observed range of $x$ are unreliable.
- **Correlation Does Not Imply Causation:** A high $R^2$ indicates strong association, not causal direction.
