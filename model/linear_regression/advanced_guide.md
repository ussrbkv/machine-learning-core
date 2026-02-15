# Advanced Guide to Linear Regression

## Table of Contents
1. [Mathematical Theory](#mathematical-theory)
2. [Algorithms](#algorithms)
3. [Optimization Techniques](#optimization-techniques)
4. [Regularization](#regularization)
5. [Diagnostics](#diagnostics)
6. [Advanced Code Examples](#advanced-code-examples)

## Mathematical Theory
Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. The linear regression model assumes that the relationship between the variables can be expressed as a linear equation:

\[ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon \]

where:
- \(Y\) is the dependent variable.
- \(\beta_0\) is the y-intercept.
- \(\beta_1, \beta_2, ..., \beta_n\) are the coefficients of the predictors.
- \(X_1, X_2, ..., X_n\) are the independent variables.
- \(\epsilon\) is the error term.

The goal of linear regression is to find the coefficients that minimize the sum of the squared residuals, which is defined as:

\[ SSR = \sum (Y_i - \hat{Y}_i)^2 \]

where \(\hat{Y}_i\) is the predicted value of \(Y_i\).

## Algorithms
The most common algorithms for fitting linear regression models include:
1. **Ordinary Least Squares (OLS)**: The most straightforward approach that minimizes the sum of squared differences between observed and predicted values.
2. **Gradient Descent**: An iterative optimization algorithm that updates the coefficients using the gradient of the cost function until convergence.
3. **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that updates the model parameters with each training example sequentially.

## Optimization Techniques
- **Batch Gradient Descent**: Computes gradients using the entire dataset, ensuring convergence but often slower on large datasets.
- **Mini-Batch Gradient Descent**: A compromise between OLS and SGD, using small batches to update parameters, which speeds up convergence.
- **L-BFGS**: An optimization algorithm that approximates the Newton method and is efficient for high-dimensional problems.

## Regularization
Regularization techniques help to prevent overfitting by adding a penalty for larger coefficients in the loss function:
1. **Lasso Regression (L1 Regularization)**: Adds the absolute value of the coefficients as a penalty to the loss function.
2. **Ridge Regression (L2 Regularization)**: Adds the squared value of the coefficients as a penalty.
3. **Elastic Net**: Combines both L1 and L2 penalties in the loss function.

## Diagnostics
After fitting a linear regression model, it’s essential to diagnose its performance:
- **Residual Analysis**: Check residuals for patterns; they should be normally distributed and homoscedastic.
- **R-squared**: Indicates how much of the variability in the dependent variable can be explained by the independent variables.
- **Adjusted R-squared**: A modified version of R-squared that adjusts for the number of predictors in the model.
- **F-statistic**: Tests whether the overall regression model is a good fit.

## Advanced Code Examples
### Example 1: Fitting a Linear Regression Model using OLS
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2']]
Y = data['target']

# Add constant term for intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(Y, X).fit()

# Print the summary
print(model.summary())
```

### Example 2: Fitting a Lasso Model
```python
from sklearn.linear_model import Lasso

# Define model
lasso_model = Lasso(alpha=0.1)

# Fit model
lasso_model.fit(X, Y)

# Coefficients
print(lasso_model.coef_)
```
## Conclusion
This guide provides a comprehensive overview of linear regression, covering its theoretical foundations, practical algorithms, and techniques to ensure robust models. For further reading, explore the latest research papers, textbooks, and online courses.