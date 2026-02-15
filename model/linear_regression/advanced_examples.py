import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Advanced Linear Regression Techniques

### Synthetic Data Creation
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2

### Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Normal Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

### Regularization Techniques
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_ridge_pred = ridge.predict(X_test)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_lasso_pred = lasso.predict(X_test)

### Model Evaluation
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
print("Ridge Regression MSE:", mean_squared_error(y_test, y_ridge_pred))
print("Lasso Regression MSE:", mean_squared_error(y_test, y_lasso_pred))

### Diagnostics
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm)
results = model.fit()
print(results.summary())

### Optimization Techniques
# This part requires more complex scenarios, can include gradient descent or other optimizers

### Visualization
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='green', label='Linear Regression')
plt.plot(X_test, y_ridge_pred, color='red', label='Ridge Regression')
plt.plot(X_test, y_lasso_pred, color='orange', label='Lasso Regression')
plt.title('Advanced Linear Regression Techniques')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()