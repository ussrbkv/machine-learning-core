"""
Comprehensive Linear Regression Examples for Beginners
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data for the linear regression
np.random.seed(0)  # For reproducibility
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Plot the results
plt.scatter(x_test, y_test, color='black', label='Actual Data')
plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Predicted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()

# Display the coefficients
print(f'Coefficient: {model.coef_[0][0]}')
print(f'Intercept: {model.intercept_[0]}')

# Evaluate the model (R^2 score)
score = model.score(x_test, y_test)
print(f'R^2 Score: {score}')