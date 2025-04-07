# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Linear Regression
# Generating a synthetic dataset for linear regression
np.random.seed(42)
X_linear = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y_linear = 4 + 3 * X_linear + np.random.randn(100, 1)  # Keeping y as 2D array to avoid shape mismatch

# Splitting data into training and testing sets
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Applying Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_lr, y_train_lr)

# Predictions
y_pred_lr = linear_model.predict(X_test_lr)

# Evaluating Linear Regression
print("Linear Regression Coefficients:", linear_model.coef_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test_lr, y_pred_lr))

# Plotting Linear Regression results
plt.scatter(X_linear, y_linear, color='blue', label='Actual Data')  # Ensure both X and y are 2D arrays
plt.plot(X_linear, linear_model.predict(X_linear), color='red', label='Regression Line')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 2. Logistic Regression
# Generating a synthetic dataset for logistic regression
X_logistic, y_logistic = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Splitting data into training and testing sets
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

# Applying Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_log, y_train_log)

# Predictions
y_pred_log = logistic_model.predict(X_test_log)

# Evaluating Logistic Regression
print("\nLogistic Regression Accuracy:", accuracy_score(y_test_log, y_pred_log))
print("Classification Report:\n", classification_report(y_test_log, y_pred_log))

# Visualizing Logistic Regression results
plt.scatter(X_logistic[:, 0], X_logistic[:, 1], c=y_logistic, cmap='coolwarm', label='Data Points')
plt.title("Logistic Regression Data Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

