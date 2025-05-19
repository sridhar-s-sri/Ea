5 Exploratory Data Analysis on 2-variable Linear Regression

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Generate random data

np.random.seed(0)

x = 2 * np.random.rand(100, 1)

y = 4 + 3 * x + np.random.randn(100, 1)

# Visualize the data

plt.scatter(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Generated Data')

plt.grid(True)

plt.show()

# Fit linear regression

model = LinearRegression()

model.fit(x, y)

# Print parameters

print("Intercept:", model.intercept_)

print("Slope:", model.coef_)

# Plot regression line

plt.scatter(x, y)

plt.plot(x, model.predict(x), color='red', label='Regression Line')

plt.xlabel('x')

plt.ylabel('y')

plt.title('Linear Regression')

plt.legend()

plt.grid(True)
plt.show()

# Correlation

correlation = np.corrcoef(x.flatten(), y.flatten())[0, 1]

print("Correlation between x and y:", correlation)

6. Regression Model With and Without Bias

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Generate random data

np.random.seed(1)

X = 2 * np.random.rand(100, 1)

y = 5 + 2 * X + np.random.randn(100, 1)

# Model with bias

model_bias = LinearRegression(fit_intercept=True)

model_bias.fit(X, y)

# Model without bias

model_no_bias = LinearRegression(fit_intercept=False)

model_no_bias.fit(X, y)

# Sort X for smooth line plotting

sorted_idx = X[:, 0].argsort()

X_sorted = X[sorted_idx]

y_sorted_bias = model_bias.predict(X)[sorted_idx]

y_sorted_no_bias = model_no_bias.predict(X)[sorted_idx]

# Plot

plt.scatter(X, y, label='Data')

plt.plot(X_sorted, y_sorted_bias, color='blue', label='With Bias')

plt.plot(X_sorted, y_sorted_no_bias, color='green', label='Without Bias')

plt.xlabel('X')

plt.ylabel('y')
plt.title('Linear Regression Comparison')

plt.legend()

plt.grid(True)

plt.show()

7. Classification using Perceptron with and without Bias

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import accuracy_score

# Load the dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

data = pd.read_csv(url, names=column_names)

# Encode labels

le = LabelEncoder()

y = le.fit_transform(data['species']) # Setosa=0, Versicolor=1, Virginica=2

X = data.drop('species', axis=1)

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Perceptron without bias

model_no_bias = Perceptron(fit_intercept=False)

model_no_bias.fit(X_train, y_train)

y_pred_no_bias = model_no_bias.predict(X_test)

accuracy_no_bias = accuracy_score(y_test, y_pred_no_bias)

print("Accuracy of perceptron without bias:", accuracy_no_bias)

# Perceptron with bias

model_with_bias = Perceptron(fit_intercept=True)

model_with_bias.fit(X_train, y_train)

y_pred_with_bias = model_with_bias.predict(X_test)

accuracy_with_bias = accuracy_score(y_test, y_pred_with_bias)

print("Accuracy of perceptron with bias:", accuracy_with_bias)
