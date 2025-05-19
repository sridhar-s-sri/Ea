import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

data = pd.DataFrame({

 'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

 'Y': [2, 4, 5, 4, 5, 8, 3, 1, 7, 4]

})

sns.scatterplot(x='X', y='Y', data=data)

plt.title("Scatter Plot of X vs Y")

plt.xlabel("X")

plt.ylabel("Y")

plt.show()

correlation = data['X'].corr(data['Y'])

print(f"Correlation coefficient: {correlation}")

x = data['X']

y = data['Y']

x_mean = x.mean()

y_mean = y.mean()

m = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

c = y_mean - m * x_mean

print(f"Slope (m): {m}")

print(f"Intercept (c): {c}")

y_pred = m * x + c

plt.scatter(x, y, label='Actual')

plt.plot(x, y_pred, color='red', label='Regression Line')

plt.xlabel('X')

plt.ylabel('Y')

plt.title("Linear Regression Model")

plt.legend()

plt.show()
