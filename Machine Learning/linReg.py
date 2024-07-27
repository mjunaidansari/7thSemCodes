import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'x': [17, 13, 12, 15, 16, 14, 16, 16, 18, 19],
    'y': [94, 73, 59, 80, 93, 85, 66, 79, 77, 91]
}

df = pd.DataFrame(data)

X = df[['x']]
y = df['y']

# create the model
model = LinearRegression()
model.fit(X, y)

print("Correlation:", df['x'].corr(df['y']))
print("Intercept: ", model.intercept_)
print("Slope: ", model.coef_[0])

# predict for a new value of x
new_x = pd.DataFrame({'x': [7]})
predicted_y = model.predict(new_x)
print(f"\nPredicted y for x: {new_x['x'][0]} is {predicted_y[0]}")

# plot the graph
plt.scatter(X, y, color='blue', label='Training Data')

# prepare values for plotting the regression line
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)

plt.plot(X_plot, y_plot, color='red', label='Regression Line') # 1D array for x axis
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
