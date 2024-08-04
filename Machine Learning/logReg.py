import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

# data = {
#     'x1': [14.5, 8.5],
#     'x2': [12.5, 4.5],
#     'y': [1, 0]
# }

data = {
    'x1': [14.5, 8.5, 15.0, 9.0, 13.0, 10.5, 16.0, 7.5, 12.0, 11.0],
    'x2': [12.5, 4.5, 13.0, 5.0, 11.5, 6.0, 14.0, 3.5, 10.0, 7.0],
    'y': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Binary target variable
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df[['x1', 'x2']].values
y = df['y'].values

# Fit a logistic regression model without scaling
model = LogisticRegression()
model.fit(X, y)

# Create a grid of values for x1 and x2
x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 100)
x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
y_prob = model.predict_proba(X_grid)[:, 1].reshape(x1_grid.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with coolwarm colormap
surf = ax.plot_surface(x1_grid, x2_grid, y_prob, cmap='coolwarm', edgecolor='none')

# Add a color bar for reference
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Probability')

# Plot the original data
ax.scatter(df['x1'], df['x2'], y, color='black', label='Data Points', edgecolor='k')

# Labels
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Probability')

plt.legend()
plt.show()
