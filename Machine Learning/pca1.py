import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load and standardize the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA result
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for target, color in zip(np.unique(y), colors):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], c=color, label=iris.target_names[target])
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
