import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Load Iris dataset
def load_data():
    iris = load_iris()
    X = iris.data  # Feature matrix
    y = iris.target  # Labels
    feature_names = iris.feature_names
    return X, y, feature_names

# Standardize the data
def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Apply PCA with scikit-learn
def apply_pca_sklearn(X, num_components=2):
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(X)
    print("Explained variance ratio (sklearn):", pca.explained_variance_ratio_)
    return principal_components

# Visualization of PCA results (2D)s
def plot_pca(X_pca, y, title='PCA'):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for target, color in zip(np.unique(y), colors):
        indices_to_keep = y == target
        plt.scatter(X_pca[indices_to_keep, 0], X_pca[indices_to_keep, 1], 
                    c=color, s=50, label=f"Class {target}")
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Setosa', 'Versicolor', 'Virginica'])
    plt.grid()
    plt.show()

# Visualization of original data in 3D
def plot_3d(X, y, title='3D Plot of Original Data'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    
    for target, color in zip(np.unique(y), colors):
        indices_to_keep = y == target
        ax.scatter(X[indices_to_keep, 0], X[indices_to_keep, 1], X[indices_to_keep, 2], 
                   c=color, s=50, label=f"Class {target}")
    
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.legend(['Setosa', 'Versicolor', 'Virginica'])
    plt.show()

# Train a simple classifier on PCA-transformed data
def train_classifier(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

def main():
    # 1. Load the data
    X, y, feature_names = load_data()

    # 2. Standardize the data
    X_scaled = standardize_data(X)

    # 3. Plot original data in 3D
    plot_3d(X_scaled, y, title="3D Plot of Original Data (Standardized)")

    # 4. Apply PCA using scikit-learn
    X_pca_sklearn = apply_pca_sklearn(X_scaled, num_components=2)
    plot_pca(X_pca_sklearn, y, title="PCA with scikit-learn")

    # 5. Train a classifier on PCA-transformed data
    X_train, X_test, y_train, y_test = train_test_split(X_pca_sklearn, y, test_size=0.2, random_state=42)
    train_classifier(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
