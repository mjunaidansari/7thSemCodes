# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Wine dataset
wine = datasets.load_wine()

# For visualization, we will take only two features (e.g., Alcohol and Malic Acid)
X = wine.data[:, :2]  # Only the first two features for 2D plot (Alcohol and Malic Acid)
y = wine.target  # Class labels

# Filter the dataset to include only two classes (class 0 and class 1)
# Here, we remove class 2 from the dataset
binary_class_indices = y != 2  # Exclude class 2
X = X[binary_class_indices]
y = y[binary_class_indices]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with linear kernel
svm_model = SVC(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualizing the decision boundary

# Define function to plot decision boundaries
def plot_decision_boundary(X, y, model):
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Plot the decision boundary by assigning a color to each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel(wine.feature_names[0])  # Label for the x-axis (Alcohol)
    plt.ylabel(wine.feature_names[1])  # Label for the y-axis (Malic Acid)
    plt.title('SVM Decision Boundary with Support Vectors')

    # Plot support vectors
    support_vectors = model.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red', marker='x', label='Support Vectors')
    plt.legend()

    plt.show()

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, svm_model)
