# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
print(data.data[0])
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)  # Using 10 trees to simplify visualization
clf.fit(X_train, y_train)

# Plot a few trees (e.g., the first, second, and third tree)
for i in range(3):
    plt.figure(figsize=(10,8))  # Set plot size for better visibility
    plot_tree(clf.estimators_[i], 
              feature_names=data.feature_names,  
              class_names=data.target_names, 
              filled=True, rounded=True)
    plt.title(f"Decision Tree {i+1} of the Random Forest")
    plt.show()