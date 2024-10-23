import pandas as pd 
import numpy as np

class DecisionTreeID3:
    def __init__(self): 
        self.tree = None
        self.feature_names = []

    def fit(self, data, target): 
        self.tree = self._build_tree(data, target)
    
    def _entropy(self, y): 
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, X, y, feature): 
        entropy_before = self._entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        entropy_after = np.sum([(counts[i] / np.sum(counts)) * self._entropy(y[X[:, feature] == values[i]]) for i in range(len(values))])
        return entropy_before - entropy_after
    
    def _best_feature(self, X, y): 
        gains = [self._information_gain(X, y, feature) for feature in range(X.shape[1])]
        return np.argmax(gains)
    
    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1: 
            return y[0]
        if X.shape[1] == 0: 
            return np.bincount(y).argmax()
        
        best_feature = self._best_feature(X, y)
        tree = { best_feature: {} }
        values = np.unique(X[:, best_feature])

        for value in values: 
            sub_data = X[X[:, best_feature] == value]
            sub_target = y[X[:, best_feature] == value]
            subtree = self._build_tree(sub_data, sub_target)
            tree[best_feature][value] = subtree

        return tree
    
    def _predict(self, tree, x):
        if not isinstance(tree, dict): 
            return tree
        feature = list(tree.keys())[0]
        value = x[feature]
        subtree = tree[feature].get(value, None)
        if subtree is None: 
            return None
        return self._predict(subtree, x)
    
    def predict(self, X): 
        return [bool(self._predict(self.tree, x)) for x in X]
    
    def print_tree(self, tree=None, feature_names=None, indent=""):
        if tree is None:
            tree = self.tree
        if feature_names is None:
            feature_names = self.feature_names
        
        if not isinstance(tree, dict):
            print(indent + "Predict:", tree)
            return
        
        feature_index = list(tree.keys())[0]
        feature_name = feature_names[feature_index]
        print(indent + feature_name + "?")
        
        for value, subtree in tree[feature_index].items():
            print(indent + " " * 4 + str(value) + ":")
            self.print_tree(subtree, feature_names, indent + " " * 8)

# Example usage
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot'],
    'Humidity': ['High', 'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'High', 'Medium', 'Medium', 'Low', 'Low', 'Low'],
    'Windy': [False, True, False, False, False, True, True, False, True, True, False, True, True],
})

# Convert categorical data to numeric
data_encoded = pd.get_dummies(data)
target = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0])  # 0: No, 1: Yes

# Set feature names
feature_names = data_encoded.columns.tolist()

# Train the decision tree
tree = DecisionTreeID3()
tree.feature_names = feature_names
tree.fit(data_encoded.values, target)

# Print the decision tree
tree.print_tree()

# New data for prediction
new_data = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['Low'],
    'Windy': [True],
})

# Convert new data to numeric
new_data_encoded = pd.get_dummies(new_data)

# Ensure new data has the same columns as the training data
new_data_encoded = new_data_encoded.reindex(columns=data_encoded.columns, fill_value=0)

# Predict new data
predictions = tree.predict(new_data_encoded.values)
print('\nPrediction for new data: ')
print(new_data)
print(predictions)
