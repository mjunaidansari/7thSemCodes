from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_pred = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
y_true = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
specificity = TN / (TN + FP)

print("Confusion Matrix:")
print(cm)

# Print individual confusion matrix components
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")

# Print a blank line for readability
print()

# Print the accuracy, precision, recall, F1 score, and specificity metrics with two decimal precision
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")

