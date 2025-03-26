import sys
sys.path.append('/Users/anekha/Documents/GitHub/jewelrecs/')

from models.features import X_train, y_train, X_val, y_val, X_test, y_test
from training_nn.train_nn import scaler, model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming your scaler and model are already fitted as per your previous code
X_test_scaled = scaler.transform(X_test)

# Predict classes with the model on the scaled test data
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate the accuracy manually or you can use the model.evaluate function as well
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print a detailed classification report
print(classification_report(y_test, y_pred_classes, target_names=np.unique(y_train).astype(str)))
