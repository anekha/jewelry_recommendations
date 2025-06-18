import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from face_detection import classify_face_shape

def load_keypoints(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def prepare_data(json_file):
    keypoints_data = load_keypoints(json_file)
    X = []
    y = []

    for path, keypoints in keypoints_data.items():
        # Convert keypoints to a flattened list of coordinates
        features = [
            keypoints['left_eye'][0], keypoints['left_eye'][1],
            keypoints['right_eye'][0], keypoints['right_eye'][1],
            keypoints['nose'][0], keypoints['nose'][1],
            keypoints['mouth_left'][0], keypoints['mouth_left'][1],
            keypoints['mouth_right'][0], keypoints['mouth_right'][1]
        ]
        X.append(features)
        label = path.split('/')[-2]  # Extract the label from the file path
        y.append(label)

    # Convert X and y into numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder.classes_

# Load data
X_test, y_test, class_labels = prepare_data('/Users/anekha/Documents/GitHub/jewelrecs/models/testing_keypoints.json')

# Classify face shapes using the provided keypoints
predicted_labels = [classify_face_shape({
    'left_eye': keypoints[0:2],
    'right_eye': keypoints[2:4],
    'nose': keypoints[4:6],
    'mouth_left': keypoints[6:8],
    'mouth_right': keypoints[8:10]
}) for keypoints in X_test]

# Decode label integers back to string labels
predicted_labels_encoded = LabelEncoder().fit_transform(predicted_labels)

# Evaluate the predictions
print("Classification Report:")
print(classification_report(y_test, predicted_labels_encoded, target_names=class_labels))
print("Accuracy:", accuracy_score(y_test, predicted_labels_encoded))
