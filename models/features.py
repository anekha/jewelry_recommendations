import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load keypoints from JSON file
def load_keypoints(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Use previously defined functions to extract features
def extract_features(keypoints):
    # Assuming your feature extraction code is defined here
    left_eye = np.array(keypoints['left_eye'])
    right_eye = np.array(keypoints['right_eye'])
    nose = np.array(keypoints['nose'])
    mouth_left = np.array(keypoints['mouth_left'])
    mouth_right = np.array(keypoints['mouth_right'])

    eye_dist = np.linalg.norm(left_eye - right_eye)
    nose_eye_left = np.linalg.norm(nose - left_eye)
    nose_eye_right = np.linalg.norm(nose - right_eye)
    mouth_eye_left = np.linalg.norm(mouth_left - left_eye)
    mouth_eye_right = np.linalg.norm(mouth_right - right_eye)

    nose_width = nose_eye_left + nose_eye_right
    face_height = mouth_eye_left + mouth_eye_right

    horizontal_proportion = eye_dist / nose_width
    vertical_proportion = face_height / ((eye_dist + nose_width) / 2)

    return [eye_dist, nose_width, face_height, horizontal_proportion, vertical_proportion]

# Function to prepare data
def prepare_data(json_file):
    keypoints_data = load_keypoints(json_file)
    X = []
    y = []

    for path, keypoints in keypoints_data.items():
        features = extract_features(keypoints)
        X.append(features)

        # Assuming 'path' is the variable containing the file path
        label = path.split('/')[-2]  # Extract the label from the second-to-last element of the path
        y.append(label)

    # Convert X and y into numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y

# Example of how to prepare data
X_train, y_train = prepare_data('/Users/anekha/Documents/GitHub/jewelrecs/models/training_keypoints.json')
X_val, y_val = prepare_data('/Users/anekha/Documents/GitHub/jewelrecs/models/validation_keypoints.json')
X_test, y_test = prepare_data('/Users/anekha/Documents/GitHub/jewelrecs/models/testing_keypoints.json')

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Unique labels:", np.unique(y_train))
print("First few features in X_train:", X_train[:5])
print("First few labels in y_train:", y_train[:5])

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("Unique labels in validation set:", np.unique(y_val))
print("First few features in X_val:", X_val[:5])
print("First few labels in y_val:", y_val[:5])

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("Unique labels in test set:", np.unique(y_test))
print("First few features in X_test:", X_test[:5])
print("First few labels in y_test:", y_test[:5])
