import numpy as np

def extract_additional_features(keypoints):
    """
    Extracts additional features from facial keypoints.

    Parameters:
    - keypoints (dict): A dictionary containing positions of facial landmarks.

    Returns:
    - np.array: Array containing additional features.
    """
    features_list = []
    for k, v in keypoints.items():
        features = extract_features(v)
        features_list.append(features)
    return np.array(features_list)

def extract_features(keypoints):
    """
    Extracts specific features from facial keypoints.

    Parameters:
    - keypoints (dict): A dictionary containing positions of facial landmarks.

    Returns:
    - list: List containing extracted features.
    """
    if not all(key in keypoints for key in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']):
        raise ValueError("Some required keypoints are missing")

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
