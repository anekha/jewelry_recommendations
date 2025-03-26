import numpy as np

def classify_face_shape(keypoints):
    """
    Classifies the shape of a face based on facial keypoints.

    Parameters:
    - keypoints (dict): A dictionary containing positions of facial landmarks.

    Returns:
    - str: A string indicating the face shape.
    """
    # Convert keypoints to numpy arrays for easier distance calculations
    left_eye = np.array(keypoints['left_eye'])
    right_eye = np.array(keypoints['right_eye'])
    nose = np.array(keypoints['nose'])
    mouth_left = np.array(keypoints['mouth_left'])
    mouth_right = np.array(keypoints['mouth_right'])

    eye_dist = np.linalg.norm(left_eye - right_eye)
    nose_width = np.linalg.norm(nose - left_eye) + np.linalg.norm(nose - right_eye)
    face_length = np.linalg.norm(mouth_left - left_eye) + np.linalg.norm(mouth_right - right_eye)

    horizontal_proportion = eye_dist / nose_width
    vertical_proportion = face_length / ((eye_dist + nose_width) / 2)

    print(f"eye_dist: {eye_dist}, nose_width: {nose_width}, face_length: {face_length}")
    print(f"horizontal_proportion: {horizontal_proportion}, vertical_proportion: {vertical_proportion}")

    if vertical_proportion > 1.1:
        return "Oval"
    elif horizontal_proportion > 1:
        return "Round"
    elif face_length < nose_width and eye_dist < nose_width:
        return "Square"
    elif horizontal_proportion > 0.9 and vertical_proportion < 1.1:
        return "Heart"
    else:
        return "Oblong"
