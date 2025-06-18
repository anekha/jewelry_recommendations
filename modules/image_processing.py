import numpy as np
from PIL import Image
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from color_analysis import classify_undertone

def detect_faces_landmarks_and_colors(image_path, num_colors=5):
    """
    Detects faces, extracts landmarks, and analyzes color from an image.

    Parameters:
    - image_path (str): Path to the image file.
    - num_colors (int): Number of dominant colors to identify.

    Returns:
    - tuple: keypoints, undertone, colors, and histogram of colors if face detected, else None for each.
    """
    # Initialize the MTCNN detector
    detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7])

    # Open and convert image to RGB
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Detect faces in the image
    results = detector.detect_faces(image_array)
    if not results:
        print("No faces detected.")
        return None, None, None, None

    # Sort faces by area (width * height) in descending order and select the largest
    results = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    result = results[0]

    # Check if the detected face is large enough to be considered valid
    if result['box'][2] * result['box'][3] < 5000:
        return None, None, None, None

    # Process face details
    keypoints = result['keypoints']
    draw_face_details(image_array, result['box'], keypoints)

    # Color analysis
    colors, hist = perform_color_analysis(image_array, num_colors)

    # Classify undertone
    undertone = classify_undertone(colors, hist)

    return keypoints, undertone, colors, hist

def draw_face_details(image_array, box, keypoints):
    """Draws rectangle and landmarks on the face."""
    x, y, width, height = box
    cv2.rectangle(image_array, (x, y), (x + width, y + height), (0, 255, 0), 2)
    for point in keypoints.values():
        cv2.circle(image_array, point, 5, (255, 0, 0), -1)

def perform_color_analysis(image_array, num_colors):
    """Performs color analysis on the image."""
    small_image = Image.fromarray(image_array).resize((600, 400), Image.Resampling.LANCZOS)
    small_pixels = np.array(small_image).reshape((-1, 3))
    clt = KMeans(n_clusters=num_colors, n_init=10)
    clt.fit(small_pixels)
    hist = np.bincount(clt.labels_, minlength=num_colors).astype('float')
    hist /= hist.sum()
    return clt.cluster_centers_, hist
