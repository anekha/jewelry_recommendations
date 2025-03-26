from mtcnn import MTCNN
import numpy as np
from PIL import Image
import json

def test_keypoint_extraction(image_path):
    detector = MTCNN()
    image = Image.open(image_path)
    image_array = np.array(image)
    detections = detector.detect_faces(image_array)

    if detections:
        keypoints = detections[0]['keypoints']
        print(f"Keypoints for {image_path}: {keypoints}")
        return keypoints
    else:
        print(f"No faces detected in {image_path}")
        return None

# Specify the path to the image file
image_path = '/Users/anekha/Documents/GitHub/jewelrecs/dataset/training_set/Heart/heart (2).jpg'
keypoints = test_keypoint_extraction(image_path)
