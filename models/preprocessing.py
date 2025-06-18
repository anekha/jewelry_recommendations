from mtcnn import MTCNN
import numpy as np
from PIL import Image
import os
import json

def extract_keypoints(dataset_dir, output_file):
    detector = MTCNN()
    keypoints_dict = {}
    total_files_processed = 0

    for shape_category in os.listdir(dataset_dir):
        shape_path = os.path.join(dataset_dir, shape_category)
        if os.path.isdir(shape_path):
            print(f"Processing images in category: {shape_category}")
            for image_file in os.listdir(shape_path):
                if image_file.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(shape_path, image_file)
                    print(f"Processing image: {image_path}")
                    try:
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        detections = detector.detect_faces(image_array)
                        total_files_processed += 1
                        if detections:
                            keypoints_dict[image_path] = detections[0]['keypoints']
                            print(f"Faces detected in image: {image_path}, Keypoints: {detections[0]['keypoints']}")
                        else:
                            print(f"No faces detected in {image_path}")
                    except Exception as e:
                        print(f"Failed to process image {image_path}: {e}")

    print(f"Total images processed: {total_files_processed}")
    if keypoints_dict:
        try:
            with open(output_file, 'w') as f:
                json.dump(keypoints_dict, f)
            print(f"Keypoints successfully written to {output_file}")
        except Exception as e:
            print(f"Failed to write keypoints to {output_file}: {e}")
    else:
        print(f"No keypoints to write for {dataset_dir}")

# Path to the directory containing preprocessed images
dataset_dir = '/Users/anekha/Documents/GitHub/jewelrecs/dataset/preprocessed/training_set'
# Output file paths for training and validation keypoints
training_output_file = '/Users/anekha/Documents/GitHub/jewelrecs/models/training_keypoints.json'
validation_output_file = '/Users/anekha/Documents/GitHub/jewelrecs/models/validation_keypoints.json'

# Extract keypoints for training set
extract_keypoints(dataset_dir, training_output_file)
# Adjust dataset_dir for validation set
dataset_dir = '/Users/anekha/Documents/GitHub/jewelrecs/dataset/preprocessed/validation_set'
extract_keypoints(dataset_dir, validation_output_file)
