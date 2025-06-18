#prepare the testing dataset
import sys
sys.path.append('/Users/anekha/Documents/GitHub/jewelrecs/')

from models.preprocessing import extract_keypoints

# Path to the directory containing preprocessed images
dataset_dir = '/Users/anekha/Documents/GitHub/jewelrecs/dataset/preprocessed/testing_set'
# Output file paths for training and validation keypoints
testing_output_file = '/Users/anekha/Documents/GitHub/jewelrecs/models/testing_keypoints.json'

# Extract keypoints for testing set
extract_keypoints(dataset_dir, testing_output_file)
