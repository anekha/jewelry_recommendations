import os
import shutil
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocess_images(source_dir, dest_dir, size=(224, 224)):
    """
    Resize, convert to RGB, and save images to a new directory, maintaining the structure.
    This function ignores non-image files.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):  # Only process directories
            new_folder_path = os.path.join(dest_dir, folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.png', '.gif')):  # Process only image files
                    img_path = os.path.join(folder_path, file)
                    try:
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize(size, Image.Resampling.LANCZOS)
                        img.save(os.path.join(new_folder_path, file), 'JPEG')
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                else:
                    print(f"Skipping non-image file: {file}")
        else:
            print(f"Skipping non-directory file: {folder_path}")

def split_training_set(train_dir, val_dir, test_size=0.2):
    """
    Split the training data into training and validation sets and move validation data to a separate directory.
    """
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for class_folder in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_folder)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.gif'))]
            train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)

            val_class_path = os.path.join(val_dir, class_folder)
            if not os.path.exists(val_class_path):
                os.makedirs(val_class_path)

            for file in val_files:
                shutil.move(os.path.join(class_path, file), os.path.join(val_class_path, file))
        else:
            print(f"Skipping non-directory file: {class_path}")

# Example usage
source_directory = 'dataset/training_set'
destination_directory = 'dataset/preprocessed'
preprocess_images(source_directory, destination_directory + '/training_set')
preprocess_images('dataset/testing_set', 'dataset/preprocessed/testing_set')

# Split training set into training and validation sets within the preprocessed directory
split_training_set(destination_directory + '/training_set', destination_directory + '/validation_set')
