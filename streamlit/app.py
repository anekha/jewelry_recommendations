import sys
import os
import numpy as np
from PIL import Image
import tempfile
import streamlit as st

# Calculate the correct path to the modules directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
modules_dir = os.path.join(parent_dir, 'modules')

# Add the modules directory to sys.path
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

# Import your custom modules
from image_processing import detect_faces_landmarks_and_colors
from jewelry_recommendations import generate_jewelry_recommendations
from utils import recommendation_results

# Set the title of your app and customize its color
st.title('Jewelry Recommendation System')
st.markdown("<style>h1 {color: #FF69B4;}</style>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your photo", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error("Error opening the image. Please try uploading again.")
    else:
        # Only display the button if the image is successfully loaded
        if st.button('Generate Recommendations'):
            st.write("Analyzing the image...")
            try:
                # Save the PIL image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name  # Preserve the file path

                # Generate recommendations using the path to the temporary file
                recommendations = generate_jewelry_recommendations(tmp_path, num_colors=5)

                # Display results
                recommendation_results(recommendations)

                # Clean up the temporary file
                os.unlink(tmp_path)
            except Exception as e:
                st.error("Error processing the image. Please try uploading again.")
else:
    st.write("Awaiting image upload...")
