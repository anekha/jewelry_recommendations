# Jewelry Recommendation System

## Overview

The **Jewelry Recommendation System** uses advanced image analysis and machine learning techniques to provide personalized jewelry recommendations based on your uploaded photo. The system analyzes your facial features, skin undertone, and color profile to suggest jewelry designs and gemstones that suit you best.

- **[JewelRecs App Link](https://jewelryrecs.streamlit.app/)**

![JewelRecs Screenshot](/presentation/app_ex.png)

## Project Poster

For a detailed overview of this work, check out the project poster:

📄 [Download the Poster (PDF)](/presentation/project_presentation.pdf)

## Features

- **Face Shape Analysis:** The system identifies your face shape (e.g., oval, round, square) to recommend suitable jewelry designs.
- **Undertone Classification:** It classifies your skin tone into warm or cool undertones to help recommend the right metal color and gemstone.
- **Color Analysis:** The system performs a color analysis to determine your dominant colors, enhancing the recommendation process.
- **Personalized Jewelry Recommendations:** Based on the analysis, it suggests jewelry styles, including earrings, necklaces, and bracelets.
- **Gemstone and Metal Color Recommendations:** It provides suggestions on gemstones and metals that complement your skin tone and facial features.

## How It Works

1. **Upload Image**: Drag and drop your photo (JPG, PNG, JPEG). The system will analyze the image and extract features like face shape and undertone.
2. **Personal Features**: The system determines your face shape and undertone classification.
3. **Jewelry Design Recommendations**: Based on your face shape and skin tone, it suggests jewelry designs (e.g., earrings, necklaces).
4. **Gemstone Recommendations**: It recommends the best gemstones (e.g., Citrine, Garnet) and gemstone shapes (e.g., oval) based on your skin tone.
5. **Metal Color Recommendations**: It suggests metal colors (e.g., yellow gold or rose gold) that complement your skin tone.

## Installation

To run this system locally, follow the steps below:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/jewelrecs.git
    ```

2. Navigate into the project directory:
    ```bash
    cd jewelrecs
    ```

3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. After installation, run the app locally using Streamlit:
    ```bash
    streamlit run streamlit/app.py
    ```

2. The app should open in your web browser. Upload an image and let the system analyze it to receive personalized jewelry recommendations.

## App Interface

- **Upload Photo**: The interface allows you to upload a photo (max 200MB, JPG, PNG, JPEG).
- **Personal Features**: Displays your face shape and undertone classification.
- **Jewelry Recommendations**: Displays jewelry design suggestions (earrings, necklaces, bracelets).
- **Gemstone Recommendations**: Provides gemstone shapes and types suitable for your skin tone.
- **Metal Color Recommendations**: Suggests metal colors (yellow gold, rose gold) based on your undertone.

## License

---

**Developed by**: Anekha Sokhal  
For any questions or suggestions, feel free to open an issue or contact me directly.
