import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

def is_warm_color(color):
    """
    Determines if a given color is 'warm'.

    Parameters:
    - color (array-like): The RGB values of the color.

    Returns:
    - bool: True if the color is warm, False otherwise.
    """
    r, g, b = color
    return r > b and r > g  # Warm color has more red

def is_cool_color(color):
    """
    Determines if a given color is 'cool'.

    Parameters:
    - color (array-like): The RGB values of the color.

    Returns:
    - bool: True if the color is cool, False otherwise.
    """
    r, g, b = color
    return b > r and b > g  # Cool color has more blue

def classify_undertone(colors, hist):
    """
    Classifies the undertone of a set of colors based on their distribution.

    Parameters:
    - colors (list): A list of RGB colors.
    - hist (list): A list of percentages representing the distribution of each color.

    Returns:
    - str: The classified undertone ('warm', 'cool', or 'neutral').
    """
    warm_count = sum(percent for percent, color in zip(hist, colors) if is_warm_color(color))
    cool_count = sum(percent for percent, color in zip(hist, colors) if is_cool_color(color))

    if warm_count > 0.7:
        return "warm"
    elif cool_count > 0.7:
        return "cool"
    else:
        return "neutral"

def detect_colors(image_path, num_colors=5):
    """
    Detects dominant colors in an image and classifies its undertone.

    Parameters:
    - image_path (str): Path to the image file.
    - num_colors (int): Number of dominant colors to identify.

    Returns:
    - None: Results are printed and displayed directly.
    """
    image = Image.open(image_path)
    image = image.convert('RGB')  # Convert image to RGB
    image = image.resize((600, 400), Image.Resampling.LANCZOS)
    pixels = np.array(image.getdata()).reshape((-1, 3))

    clt = KMeans(n_clusters=num_colors, n_init=10)
    clt.fit(pixels)

    colors = clt.cluster_centers_
    hist = np.bincount(clt.labels_, minlength=num_colors).astype('float')
    hist /= hist.sum()

    display_color_block(colors, hist)
    generate_color_info(colors, hist)

def display_color_block(colors, hist):
    """
    Displays a color block representing the color distribution.

    Parameters:
    - colors (array): Array of RGB colors.
    - hist (array): Histogram of color distribution.
    """
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    start = 0
    for percent, color in zip(hist, colors):
        end = start + int(percent * 300)
        color_bgr = color.astype("uint8")[::-1]  # Convert RGB to BGR for OpenCV
        cv2.rectangle(rect, (start, 0), (end, 50), color_bgr.tolist(), -1)
        start = end

    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
    plt.show()

def generate_color_info(colors, hist):
    """
    Prints detailed information about the dominant colors.

    Parameters:
    - colors (array): Array of RGB colors.
    - hist (array): Histogram of color distribution.
    """
    print("\nColor Detection Summary:")
    print("- Algorithm: K-means clustering")
    print("- Description: A method for identifying dominant colors in an image.")
    print("- Number of Colors Detected:", len(colors))
    print("Dominant Colors:")
    for i, (percent, color) in enumerate(zip(hist, colors)):
        color_rgb = tuple(color.astype(int))
        print(f"  Color {i+1}: RGB: {color_rgb} - {percent:.2%}")
