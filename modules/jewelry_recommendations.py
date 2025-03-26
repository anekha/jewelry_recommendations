

def recommend_jewelry_face_shape(face_shape):
    """Returns jewelry recommendations based on the face shape."""
    recommendations = {
        "oval": ["Drop earrings, hoops, or studs with elongated shapes",
                 "Layered necklaces or pendants with medium to long lengths",
                 "Bangle bracelets or cuffs to add symmetry to the face shape"],
        "round": ["Long, angular earrings or dangling earrings to create vertical lines",
                  "Angular or geometric pendant necklaces to elongate the face",
                  "Slim, elongated bracelets or stacked bangles to add length to the arms"],
        "square": ["Soft, curved earrings or hoops to soften angular features",
                   "Long, layered necklaces or pendant necklaces with rounded shapes",
                   "Curved or rounded bracelets to balance the angular jawline"],
        "heart": ["Tear-drop or chandelier earrings to complement the wider forehead",
                  "V-shaped or sweetheart necklaces to accentuate the neckline",
                  "Delicate, heart-shaped bracelets or charm bracelets to add femininity"],
        "oblong": ["Large hoop or teardrop earrings to add width to the face",
                   "Shorter necklaces to reduce the appearance of length",
                   "Wide bracelets to balance facial proportions"]
    }
    return recommendations.get(face_shape.lower(), "No specific recommendations")

def recommend_gemstone_shape(face_shape):
    """Returns recommended gemstone shapes based on the face shape."""
    shapes = {
        "oval": "Oval-shaped gemstones to enhance the face's symmetry",
        "round": "Angular gemstone shapes like rectangles or squares to add definition",
        "square": "Oval or pear-shaped gemstones to soften facial angles",
        "heart": "Heart-shaped gemstones for a romantic look",
        "oblong": "Round or cushion-cut gemstones to add softness and reduce perceived length"
    }
    return shapes.get(face_shape.lower(), "No specific recommendations")

def recommend_gemstone_skin_tone(undertone):
    """Returns gemstone recommendations based on skin undertone."""
    recommendations = {
        "cool": ["Diamonds, sapphires, or amethysts to complement cooler skin tones"],
        "warm": ["Citrine, garnet, or topaz to enhance warmer skin tones"],
        "neutral": ["Aquamarine, emerald, or pearls for versatile options that suit most skin tones"]
    }
    return recommendations.get(undertone.lower(), [])

def recommend_metal_color_skin_tone(undertone):
    """Returns recommended metal colors based on skin undertone."""
    recommendations = {
        "cool": ["White gold or platinum for a flattering contrast with cooler skin tones"],
        "warm": ["Yellow gold or rose gold to complement warmer skin tones"],
        "neutral": ["Both white gold and yellow gold can work well, depending on personal preference"]
    }
    return recommendations.get(undertone.lower(), [])

def generate_jewelry_recommendations(image_path, num_colors=5):
    """Generates jewelry recommendations based on detected facial features and colors."""
    from image_processing import detect_faces_landmarks_and_colors  # Import here to avoid circular imports
    from face_detection import classify_face_shape  # Assuming classify_face_shape is in face_detection module

    keypoints, undertone, colors, hist = detect_faces_landmarks_and_colors(image_path, num_colors)
    recommendations = {}

    if keypoints:
        face_shape = classify_face_shape(keypoints).lower()
        recommendations['face_shape'] = face_shape
        recommendations['jewelry_design'] = recommend_jewelry_face_shape(face_shape)
        recommendations['gemstone_shape'] = recommend_gemstone_shape(face_shape)
        recommendations['gemstone_recommendation'] = recommend_gemstone_skin_tone(undertone.lower())
        recommendations['metal_color_recommendation'] = recommend_metal_color_skin_tone(undertone.lower())
        recommendations['undertone'] = undertone.lower()  # Add undertone to recommendations
        recommendations['colors'] = colors  # Add colors to recommendations
        recommendations['hist'] = hist  # Add histogram of colors to recommendations
    else:
        default_message = "No face detected or too small to analyze."
        recommendations.update({
            'face_shape': None,
            'jewelry_design': default_message,
            'gemstone_shape': default_message,
            'gemstone_recommendation': default_message,
            'metal_color_recommendation': default_message,
            'undertone': None,  # Set undertone to None when no keypoints are detected
            'colors': None,  # Set colors to None when no keypoints are detected
            'hist': None  # Set histogram of colors to None when no keypoints are detected
        })

    return recommendations
