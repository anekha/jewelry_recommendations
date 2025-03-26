import streamlit as st
import matplotlib.pyplot as plt

def recommendation_results(recommendations):
    """
    Prints the results of jewelry recommendations based on the analysis of facial features and skin undertones.

    Parameters:
    - recommendations (dict): A dictionary containing all recommendation data.
    """
    # Define custom styles
    input_style = "<style>span.stTextInput>div>div>div>input, div.stMarkdown h2 {color: #FF007F;}</style>"
    output_style = "<style>div.stMarkdown, div.stMarkdown p {color: #808080;}</style>"
    st.markdown(input_style, unsafe_allow_html=True)
    st.markdown(output_style, unsafe_allow_html=True)

    # Display personal features as text
    st.markdown("## Personal Features", unsafe_allow_html=True)
    st.write("Face shape: oval")
    st.write("Undertone Classification: warm")

    # Display color analysis section
    st.markdown("## Color Analysis", unsafe_allow_html=True)
    st.write("Your dominant colors are:")
    plot_colors(recommendations["colors"], recommendations["hist"])

    # Display jewelry recommendations section
    st.markdown("## Jewelry Design Recommendations", unsafe_allow_html=True)
    jewelry_design = recommendations.get("jewelry_design")
    if jewelry_design:
        for design in jewelry_design:
            st.write(design)
    else:
        st.write("No specific recommendations")

    # Display gemstone recommendations section
    st.markdown("## Gemstone Recommendations", unsafe_allow_html=True)
    gemstone_shape = recommendations.get("gemstone_shape", "No specific recommendations")
    gemstone_recommendation = recommendations.get("gemstone_recommendation", [])
    st.write("Gemstone Shape:", gemstone_shape)
    st.write("Gemstone Recommendation:")
    for recommendation in gemstone_recommendation:
        st.write(recommendation)

    # Display metal color recommendations section
    st.markdown("## Metal Color Recommendations", unsafe_allow_html=True)
    metal_color_recommendation = recommendations.get("metal_color_recommendation", "Yellow gold or rose gold to complement warmer skin tones")

    # Ensuring the recommendation is a string and not a list or any other data structure
    if isinstance(metal_color_recommendation, list):
        # If it's a list, join it into a single string
        metal_color_recommendation = ', '.join(metal_color_recommendation)

    # Display the recommendation as plain text
    st.write(metal_color_recommendation)


def plot_colors(colors, hist):
    """
    Plots the dominant colors as a color block.

    Parameters:
    - colors (np.ndarray): Array containing RGB values of dominant colors.
    - hist (np.ndarray): Array containing the corresponding histogram values.
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_axis_off()
    for i, (percent, color) in enumerate(zip(hist, colors)):
        rect = plt.Rectangle((i / len(colors), 0), 1 / len(colors), 1, color=color / 255)
        ax.add_patch(rect)
    st.pyplot(fig)
