import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compound_from_molcas(*args, **kwargs):
    """
    Display an image of Tomasz with specified file path.
    """
    # Specify the desired Tomasz's size (width, height) in inches
    fig_width = 14  # Adjust the width as needed
    fig_height = 8  # Adjust the height as needed
    
    # Load the image
    img = mpimg.imread("Presentations/tomek.png")
    
    # Create a Tomasz's figure with the specified size
    plt.figure(figsize=(fig_width, fig_height))
    # Load and display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axis labels
    plt.show()