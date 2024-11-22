import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def random_lesion(image, intensity_range=(0.2, 0.8), size_range=(10, 50), texture_strength=0.5):
    """
    Adds a synthetic lesion to the input image with random intensity and texture.

    Parameters:
    - image (np.ndarray): Input image as a 2D numpy array (grayscale).
    - intensity_range (tuple): Min and max intensity for the lesion (relative to the image range).
    - size_range (tuple): Min and max size of the lesion in pixels (radius).
    - texture_strength (float): Strength of the texture pattern (0 for smooth, 1 for highly textured).

    Returns:
    - np.ndarray: Image with the generated lesion.
    """
    # Validate input
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D array.")

    # Generate random lesion parameters
    intensity = np.random.uniform(*intensity_range)
    size = np.random.randint(*size_range)
    center_x = np.random.randint(size, image.shape[1] - size)
    center_y = np.random.randint(size, image.shape[0] - size)

    # Create lesion shape (Gaussian blob)
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    lesion = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * size**2))

    # Add texture to the lesion
    texture = np.random.uniform(-1, 1, image.shape)
    texture = gaussian_filter(texture, sigma=size * (1 - texture_strength))
    lesion *= 1 + texture_strength * texture

    # Scale lesion intensity
    lesion = lesion * intensity

    # Add lesion to the image
    augmented_image = image + lesion
    return np.clip(augmented_image, 0, 1)  # Keep pixel values within range