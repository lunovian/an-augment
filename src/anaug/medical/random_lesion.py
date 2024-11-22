import numpy as np
from scipy.ndimage import gaussian_filter

def random_lesion(
    image, 
    intensity_range=(0.2, 0.8), 
    size_range=(10, 50), 
    shape='circle', 
    location=None, 
    texture_strength=0.5
):
    """
    Generates a random lesion with specified size, shape, and location.

    Parameters:
    - image (np.ndarray): Input image as a 2D numpy array.
    - intensity_range (tuple): Min and max intensity for the lesion.
    - size_range (tuple): Min and max size of the lesion (radius for circles or largest dimension for other shapes).
    - shape (str): Shape of the lesion ('circle', 'ellipse', 'irregular').
    - location (tuple): Center of the lesion as (x, y). If None, a random location is chosen.
    - texture_strength (float): Strength of texture variation (0 for smooth, 1 for highly textured).

    Returns:
    - np.ndarray: Image with the generated lesion.
    """
    # Validate inputs
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D numpy array.")
    if shape not in ['circle', 'ellipse', 'irregular']:
        raise ValueError("Shape must be one of 'circle', 'ellipse', or 'irregular'.")

    # Generate lesion properties
    intensity = np.random.uniform(*intensity_range)
    size = np.random.randint(*size_range)
    if location is None:
        center_x = np.random.randint(size, image.shape[1] - size)
        center_y = np.random.randint(size, image.shape[0] - size)
    else:
        center_x, center_y = location

    # Create lesion shape
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    if shape == 'circle':
        lesion = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * size**2))
    elif shape == 'ellipse':
        size_y = size
        size_x = size * np.random.uniform(0.5, 1.5)  # Randomize x-to-y ratio
        lesion = np.exp(-(((x - center_x)**2 / (2 * size_x**2)) + ((y - center_y)**2 / (2 * size_y**2))))
    elif shape == 'irregular':
        lesion = np.random.uniform(-1, 1, image.shape)
        lesion = gaussian_filter(lesion, sigma=size / 2)
        lesion = (lesion - lesion.min()) / (lesion.max() - lesion.min())  # Normalize to [0, 1]

    # Add texture
    texture = np.random.uniform(-1, 1, image.shape)
    texture = gaussian_filter(texture, sigma=size * (1 - texture_strength))
    lesion *= 1 + texture_strength * texture

    # Scale lesion intensity
    lesion = lesion * intensity

    # Add lesion to the image
    augmented_image = image + lesion
    return np.clip(augmented_image, 0, 1)  # Ensure values remain within [0, 1]