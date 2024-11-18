from scipy.ndimage import gaussian_filter, uniform_filter

def blur(image, blur_type='gaussian', blur_radius=1):
    """
    Applies blur to a given image.
    
    Parameters:
    - image (np.array): Input image as a 2D or 3D numpy array.
    - blur_type (str): Type of blur to apply ('gaussian' or 'uniform').
    - blur_radius (float): Standard deviation for Gaussian kernel or size for uniform filter. Higher values increase blur.

    Returns:
    - np.array: Blurred image with the same shape as input.
    """
    if blur_type == 'gaussian':
        return gaussian_filter(image, sigma=blur_radius)
    elif blur_type == 'uniform':
        return uniform_filter(image, size=blur_radius)
    else:
        raise ValueError("Unsupported blur type. Use 'gaussian' or 'uniform'.")