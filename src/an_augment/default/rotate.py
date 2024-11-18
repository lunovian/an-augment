from scipy.ndimage import rotate as scipy_rotate

def rotate(image, angle):
    """
    Rotate the image by the specified angle.
    
    Parameters:
    - image (np.array): Input image as a 2D or 3D numpy array.
    - angle (float): Angle by which to rotate the image.
    
    Returns:
    - np.array: Rotated image.
    """
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a numeric value.")
    
    return scipy_rotate(image, angle, reshape=False, mode='nearest')