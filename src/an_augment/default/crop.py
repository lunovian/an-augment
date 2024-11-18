def crop(image, top, left, height, width):
    """
    Crop an image to the specified size and position.
    
    Args:
        image (numpy.ndarray): Input image.
        top (int): Top pixel coordinate.
        left (int): Left pixel coordinate.
        height (int): Desired height.
        width (int): Desired width.
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    if top < 0 or left < 0 or height <= 0 or width <= 0 or top + height > image.shape[0] or left + width > image.shape[1]:
        raise ValueError("Invalid crop parameters.")
    return image[top:top+height, left:left+width]