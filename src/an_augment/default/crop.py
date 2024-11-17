def crop(image, crop_size):
    """
    Crop an image to the specified size.
    
    Args:
        image (numpy.ndarray): Input image.
        crop_size (tuple): Desired (height, width).
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    h, w = crop_size
    return image[:h, :w]
