import cv2

def rotate(image, angle):
    """
    Rotate an image by a given angle.
    
    Args:
        image (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        
    Returns:
        numpy.ndarray: Rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))
