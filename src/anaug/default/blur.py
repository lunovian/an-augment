import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from typing import Optional, Tuple, Union


def motion_blur(
    image: np.ndarray,
    length: int = 15,
    angle: float = 0
) -> np.ndarray:
    """
    Applies motion blur to an image.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - length (int): Length of the motion blur effect. Must be positive.
    - angle (float): Angle of motion blur in degrees. Range: [0, 360).

    Returns:
    - np.ndarray: Motion-blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If 'length' is not positive or 'angle' is out of range.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Parameter 'length' must be a positive integer.")
    
    if not isinstance(angle, (int, float)) or not (0 <= angle < 360):
        raise ValueError("Parameter 'angle' must be a float in the range [0, 360).")

    # Ensure the kernel size is odd to have a central pixel
    kernel_size = length
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create the horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = np.ones(kernel_size, dtype=np.float32)

    # Normalize the kernel to maintain brightness
    kernel /= kernel_size

    # Rotate the kernel to the specified angle
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    
    # Apply the motion blur kernel to the image
    if image.ndim == 2:
        blurred = cv2.filter2D(image, -1, rotated_kernel)
    elif image.ndim == 3:
        # Apply the filter to each channel independently
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, rotated_kernel)
    else:
        raise ValueError("Input image must be either a 2D or 3D numpy array.")
    
    return blurred


def bilateral_blur(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Applies bilateral blur to an image, preserving edges while reducing noise.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - diameter (int): Diameter of each pixel neighborhood.
    - sigma_color (float): Filter sigma in the color space. Larger values mean colors farther apart are mixed.
    - sigma_space (float): Filter sigma in the coordinate space. Larger values mean pixels farther apart influence each other.

    Returns:
    - np.ndarray: Bilaterally blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If parameters are out of acceptable ranges.
    """
    if not isinstance(diameter, int) or diameter <= 0:
        raise ValueError("Parameter 'diameter' must be a positive integer.")
    if not isinstance(sigma_color, (int, float)) or sigma_color < 0:
        raise ValueError("Parameter 'sigma_color' must be a non-negative float.")
    if not isinstance(sigma_space, (int, float)) or sigma_space < 0:
        raise ValueError("Parameter 'sigma_space' must be a non-negative float.")

    blurred = cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return blurred


def adaptive_blur(
    image: np.ndarray,
    max_kernel_size: int = 15,
    adaptive_type: str = 'mean',
    block_size: int = 11,
    C: float = 2
) -> np.ndarray:
    """
    Applies adaptive blur to an image, adjusting the blurring based on local image regions.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - max_kernel_size (int): Maximum size of the blurring kernel.
    - adaptive_type (str): Type of adaptive blur ('mean', 'gaussian').
    - block_size (int): Size of the neighborhood area.
    - C (float): Constant subtracted from the mean or weighted mean.

    Returns:
    - np.ndarray: Adaptively blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If parameters are invalid.
    """
    if not isinstance(max_kernel_size, int) or max_kernel_size <= 0 or max_kernel_size % 2 == 0:
        raise ValueError("Parameter 'max_kernel_size' must be a positive odd integer.")
    if adaptive_type not in ['mean', 'gaussian']:
        raise ValueError("Parameter 'adaptive_type' must be either 'mean' or 'gaussian'.")
    if not isinstance(block_size, int) or block_size <= 1 or block_size % 2 == 0:
        raise ValueError("Parameter 'block_size' must be an odd integer greater than 1.")
    if not isinstance(C, (int, float)):
        raise ValueError("Parameter 'C' must be a numeric value.")

    # Convert image to uint8 if necessary
    original_dtype = image.dtype
    if original_dtype != np.uint8:
        image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    if adaptive_type == 'mean':
        blurred = cv2.adaptiveThreshold(
            image_uint8,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        # Invert to get the blurred effect
        blurred = cv2.bitwise_not(blurred)
    elif adaptive_type == 'gaussian':
        blurred = cv2.adaptiveThreshold(
            image_uint8,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        # Invert to get the blurred effect
        blurred = cv2.bitwise_not(blurred)
    else:
        # This should never happen due to earlier validation
        raise ValueError(f"Unhandled adaptive_type '{adaptive_type}'.")

    # Convert back to original dtype
    if original_dtype != np.uint8:
        blurred = blurred.astype(original_dtype) / 255.0

    return blurred


def box_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    border_type: str = 'reflect'
) -> np.ndarray:
    """
    Applies box blur (uniform blur) to an image using a rectangular kernel.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - kernel_size (int): Size of the kernel. Must be a positive odd integer.
    - border_type (str): Border handling method ('reflect', 'constant', 'nearest', 'mirror', 'wrap').

    Returns:
    - np.ndarray: Box-blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If parameters are invalid.
    """
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Parameter 'kernel_size' must be a positive odd integer.")
    
    # Map string border_type to OpenCV border constants
    border_types = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
        'reflect_101': cv2.BORDER_REFLECT_101,
        'mirror': cv2.BORDER_REFLECT
    }
    if border_type not in border_types:
        raise ValueError(f"Unsupported border_type '{border_type}'. Supported types: {list(border_types.keys())}.")
    cv_border_type = border_types[border_type]

    blurred = cv2.blur(image, (kernel_size, kernel_size), borderType=cv_border_type)
    return blurred


def blur(
    image: np.ndarray,
    blur_type: str = 'gaussian',
    blur_radius: Union[int, float] = 1,
    border_type: str = 'reflect',
    **kwargs
) -> np.ndarray:
    """
    Applies various types of blur to a given image.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - blur_type (str): Type of blur to apply ('gaussian', 'uniform', 'median', 'motion', 'bilateral', 'box', 'adaptive').
    - blur_radius (Union[int, float]): 
        - For 'gaussian': Standard deviation (sigma). Must be non-negative.
        - For 'uniform' and 'median': Kernel size. Must be positive odd integer.
        - For 'motion': Length of motion blur. Must be positive integer.
        - For 'bilateral': Not used; instead, use 'diameter', 'sigma_color', 'sigma_space'.
        - For 'box': Kernel size. Must be positive odd integer.
        - For 'adaptive': Not used; instead, use 'max_kernel_size', 'adaptive_type', 'block_size', 'C'.
    - border_type (str): Border handling method ('reflect', 'constant', 'nearest', 'mirror', 'wrap').
    - **kwargs: Additional parameters for specific blur types.
        - For 'motion': 
            - 'angle' (float): Angle of motion blur in degrees.
        - For 'bilateral':
            - 'diameter' (int): Diameter of each pixel neighborhood.
            - 'sigma_color' (float): Filter sigma in the color space.
            - 'sigma_space' (float): Filter sigma in the coordinate space.
        - For 'adaptive':
            - 'max_kernel_size' (int): Maximum size of the blurring kernel.
            - 'adaptive_type' (str): Type of adaptive blur ('mean', 'gaussian').
            - 'block_size' (int): Size of the neighborhood area.
            - 'C' (float): Constant subtracted from the mean or weighted mean.
    
    Returns:
    - np.ndarray: Blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If an unsupported blur type is provided or parameters are invalid.
    """
    supported_blurs = ['gaussian', 'uniform', 'median', 'motion', 'bilateral', 'box', 'adaptive']
    if blur_type not in supported_blurs:
        raise ValueError(f"Unsupported blur type '{blur_type}'. Supported types: {supported_blurs}.")

    # Map string border_type to OpenCV border constants
    border_types = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
        'reflect_101': cv2.BORDER_REFLECT_101,
        'mirror': cv2.BORDER_REFLECT
    }
    if border_type not in border_types:
        raise ValueError(f"Unsupported border_type '{border_type}'. Supported types: {list(border_types.keys())}.")
    cv_border_type = border_types[border_type]

    if blur_type == 'gaussian':
        if not isinstance(blur_radius, (int, float)) or blur_radius < 0:
            raise ValueError("For 'gaussian' blur_type, 'blur_radius' must be a non-negative number.")
        if blur_radius == 0:
            return image.copy()  # No blurring needed
        # Calculate kernel size from sigma. A common choice is kernel_size = 6*sigma +1
        kernel_size = int(6 * blur_radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=blur_radius, borderType=cv_border_type)
        return blurred

    elif blur_type == 'uniform':
        if not isinstance(blur_radius, int) or blur_radius <= 0:
            raise ValueError("For 'uniform' blur_type, 'blur_radius' must be a positive odd integer.")
        # Ensure kernel size is odd
        kernel_size = blur_radius
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.blur(image, (kernel_size, kernel_size), borderType=cv_border_type)
        return blurred

    elif blur_type == 'median':
        if not isinstance(blur_radius, int) or blur_radius <= 0:
            raise ValueError("For 'median' blur_type, 'blur_radius' must be a positive odd integer.")
        # Median blur kernel size must be odd and greater than 1
        kernel_size = blur_radius
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        if image.ndim == 2:
            blurred = cv2.medianBlur(image, kernel_size)
        elif image.ndim == 3:
            # Apply median blur to each channel independently
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = cv2.medianBlur(image[:, :, c], kernel_size)
        else:
            raise ValueError("Input image must be either a 2D or 3D numpy array.")
        return blurred

    elif blur_type == 'motion':
        if not isinstance(blur_radius, int) or blur_radius <= 0:
            raise ValueError("For 'motion' blur_type, 'blur_radius' must be a positive integer representing the length of motion.")
        angle = kwargs.get('angle', 0)
        return motion_blur(image, length=blur_radius, angle=angle)

    elif blur_type == 'bilateral':
        # Retrieve bilateral filter parameters
        diameter = kwargs.get('diameter', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return bilateral_blur(image, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)

    elif blur_type == 'box':
        if not isinstance(blur_radius, int) or blur_radius <= 0 or blur_radius % 2 == 0:
            raise ValueError("For 'box' blur_type, 'blur_radius' must be a positive odd integer.")
        return box_blur(image, kernel_size=blur_radius, border_type=border_type)

    elif blur_type == 'adaptive':
        # Retrieve adaptive blur parameters
        max_kernel_size = kwargs.get('max_kernel_size', 15)
        adaptive_type = kwargs.get('adaptive_type', 'mean')
        block_size = kwargs.get('block_size', 11)
        C = kwargs.get('C', 2)
        return adaptive_blur(
            image,
            max_kernel_size=max_kernel_size,
            adaptive_type=adaptive_type,
            block_size=block_size,
            C=C
        )

    else:
        # This should never happen due to the earlier check
        raise ValueError(f"Unhandled blur_type '{blur_type}'.")


def bilateral_blur(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Applies bilateral blur to an image, preserving edges while reducing noise.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - diameter (int): Diameter of each pixel neighborhood.
    - sigma_color (float): Filter sigma in the color space. Larger values mean colors farther apart are mixed.
    - sigma_space (float): Filter sigma in the coordinate space. Larger values mean pixels farther apart influence each other.

    Returns:
    - np.ndarray: Bilaterally blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If parameters are out of acceptable ranges.
    """
    if not isinstance(diameter, int) or diameter <= 0:
        raise ValueError("Parameter 'diameter' must be a positive integer.")
    if not isinstance(sigma_color, (int, float)) or sigma_color < 0:
        raise ValueError("Parameter 'sigma_color' must be a non-negative float.")
    if not isinstance(sigma_space, (int, float)) or sigma_space < 0:
        raise ValueError("Parameter 'sigma_space' must be a non-negative float.")

    blurred = cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return blurred


def box_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    border_type: str = 'reflect'
) -> np.ndarray:
    """
    Applies box blur (uniform blur) to an image using a rectangular kernel.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - kernel_size (int): Size of the kernel. Must be a positive odd integer.
    - border_type (str): Border handling method ('reflect', 'constant', 'nearest', 'mirror', 'wrap').

    Returns:
    - np.ndarray: Box-blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If parameters are invalid.
    """
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Parameter 'kernel_size' must be a positive odd integer.")
    
    # Map string border_type to OpenCV border constants
    border_types = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
        'reflect_101': cv2.BORDER_REFLECT_101,
        'mirror': cv2.BORDER_REFLECT
    }
    if border_type not in border_types:
        raise ValueError(f"Unsupported border_type '{border_type}'. Supported types: {list(border_types.keys())}.")
    cv_border_type = border_types[border_type]

    blurred = cv2.blur(image, (kernel_size, kernel_size), borderType=cv_border_type)
    return blurred


def adaptive_blur(
    image: np.ndarray,
    max_kernel_size: int = 15,
    adaptive_type: str = 'mean',
    block_size: int = 11,
    C: float = 2
) -> np.ndarray:
    """
    Applies adaptive blur to an image, adjusting the blurring based on local image regions.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - max_kernel_size (int): Maximum size of the blurring kernel. Must be a positive odd integer.
    - adaptive_type (str): Type of adaptive blur ('mean', 'gaussian').
    - block_size (int): Size of the neighborhood area. Must be a positive odd integer greater than 1.
    - C (float): Constant subtracted from the mean or weighted mean.

    Returns:
    - np.ndarray: Adaptively blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If parameters are invalid.
    """
    if not isinstance(max_kernel_size, int) or max_kernel_size <= 0 or max_kernel_size % 2 == 0:
        raise ValueError("Parameter 'max_kernel_size' must be a positive odd integer.")
    if adaptive_type not in ['mean', 'gaussian']:
        raise ValueError("Parameter 'adaptive_type' must be either 'mean' or 'gaussian'.")
    if not isinstance(block_size, int) or block_size <= 1 or block_size % 2 == 0:
        raise ValueError("Parameter 'block_size' must be an odd integer greater than 1.")
    if not isinstance(C, (int, float)):
        raise ValueError("Parameter 'C' must be a numeric value.")

    # Convert image to uint8 if necessary
    original_dtype = image.dtype
    if original_dtype != np.uint8:
        image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    if adaptive_type == 'mean':
        # Apply adaptive mean filter
        blurred = cv2.adaptiveThreshold(
            image_uint8,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        # Invert to get the blurred effect
        blurred = cv2.bitwise_not(blurred)
    elif adaptive_type == 'gaussian':
        # Apply adaptive Gaussian filter
        blurred = cv2.adaptiveThreshold(
            image_uint8,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        # Invert to get the blurred effect
        blurred = cv2.bitwise_not(blurred)
    else:
        # This should never happen due to earlier validation
        raise ValueError(f"Unhandled adaptive_type '{adaptive_type}'.")

    # Convert back to original dtype if necessary
    if original_dtype != np.uint8:
        blurred = blurred.astype(original_dtype) / 255.0

    return blurred


def blur(
    image: np.ndarray,
    blur_type: str = 'gaussian',
    blur_radius: Union[int, float] = 1,
    border_type: str = 'reflect',
    **kwargs
) -> np.ndarray:
    """
    Applies various types of blur to a given image.

    Parameters:
    - image (np.ndarray): Input image as a 2D or 3D numpy array.
    - blur_type (str): Type of blur to apply ('gaussian', 'uniform', 'median', 'motion', 'bilateral', 'box', 'adaptive').
    - blur_radius (Union[int, float]): 
        - For 'gaussian': Standard deviation (sigma). Must be non-negative.
        - For 'uniform' and 'median': Kernel size. Must be positive odd integer.
        - For 'motion': Length of motion blur. Must be positive integer.
        - For 'bilateral': Not used; instead, use 'diameter', 'sigma_color', 'sigma_space'.
        - For 'box': Kernel size. Must be positive odd integer.
        - For 'adaptive': Not used; instead, use 'max_kernel_size', 'adaptive_type', 'block_size', 'C'.
    - border_type (str): Border handling method ('reflect', 'constant', 'nearest', 'mirror', 'wrap').
    - **kwargs: Additional parameters for specific blur types.
        - For 'motion': 
            - 'angle' (float): Angle of motion blur in degrees.
        - For 'bilateral':
            - 'diameter' (int): Diameter of each pixel neighborhood.
            - 'sigma_color' (float): Filter sigma in the color space.
            - 'sigma_space' (float): Filter sigma in the coordinate space.
        - For 'adaptive':
            - 'max_kernel_size' (int): Maximum size of the blurring kernel.
            - 'adaptive_type' (str): Type of adaptive blur ('mean', 'gaussian').
            - 'block_size' (int): Size of the neighborhood area.
            - 'C' (float): Constant subtracted from the mean or weighted mean.

    Returns:
    - np.ndarray: Blurred image with the same shape and dtype as input.
    
    Raises:
    - ValueError: If an unsupported blur type is provided or parameters are invalid.
    """
    supported_blurs = ['gaussian', 'uniform', 'median', 'motion', 'bilateral', 'box', 'adaptive']
    if blur_type not in supported_blurs:
        raise ValueError(f"Unsupported blur type '{blur_type}'. Supported types: {supported_blurs}.")

    # Map string border_type to OpenCV border constants
    border_types = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
        'reflect_101': cv2.BORDER_REFLECT_101,
        'mirror': cv2.BORDER_REFLECT
    }
    if border_type not in border_types:
        raise ValueError(f"Unsupported border_type '{border_type}'. Supported types: {list(border_types.keys())}.")
    cv_border_type = border_types[border_type]

    if blur_type == 'gaussian':
        if not isinstance(blur_radius, (int, float)) or blur_radius < 0:
            raise ValueError("For 'gaussian' blur_type, 'blur_radius' must be a non-negative number.")
        if blur_radius == 0:
            return image.copy()  # No blurring needed
        # Calculate kernel size from sigma. A common choice is kernel_size = 6*sigma +1
        kernel_size = int(6 * blur_radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=blur_radius, borderType=cv_border_type)
        return blurred

    elif blur_type == 'uniform':
        if not isinstance(blur_radius, int) or blur_radius <= 0:
            raise ValueError("For 'uniform' blur_type, 'blur_radius' must be a positive odd integer.")
        # Ensure kernel size is odd
        kernel_size = blur_radius
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.blur(image, (kernel_size, kernel_size), borderType=cv_border_type)
        return blurred

    elif blur_type == 'median':
        if not isinstance(blur_radius, int) or blur_radius <= 0:
            raise ValueError("For 'median' blur_type, 'blur_radius' must be a positive odd integer.")
        # Median blur kernel size must be odd and greater than 1
        kernel_size = blur_radius
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        if image.ndim == 2:
            blurred = cv2.medianBlur(image, kernel_size)
        elif image.ndim == 3:
            # Apply median blur to each channel independently
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = cv2.medianBlur(image[:, :, c], kernel_size)
        else:
            raise ValueError("Input image must be either a 2D or 3D numpy array.")
        return blurred

    elif blur_type == 'motion':
        if not isinstance(blur_radius, int) or blur_radius <= 0:
            raise ValueError("For 'motion' blur_type, 'blur_radius' must be a positive integer representing the length of motion.")
        angle = kwargs.get('angle', 0)
        return motion_blur(image, length=blur_radius, angle=angle)

    elif blur_type == 'bilateral':
        # Retrieve bilateral filter parameters
        diameter = kwargs.get('diameter', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return bilateral_blur(image, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)

    elif blur_type == 'box':
        if not isinstance(blur_radius, int) or blur_radius <= 0 or blur_radius % 2 == 0:
            raise ValueError("For 'box' blur_type, 'blur_radius' must be a positive odd integer.")
        return box_blur(image, kernel_size=blur_radius, border_type=border_type)

    elif blur_type == 'adaptive':
        # Retrieve adaptive blur parameters
        max_kernel_size = kwargs.get('max_kernel_size', 15)
        adaptive_type = kwargs.get('adaptive_type', 'mean')
        block_size = kwargs.get('block_size', 11)
        C = kwargs.get('C', 2)
        return adaptive_blur(
            image,
            max_kernel_size=max_kernel_size,
            adaptive_type=adaptive_type,
            block_size=block_size,
            C=C
        )

    else:
        # This should never happen due to the earlier check
        raise ValueError(f"Unhandled blur_type '{blur_type}'.")
