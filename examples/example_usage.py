import cv2
import matplotlib.pyplot as plt
from anaug.default import scale, flip, noise, random_rotation, random_crop, intensity, elastic_deformation, occlusion, blur
from anaug.medical.random_lesion import random_lesion

# Load the image
image_path = "images/mri.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"The image at path '{image_path}' could not be loaded. Ensure the path is correct.")

image = image / 255.0  # Normalize the image to [0, 1]

# Define augmentation parameters
params = {
    'blur': {'blur_radius': 2},
    'elastic_deformation': {'alpha': 30, 'sigma': 4},
    'flip': {'flip_horizontal': True, 'flip_vertical': False},
    'intensity': {'brightness_factor': 1.2, 'contrast_factor': 1.3},
    'noise': {'noise_type': 'gaussian', 'noise_intensity': 0.1},
    'occlusion': {'mask_shape': 'rectangle', 'mask_size_range': (0.1, 0.2)},
    'random_rotation': {'angle_range': (-15, 15)},
    'random_crop': {'crop_size': (0.8, 0.8), 'scaling_factor': 1.0},
    'scale': {'scale_factor': 0.8},
    'random_lesion': {  # Added parameters for random lesion
        'intensity_range': (0.3, 0.7),
        'size_range': (10, 50),
        'shape': 'circle',  # Options: 'circle', 'ellipse', 'irregular'
        'location': None,  # Set to None for random locations
        'texture_strength': 0.5,
        'num_lesions': 1,
        'blending_mode': 'additive',  # Options: 'additive', 'overlay'
        'seed': 42  # For reproducibility; set to None for randomness
    }
}

def display_multiple_augmented_images(original, augmented, lesion_applied):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Augmented Image without Lesion")
    plt.imshow(augmented, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Augmented Image with Lesion")
    plt.imshow(lesion_applied, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Apply augmentations (manually applying each augmentation)
try:
    augmented_without_lesion = blur(image, **params['blur'])
    augmented_without_lesion = elastic_deformation(augmented_without_lesion, **params['elastic_deformation'])
    augmented_without_lesion = flip(augmented_without_lesion, **params['flip'])
    augmented_without_lesion = intensity(augmented_without_lesion, **params['intensity'])
    augmented_without_lesion = noise(augmented_without_lesion, **params['noise'])
    augmented_without_lesion = occlusion(augmented_without_lesion, **params['occlusion'])
    augmented_without_lesion = random_rotation(augmented_without_lesion, **params['random_rotation'])
    augmented_without_lesion = random_crop(augmented_without_lesion, **params['random_crop'])
    augmented_without_lesion = scale(augmented_without_lesion, **params['scale'])
    
    augmented_with_lesion = random_lesion(augmented_without_lesion, **params['random_lesion'])
    
    display_multiple_augmented_images(image, augmented_without_lesion, augmented_with_lesion)
except Exception as e:
    raise RuntimeError(f"An error occurred while applying augmentations: {e}")