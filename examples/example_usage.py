import cv2
import matplotlib.pyplot as plt
from  import MedicalAugmentation

# Load the image
image = cv2.imread("images/mri.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

# Initialize the augmentor
augmentor = MedicalAugmentation()

# Define augmentation parameters
params = {
    'elastic_deformation': {'alpha': 30, 'sigma': 4},
    'intensity_scaling': {'brightness_factor': 1.2, 'contrast_factor': 1.3},
    'gaussian_blur': {'blur_radius': 2},
    'random_rotation': {'angle_range': (-15, 15)},
    'flip': {'flip_horizontal': True, 'flip_vertical': False},
    'random_crop_and_scale': {'crop_size': (0.8, 0.8), 'scaling_factor': 1.0},
    'add_noise': {'noise_type': 'gaussian', 'noise_intensity': 0.05},
    'occlusion': {'mask_shape': 'rectangle', 'mask_size_range': (0.1, 0.2)}
}

# Apply augmentations
augmented_image = augmentor.apply_augmentations(image, **params)

# Display original and augmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Augmented Image")
plt.imshow(augmented_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
