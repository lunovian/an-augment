import cv2
import matplotlib.pyplot as plt
from anaug.default.blur import blur

# Load a sample grayscale image
image_path = "images/mri.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"The image at path '{image_path}' could not be loaded. Ensure the path is correct.")

# Apply Adaptive Blur with Mean Method
blur_params = {
    'blur_type': 'adaptive',
    'blur_radius': 0,  # Not used for adaptive
    'border_type': 'reflect',
    'max_kernel_size': 15,
    'adaptive_type': 'mean',
    'block_size': 11,
    'C': 2
}
blurred_image = blur(image, **blur_params)

# Display Original and Adaptively Blurred Images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Grayscale Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Adaptively Blurred Image (Mean)")
plt.imshow(blurred_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()