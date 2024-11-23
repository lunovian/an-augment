import cv2
import matplotlib.pyplot as plt
from anaug.default import blur  # Adjust the import path as necessary

# Load a sample color image
image_path = "images/mri.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"The image at path '{image_path}' could not be loaded. Ensure the path is correct.")

# Apply Bilateral Blur
blur_params = {
    'blur_type': 'bilateral',
    'blur_radius': 0,  # Not used for bilateral
    'border_type': 'reflect',
    'diameter': 15,
    'sigma_color': 75,
    'sigma_space': 75
}
blurred_image = blur(image, **blur_params)

# Display Original and Blurred Images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Color Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Bilateral Blurred Image")
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
