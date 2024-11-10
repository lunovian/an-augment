# tests/test_gaussian_blur.py
import unittest
import numpy as np
from src.augmentations.gaussian_blur import gaussian_blur

class TestGaussianBlur(unittest.TestCase):
    def setUp(self):
        # Create a test image (e.g., 128x128 pixels with random values)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        # Check if the output shape matches the input shape
        blurred_image = gaussian_blur(self.image, blur_radius=2)
        self.assertEqual(self.image.shape, blurred_image.shape)

    def test_blur_effect(self):
        # Test with different blur radii
        blurred_image_low = gaussian_blur(self.image, blur_radius=1)
        blurred_image_high = gaussian_blur(self.image, blur_radius=5)
        # Ensure the images are different as blur effect increases
        self.assertFalse(np.array_equal(blurred_image_low, blurred_image_high))

if __name__ == "__main__":
    unittest.main()
