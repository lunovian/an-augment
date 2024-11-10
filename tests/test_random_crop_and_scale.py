# tests/test_random_crop_and_scale.py
import unittest
import numpy as np
from src.augmentations.random_crop_and_scale import random_crop_and_scale

class TestRandomCropAndScale(unittest.TestCase):
    def setUp(self):
        # Create a test image (e.g., 128x128 pixels with random values)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        # Check if the output shape matches the input shape
        cropped_scaled_image = random_crop_and_scale(self.image, crop_size=(0.8, 0.8), scaling_factor=1.0)
        self.assertEqual(self.image.shape, cropped_scaled_image.shape)

    def test_crop_size_effect(self):
        # Test with different crop sizes
        cropped_image_small = random_crop_and_scale(self.image, crop_size=(0.5, 0.5))
        cropped_image_large = random_crop_and_scale(self.image, crop_size=(0.9, 0.9))
        # Ensure the images are different
        self.assertFalse(np.array_equal(cropped_image_small, cropped_image_large))

    def test_scaling_factor_effect(self):
        # Test with different scaling factors
        scaled_image_low = random_crop_and_scale(self.image, crop_size=(0.8, 0.8), scaling_factor=0.5)
        scaled_image_high = random_crop_and_scale(self.image, crop_size=(0.8, 0.8), scaling_factor=1.5)
        # Ensure the images are different
        self.assertFalse(np.array_equal(scaled_image_low, scaled_image_high))

if __name__ == "__main__":
    unittest.main()
