# tests/test_intensity_scaling.py
import unittest
import numpy as np
from src.augmentations.intensity_scaling import intensity_scaling

class TestIntensityScaling(unittest.TestCase):
    def setUp(self):
        # Create a test image (e.g., 128x128 pixels with random values between 0 and 1)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        # Check if the output shape matches the input shape
        scaled_image = intensity_scaling(self.image, brightness_factor=1.2, contrast_factor=1.2)
        self.assertEqual(self.image.shape, scaled_image.shape)

    def test_brightness_factor(self):
        # Test with different brightness factors
        brighter_image = intensity_scaling(self.image, brightness_factor=1.5)
        darker_image = intensity_scaling(self.image, brightness_factor=0.5)
        # Ensure the images are different
        self.assertFalse(np.array_equal(brighter_image, darker_image))

    def test_contrast_factor(self):
        # Test with different contrast factors
        higher_contrast_image = intensity_scaling(self.image, contrast_factor=1.5)
        lower_contrast_image = intensity_scaling(self.image, contrast_factor=0.5)
        # Ensure the images are different
        self.assertFalse(np.array_equal(higher_contrast_image, lower_contrast_image))

if __name__ == "__main__":
    unittest.main()
