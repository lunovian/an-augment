"""Test the output shape of the function."""

import unittest
import numpy as np
from src.med_augment.augmentations.intensity_scaling import intensity_scaling


class TestIntensityScaling(unittest.TestCase):
    """
    Test suite for the `intensity_scaling` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """Test if the output shape matches the input shape."""
        scaled_image = intensity_scaling(
            self.image,
            brightness_factor=1.2,
            contrast_factor=1.2
        )
        self.assertEqual(self.image.shape, scaled_image.shape)

    def test_brightness_factor(self):
        """Test the effect of varying brightness factors."""
        brighter_image = intensity_scaling(
            self.image, brightness_factor=1.5
        )
        darker_image = intensity_scaling(
            self.image, brightness_factor=0.5
        )
        # Ensure the images are different
        self.assertFalse(np.array_equal(brighter_image, darker_image))

    def test_contrast_factor(self):
        """Test the effect of varying contrast factors."""
        higher_contrast_image = intensity_scaling(
            self.image,
            contrast_factor=1.5
        )
        lower_contrast_image = intensity_scaling(
            self.image,
            contrast_factor=0.5
        )
        # Ensure the images are different
        self.assertFalse(
            np.array_equal(higher_contrast_image, lower_contrast_image)
        )


if __name__ == "__main__":
    unittest.main()
