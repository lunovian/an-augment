"""
Unit tests for the `occlusion` function in the MedAugment library.

These tests validate the functionality of applying occlusion to medical images.
"""

import unittest
import numpy as np
from src.an_augment.medical.occlusion import occlusion


class TestOcclusion(unittest.TestCase):
    """
    Test suite for the `occlusion` function.
    """

    def setUp(self):
        """
        Set up a test image for use in all test cases.
        """
        # Create a test image (e.g., 128x128 pixels with random values between 0 and 1)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """
        Test if the output shape matches the input shape.
        """
        occluded_image = occlusion(
            self.image, mask_shape='rectangle', mask_size_range=(0.1, 0.3)
        )
        self.assertEqual(self.image.shape, occluded_image.shape)

    def test_occlusion_effect(self):
        """
        Test if occlusion alters the image.
        """
        occluded_image = occlusion(
            self.image, mask_shape='rectangle', mask_size_range=(0.2, 0.3)
        )
        self.assertFalse(np.array_equal(self.image, occluded_image))


if __name__ == "__main__":
    unittest.main()
