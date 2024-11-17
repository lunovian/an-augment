"""
Unit tests for the `random_rotation` function in the MedAugment library.

These tests validate the functionality of applying random rotations to medical images.
"""

import unittest
import numpy as np
from src.med_augment.augmentations.random_rotation import random_rotation


class TestRandomRotation(unittest.TestCase):
    """
    Test suite for the `random_rotation` function.
    """

    def setUp(self):
        """
        Set up a test image for use in all test cases.
        """
        # Create a test image (e.g., 128x128 pixels with random values)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """
        Test if the output shape matches the input shape.
        """
        rotated_image = random_rotation(self.image, angle_range=(-30, 30))
        self.assertEqual(self.image.shape, rotated_image.shape)

    def test_angle_range_effect(self):
        """
        Test the effect of different angle ranges on rotation.
        """
        rotated_image_low = random_rotation(self.image, angle_range=(-15, 15))
        rotated_image_high = random_rotation(self.image, angle_range=(-90, 90))
        # Ensure the images are different
        self.assertFalse(np.array_equal(rotated_image_low, rotated_image_high))


if __name__ == "__main__":
    unittest.main()
