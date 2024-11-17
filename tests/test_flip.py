"""
Unit tests for the `flip` function in the MedAugment library.

These tests validate the functionality of flipping medical images horizontally,
vertically, or both.
"""

import unittest
import numpy as np
from src.med_augment.augmentations.flip import flip


class TestFlip(unittest.TestCase):
    """
    Test suite for the `flip` function.
    """

    def setUp(self):
        """
        Set up a test image for use in all test cases.
        """
        # Create a test image (e.g., 128x128 pixels with random values)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """
        Test if the output shape matches the input shape for flips.
        """
        flipped_image = flip(
            self.image, flip_horizontal=True, flip_vertical=True
        )
        self.assertEqual(self.image.shape, flipped_image.shape)

    def test_horizontal_flip(self):
        """
        Test horizontal flip and ensure it alters the image correctly.
        """
        flipped_image = flip(
            self.image, flip_horizontal=True, flip_vertical=False
        )
        self.assertFalse(np.array_equal(self.image, flipped_image))
        # Check if flipping horizontally twice restores the original image
        restored_image = flip(
            flipped_image, flip_horizontal=True, flip_vertical=False
        )
        np.testing.assert_array_equal(self.image, restored_image)

    def test_vertical_flip(self):
        """
        Test vertical flip and ensure it alters the image correctly.
        """
        flipped_image = flip(
            self.image, flip_horizontal=False, flip_vertical=True
        )
        self.assertFalse(np.array_equal(self.image, flipped_image))
        # Check if flipping vertically twice restores the original image
        restored_image = flip(
            flipped_image, flip_horizontal=False, flip_vertical=True
        )
        np.testing.assert_array_equal(self.image, restored_image)


if __name__ == "__main__":
    unittest.main()
