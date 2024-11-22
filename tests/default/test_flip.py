"""Test the output shape and functionality of the flip function."""

import unittest
import numpy as np
from src.anaug.default.flip import flip


class TestFlip(unittest.TestCase):
    """
    Test suite for the `flip` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_horizontal_flip(self):
        """Test if the image is flipped horizontally."""
        flipped_image = flip(self.image, flip_horizontal=True, flip_vertical=False)
        self.assertTrue(np.array_equal(flipped_image, np.fliplr(self.image)))

    def test_vertical_flip(self):
        """Test if the image is flipped vertically."""
        flipped_image = flip(self.image, flip_horizontal=False, flip_vertical=True)
        self.assertTrue(np.array_equal(flipped_image, np.flipud(self.image)))

    def test_both_flips(self):
        """Test if the image is flipped both horizontally and vertically."""
        flipped_image = flip(self.image, flip_horizontal=True, flip_vertical=True)
        self.assertTrue(np.array_equal(flipped_image, np.flipud(np.fliplr(self.image))))


if __name__ == "__main__":
    unittest.main()