"""Test the output shape and functionality of the rotate function."""

import unittest
import numpy as np
from src.an_augment.default.rotate import rotate


class TestRotate(unittest.TestCase):
    """
    Test suite for the `rotate` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_rotate_90_degrees(self):
        """Test if the image is rotated by 90 degrees."""
        rotated_image = rotate(self.image, angle=90)
        expected_image = np.rot90(self.image, k=1)
        self.assertTrue(np.allclose(rotated_image, expected_image, atol=1e-6))

    def test_rotate_180_degrees(self):
        """Test if the image is rotated by 180 degrees."""
        rotated_image = rotate(self.image, angle=180)
        expected_image = np.rot90(self.image, k=2)
        self.assertTrue(np.allclose(rotated_image, expected_image, atol=1e-6))

    def test_rotate_270_degrees(self):
        """Test if the image is rotated by 270 degrees."""
        rotated_image = rotate(self.image, angle=270)
        expected_image = np.rot90(self.image, k=3)
        self.assertTrue(np.allclose(rotated_image, expected_image, atol=1e-6))

    def test_rotate_arbitrary_angle(self):
        """Test if the image is rotated by an arbitrary angle."""
        angle = 45
        rotated_image = rotate(self.image, angle=angle)
        # Since we can't easily predict the exact result of an arbitrary rotation,
        # we can check properties like shape and type.
        self.assertEqual(rotated_image.shape, self.image.shape)
        self.assertEqual(rotated_image.dtype, self.image.dtype)

    def test_invalid_angle(self):
        """Test if the function raises a ValueError for invalid angles."""
        with self.assertRaises(ValueError):
            rotate(self.image, angle='invalid')


if __name__ == "__main__":
    unittest.main()