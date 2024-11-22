"""Test the output shape and functionality of the crop function."""

import unittest
import numpy as np
from src.anaug.default.crop import crop


class TestCrop(unittest.TestCase):
    """
    Test suite for the `crop` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """Test if the output shape matches the expected shape after crop."""
        cropped_image = crop(self.image, top=10, left=10, height=50, width=50)
        self.assertEqual(cropped_image.shape, (50, 50))

    def test_crop_effect(self):
        """Test if crop alters the image correctly."""
        cropped_image = crop(self.image, top=10, left=10, height=50, width=50)
        self.assertFalse(np.array_equal(self.image, cropped_image))
        self.assertTrue(np.array_equal(self.image[10:60, 10:60], cropped_image))

    def test_invalid_crop_parameters(self):
        """Test if crop handles invalid parameters correctly."""
        with self.assertRaises(ValueError):
            crop(self.image, top=-10, left=10, height=50, width=50)
        with self.assertRaises(ValueError):
            crop(self.image, top=10, left=10, height=150, width=50)
        with self.assertRaises(ValueError):
            crop(self.image, top=10, left=10, height=50, width=150)


if __name__ == "__main__":
    unittest.main()