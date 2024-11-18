"""Test the output shape of the blur function."""

import unittest
import numpy as np
from src.an_augment.default.blur import blur


class TestBlur(unittest.TestCase):
    """
    Test suite for the `blur` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """Test if the output shape matches the input shape after blur."""
        blurred_image = blur(self.image, blur_type='gaussian', blur_radius=1)
        self.assertEqual(self.image.shape, blurred_image.shape)

    def test_blur_effect(self):
        """Test if blur alters the image."""
        blurred_image = blur(self.image, blur_type='gaussian', blur_radius=1)
        self.assertFalse(np.array_equal(self.image, blurred_image))

    def test_different_blur_types(self):
        """Test if blur works with different blur types."""
        blurred_image_gaussian = blur(self.image, blur_type='gaussian', blur_radius=1)
        blurred_image_uniform = blur(self.image, blur_type='uniform', blur_radius=3)
        self.assertEqual(self.image.shape, blurred_image_gaussian.shape)
        self.assertEqual(self.image.shape, blurred_image_uniform.shape)
        self.assertFalse(np.array_equal(self.image, blurred_image_gaussian))
        self.assertFalse(np.array_equal(self.image, blurred_image_uniform))


if __name__ == "__main__":
    unittest.main()