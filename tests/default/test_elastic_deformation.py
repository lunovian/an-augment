"""Test the output shape and functionality of the elastic_deformation function."""

import unittest
import numpy as np
from src.anaug.default.elastic_deformation import elastic_deformation


class TestElasticDeformation(unittest.TestCase):
    """
    Test suite for the `elastic_deformation` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """Test if the output shape matches the input shape after elastic deformation."""
        deformed_image = elastic_deformation(self.image, alpha=34, sigma=4)
        self.assertEqual(self.image.shape, deformed_image.shape)

    def test_deformation_effect(self):
        """Test if elastic deformation alters the image."""
        deformed_image = elastic_deformation(self.image, alpha=34, sigma=4)
        self.assertFalse(np.array_equal(self.image, deformed_image))

    def test_invalid_parameters(self):
        """Test if elastic deformation handles invalid parameters correctly."""
        with self.assertRaises(ValueError):
            elastic_deformation(self.image, alpha=-34, sigma=4)
        with self.assertRaises(ValueError):
            elastic_deformation(self.image, alpha=34, sigma=-4)


if __name__ == "__main__":
    unittest.main()