"""Test the output shape of the function."""

import unittest
import numpy as np
from src.med_augment.augmentations.elastic_deformation import elastic_deformation


class TestElasticDeformation(unittest.TestCase):
    """
    Test suite for the `elastic_deformation` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """Test if the output shape matches the input shape."""
        deformed_image = elastic_deformation(self.image, alpha=34, sigma=4)
        self.assertEqual(self.image.shape, deformed_image.shape)

    def test_alpha_effect(self):
        """Test the effect of varying alpha on the deformation."""
        deformed_image_low_alpha = elastic_deformation(
            self.image, alpha=10, sigma=4
        )
        deformed_image_high_alpha = elastic_deformation(
            self.image, alpha=50, sigma=4
        )
        # Check that images differ, as alpha affects displacement
        self.assertFalse(
            np.array_equal(deformed_image_low_alpha, deformed_image_high_alpha)
        )

    def test_sigma_effect(self):
        """Test the effect of varying sigma on the deformation."""
        deformed_image_low_sigma = elastic_deformation(
            self.image, alpha=34, sigma=2
        )
        deformed_image_high_sigma = elastic_deformation(
            self.image, alpha=34, sigma=6
        )
        # Check that images differ, as sigma affects smoothness
        self.assertFalse(
            np.array_equal(deformed_image_low_sigma, deformed_image_high_sigma)
        )


if __name__ == "__main__":
    unittest.main()
