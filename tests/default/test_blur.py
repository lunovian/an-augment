"""Test the output shape of the function."""

import unittest
import numpy as np
from an_augment.default.noise import add_noise


class TestAddNoise(unittest.TestCase):
    """
    Test suite for the `add_noise` function.
    """

    def setUp(self):
        """Set up a test image for use in all test cases."""
        # Create a test image (128x128)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        """Test if the output shape matches the input shape for Gaussian noise."""
        noisy_image = add_noise(
            self.image, noise_type='gaussian', noise_intensity=0.1
        )
        self.assertEqual(self.image.shape, noisy_image.shape)

    def test_gaussian_noise(self):
        """Test if Gaussian noise alters the image."""
        noisy_image = add_noise(
            self.image, noise_type='gaussian', noise_intensity=0.1
        )
        self.assertFalse(np.array_equal(self.image, noisy_image))

    def test_salt_and_pepper_noise(self):
        """Test if salt-and-pepper noise alters the image."""
        noisy_image = add_noise(
            self.image, noise_type='salt_and_pepper', noise_intensity=0.05
        )
        self.assertFalse(np.array_equal(self.image, noisy_image))


if __name__ == "__main__":
    unittest.main()
