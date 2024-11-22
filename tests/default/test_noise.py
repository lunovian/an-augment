import unittest
import numpy as np
from src.anaug.default.noise import noise

"""Unit tests for the noise addition functions in an_augment.default.noise.py."""

class TestNoise(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing
        self.image = np.ones((100, 100), dtype=np.float32)

    def test_add_gaussian_noise(self):
        noisy_image = noise(self.image, noise_type='gaussian', noise_intensity=0.1)
        self.assertEqual(noisy_image.shape, self.image.shape)
        self.assertTrue(np.any(noisy_image != self.image))
        self.assertTrue(np.all(noisy_image >= 0) and np.all(noisy_image <= 1))

    def test_add_salt_and_pepper_noise(self):
        noisy_image = noise(self.image, noise_type='salt_and_pepper', noise_intensity=0.1)
        self.assertEqual(noisy_image.shape, self.image.shape)
        self.assertTrue(np.any(noisy_image != self.image))
        self.assertTrue(np.all(noisy_image >= 0) and np.all(noisy_image <= 1))

    def test_invalid_noise_type(self):
        with self.assertRaises(ValueError):
            noise(self.image, noise_type='invalid', noise_intensity=0.1)

    def test_zero_intensity_gaussian_noise(self):
        noisy_image = noise(self.image, noise_type='gaussian', noise_intensity=0)
        np.testing.assert_array_equal(noisy_image, self.image)

    def test_zero_intensity_salt_and_pepper_noise(self):
        noisy_image = noise(self.image, noise_type='salt_and_pepper', noise_intensity=0)
        np.testing.assert_array_equal(noisy_image, self.image)

    def test_add_poisson_noise(self):
        noisy_image = noise(self.image, noise_type='poisson', scale=5)
        self.assertEqual(noisy_image.shape, self.image.shape)
        self.assertTrue(np.any(noisy_image != self.image))
        self.assertTrue(np.all(noisy_image >= 0) and np.all(noisy_image <= 1))

if __name__ == '__main__':
    unittest.main()