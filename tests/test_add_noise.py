# tests/test_add_noise.py
import unittest
import numpy as np
from src.augmentations.add_noise import add_noise

class TestAddNoise(unittest.TestCase):
    def setUp(self):
        # Create a test image (e.g., 128x128 pixels with random values between 0 and 1)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        # Check if the output shape matches the input shape
        noisy_image = add_noise(self.image, noise_type='gaussian', noise_intensity=0.1)
        self.assertEqual(self.image.shape, noisy_image.shape)

    def test_gaussian_noise(self):
        # Apply Gaussian noise and check that the image is different
        noisy_image = add_noise(self.image, noise_type='gaussian', noise_intensity=0.1)
        self.assertFalse(np.array_equal(self.image, noisy_image))

    def test_salt_and_pepper_noise(self):
        # Apply salt-and-pepper noise and check that the image is different
        noisy_image = add_noise(self.image, noise_type='salt_and_pepper', noise_intensity=0.05)
        self.assertFalse(np.array_equal(self.image, noisy_image))

if __name__ == "__main__":
    unittest.main()
