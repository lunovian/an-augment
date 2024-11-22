import unittest
import numpy as np
from src.anaug.medical.random_lesion import generate_random_lesion

"""Unit tests for the random lesion generation function."""

class TestRandomLesion(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing
        self.image = np.zeros((256, 256), dtype=np.float32)

    def test_random_lesion_shape(self):
        # Test if the output image has the same shape as the input
        lesion_image = generate_random_lesion(self.image)
        self.assertEqual(lesion_image.shape, self.image.shape)

    def test_random_lesion_intensity_range(self):
        # Test if the added lesion has intensity within the specified range
        lesion_image = generate_random_lesion(self.image, intensity_range=(0.3, 0.7))
        self.assertTrue(np.max(lesion_image) <= 0.7)
        self.assertTrue(np.max(lesion_image) >= 0.3)

    def test_random_lesion_size_range(self):
        # Ensure the lesion size is within the given range
        lesion_image = generate_random_lesion(self.image, size_range=(20, 50))
        lesion_diff = lesion_image - self.image
        lesion_area = np.sum(lesion_diff > 0)  # Count lesion pixels
        self.assertTrue(lesion_area > 20**2)  # Minimum lesion area
        self.assertTrue(lesion_area < 50**2 * np.pi)  # Approximate maximum lesion area

    def test_random_lesion_texture_strength(self):
        # Test the effect of texture_strength parameter
        lesion_smooth = generate_random_lesion(self.image, texture_strength=0.1)
        lesion_textured = generate_random_lesion(self.image, texture_strength=0.9)

        # Ensure lesions with high texture_strength have more variability
        self.assertGreater(np.std(lesion_textured), np.std(lesion_smooth))

    def test_random_lesion_clipping(self):
        # Test if the function clips the output values within [0, 1]
        lesion_image = generate_random_lesion(self.image, intensity_range=(0.5, 1.2))
        self.assertTrue(np.all(lesion_image >= 0))
        self.assertTrue(np.all(lesion_image <= 1))

    def test_random_lesion_invalid_input(self):
        # Test if invalid input raises appropriate errors
        with self.assertRaises(ValueError):
            generate_random_lesion(np.ones((256, 256, 3)), intensity_range=(0.2, 0.8))  # 3D array not allowed
        with self.assertRaises(ValueError):
            generate_random_lesion(self.image, intensity_range=(1.5, 2))  # Invalid intensity range
        with self.assertRaises(ValueError):
            generate_random_lesion(self.image, size_range=(-10, 50))  # Negative lesion size

if __name__ == '__main__':
    unittest.main()
