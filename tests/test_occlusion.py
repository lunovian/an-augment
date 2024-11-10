# tests/test_occlusion.py
import unittest
import numpy as np
from src.augmentations.occlusion import occlusion

class TestOcclusion(unittest.TestCase):
    def setUp(self):
        # Create a test image (e.g., 128x128 pixels with random values between 0 and 1)
        self.image = np.random.rand(128, 128)

    def test_output_shape(self):
        # Check if the output shape matches the input shape
        occluded_image = occlusion(self.image, mask_shape='rectangle', mask_size_range=(0.1, 0.3))
        self.assertEqual(self.image.shape, occluded_image.shape)

    def test_occlusion_effect(self):
        # Apply occlusion and check that the image is different
        occluded_image = occlusion(self.image, mask_shape='rectangle', mask_size_range=(0.2, 0.3))
        self.assertFalse(np.array_equal(self.image, occluded_image))

if __name__ == "__main__":
    unittest.main()
