import unittest
import numpy as np
from src.medical_augmentation import MedicalAugmentation

class TestMedicalAugmentation(unittest.TestCase):
    def setUp(self):
        # Use a dummy image instead of loading from a file
        self.image = np.random.rand(128, 128).astype(np.float32)  # Random 128x128 grayscale image
        self.augmentor = MedicalAugmentation()

    def test_add_noise(self):
        params = {'add_noise': {'noise_type': 'gaussian', 'noise_intensity': 0.05}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    # Add additional tests here, using self.image as the base image for augmentations

if __name__ == "__main__":
    unittest.main()