"""Test the output shape of the function."""

import unittest
import numpy as np
from src.an_augment.medical.medical_augmentation import MedicalAugmentation


class TestMedicalAugmentation(unittest.TestCase):
    """Test suite for the `MedicalAugmentation` class."""

    def setUp(self):
        """
        Set up a dummy image and initialize the MedicalAugmentation instance.

        This method creates a random 128x128 image as input data for testing.
        """
        # Use a dummy image instead of loading from a file
        self.image = np.random.rand(128, 128).astype(np.float32)
        self.augmentor = MedicalAugmentation()

    def test_add_noise(self):
        """Test the `add_noise` augmentation."""
        params = {
            'add_noise': {
                'noise_type': 'gaussian',
                'noise_intensity': 0.05
            }
        }
        augmented_image = self.augmentor.apply_augmentations(
            self.image,
            **params
        )
        self.assertEqual(augmented_image.shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
