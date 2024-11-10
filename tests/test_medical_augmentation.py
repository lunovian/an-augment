import unittest
import numpy as np
import cv2
from src.medical_augmentation import MedicalAugmentation

class TestMedicalAugmentation(unittest.TestCase):
    def setUp(self):
        # Load the image from the file system
        self.image = cv2.imread("images/mri.jpg", cv2.IMREAD_GRAYSCALE) / 255.0  # Normalize to [0, 1] range
        self.augmentor = MedicalAugmentation()

    def test_elastic_deformation(self):
        params = {'elastic_deformation': {'alpha': 30, 'sigma': 4}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_intensity_scaling(self):
        params = {'intensity_scaling': {'brightness_factor': 1.2, 'contrast_factor': 1.3}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_gaussian_blur(self):
        params = {'gaussian_blur': {'blur_radius': 2}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_random_rotation(self):
        params = {'random_rotation': {'angle_range': (-15, 15)}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_flip(self):
        params = {'flip': {'flip_horizontal': True, 'flip_vertical': False}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_random_crop_and_scale(self):
        params = {'random_crop_and_scale': {'crop_size': (0.8, 0.8), 'scaling_factor': 1.0}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_add_noise(self):
        params = {'add_noise': {'noise_type': 'gaussian', 'noise_intensity': 0.05}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_occlusion(self):
        params = {'occlusion': {'mask_shape': 'rectangle', 'mask_size_range': (0.1, 0.2)}}
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

    def test_combined_augmentations(self):
        params = {
            'elastic_deformation': {'alpha': 30, 'sigma': 4},
            'intensity_scaling': {'brightness_factor': 1.2, 'contrast_factor': 1.3},
            'gaussian_blur': {'blur_radius': 2},
            'random_rotation': {'angle_range': (-15, 15)},
            'flip': {'flip_horizontal': True, 'flip_vertical': False},
            'random_crop_and_scale': {'crop_size': (0.8, 0.8), 'scaling_factor': 1.0},
            'add_noise': {'noise_type': 'gaussian', 'noise_intensity': 0.05},
            'occlusion': {'mask_shape': 'rectangle', 'mask_size_range': (0.1, 0.2)}
        }
        augmented_image = self.augmentor.apply_augmentations(self.image, **params)
        self.assertEqual(augmented_image.shape, self.image.shape)

if __name__ == "__main__":
    unittest.main()
