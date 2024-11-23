# tests/default/test_blur.py

import unittest
import numpy as np
from anaug.default import blur


class TestBlurFunctions(unittest.TestCase):
    
    def setUp(self):
        # Create a synthetic grayscale and color image
        self.gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_gaussian_blur(self):
        blurred = blur(self.gray_image, blur_type='gaussian', blur_radius=2)
        self.assertEqual(blurred.shape, self.gray_image.shape)
        self.assertEqual(blurred.dtype, self.gray_image.dtype)
    
    def test_uniform_blur(self):
        blurred = blur(self.gray_image, blur_type='uniform', blur_radius=5)
        self.assertEqual(blurred.shape, self.gray_image.shape)
        self.assertEqual(blurred.dtype, self.gray_image.dtype)
    
    def test_median_blur_valid_radius(self):
        # Test with a valid odd blur_radius >=3
        blurred = blur(self.gray_image, blur_type='median', blur_radius=3)
        self.assertEqual(blurred.shape, self.gray_image.shape)
        self.assertEqual(blurred.dtype, self.gray_image.dtype)
    
    def test_motion_blur(self):
        blurred = blur(self.color_image, blur_type='motion', blur_radius=5, angle=45)
        self.assertEqual(blurred.shape, self.color_image.shape)
        self.assertEqual(blurred.dtype, self.color_image.dtype)
    
    def test_bilateral_blur(self):
        blurred = blur(
            self.color_image,
            blur_type='bilateral',
            blur_radius=0,  # Not used for bilateral
            border_type='reflect',
            diameter=15,
            sigma_color=75,
            sigma_space=75
        )
        self.assertEqual(blurred.shape, self.color_image.shape)
        self.assertEqual(blurred.dtype, self.color_image.dtype)
    
    def test_box_blur(self):
        blurred = blur(
            self.gray_image,
            blur_type='box',
            blur_radius=5,
            border_type='reflect'
        )
        self.assertEqual(blurred.shape, self.gray_image.shape)
        self.assertEqual(blurred.dtype, self.gray_image.dtype)
    
    def test_adaptive_blur_mean(self):
        blurred = blur(
            self.gray_image,
            blur_type='adaptive',
            blur_radius=0,  # Not used for adaptive
            border_type='reflect',
            max_kernel_size=15,
            adaptive_type='mean',
            block_size=11,
            C=2
        )
        self.assertEqual(blurred.shape, self.gray_image.shape)
        self.assertEqual(blurred.dtype, self.gray_image.dtype)
    
    def test_adaptive_blur_gaussian(self):
        blurred = blur(
            self.gray_image,
            blur_type='adaptive',
            blur_radius=0,  # Not used for adaptive
            border_type='reflect',
            max_kernel_size=15,
            adaptive_type='gaussian',
            block_size=11,
            C=2
        )
        self.assertEqual(blurred.shape, self.gray_image.shape)
        self.assertEqual(blurred.dtype, self.gray_image.dtype)
    
    def test_invalid_blur_type(self):
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='invalid_blur', blur_radius=5)
    
    def test_invalid_blur_radius_gaussian(self):
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='gaussian', blur_radius=-1)
    
    def test_invalid_blur_radius_uniform(self):
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='uniform', blur_radius=0)
    
    def test_invalid_blur_radius_median_even(self):
        # Test with an even blur_radius which should raise ValueError
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='median', blur_radius=2)  # Even number
    
    def test_invalid_blur_radius_median_zero(self):
        # Test with a blur_radius of 0 which should raise ValueError
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='median', blur_radius=0)  # Zero
    
    def test_invalid_blur_radius_median_less_than_three(self):
        # Since blur_radius=1 is allowed and returns the original image,
        # this test should **not** expect a ValueError.
        # Instead, it should verify that the original image is returned.
        blurred = blur(self.gray_image, blur_type='median', blur_radius=1)
        np.testing.assert_array_almost_equal(blurred, self.gray_image)
    
    def test_invalid_bilateral_parameters(self):
        with self.assertRaises(ValueError):
            blur(
                self.color_image,
                blur_type='bilateral',
                blur_radius=0,  # Not used for bilateral
                border_type='reflect',
                diameter=-1,
                sigma_color=75,
                sigma_space=75
            )
    
    def test_invalid_box_parameters(self):
        with self.assertRaises(ValueError):
            blur(
                self.gray_image,
                blur_type='box',
                blur_radius=4,  # Even number
                border_type='reflect'
            )
    
    def test_invalid_adaptive_parameters(self):
        with self.assertRaises(ValueError):
            blur(
                self.gray_image,
                blur_type='adaptive',
                blur_radius=0,
                border_type='reflect',
                max_kernel_size=14,  # Even number
                adaptive_type='mean',
                block_size=11,
                C=2
            )
    
    def test_no_blur_gaussian_zero_radius(self):
        blurred = blur(self.gray_image, blur_type='gaussian', blur_radius=0)
        np.testing.assert_array_almost_equal(blurred, self.gray_image)
    
    def test_no_blur_uniform_radius_one(self):
        # For 'uniform', blur_radius=1 should be treated as kernel_size=1 (no blur)
        blurred = blur(
            self.gray_image,
            blur_type='uniform',
            blur_radius=1,
            border_type='reflect'
        )
        np.testing.assert_array_almost_equal(blurred, self.gray_image)
    
    def test_no_blur_median_radius_one(self):
        # For 'median', blur_radius=1 should return the original image
        blurred = blur(self.gray_image, blur_type='median', blur_radius=1)
        np.testing.assert_array_almost_equal(blurred, self.gray_image)


if __name__ == '__main__':
    unittest.main()