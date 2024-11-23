import unittest
import numpy as np
from src.anaug.default.blur import blur


class TestBlur(unittest.TestCase):
    
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
    
    def test_median_blur(self):
        blurred = blur(self.gray_image, blur_type='median', blur_radius=5)
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
            blur_radius=0,  # Not used
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
            blur_radius=0,  # Not used
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
    
    def test_invalid_blur_radius_median(self):
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='median', blur_radius=2)  # Even number
    
    def test_invalid_blur_radius_motion(self):
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='motion', blur_radius=-5, angle=45)
    
    def test_invalid_bilateral_parameters(self):
        with self.assertRaises(ValueError):
            blur(
                self.color_image,
                blur_type='bilateral',
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
                max_kernel_size=14,  # Even number
                adaptive_type='mean',
                block_size=11,
                C=2
            )
    
    def test_no_blur_gaussian_zero_radius(self):
        blurred = blur(self.gray_image, blur_type='gaussian', blur_radius=0)
        np.testing.assert_array_almost_equal(blurred, self.gray_image)
    
    def test_no_blur_uniform_radius_one(self):
        blurred = blur(self.gray_image, blur_type='uniform', blur_radius=1)
        # Uniform blur with kernel size 1 should return the original image
        np.testing.assert_array_almost_equal(blurred, self.gray_image)
    
    def test_no_blur_median_radius_one(self):
        blurred = blur(self.gray_image, blur_type='median', blur_radius=1)
        # Median blur with kernel size 1 should return the original image
        np.testing.assert_array_almost_equal(blurred, self.gray_image)
    
    def test_no_blur_motion_zero_length(self):
        with self.assertRaises(ValueError):
            blur(self.gray_image, blur_type='motion', blur_radius=0, angle=0)
    
    def test_no_blur_bilateral_zero_diameter(self):
        with self.assertRaises(ValueError):
            blur(
                self.color_image,
                blur_type='bilateral',
                diameter=0,
                sigma_color=75,
                sigma_space=75
            )
    
    def test_no_blur_box_radius_one(self):
        blurred = blur(
            self.gray_image,
            blur_type='box',
            blur_radius=1,
            border_type='reflect'
        )
        # Box blur with kernel size 1 should return the original image
        np.testing.assert_array_almost_equal(blurred, self.gray_image)


if __name__ == '__main__':
    unittest.main()
