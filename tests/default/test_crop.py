import unittest
import numpy as np
from an_augment.default.crop import crop

class TestCropFunction(unittest.TestCase):
    
    def test_crop_center(self):
        image = np.random.rand(100, 100, 3)
        crop_size = (50, 50)
        cropped_image = crop(image, crop_size)
        self.assertEqual(cropped_image.shape, (50, 50, 3))
    
    def test_crop_larger_than_image(self):
        image = np.random.rand(50, 50, 3)
        crop_size = (100, 100)
        cropped_image = crop(image, crop_size)
        self.assertEqual(cropped_image.shape, (50, 50, 3))
    
    def test_crop_exact_size(self):
        image = np.random.rand(100, 100, 3)
        crop_size = (100, 100)
        cropped_image = crop(image, crop_size)
        self.assertEqual(cropped_image.shape, (100, 100, 3))
    
    def test_crop_smaller_image(self):
        image = np.random.rand(30, 30, 3)
        crop_size = (20, 20)
        cropped_image = crop(image, crop_size)
        self.assertEqual(cropped_image.shape, (20, 20, 3))
    
    def test_crop_grayscale_image(self):
        image = np.random.rand(100, 100)
        crop_size = (50, 50)
        cropped_image = crop(image, crop_size)
        self.assertEqual(cropped_image.shape, (50, 50))

if __name__ == '__main__':
    unittest.main()