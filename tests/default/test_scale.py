import unittest
import numpy as np
from src.anaug.default import scale

class TestScale(unittest.TestCase):
    """Test cases for the scale augmentation function"""

    def setUp(self):
        """Set up test image"""
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    def test_upscale(self):
        """Test scaling up an image"""
        scale_factor = 2.0
        result = scale(self.test_image, scale_factor)
        self.assertEqual(result.shape, (200, 200, 3))

    def test_downscale(self):
        """Test scaling down an image"""
        scale_factor = 0.5
        result = scale(self.test_image, scale_factor)
        self.assertEqual(result.shape, (50, 50, 3))

    def test_scale_unchanged(self):
        """Test scaling with factor 1.0"""
        scale_factor = 1.0
        result = scale(self.test_image, scale_factor)
        self.assertEqual(result.shape, self.test_image.shape)
        
    def test_scale_grayscale(self):
        """Test scaling grayscale image"""
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        scale_factor = 1.5
        result = scale(gray_image, scale_factor)
        self.assertEqual(result.shape, (150, 150))

    def test_type_preservation(self):
        """Test if output maintains same dtype as input"""
        scale_factor = 1.5
        result = scale(self.test_image, scale_factor)
        self.assertEqual(result.dtype, self.test_image.dtype)

if __name__ == '__main__':
    unittest.main()