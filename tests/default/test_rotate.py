import unittest
import numpy as np
import cv2
from an_augment.default.rotate import rotate

class TestRotate(unittest.TestCase):
    """Test cases for the rotate function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (25, 25), (75, 75), (255, 255, 255), -1)

    def test_rotate_90_degrees(self):
        """Test rotating image by 90 degrees."""
        rotated = rotate(self.test_image, 90)
        self.assertEqual(rotated.shape, self.test_image.shape)
        self.assertTrue(np.any(rotated != self.test_image))

    def test_rotate_360_degrees(self):
        """Test rotating image by 360 degrees should give similar result."""
        rotated = rotate(self.test_image, 360)
        np.testing.assert_array_almost_equal(rotated, self.test_image)

    def test_rotate_0_degrees(self):
        """Test rotating image by 0 degrees should return same image."""
        rotated = rotate(self.test_image, 0)
        np.testing.assert_array_equal(rotated, self.test_image)

    def test_image_dimensions(self):
        """Test if output image maintains same dimensions as input."""
        angles = [45, 90, 180, 270]
        for angle in angles:
            rotated = rotate(self.test_image, angle)
            self.assertEqual(rotated.shape, self.test_image.shape)

if __name__ == '__main__':
    unittest.main()