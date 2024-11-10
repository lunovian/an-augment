# tests/test_elastic_deformation.py
import unittest
import numpy as np
from src.augmentations.elastic_deformation import elastic_deformation

class TestElasticDeformation(unittest.TestCase):
    def test_output_shape(self):
        image = np.random.rand(128, 128)
        transformed_image = elastic_deformation(image)
        self.assertEqual(image.shape, transformed_image.shape)

    # Add more tests for parameters like alpha and sigma

if __name__ == "__main__":
    unittest.main()
