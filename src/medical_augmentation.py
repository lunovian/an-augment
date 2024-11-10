# src/medical_augmentation.py
from .augmentations.elastic_deformation import elastic_deformation
from .augmentations.intensity_scaling import intensity_scaling
from .augmentations.gaussian_blur import gaussian_blur
from .augmentations.random_rotation import random_rotation
from .augmentations.flip import flip
from .augmentations.random_crop_and_scale import random_crop_and_scale
from .augmentations.add_noise import add_noise
from .augmentations.occlusion import occlusion

class MedicalAugmentation:
    def __init__(self):
        pass

    def apply_augmentations(self, image, **kwargs):
        # Example application logic for chaining augmentations
        if kwargs.get('elastic_deformation', False):
            image = elastic_deformation(image)
        # Repeat for other augmentations based on kwargs
        return image
