from src.augmentations.elastic_deformation import elastic_deformation
from src.augmentations.intensity_scaling import intensity_scaling
from src.augmentations.gaussian_blur import gaussian_blur
from src.augmentations.random_rotation import random_rotation
from src.augmentations.flip import flip
from src.augmentations.random_crop_and_scale import random_crop_and_scale
from src.augmentations.add_noise import add_noise
from src.augmentations.occlusion import occlusion

class MedicalAugmentation:
    def __init__(self):
        pass

    def apply_augmentations(self, image, **kwargs):
        """
        Applies a series of augmentations to the image based on provided parameters.
        
        Parameters:
        - image (np.array): Input image to be augmented.
        - kwargs (dict): Dictionary of augmentation options and parameters.
        
        Returns:
        - np.array: Augmented image.
        """
        if kwargs.get('elastic_deformation'):
            image = elastic_deformation(image, **kwargs['elastic_deformation'])
        if kwargs.get('intensity_scaling'):
            image = intensity_scaling(image, **kwargs['intensity_scaling'])
        if kwargs.get('gaussian_blur'):
            image = gaussian_blur(image, **kwargs['gaussian_blur'])
        if kwargs.get('random_rotation'):
            image = random_rotation(image, **kwargs['random_rotation'])
        if kwargs.get('flip'):
            image = flip(image, **kwargs['flip'])
        if kwargs.get('random_crop_and_scale'):
            image = random_crop_and_scale(image, **kwargs['random_crop_and_scale'])
        if kwargs.get('add_noise'):
            image = add_noise(image, **kwargs['add_noise'])
        if kwargs.get('occlusion'):
            image = occlusion(image, **kwargs['occlusion'])
        
        return image
