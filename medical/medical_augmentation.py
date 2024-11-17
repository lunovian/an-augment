"""Initialize the MedicalAugmentation class."""

from elastic_deformation import elastic_deformation
from intensity_scaling import intensity_scaling
from gaussian_blur import gaussian_blur
from random_rotation import random_rotation
from flip import flip
from random_crop_and_scale import random_crop_and_scale
from add_noise import add_noise
from occlusion import occlusion


class MedicalAugmentation:
    """
    A class to apply multiple augmentations to medical images.

    This class provides methods for chaining various augmentation techniques
    to simulate realistic variations in medical imaging data.
    """

    def __init__(self):
        """Initialize the MedicalAugmentation class."""
        pass

    def apply_augmentations(self, image, **kwargs):
        """
        Apply a series of augmentations to the image based on provided parameters.

        Parameters:
            image (np.array): Input image to be augmented.
            kwargs (dict): Dictionary of augmentation options and their parameters.

        Returns:
            np.array: Augmented image with the specified transformations applied.
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
            image = random_crop_and_scale(
                image, **kwargs['random_crop_and_scale']
            )
        if kwargs.get('add_noise'):
            image = add_noise(image, **kwargs['add_noise'])
        if kwargs.get('occlusion'):
            image = occlusion(image, **kwargs['occlusion'])

        return image
