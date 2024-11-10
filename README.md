# MedAugment

**MedAugment** is a Python library designed for data augmentation in medical imaging, tailored specifically to modalities like MRI, CT, and X-ray. By simulating realistic clinical variations through domain-specific transformations, MedAugment enhances dataset diversity, helping to build more robust machine learning and deep learning models in medical AI.

## Features

- **Medical-Specific Augmentations**: Elastic deformation, intensity scaling, Gaussian blurring, and occlusion, all optimized for medical images.
- **Fine-Grained Control**: Adjustable parameters for each augmentation allow precise control over transformation intensity and probability.
- **Batch Processing & Randomization**: Apply augmentations to image batches with flexible randomization settings.
- **Visualization Tools**: Easily visualize original and augmented images side-by-side for quick quality checks.

## Installation

To install MedAugment, clone the repository and install the requirements:

```bash
git clone https://github.com/yourusername/medaugment.git
cd medaugment
pip install -r requirements.txt
```

## Usage

Here's a quick example of how to use MedAugment:

```python
from medaugment import MedicalAugmentation

# Initialize augmentation class
augmentor = MedicalAugmentation()

# Load an image (assumed as a numpy array)
image = load_your_image_function()

# Apply augmentations
augmented_image = augmentor.apply_augmentations(image, elastic_deformation=True, intensity_scaling=True)

# Visualize
augmentor.visualize(image, augmented_image)
```

## Augmentations

- **Elastic Deformation**: Adds tissue-like distortions with controlled intensity.
- **Intensity Scaling**: Simulates variations in brightness and contrast.
- **Gaussian Blur**: Mimics lower-resolution scanner settings.
- **Random Rotation**: Applies small random rotations, helpful for MRI/CT.
- **Flipping**: Horizontal/vertical flips, useful for symmetrical structures.
- **Occlusion**: Adds random occlusions to simulate partial visibility.

## Contributing

Feel free to contribute! Fork the repository, make your changes, and submit a pull request.

## License

MedAugment is licensed under the Apache-2.0 License.
