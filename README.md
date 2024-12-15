# ANAugment (Advanced and Novel Augmentation)

ANAugment is a Python library for advanced and novel data augmentation, blending traditional techniques such as cropping and blurring with cutting-edge generative AI methods like style transfer, image inpainting, and latent space interpolation. It empowers users to enhance data diversity, improving the training and performance of machine learning models across images, text, and tabular datasets.

---

## Features

### Traditional Data Augmentation

- Blur
- Crop
- Elastic Deformation
- Flip
- Intensity
- Noise
- Random rotation
- Rotation
- Scale

### Generative AI-Based Augmentation

- **Style Transfer**: Apply artistic or domain-specific styles to images.
- **Image Inpainting**: Fill missing regions or simulate occlusions with AI-generated content.
- **Latent Space Interpolation**: Generate intermediate samples using GANs or VAEs.
- **Synthetic Data Generation**: Create new samples for image, text, or tabular datasets using generative models.

---

## Installation

Install ANAugment from source:

```bash
git clone https://github.com/lunovian/an-augment.git
cd an_augment
pip install .
```

---

## Quick Start

Here is an example of how to use ANAugment:

```python
from anaug import default, generative as d, g

# Apply basic transformations
cropped_image = d.crop(image, box=(50, 50, 200, 200))
blurred_image = d.blur(image, sigma=2)

# Apply generative AI augmentation
styled_image = g.style_transfer(image, style="VanGogh")
inpainted_image = g.inpaint(image, mask)
```

For more examples, check the [`examples`](examples/) folder.

---

## Documentation

Detailed documentation is available in the [`docs`](docs/) folder.

- [Installation Guide](docs/SETUP_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Tutorials](docs/TUTORIALS.md)

---

## Contributing

We welcome contributions from the community! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and test them.
4. Submit a pull request.

Please see our [CONTRIBUTING.md](docs/CONTRIBUTING.md) for more information.

---

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Special thanks to the open-source projects and pre-trained models that make generative AI augmentation possible.
