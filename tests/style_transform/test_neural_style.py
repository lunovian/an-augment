import os
import torch
from src.anaug.style_transform.neural_style import load_vgg19_model, process_image, perform_style_transfer

# Test loading VGG19 model
def test_load_vgg19_model():
    model = load_vgg19_model()
    assert model is not None, "VGG19 model loading failed!"
    assert isinstance(model, torch.nn.Module), "Loaded model is not a PyTorch module."

# Test image preprocessing (ensure tensor output)
def test_process_image():
    image_path = "./images/starry_night.jpg"  # Replace with actual image or mock input.
    image_tensor = process_image(image_path)
    assert isinstance(image_tensor, torch.Tensor), "The processed image is not a tensor."
    assert image_tensor.size(0) == 1, "Batch size is not 1 for the input image."
    assert image_tensor.size(1) == 3, "Image channels are not 3."
    assert image_tensor.size(2) == 512 and image_tensor.size(3) == 512, "Image size is not 512x512."

# Test style transfer end-to-end
def test_perform_style_transfer():
    content_image_path = "./images/starry_night.jpg"  # Replace with actual path
    style_image_path = "./images/realistic_starry_night.jpg"  # Replace with actual path
    output_image_path = "./images/output_styled_image.jpg"  # Specify output image path
    
    # For testing purposes, you may mock the output instead of actual file I/O.
    result_image = perform_style_transfer(content_image_path, style_image_path, output_image_path=output_image_path)
    
    assert os.path.exists(result_image), "Output style-transferred image does not exist."
    assert result_image.endswith(".jpg"), "The output image does not have a .jpg extension."