import os
import torch
from src.anaug.style_transform.vgg16 import load_vgg16_model, process_image, extract_features, gram_matrix, compute_content_loss, compute_style_loss, perform_style_transfer

# Test loading VGG16 model
def test_load_vgg16_model():
    model = load_vgg16_model()
    assert model is not None, "VGG16 model loading failed!"
    assert isinstance(model, torch.nn.Module), "Loaded model is not a PyTorch module."

# Test image preprocessing (ensure tensor output)
def test_process_image():
    image_path = "./images/starry_night.jpg"
    image_tensor = process_image(image_path)
    assert isinstance(image_tensor, torch.Tensor), "The processed image is not a tensor."
    assert image_tensor.size(0) == 1, "Batch size is not 1 for the input image."
    assert image_tensor.size(1) == 3, "Image channels are not 3."
    assert image_tensor.size(2) == 224 and image_tensor.size(3) == 224, "Image size is not 224x224."

# Test feature extraction
def test_extract_features():
    model = load_vgg16_model()
    image_path = "./images/starry_night.jpg"
    image_tensor = process_image(image_path)

    layers_to_test = ['21']

    # Test for specific layers
    features = extract_features(image_tensor, model, layers=layers_to_test)

    assert '21' in features, "Feature extraction failed for layer '21'."
    assert features['21'].dim() == 4, "Extracted features should be 4D."

# Test Gram matrix calculation
def test_gram_matrix():
    tensor = torch.randn(1, 3, 224, 224)
    gram = gram_matrix(tensor)

    assert gram.size(0) == 3, "Gram matrix rows should equal the number of channels."
    assert gram.size(1) == 3, "Gram matrix columns should equal the number of channels."

# Test loss functions (check non-zero losses)
def test_loss_functions():
    target_tensor = torch.randn(1, 512, 7, 7)
    content_tensor = torch.randn(1, 512, 7, 7)
    style_tensor = torch.randn(1, 512, 7, 7)
    
    content_loss = compute_content_loss(target_tensor, content_tensor)
    style_loss = compute_style_loss(target_tensor, style_tensor)
    
    assert content_loss.item() >= 0, "Content loss is negative!"
    assert style_loss.item() >= 0, "Style loss is negative!"

# Test style transfer end-to-end
def test_perform_style_transfer():
    content_image_path = "./images/starry_night.jpg"
    style_image_path = "./images/realistic_starry_night.jpg"
    
    # For testing purposes, you may mock the output instead of actual file I/O.
    result_image = perform_style_transfer(content_image_path, style_image_path, load_vgg16_model(), num_iterations=1, content_layers=['21'], style_layers=['0', '5', '10', '19', '21', '28'])
    
    assert os.path.exists(result_image), "Output style-transferred image does not exist."
    assert result_image.endswith(".jpg"), "The output image does not have a .jpg extension."
