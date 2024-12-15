import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

# Load VGG16 pre-trained model
def load_vgg16_model(weights=VGG16_Weights.DEFAULT):
    """
    Load and return the VGG16 model, with an option to use pre-trained weights.

    Args:
    - use_pretrained (bool): If True, load pre-trained weights. If False, use random weights.

    Returns:
    - model (torch.nn.Module): The loaded VGG16 model.
    """
    if weights is not None:
        # Load the VGG16 model with pre-trained weights (ImageNet)
        model = vgg16(weights=weights)
    else:
        # Load the VGG16 model without pre-trained weights (random initialization)
        model = vgg16(weights=None)
    
    model.eval()  # Set the model to evaluation mode
    return model

def process_image(image_path):
    """
    Preprocess the input image so it's ready to be passed to the VGG16 model.
    Args:
    - image_path (str): Path to the input image.

    Returns:
    - torch.Tensor: Processed image tensor with required shape and normalization.
    """
    input_image = Image.open(image_path)
    
    # Preprocessing pipeline matching VGG16 requirements (224x224 and normalization)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VGG16's expected size
        transforms.ToTensor(),  # Convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize like ImageNet images
    ])
    # Add a batch dimension (1, C, H, W)
    return preprocess(input_image).unsqueeze(0)

def extract_features(image_tensor, model, layers=None):
    """
    Extract features from specified layers of the VGG16 model.
    
    Args:
    - image_tensor (torch.Tensor): Input tensor (batch, channels, height, width).
    - model (torch.nn.Module): The VGG16 model.
    - layers (list, optional): List of layer names to extract (default: None extracts all layers).
    
    Returns:
    - dict: Dictionary containing extracted features for each selected layer.
    """
    if layers is None:
        layers = ['0', '5', '10', '19', '21', '28']  # VGG16 layer indices corresponding to common features

    x = image_tensor
    features = {}

    # Pass through each layer of the model
    for name, layer in model.features._modules.items():
        x = layer(x)
        
        # Check if current layer is part of layers to extract
        if name in layers:
            features[name] = x

    return features

def gram_matrix(tensor):
    """
    Compute the Gram Matrix for style loss comparison.
    
    Args:
    - tensor (torch.Tensor): Input tensor.
    
    Returns:
    - torch.Tensor: Gram matrix (N x N) where N is number of channels x width x height.
    """
    # The tensor shape: (batch_size, channels, height, width)
    _, channels, height, width = tensor.size()

    # Flatten the feature maps
    tensor = tensor.view(channels, height * width)

    # Compute the Gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

def compute_content_loss(target, content):
    """
    Compute content loss based on the mean squared error.
    
    Args:
    - target (torch.Tensor): Target features.
    - content (torch.Tensor): Content features.
    
    Returns:
    - torch.Tensor: Content loss (MSE).
    """
    return F.mse_loss(target, content)

def compute_style_loss(target, style):
    """
    Compute style loss based on the difference between the Gram matrices.
    
    Args:
    - target (torch.Tensor): Features from the generated image.
    - style (torch.Tensor): Features from the style image.
    
    Returns:
    - torch.Tensor: Style loss (based on Gram matrices).
    """
    target_gram = gram_matrix(target)
    style_gram = gram_matrix(style)
    return F.mse_loss(target_gram, style_gram)

def compute_total_loss(content_loss, style_loss, alpha=1, beta=1e6):
    """
    Combine content and style losses with weighted loss factors (alpha and beta).
    
    Args:
    - content_loss (torch.Tensor): Content loss.
    - style_loss (torch.Tensor): Style loss.
    - alpha (float): Weight for the content loss.
    - beta (float): Weight for the style loss.
    
    Returns:
    - torch.Tensor: Combined total loss.
    """
    return alpha * content_loss + beta * style_loss

def perform_style_transfer(content_image_path, style_image_path, model, num_iterations=500, alpha=1, beta=1e6, content_layers=['21'], style_layers=['0', '5', '10', '19', '21', '28'], output_image_path='styled_output.jpg'):
    """
    Perform style transfer using the VGG16 model.
    
    Args:
    - content_image_path (str): Path to content image.
    - style_image_path (str): Path to style image.
    - model (torch.nn.Module): The VGG16 model.
    - num_iterations (int): Number of iterations for optimization.
    - alpha (float): Weight for content loss.
    - beta (float): Weight for style loss.
    - content_layers (list): List of layers to use for content extraction.
    - style_layers (list): List of layers to use for style extraction.
    - output_image_path (str): Path to save the output style-transferred image.
    
    Returns:
    - str: Path to the output style-transferred image.
    """
    content_image = process_image(content_image_path)
    style_image = process_image(style_image_path)

    # Extract features for content and style images
    content_features = extract_features(content_image, model, layers=content_layers)
    style_features = extract_features(style_image, model, layers=style_layers)

    # Clone the content image to use as our starting point
    generated_image = content_image.clone().requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.LBFGS([generated_image])

    for i in range(num_iterations):
        def closure():
            optimizer.zero_grad()

            # Extract features for the generated image
            generated_features = extract_features(generated_image, model, layers=content_layers + style_layers)

            # Content and Style losses
            content_loss = compute_content_loss(generated_features[content_layers[0]], content_features[content_layers[0]])
            style_loss = 0.0
            for layer in style_layers:
                style_loss += compute_style_loss(generated_features[layer], style_features[layer])

            total_loss = compute_total_loss(content_loss, style_loss, alpha, beta)
            total_loss.backward(retain_graph=True)

            return total_loss

        loss = optimizer.step(closure)

        # Optional: Print loss at intervals
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    # Post-process output image
    output_image = generated_image.detach().cpu().squeeze(0)
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_image_path)

    return output_image_path
