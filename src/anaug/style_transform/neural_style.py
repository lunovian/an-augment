import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import copy

# Load VGG19 model
def load_vgg19_model(weights=VGG19_Weights):
    """
    Load and return the VGG19 model pre-trained on ImageNet.

    Returns:
    - model (torch.nn.Module): The loaded VGG19 model.
    """
    model = vgg19(weights=weights).features
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # Set the model to evaluation mode
    return model

def process_image(image_path, max_size=512):
    """
    Preprocess the input image so it's ready to be passed to the VGG19 model.
    Args:
    - image_path (str): Path to the input image.
    - max_size (int): Maximum size of the image's width or height.

    Returns:
    - torch.Tensor: Processed image tensor with required shape and normalization.
    """
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing pipeline matching VGG19 requirements
    preprocess = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def gram_matrix(tensor):
    """
    Compute the Gram Matrix for style loss comparison.
    
    Args:
    - tensor (torch.Tensor): Input tensor.
    
    Returns:
    - torch.Tensor: Gram matrix (N x N) where N is number of channels x width x height.
    """
    _, n_channels, height, width = tensor.size()
    tensor = tensor.view(n_channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, style_img, content_img, content_layers, style_layers):
    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    model = nn.Sequential()

    i = 0  # Increment every time we see a conv layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]

    return model, style_losses, content_losses

def perform_style_transfer(content_image_path, style_image_path, output_image_path='styled_output.jpg', num_steps=300, style_weight=1000000, content_weight=1):
    """
    Perform style transfer using the VGG19 model.
    
    Args:
    - content_image_path (str): Path to content image.
    - style_image_path (str): Path to style image.
    - output_image_path (str): Path to save the output style-transferred image.
    - num_steps (int): Number of steps for optimization.
    - style_weight (float): Weight for style loss.
    - content_weight (float): Weight for content loss.
    
    Returns:
    - str: Path to the output style-transferred image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = process_image(content_image_path).to(device)
    style_img = process_image(style_image_path).to(device)

    assert content_img.size() == style_img.size(), "Content and style images must be of the same size"

    cnn = load_vgg19_model().to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, content_layers, style_layers)

    input_img = content_img.clone()

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}:")
                print(f"Style Loss : {style_score.item()} Content Loss: {content_score.item()}")

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    # Post-process output image
    output_image = input_img.cpu().clone().squeeze(0)
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_image_path)

    return output_image_path