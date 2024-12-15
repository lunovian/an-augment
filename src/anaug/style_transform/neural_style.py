from torchvision import models
import torch.nn.functional as F
import torch

fast_style_model = torch.load("path_to_model/fast_style_model.pth")
fast_style_model.eval()
