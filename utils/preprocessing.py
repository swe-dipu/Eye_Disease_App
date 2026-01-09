import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from config.config import IMG_SIZE, MEAN, STD

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def preprocess_image(image):
    """
    Preprocess uploaded image for model prediction
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def denormalize_image(tensor):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        numpy.ndarray: Denormalized image
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor.squeeze().permute(1, 2, 0).numpy()