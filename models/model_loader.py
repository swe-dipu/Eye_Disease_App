import torch
import torch.nn as nn
from torchvision import models
from config.config import MODEL_PATH, NUM_CLASSES

def create_fusioneyenet_model(num_classes=NUM_CLASSES):
    """
    Create MobileNetV2 model architecture matching training
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        torch.nn.Module: FusionEyeNet model
    """
    # Fixed: Use weights=None instead of pretrained=False
    model = models.mobilenet_v2(weights=None)
    
    # Get input features for classifier
    num_ftrs = model.classifier[1].in_features
    
    # Replace classifier to match training architecture
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(128, num_classes)
    )
    
    return model

def load_model(model_path=MODEL_PATH, device=None):
    """
    Load trained model weights
    
    Args:
        model_path: Path to model weights
        device: Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_fusioneyenet_model()
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f" Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        raise Exception(f"Model file not found: {model_path}\nPlease ensure best_fusioneyenet.pth is in the models/ directory")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def get_target_layer(model):
    """
    Get target layer for Grad-CAM visualization
    
    Args:
        model: PyTorch model
        
    Returns:
        nn.Module: Target layer
    """
    # For FusionEyeNet, use last convolutional layer
    return model.features[-1]