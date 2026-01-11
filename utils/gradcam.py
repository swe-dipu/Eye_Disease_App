import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.preprocessing import denormalize_image

class GradCAMVisualizer:
    """Grad-CAM visualization for model interpretability"""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.cam = GradCAM(model=model, target_layers=[target_layer])
    
    def generate_heatmap(self, input_tensor, target_class):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Preprocessed image tensor
            target_class: Target class index
            
        Returns:
            numpy.ndarray: Heatmap overlay on original image
        """
        # Generate CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Denormalize input for visualization
        rgb_img = denormalize_image(input_tensor.cpu().squeeze())
        
        # Create heatmap overlay
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return visualization
    
    def cleanup(self):
        """Clean up resources"""
        del self.cam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()