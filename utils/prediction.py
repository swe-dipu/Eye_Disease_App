import torch
import torch.nn.functional as F
from config.config import CLASS_NAMES

def predict(model, image_tensor, device):
    """
    Make prediction on preprocessed image
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = {
            CLASS_NAMES[i]: probabilities[0][i].item() * 100 
            for i in range(len(CLASS_NAMES))
        }
        
    return {
        'class': predicted_class,
        'confidence': confidence_score,
        'class_idx': predicted_idx.item(),
        'probabilities': all_probs
    }