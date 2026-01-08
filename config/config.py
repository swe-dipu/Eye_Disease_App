import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_fusioneyenet.pth"
ASSETS_DIR = BASE_DIR / "assets"

# Model configuration
CLASS_NAMES = ['Cataract', 'Conjunctivitis', 'Eyelid', 'Normal', 'Uveitis']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)

# Model parameters (ImageNet normalization)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Disease descriptions with detailed information
DISEASE_INFO = {
    'Cataract': {
        'description': 'Clouding of the eye lens causing blurred vision and reduced visual acuity',
        'severity': 'Moderate to Severe',
        'common_symptoms': 'Blurry vision, faded colors, glare sensitivity, poor night vision',
        'age_group': 'Primarily affects elderly (60+), but can occur at any age'
    },
    'Conjunctivitis': {
        'description': 'Inflammation of the conjunctiva (pink eye), often caused by infection or allergies',
        'severity': 'Mild to Moderate',
        'common_symptoms': 'Redness, itching, discharge, tearing, gritty feeling',
        'age_group': 'All ages, common in children and adults'
    },
    'Eyelid': {
        'description': 'Eyelid disorders including styes, chalazion, blepharitis, or eyelid inflammation',
        'severity': 'Mild to Moderate',
        'common_symptoms': 'Swelling, redness, pain, tenderness, bump formation',
        'age_group': 'All ages'
    },
    'Normal': {
        'description': 'Healthy eye with no detected abnormalities or pathological conditions',
        'severity': 'None',
        'common_symptoms': 'No symptoms - healthy eye appearance',
        'age_group': 'All ages'
    },
    'Uveitis': {
        'description': 'Inflammation of the uvea (middle layer of the eye), requiring prompt medical attention',
        'severity': 'Moderate to Severe',
        'common_symptoms': 'Eye pain, redness, blurred vision, light sensitivity, floaters',
        'age_group': 'Most common in adults 20-60 years'
    }
}

# OpenAI configuration
#OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective and capable
# Alternative: "gpt-3.5-turbo" for even lower cost
#MAX_TOKENS = 600  # Increased for more detailed responses
#TEMPERATURE = 0.7  # Balanced creativity and consistency

# Google Gemini configuration
GEMINI_MODEL = "gemini-2.5-flash"  # Using the requested 2.5 flash model
MAX_TOKENS = 800  # You can give Gemini a bit more space
TEMPERATURE = 0.7  # Balanced creativity and consistency

# Application settings
APP_TITLE = "Eye Disease Diagnosis System"
APP_ICON = "👁️"
MAX_IMAGE_SIZE_MB = 10  # Maximum upload size in MB
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']

# Confidence thresholds
CONFIDENCE_HIGH = 80.0  # High confidence threshold
CONFIDENCE_MEDIUM = 60.0  # Medium confidence threshold
# Below CONFIDENCE_MEDIUM = Low confidence