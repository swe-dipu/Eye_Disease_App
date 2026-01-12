import os
import google.generativeai as genai
from dotenv import load_dotenv
# This line now imports your "gemini-2.5-flash" model name
from config.config import GEMINI_MODEL, MAX_TOKENS, TEMPERATURE, DISEASE_INFO, CONFIDENCE_MEDIUM

# Load environment variables
load_dotenv()

class LLMRecommendationEngine:
    """Google Gemini-based recommendation system for eye diseases"""
    
    def __init__(self):
        """Initialize Google Gemini client"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        # Configure the client
        genai.configure(api_key=api_key)
        
        # Set up generation config
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        # This will be "gemini-2.5-flash" from your config
        self.model_name = GEMINI_MODEL

    def _build_system_prompt(self):
        """Creates a robust system prompt for the AI assistant."""
        return """
You are an expert ophthalmologist assistant AI. 
Your tone is professional, empathetic, and serious.
Your primary goal is to provide guidance and explain potential conditions based on an AI analysis.

IMPORTANT RULES:
1.  **You are NOT a doctor.** You cannot provide a definitive diagnosis or a treatment plan.
2.  Your guidance is based on a prediction, not a medical exam.
3.  **ALWAYS** emphasize that the user MUST consult a qualified ophthalmologist for a real diagnosis.
4.  Use clear Markdown formatting (headings, lists) for readability.
5.  **MANDATORY DISCLAIMER:** You MUST end every single response with the following disclaimer, exactly as written:
    ---
    **Disclaimer: This is AI-generated guidance and not a medical diagnosis. This tool is for informational purposes only. Please consult a qualified ophthalmologist for a definitive diagnosis and treatment plan.**
"""

    def _get_low_confidence_response(self, confidence_score):
        """
        Return a static, safe response for low-confidence predictions.
        """
        return f"""
## Analysis Inconclusive

**Confidence Score:** {confidence_score:.1f}%

This confidence level is too low for the AI to provide a meaningful recommendation. 
Low confidence can be caused by poor image quality (e.g., blur, poor lighting, wrong angle) or because the image features are ambiguous.

### Recommended Action

It is **strongly recommended** that you consult an ophthalmologist for a professional evaluation, especially if you are experiencing any symptoms like pain, vision changes, or redness.

---
**Disclaimer: This is AI-generated guidance and not a medical diagnosis. This tool is for informational purposes only. Please consult a qualified ophthalmologist for a definitive diagnosis and treatment plan.**
"""

    def _build_normal_prompt(self, confidence_score, user_query=None):
        """Builds a prompt for when the 'Normal' class is detected."""
        
        if user_query:
            return f"""
The analysis result is **'Normal'** with **{confidence_score:.1f}%** confidence.

Please answer the user's specific question based on this 'Normal' finding.
Remember to be reassuring and promote general eye health.

**User's question:** {user_query}
"""
        else:
            return f"""
The analysis result is **'Normal'** with **{confidence_score:.1f}%** confidence.

Please provide a reassuring response that covers the following:
1.  **## What "Normal" Means:** Explain that the AI did not detect signs of the scanned diseases.
2.  **## General Eye Health Tips:** Provide 3-5 actionable tips for maintaining good eye health (e.g., the 20-20-20 rule, regular check-ups, UV protection).
3.  **## When to See a Doctor:** Remind them to still seek medical advice if they *do* experience any new or persistent symptoms (like pain, redness, or vision changes) despite this result.
"""

    def _build_disease_prompt(self, disease_class, confidence_score, disease_info, user_query=None):
        """Builds a comprehensive prompt for a detected disease."""
        
        if confidence_score >= 80.0:
            confidence_statement = f"The analysis **strongly suggests** this condition"
        else:
            confidence_statement = f"The analysis indicates a **possibility** of this condition"

        prompt_context = f"""
**Analysis Context:**
-   **Predicted Condition:** {disease_class}
-   **Confidence Score:** {confidence_score:.1f}%
-   **Confidence Statement:** {confidence_statement}.
-   **Condition Description:** {disease_info.get('description', 'N/A')}
-   **Reported Severity:** {disease_info.get('severity', 'N/A')}
-   **Common Symptoms:** {disease_info.get('common_symptoms', 'N/A')}
-   **Typical Age Group:** {disease_info.get('age_group', 'N/A')}
"""
        
        if user_query:
            return f"""
{prompt_context}

Based on all the context above, please answer the user's specific question in a professional and empathetic manner.

**User's question:** {user_query}
"""
        else:
            return f"""
{prompt_context}

Based on all the context above, please provide a comprehensive guidance plan. 
Use these exact Markdown headings:

1.  **## What This Condition Means**
    (Briefly explain {disease_class}, referencing the description and common symptoms.)
2.  **## Recommended Immediate Actions**
    (What should the user do *right now*? e.g., schedule an appointment, rest eyes. Be very clear.)
3.  **## What to Avoid**
    (List actions or environments that could worsen the condition, e.g., "avoiding rubbing your eyes," "reducing screen time.")
4.  **## When to Seek Urgent Medical Help**
    (List "red flag" symptoms that require immediate medical attention, e.g., "sudden vision loss," "severe pain.")
"""

    def generate_recommendation(self, disease_class, confidence_score, user_query=None):
        """Generate recommendations using the Google Gemini API"""
        
        disease_info = DISEASE_INFO.get(disease_class, {})

        if disease_class != 'Normal' and confidence_score < CONFIDENCE_MEDIUM:
            return self._get_low_confidence_response(confidence_score)

        if disease_class == 'Normal':
            system_prompt = "You are a helpful and reassuring AI assistant for an eye health application. Your role is to provide reassurance and general eye health tips. You MUST end your response with the disclaimer: --- \n**Disclaimer: This is AI-generated guidance and not a medical diagnosis. This tool is for informational purposes only. Please consult a qualified ophthalmologist for a definitive diagnosis and treatment plan.**"
            prompt = self._build_normal_prompt(confidence_score, user_query)
        else:
            system_prompt = self._build_system_prompt()
            prompt = self._build_disease_prompt(disease_class, confidence_score, disease_info, user_query)
        
        try:
            # Initialize the model with the system prompt
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt
            )
            
            # Generate the content
            response = model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Handle potential safety blocks
            if not response.parts:
                return self._get_fallback_recommendation(disease_class, confidence_score, disease_info, "Response was blocked for safety reasons.")

            return response.text
            
        except Exception as e:
            # Fallback recommendations
            return self._get_fallback_recommendation(disease_class, confidence_score, disease_info, str(e))
    
    def _get_fallback_recommendation(self, disease_class, confidence_score, disease_info, error_msg):
        """Provide fallback recommendations"""
        print(f"LLM API Error: {error_msg}") # For server logs
        
        return f"""
## Medical Guidance for {disease_class}

**Confidence Level:** {confidence_score:.1f}%
**Condition:** {disease_info.get('description', 'Professional evaluation recommended')}

**Recommended Actions:**
1.  **Consult an Ophthalmologist**: Due to a temporary service interruption, our AI cannot provide detailed advice. Please schedule an appointment with an ophthalmologist for a proper diagnosis.
2.  **Monitor Symptoms**: Pay close attention to any changes in your symptoms.
3.  **Seek Urgent Care**: If you experience severe pain or sudden vision loss, seek urgent medical help.

*Note: The AI recommendation service is temporarily unavailable ({error_msg}). The guidance above is a general fallback. Please consult a healthcare professional.*
"""