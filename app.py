import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Import custom modules
from models.model_loader import load_model, get_target_layer
from utils.preprocessing import preprocess_image
from utils.prediction import predict
from utils.gradcam import GradCAMVisualizer
from utils.llm_handler import LLMRecommendationEngine
from config.config import CLASS_NAMES, DISEASE_INFO

# Page configuration
st.set_page_config(
    page_title="Eye Disease Diagnosis System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #F3E5F5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #7B1FA2;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load model and initialize system components"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    target_layer = get_target_layer(model)
    
    # Initialize LLM engine with multiple fallbacks
    llm_engine = None
    try:
        llm_engine = LLMRecommendationEngine()
        st.sidebar.success("🤖 AI Recommendations: Enabled")
    except Exception as e:
        st.sidebar.warning(f"🤖 AI Recommendations: Basic Mode")
        # Create basic fallback
        class BasicEngine:
            def generate_recommendation(self, disease_class, confidence_score, user_query=None):
                return f"""
## Basic Guidance for {disease_class}

**Confidence:** {confidence_score:.1f}%

**Recommendations:**
1. **Consult an Ophthalmologist**: Schedule a comprehensive eye examination
2. **Professional Diagnosis**: Get proper medical evaluation and treatment plan
3. **Symptom Monitoring**: Keep track of any changes in your condition
4. **Follow Medical Advice**: Adhere to prescribed treatments and follow-up appointments

*For detailed AI-powered recommendations, please ensure your LLM API configuration is correct.*
"""
        llm_engine = BasicEngine()
    
    return model, target_layer, device, llm_engine

def display_probabilities(probabilities):
    """Display probability chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#2E7D32' if p == max(probs) else '#1E88E5' for p in probs]
    
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_confidence_class(confidence):
    """Get confidence level classification"""
    if confidence >= 80:
        return "High", "confidence-high"
    elif confidence >= 60:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Eye Disease Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered External Eye Disease Classification with Explainable AI & Personalized Recommendations</p>', unsafe_allow_html=True)
    
    # Initialize system
    try:
        model, target_layer, device, llm_engine = load_system()
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        # Fixed: Remove use_container_width or use appropriate method
        st.markdown("### Eye Health AI")
        st.markdown("---")
        
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** an eye image (JPG, PNG, JPEG)
        2. **View** AI prediction with confidence scores
        3. **Analyze** Grad-CAM visualization
        4. **Get** personalized recommendations from AI
        5. **Ask** specific questions about your condition
        """)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info("""
        This system uses deep learning (FusionEyeNet) to classify external eye diseases:
        - **Cataract**
        - **Conjunctivitis**
        - **Eyelid Disorders**
        - **Normal Eye**
        - **Uveitis**
        """)
        
        st.warning("⚠️ **Disclaimer**: This is a diagnostic aid tool. Always consult a qualified ophthalmologist for proper medical advice.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Eye Image")
        uploaded_file = st.file_uploader(
            "Choose an eye image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the affected eye"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            # Fixed: Use width parameter instead of use_container_width
            st.image(image, caption="Uploaded Image", width=300)
            
            # Add analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("🧠 AI is analyzing the image..."):
                    # Preprocess
                    image_tensor = preprocess_image(image)
                    
                    # Predict
                    results = predict(model, image_tensor, device)
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.image_tensor = image_tensor
                    st.session_state.image = image
                    
                st.success("✅ Analysis complete!")
                st.rerun()
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display prediction
            confidence_level, confidence_class = get_confidence_class(results['confidence'])
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Predicted Condition</h2>
                <h1 style="color: #1E88E5; margin: 10px 0;">{results['class']}</h1>
                <h3>Confidence: <span class="{confidence_class}">{results['confidence']:.2f}% ({confidence_level})</span></h3>
                <p><strong>Description:</strong> {DISEASE_INFO[results['class']]['description']}</p>
                <p><strong>Severity:</strong> {DISEASE_INFO[results['class']]['severity']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probability chart
            st.subheader("📈 Classification Probabilities")
            fig = display_probabilities(results['probabilities'])
            st.pyplot(fig)
            
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Grad-CAM Visualization
    if 'results' in st.session_state:
        st.divider()
        st.header("🔥 Explainable AI - Grad-CAM Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.image, width=300)
        
        with col2:
            st.subheader("Attention Heatmap")
            with st.spinner("Generating Grad-CAM..."):
                try:
                    gradcam_viz = GradCAMVisualizer(model, target_layer)
                    heatmap = gradcam_viz.generate_heatmap(
                        st.session_state.image_tensor,
                        st.session_state.results['class_idx']
                    )
                    st.image(heatmap, width=300)
                    gradcam_viz.cleanup()
                    
                    st.caption("🔍 Red regions indicate areas that most influenced the AI's decision")
                except Exception as e:
                    st.error(f"Error generating Grad-CAM: {str(e)}")
    
    # LLM Recommendations
    if 'results' in st.session_state and llm_engine is not None:
        st.divider()
        st.header("🤖 AI-Powered Personalized Recommendations")
        
        # Default recommendations
        if st.button("📋 Get General Recommendations"):
            with st.spinner("🧠 AI is generating personalized recommendations..."):
                try:
                    recommendation = llm_engine.generate_recommendation(
                        st.session_state.results['class'],
                        st.session_state.results['confidence']
                    )
                    st.session_state.recommendation = recommendation
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
        
        # Display recommendation if exists
        if 'recommendation' in st.session_state:
            st.markdown(f"""
            <div class="recommendation-box">
                {st.session_state.recommendation}
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Custom query
        st.subheader("💬 Ask Specific Questions")
        user_query = st.text_area(
            "Ask the AI assistant about your condition:",
            placeholder="Example: What will patient primarily do and what should they avoid? Also what the syndrome?",
            height=100
        )
        
        if st.button("🔮 Get Custom Advice"):
            if user_query.strip():
                with st.spinner("🧠 AI is generating custom advice..."):
                    try:
                        custom_recommendation = llm_engine.generate_recommendation(
                            st.session_state.results['class'],
                            st.session_state.results['confidence'],
                            user_query
                        )
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h3>📝 Custom Response</h3>
                            {custom_recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating custom advice: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question first")
    elif 'results' in st.session_state:
        st.info("🤖 AI recommendation service is currently unavailable. Basic classification results are still available.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #757575; padding: 20px;">
        <p>🏥 <strong>Eye Disease Diagnosis System</strong> | Powered by Deep Learning & AI</p>
        <p>⚠️ For educational and screening purposes only. Not a substitute for professional medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()