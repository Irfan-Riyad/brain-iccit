import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import time

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 HybridCNN Brain Tumor Classifier")
st.markdown("**Powered by ResNet50 + DenseNet121 Fusion Architecture**")

# ----------------------------
# Session State Initialization
# ----------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ----------------------------
# Hybrid CNN Model Definition (Exact Match)
# ----------------------------
class HybridCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbones=True, hidden=1024, p=0.5):
        super().__init__()
        r_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        d_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        # --- Backbone A: ResNet50 ---
        self.resnet = models.resnet50(weights=r_weights)
        self.resnet.fc = nn.Identity()

        # --- Backbone B: DenseNet121 ---
        self.densenet = models.densenet121(weights=d_weights)
        self.densenet.classifier = nn.Identity()

        feat_dim = 2048 + 1024  # ResNet50 output + DenseNet121 output

        if freeze_backbones:
            for m in [self.resnet, self.densenet]:
                for p_ in m.parameters():
                    p_.requires_grad = False

        # --- Fusion + Classifier Head ---
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        fused = torch.cat([f1, f2], dim=1)
        return self.head(fused)

# ----------------------------
# Optimized Model Loader with Caching
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(model_bytes, num_classes):
    """
    Load HybridCNN model with caching for performance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create HybridCNN model with exact training parameters
    model = HybridCNN(
        num_classes=num_classes,
        pretrained=False,  # No pretrained weights needed for inference
        freeze_backbones=False,  # All params should be loaded from checkpoint
        hidden=1024,
        p=0.5
    )
    
    # Load checkpoint from bytes
    checkpoint = torch.load(io.BytesIO(model_bytes), map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    
    return model

# ----------------------------
# Exact Inference Transforms (Matching Training)
# ----------------------------
def get_inference_transform(img_size=224):
    """
    Returns transforms that exactly match validation/test transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ----------------------------
# Load Classes from File
# ----------------------------
def load_classes(classes_file):
    """
    Load class names from uploaded text file
    """
    content = classes_file.read().decode('utf-8')
    classes = [line.strip() for line in content.split('\n') if line.strip()]
    return classes

# ----------------------------
# Optimized Prediction Function
# ----------------------------
@torch.no_grad()
def predict_image(model, image, transform, device):
    """
    Fast prediction with proper preprocessing
    """
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    start_time = time.time()
    
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    
    inference_time = time.time() - start_time
    
    # Get predictions
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    return predicted_idx.item(), confidence.item(), probabilities.cpu(), inference_time

# ----------------------------
# Sidebar Configuration
# ----------------------------
st.sidebar.header("⚙️ Model Configuration")

# Model file uploader
model_file = st.sidebar.file_uploader(
    "📦 Upload Model (.pth)",
    type=['pth', 'pt'],
    help="Upload your trained HybridCNN model checkpoint"
)

# Classes file uploader
classes_file = st.sidebar.file_uploader(
    "📋 Upload Classes (.txt)",
    type=['txt'],
    help="Upload text file with one class name per line"
)

# Image size configuration
img_size = st.sidebar.number_input(
    "🖼️ Input Image Size",
    min_value=128,
    max_value=512,
    value=224,
    step=32,
    help="Must match training image size (default: 224)"
)

# Load model button
if st.sidebar.button("🔄 Load Model", type="primary", use_container_width=True):
    if model_file is None:
        st.sidebar.error("❌ Please upload a model file!")
    elif classes_file is None:
        st.sidebar.error("❌ Please upload a classes file!")
    else:
        with st.spinner("⏳ Loading model... This may take a moment."):
            try:
                # Load class names
                class_names = load_classes(classes_file)
                num_classes = len(class_names)
                
                # Read model file as bytes for caching
                model_bytes = model_file.read()
                
                # Load model with caching
                model = load_model_cached(model_bytes, num_classes)
                
                # Save to session state
                st.session_state.model = model
                st.session_state.class_names = class_names
                st.session_state.model_loaded = True
                
                st.sidebar.success(f"✅ Model Loaded Successfully!")
                st.sidebar.metric("Classes", num_classes)
                st.sidebar.metric("Device", str(st.session_state.device).upper())
                
                # Show model info
                total_params = sum(p.numel() for p in model.parameters())
                st.sidebar.info(f"📊 Total Parameters: {total_params:,}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error loading model: {str(e)}")
                st.session_state.model_loaded = False

# Display loaded classes
if st.session_state.class_names and st.session_state.model_loaded:
    with st.sidebar.expander("📋 Loaded Classes", expanded=False):
        for idx, class_name in enumerate(st.session_state.class_names):
            st.text(f"{idx}: {class_name}")

st.sidebar.divider()

# Model status indicator
if st.session_state.model_loaded:
    st.sidebar.success("🟢 Model Ready")
else:
    st.sidebar.warning("🟡 Model Not Loaded")

st.sidebar.caption("🧠 HybridCNN v1.0 | ResNet50 + DenseNet121")

# ----------------------------
# Main Content Area
# ----------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Upload Image")
    
    uploaded_image = st.file_uploader(
        "Choose a brain MRI scan",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a brain MRI image for classification"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image info
        st.caption(f"📏 Size: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.header("🔍 Prediction Results")
    
    if uploaded_image is not None:
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model first using the sidebar!")
            st.info("👈 Upload your .pth model and classes.txt file, then click 'Load Model'")
        else:
            if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                try:
                    # Get transform
                    transform = get_inference_transform(img_size)
                    
                    # Predict with progress
                    with st.spinner("🔬 Analyzing image..."):
                        predicted_idx, confidence, probabilities, inference_time = predict_image(
                            st.session_state.model,
                            image,
                            transform,
                            st.session_state.device
                        )
                    
                    predicted_class = st.session_state.class_names[predicted_idx]
                    confidence_score = confidence * 100
                    
                    # Display main prediction with custom styling
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="margin:0;">Predicted: {predicted_class}</h2>
                        <h1 style="margin:10px 0;">{confidence_score:.2f}%</h1>
                        <p style="margin:0;">Inference Time: {inference_time*1000:.2f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.progress(confidence_score / 100)
                    
                    st.divider()
                    
                    # Detailed predictions
                    st.subheader("📊 Detailed Probabilities")
                    
                    # Sort by probability
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                    
                    # Create columns for better layout
                    for i in range(len(st.session_state.class_names)):
                        idx = sorted_indices[i].item()
                        prob = sorted_probs[i].item() * 100
                        class_name = st.session_state.class_names[idx]
                        
                        # Create columns for class name and probability
                        c1, c2, c3 = st.columns([3, 1, 2])
                        
                        with c1:
                            # Highlight predicted class
                            if idx == predicted_idx:
                                st.markdown(f"**🎯 {class_name}**")
                            else:
                                st.markdown(f"{class_name}")
                        
                        with c2:
                            # Color-coded probability
                            if prob > 50:
                                st.markdown(f"<span style='color:#00C851;font-weight:bold;'>{prob:.2f}%</span>", unsafe_allow_html=True)
                            elif prob > 20:
                                st.markdown(f"<span style='color:#ffbb33;font-weight:bold;'>{prob:.2f}%</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"{prob:.2f}%")
                        
                        with c3:
                            st.progress(prob / 100)
                    
                    # Additional metrics
                    st.divider()
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Top Confidence", f"{confidence_score:.2f}%")
                    with metric_col2:
                        second_prob = sorted_probs[1].item() * 100
                        st.metric("2nd Highest", f"{second_prob:.2f}%")
                    with metric_col3:
                        margin = confidence_score - second_prob
                        st.metric("Confidence Margin", f"{margin:.2f}%")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Upload an image to get started")

# ----------------------------
# Instructions Section
# ----------------------------
st.divider()

with st.expander("📖 How to Use This App", expanded=False):
    st.markdown("""
    ### 🚀 Quick Start Guide:
    
    #### Step 1: Prepare Your Files
    - **Model File (.pth)**: Your trained HybridCNN checkpoint
    - **Classes File (.txt)**: Text file with class names (one per line)
    
    Example `classes.txt`:
    ```
    glioma
    meningioma
    notumor
    pituitary
    ```
    
    #### Step 2: Load Model
    1. Click "📦 Upload Model" in sidebar → Select your `.pth` file
    2. Click "📋 Upload Classes" → Select your `.txt` file
    3. Verify image size matches training (default: 224)
    4. Click "🔄 Load Model" button
    5. Wait for success message ✅
    
    #### Step 3: Classify Images
    1. Upload a brain MRI image in the main area
    2. Click "🚀 Classify Image"
    3. View detailed predictions and probabilities
    
    ### 🎯 Model Architecture:
    - **HybridCNN**: Fusion of ResNet50 + DenseNet121
    - **Feature Dimension**: 3072 (2048 + 1024)
    - **Classifier**: 1024 hidden units with 0.5 dropout
    - **Normalization**: ImageNet mean/std
    
    ### ⚡ Performance Tips:
    - Use GPU for faster inference (detected automatically)
    - Model is cached after first load (faster subsequent predictions)
    - Supports CUDA automatic mixed precision
    - Typical inference time: 50-200ms per image
    
    ### 🔧 Troubleshooting:
    - **Model won't load**: Ensure architecture matches training
    - **Wrong predictions**: Verify image size and class order
    - **Slow inference**: Check if GPU is available in sidebar
    - **Memory errors**: Try smaller batch of images
    """)

# ----------------------------
# Footer with Statistics
# ----------------------------
st.divider()

footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    st.metric("Device", str(st.session_state.device).upper())

with footer_col2:
    if st.session_state.model_loaded:
        st.metric("Status", "✅ Ready")
    else:
        st.metric("Status", "⏳ Waiting")

with footer_col3:
    if st.session_state.class_names:
        st.metric("Classes", len(st.session_state.class_names))
    else:
        st.metric("Classes", "Not Loaded")

with footer_col4:
    if torch.cuda.is_available():
        st.metric("CUDA", "✅ Available")
    else:
        st.metric("CUDA", "❌ Not Available")

st.caption("Built with ❤️ using Streamlit + PyTorch | HybridCNN Architecture")
