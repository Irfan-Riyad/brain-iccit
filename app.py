import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier", 
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Classification App")
st.write("Upload your model and classes file, then classify images with confidence scores.")

# ----------------------------
# Session State for Model Caching
# ----------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hybrid CNN Model Definition
# ----------------------------
class HybridCNN(nn.Module):
    def __init__(self, num_classes, pretrained=False, freeze_backbones=False, hidden=1024, p=0.5):
        super().__init__()
        r_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        d_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        # --- Backbone A: ResNet50 ---
        self.resnet = models.resnet50(weights=r_weights)
        self.resnet.fc = nn.Identity()

        # --- Backbone B: DenseNet121 ---
        self.densenet = models.densenet121(weights=d_weights)
        self.densenet.classifier = nn.Identity()

        feat_dim = 2048 + 1024 # ResNet50 output + DenseNet121 output

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
# Model Loader
# ----------------------------
@st.cache_resource
def load_model(model_file, num_classes):
    """
    Load HybridCNN model from checkpoint
    """
    device = st.session_state.device
    
    # Create HybridCNN model
    model = HybridCNN(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbones=False,
        hidden=1024,
        p=0.5
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image, img_size=224):
    """
    Preprocess image with standard ImageNet normalization
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

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
# Sidebar Configuration
# ----------------------------
st.sidebar.header("⚙️ Model Configuration")

# File uploaders
model_file = st.sidebar.file_uploader(
    "Upload Model (.pth file)",
    type=['pth', 'pt'],
    help="Upload your trained PyTorch model file"
)

classes_file = st.sidebar.file_uploader(
    "Upload Classes (.txt file)",
    type=['txt'],
    help="Upload a text file with one class name per line"
)

# Image size
img_size = st.sidebar.number_input(
    "Input Image Size",
    min_value=64,
    max_value=512,
    value=224,
    step=32,
    help="Image size used during training"
)



# Load model button
if st.sidebar.button("🔄 Load Model", type="primary"):
    if model_file is None:
        st.sidebar.error("Please upload a model file!")
    else:
        with st.spinner("Loading model..."):
            try:
                # Load class names
                if classes_file is not None:
                    class_names = load_classes(classes_file)
                else:
                    st.sidebar.error("Please upload a classes.txt file!")
                    st.stop()
                
                # Load model
                num_classes = len(class_names)
                model = load_model(model_file, num_classes)
                
                # Save to session state
                st.session_state.model = model
                st.session_state.class_names = class_names
                
                st.sidebar.success(f"✅ Model loaded! ({num_classes} classes)")
                st.sidebar.info(f"Device: {st.session_state.device}")
                
            except Exception as e:
                st.sidebar.error(f"Error loading model: {str(e)}")

# Display loaded classes
if st.session_state.class_names:
    with st.sidebar.expander("📋 Loaded Classes"):
        for idx, class_name in enumerate(st.session_state.class_names):
            st.write(f"{idx}: {class_name}")

st.sidebar.divider()
st.sidebar.caption("🧠 Brain Tumor Classifier v1.0")

# ----------------------------
# Main Content
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_image = st.file_uploader(
        "Choose an image to classify",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a brain MRI scan image"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("🔍 Prediction Results")
    
    if uploaded_image is not None:
        if st.session_state.model is None:
            st.warning("⚠️ Please load a model first using the sidebar!")
        else:
            if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                try:
                    with st.spinner("Analyzing image..."):
                        # Preprocess
                        input_tensor = preprocess_image(image, img_size).to(st.session_state.device)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = st.session_state.model(input_tensor)
                            probabilities = torch.softmax(outputs, dim=1)[0]
                            confidence, predicted_idx = torch.max(probabilities, 0)
                        
                        predicted_class = st.session_state.class_names[predicted_idx.item()]
                        confidence_score = confidence.item() * 100
                        
                        # Display main prediction
                        st.success(f"### Predicted: **{predicted_class}**")
                        st.metric("Confidence", f"{confidence_score:.2f}%")
                        
                        # Progress bar for confidence
                        st.progress(confidence_score / 100)
                        
                        st.divider()
                        
                        # Top predictions
                        st.subheader("📊 All Class Probabilities")
                        
                        # Sort by probability
                        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                        
                        for i in range(len(st.session_state.class_names)):
                            idx = sorted_indices[i].item()
                            prob = sorted_probs[i].item() * 100
                            class_name = st.session_state.class_names[idx]
                            
                            # Color code based on probability
                            if prob > 50:
                                st.markdown(f"**{class_name}**: :green[{prob:.2f}%]")
                            elif prob > 20:
                                st.markdown(f"**{class_name}**: :orange[{prob:.2f}%]")
                            else:
                                st.markdown(f"**{class_name}**: {prob:.2f}%")
                            
                            st.progress(prob / 100)
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("👆 Upload an image to get started")

# ----------------------------
# Instructions
# ----------------------------
with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Upload Model File (.pth)**
       - Click "Upload Model" in the sidebar
       - Select your trained PyTorch model file
    
    2. **Upload Classes File (.txt)**
       - Create a text file with one class name per line
       - Example format:
         ```
         glioma
         meningioma
         pituitary
         notumor
         ```
    
    3. **Configure Settings**
       - Set the input image size (default: 224)
    
    4. **Load Model**
       - Click the "🔄 Load Model" button
       - Wait for confirmation message
    
    5. **Upload & Classify**
       - Upload an image in the main area
       - Click "🚀 Classify Image"
       - View results with confidence scores
    
    ### Tips:
    - The app uses HybridCNN (ResNet50 + DenseNet121)
    - Image size should match your training configuration
    - The app supports GPU if available (check sidebar)
    - Classes must match the order used during training
    """)

# ----------------------------
# Footer
# ----------------------------
st.divider()
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Device", str(st.session_state.device).upper())
with col_b:
    if st.session_state.model:
        st.metric("Classes", len(st.session_state.class_names))
    else:
        st.metric("Classes", "Not loaded")
with col_c:
    if st.session_state.model:
        st.metric("Status", "✅ Ready")
    else:
        st.metric("Status", "⏳ Waiting")
