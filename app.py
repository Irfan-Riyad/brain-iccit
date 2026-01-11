import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import time
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier - Debug Mode", 
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 HybridCNN Brain Tumor Classifier - Debug Mode")
st.markdown("**Debugging wrong predictions**")

# ----------------------------
# Session State
# ----------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'checkpoint_classes' not in st.session_state:
    st.session_state.checkpoint_classes = None

# ----------------------------
# HybridCNN Model (Exact Match)
# ----------------------------
class HybridCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbones=True, hidden=1024, p=0.5):
        super().__init__()
        r_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        d_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        self.resnet = models.resnet50(weights=r_weights)
        self.resnet.fc = nn.Identity()

        self.densenet = models.densenet121(weights=d_weights)
        self.densenet.classifier = nn.Identity()

        feat_dim = 2048 + 1024

        if freeze_backbones:
            for m in [self.resnet, self.densenet]:
                for p_ in m.parameters():
                    p_.requires_grad = False

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
# Model Loader with Debug Info
# ----------------------------
def load_model_debug(model_file, num_classes):
    """
    Load model with extensive debugging
    """
    device = st.session_state.device
    
    st.info("🔍 Loading checkpoint...")
    checkpoint = torch.load(model_file, map_location=device)
    
    # Debug checkpoint structure
    st.write("**Checkpoint Structure:**")
    if isinstance(checkpoint, dict):
        st.write(f"- Type: Dictionary")
        st.write(f"- Keys: {list(checkpoint.keys())}")
        
        # Check for classes in checkpoint
        if 'classes' in checkpoint:
            st.session_state.checkpoint_classes = checkpoint['classes']
            st.success(f"✅ Found classes in checkpoint: {checkpoint['classes']}")
        else:
            st.warning("⚠️ No 'classes' key found in checkpoint")
    else:
        st.write(f"- Type: {type(checkpoint)}")
    
    # Create model
    st.info(f"🏗️ Creating HybridCNN with {num_classes} classes...")
    model = HybridCNN(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbones=False,
        hidden=1024,
        p=0.5
    )
    
    # Load state dict
    try:
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                st.success("✅ Loaded from 'state_dict' key")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Loaded from 'model_state_dict' key")
            else:
                model.load_state_dict(checkpoint)
                st.success("✅ Loaded checkpoint directly")
        else:
            model = checkpoint
            st.success("✅ Checkpoint is the model itself")
    except Exception as e:
        st.error(f"❌ Error loading state dict: {e}")
        return None
    
    model.to(device)
    model.eval()
    
    # Verify model structure
    st.write("**Model Structure Verification:**")
    st.write(f"- Final layer output features: {model.head[-1].out_features}")
    st.write(f"- Expected classes: {num_classes}")
    
    if model.head[-1].out_features != num_classes:
        st.error(f"❌ MISMATCH! Model has {model.head[-1].out_features} outputs but you provided {num_classes} classes!")
    
    return model

# ----------------------------
# Load Classes
# ----------------------------
def load_classes(classes_file):
    content = classes_file.read().decode('utf-8')
    classes = [line.strip() for line in content.split('\n') if line.strip()]
    return classes

# ----------------------------
# Multiple Transform Options for Testing
# ----------------------------
def get_transform_v1(img_size=224):
    """Original transform - exactly as training"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_transform_v2(img_size=224):
    """Alternative: No lambda expansion"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_transform_v3(img_size=224):
    """Alternative: With center crop"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ----------------------------
# Prediction with Debug Info
# ----------------------------
@torch.no_grad()
def predict_with_debug(model, image, transform, device, class_names):
    """
    Predict with extensive debugging
    """
    model.eval()
    
    # Show original image stats
    img_array = np.array(image)
    st.write("**Input Image Stats:**")
    st.write(f"- Shape: {img_array.shape}")
    st.write(f"- Data type: {img_array.dtype}")
    st.write(f"- Min value: {img_array.min()}")
    st.write(f"- Max value: {img_array.max()}")
    st.write(f"- Mean: {img_array.mean():.2f}")
    
    # Apply transform
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    st.write("**After Transform:**")
    st.write(f"- Tensor shape: {input_tensor.shape}")
    st.write(f"- Tensor device: {input_tensor.device}")
    st.write(f"- Tensor min: {input_tensor.min().item():.4f}")
    st.write(f"- Tensor max: {input_tensor.max().item():.4f}")
    st.write(f"- Tensor mean: {input_tensor.mean().item():.4f}")
    st.write(f"- Tensor std: {input_tensor.std().item():.4f}")
    
    # Forward pass
    logits = model(input_tensor)
    
    st.write("**Model Output (Logits):**")
    st.write(f"- Logits shape: {logits.shape}")
    st.write(f"- Logits: {logits[0].cpu().numpy()}")
    
    # Softmax
    probabilities = torch.softmax(logits, dim=1)[0]
    
    st.write("**Top 5 Predictions:**")
    # Use checkpoint classes if available, otherwise use provided class_names
    display_classes = st.session_state.checkpoint_classes if st.session_state.checkpoint_classes else class_names
    
    # Get top 5 predictions sorted by probability (descending)
    probs_np = probabilities.cpu().numpy()
    top5_indices = np.argsort(probs_np)[-5:][::-1]  # Get top 5 indices in descending order
    
    # Create data for table
    import pandas as pd
    top5_data = {
        'Rank': list(range(1, 6)),
        'Class': [display_classes[idx] for idx in top5_indices],
        'Probability': [f"{probs_np[idx]:.6f}" for idx in top5_indices],
        'Percentage': [f"{probs_np[idx]*100:.2f}%" for idx in top5_indices]
    }
    
    df = pd.DataFrame(top5_data)
    st.table(df)
    
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    return predicted_idx.item(), confidence.item(), probabilities.cpu()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Configuration")

model_file = st.sidebar.file_uploader("📦 Upload Model (.pth)", type=['pth', 'pt'])
classes_file = st.sidebar.file_uploader("📋 Upload Classes (.txt)", type=['txt'])

img_size = st.sidebar.number_input("🖼️ Image Size", min_value=128, max_value=512, value=224, step=32)

transform_version = st.sidebar.selectbox(
    "🔄 Transform Version",
    ["v1 (Original)", "v2 (No Lambda)", "v3 (Center Crop)"],
    help="Try different transforms to see which matches training"
)

if st.sidebar.button("🔄 Load Model", type="primary"):
    if model_file and classes_file:
        with st.spinner("Loading..."):
            try:
                class_names = load_classes(classes_file)
                st.sidebar.write(f"**Classes from file ({len(class_names)}):**")
                for i, cls in enumerate(class_names):
                    st.sidebar.write(f"{i}: {cls}")
                
                model_file.seek(0)
                model = load_model_debug(model_file, len(class_names))
                
                if model:
                    st.session_state.model = model
                    st.session_state.class_names = class_names
                    st.sidebar.success("✅ Model loaded!")
                    
                    # Check class order mismatch
                    if st.session_state.checkpoint_classes:
                        st.sidebar.write("**Classes from checkpoint:**")
                        for i, cls in enumerate(st.session_state.checkpoint_classes):
                            st.sidebar.write(f"{i}: {cls}")
                        
                        if st.session_state.checkpoint_classes != class_names:
                            st.sidebar.error("⚠️ CLASS ORDER MISMATCH DETECTED!")
                            st.sidebar.write("Your classes.txt order doesn't match the checkpoint!")
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                import traceback
                st.sidebar.code(traceback.format_exc())
    else:
        st.sidebar.error("Please upload both files!")

st.sidebar.divider()
if st.session_state.model:
    st.sidebar.success("🟢 Model Ready")
else:
    st.sidebar.warning("🟡 No Model Loaded")

# ----------------------------
# Main Content
# ----------------------------
st.header("📤 Upload Test Image")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if st.session_state.model:
            if st.button("🚀 Predict with Debug Info", type="primary"):
                st.subheader("🔍 Debug Information")
                
                # Select transform
                if transform_version == "v1 (Original)":
                    transform = get_transform_v1(img_size)
                elif transform_version == "v2 (No Lambda)":
                    transform = get_transform_v2(img_size)
                else:
                    transform = get_transform_v3(img_size)
                
                st.write(f"**Using transform: {transform_version}**")
                
                try:
                    predicted_idx, confidence, probabilities = predict_with_debug(
                        st.session_state.model,
                        image,
                        transform,
                        st.session_state.device,
                        st.session_state.class_names
                    )
                    
                    st.divider()
                    st.subheader("🎯 Final Prediction")
                    
                    # Always use checkpoint classes if available
                    if st.session_state.checkpoint_classes:
                        checkpoint_pred = st.session_state.checkpoint_classes[predicted_idx]
                        st.success(f"**Predicted: {checkpoint_pred}**")
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    else:
                        predicted_class = st.session_state.class_names[predicted_idx]
                        st.success(f"**Predicted: {predicted_class}**")
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                        st.warning("⚠️ No checkpoint classes found - using uploaded classes.txt")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please load model first")

st.divider()
st.caption("Debug Mode - Identifies prediction issues")
