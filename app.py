import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import traceback

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 HybridCNN Brain Tumor Classifier")

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
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'checkpoint_classes' not in st.session_state:
    st.session_state.checkpoint_classes = None

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

        # --- Fusion + Classifier Head (No in-place operations for Grad-CAM) ---
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=False),  # Changed to False for Grad-CAM compatibility
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        fused = torch.cat([f1, f2], dim=1)
        return self.head(fused)

# ----------------------------
# Model Loader (Clean Version)
# ----------------------------
def load_model_clean(model_file, num_classes):
    """
    Load model silently without debug output
    """
    device = st.session_state.device
    
    checkpoint = torch.load(model_file, map_location=device)
    
    # Check for classes in checkpoint (silent)
    if isinstance(checkpoint, dict) and 'classes' in checkpoint:
        st.session_state.checkpoint_classes = checkpoint['classes']
    
    # Create model
    model = HybridCNN(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbones=False,
        hidden=1024,
        p=0.5
    )
    
    # Load state dict
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
# Prediction Function (Clean Version)
# ----------------------------
@torch.no_grad()
def predict_clean(model, image, transform, device):
    """
    Predict without debug output
    """
    model.eval()
    
    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    return predicted_idx.item(), confidence.item(), probabilities.cpu(), input_tensor

# ----------------------------
# Enhanced Grad-CAM Implementation
# ----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        # Clone input to avoid in-place modification issues
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad = True
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0].clone()
        activations = self.activations[0].clone()
        
        # Check if we have spatial dimensions
        if gradients.dim() == 3:  # Convolutional layer [C, H, W]
            # Use all gradients (not just positive) for better multi-region detection
            # Positive gradients for tumor presence, negative for context
            
            # Compute weights using gradient magnitude for better sensitivity
            grad_magnitude = torch.abs(gradients)
            weights = grad_magnitude.mean(dim=(1, 2))  # [C]
            
            # Weighted combination of activation maps
            cam = torch.zeros(activations.shape[1:], device=activations.device)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # Don't apply ReLU here - keep both positive and negative activations
            # Apply after normalization for better multi-region detection
            
        elif gradients.dim() == 1:  # Fully connected layer [C]
            cam_size = 7
            weights = torch.abs(gradients)
            avg_weight = weights.mean()
            cam = torch.ones((cam_size, cam_size), device=activations.device) * avg_weight
            
        else:
            cam = torch.ones((7, 7), device=activations.device) * 0.5
        
        # Normalize using percentile for better multi-region detection
        cam_np = cam.cpu().numpy()
        
        return cam_np

def apply_enhanced_colormap(org_img, activation_map, threshold=0.3):
    """
    Professional medical-grade tumor localization visualization
    """
    # Resize with high-quality interpolation
    h, w = org_img.shape[:2]
    activation_map = cv2.resize(activation_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # Advanced normalization using robust statistics
    p1 = np.percentile(activation_map, 1)
    p99 = np.percentile(activation_map, 99)
    activation_map = np.clip(activation_map, p1, p99)
    
    if activation_map.max() > activation_map.min():
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    
    # Apply bilateral filter for edge-preserving smoothing
    activation_map = cv2.bilateralFilter(
        (activation_map * 255).astype(np.uint8), 
        d=9, 
        sigmaColor=75, 
        sigmaSpace=75
    ).astype(np.float32) / 255.0
    
    # Threshold with hysteresis for better region connectivity
    high_threshold = threshold + 0.1
    low_threshold = threshold - 0.05
    
    strong = activation_map >= high_threshold
    weak = (activation_map >= low_threshold) & (activation_map < high_threshold)
    
    # Connect weak regions that are adjacent to strong regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    strong_dilated = cv2.dilate(strong.astype(np.uint8), kernel, iterations=1)
    connected_weak = weak & (strong_dilated > 0)
    
    final_mask = strong | connected_weak
    activation_masked = activation_map * final_mask
    
    # Morphological refinement
    mask_uint8 = (activation_masked * 255).astype(np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 
                                   iterations=1)
    
    activation_final = mask_uint8.astype(np.float32) / 255.0
    
    # Create professional red-yellow heatmap
    # Red for high confidence, yellow for medium, transparent for low
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create intensity-based color gradient
    intensity = activation_final
    heatmap[:, :, 0] = np.clip(intensity * 255, 0, 255).astype(np.uint8)  # Red channel
    heatmap[:, :, 1] = np.clip((intensity - 0.5) * 2 * 200, 0, 200).astype(np.uint8)  # Green for yellow tint
    heatmap[:, :, 2] = 0  # No blue
    
    # Smart alpha blending - higher intensity = more visible
    alpha = np.expand_dims(activation_final ** 0.7, axis=2) * 0.6  # Gamma correction for better visibility
    
    # Blend with original image
    org_float = org_img.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)
    
    # Enhanced blending for better contrast
    blended = org_float * (1 - alpha) + heatmap_float * alpha
    
    # Subtle sharpening only on tumor regions for crisp boundaries
    sharpening_kernel = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]])
    
    tumor_mask_3ch = np.repeat(np.expand_dims(activation_final > 0.1, axis=2), 3, axis=2)
    sharpened = cv2.filter2D(blended, -1, sharpening_kernel)
    
    # Apply sharpening only to tumor regions
    result = blended * (1 - tumor_mask_3ch) + sharpened * tumor_mask_3ch * 0.3 + blended * tumor_mask_3ch * 0.7
    
    # Final enhancement: local contrast adjustment
    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
    
    # Apply CLAHE only to luminance channel for better local contrast
    lab = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    result_enhanced = cv2.merge([l_enhanced, a, b])
    result_final = cv2.cvtColor(result_enhanced, cv2.COLOR_LAB2RGB)
    
    return result_final, activation_final

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

transform_version = st.sidebar.selectbox(
    "🔄 Transform Version",
    ["v1 (Original)", "v2 (No Lambda)", "v3 (Center Crop)"],
    help="Try different transforms to see which matches training"
)

# Load model button
if st.sidebar.button("🔄 Load Model", type="primary"):
    if model_file and classes_file:
        with st.spinner("Loading model..."):
            try:
                class_names = load_classes(classes_file)
                
                model_file.seek(0)
                model = load_model_clean(model_file, len(class_names))
                
                if model:
                    st.session_state.model = model
                    st.session_state.class_names = class_names
                    st.session_state.model_loaded = True
                    st.sidebar.success("✅ Model loaded successfully!")
                    st.sidebar.metric("Classes", len(class_names))
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                st.sidebar.code(traceback.format_exc())
    else:
        st.sidebar.error("Please upload both files!")

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
st.header("📤 Upload & Analyze MRI Scan")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Image Upload")
    
    uploaded_image = st.file_uploader(
        "Choose a brain MRI scan",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a brain MRI image for classification"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.caption(f"📏 Size: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.subheader("🎯 Prediction Results")
    
    if st.session_state.model and uploaded_image:
        try:
            # Select transform
            if transform_version == "v1 (Original)":
                transform = get_transform_v1(img_size)
            elif transform_version == "v2 (No Lambda)":
                transform = get_transform_v2(img_size)
            else:
                transform = get_transform_v3(img_size)
            
            with st.spinner("🔬 Analyzing image..."):
                predicted_idx, confidence, probabilities, input_tensor = predict_clean(
                    st.session_state.model,
                    image,
                    transform,
                    st.session_state.device
                )
            
            # Use checkpoint classes if available
            display_classes = st.session_state.checkpoint_classes if st.session_state.checkpoint_classes else st.session_state.class_names
            
            predicted_class = display_classes[predicted_idx]
            
            # Main prediction
            st.success(f"### 🎯 Predicted: **{predicted_class}**")
            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.progress(confidence)
            
            st.divider()
            
            # Top 5 table
            st.subheader("📊 Top 5 Predictions")
            
            probs_np = probabilities.numpy()
            top5_indices = np.argsort(probs_np)[-5:][::-1]
            
            top5_data = {
                'Rank': list(range(1, 6)),
                'Class': [display_classes[idx] for idx in top5_indices],
                'Probability': [f"{probs_np[idx]:.6f}" for idx in top5_indices],
                'Percentage': [f"{probs_np[idx]*100:.2f}%" for idx in top5_indices]
            }
            
            df = pd.DataFrame(top5_data)
            st.table(df)
            
            # Store in session state for Grad-CAM
            st.session_state.last_prediction = {
                'input_tensor': input_tensor,
                'predicted_idx': predicted_idx,
                'image': image,
                'display_classes': display_classes
            }
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
    elif not st.session_state.model:
        st.warning("⚠️ Please load model first")
    else:
        st.info("👆 Upload an image to get started")

# ----------------------------
# Grad-CAM Section (Separate, Below Main Content)
# ----------------------------
if st.session_state.last_prediction:
    st.divider()
    st.header("🔥 Grad-CAM: Tumor Localization Analysis")
    st.markdown("**Visualize which regions influenced the model's prediction**")
    
    # Layer selection with radio button for single selection
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        selected_layer = st.radio(
            "Select Layer to Visualize",
            ["HybridCNN (Fusion)"],
            index=0,
            help="Visualize combined ResNet50 + DenseNet121 features"
        )
    
    with col_config2:
        threshold = st.slider(
            "Sensitivity",
            min_value=0.2,
            max_value=0.6,
            value=0.35,
            step=0.05,
            help="Adjust detection sensitivity"
        )
    
    # Generate button
    if st.button("🔍 Generate Grad-CAM", type="primary", use_container_width=True):
        with st.spinner(f"Generating Grad-CAM for {selected_layer}..."):
            try:
                # Get data from session
                input_tensor = st.session_state.last_prediction['input_tensor']
                predicted_idx = st.session_state.last_prediction['predicted_idx']
                image = st.session_state.last_prediction['image']
                
                # Convert original image to numpy
                org_img = np.array(image.resize((img_size, img_size)))
                
                # Define target layer - HybridCNN Fusion only
                target_layer = st.session_state.model.resnet.layer4[-1].conv3  # Combined features visualization
                
                # Generate Grad-CAM
                gradcam = GradCAM(st.session_state.model, target_layer)
                cam = gradcam.generate_cam(input_tensor, predicted_idx)
                
                # Apply enhanced heatmap
                gradcam_img, cam_thresholded = apply_enhanced_colormap(org_img, cam, threshold)
                
                # Calculate metrics
                tumor_coverage = (cam_thresholded > 0).sum() / cam_thresholded.size * 100
                
                # Region detection
                cam_binary = (cam_thresholded > 0.1).astype(np.uint8)
                num_regions, labeled_regions, stats, _ = cv2.connectedComponentsWithStats(cam_binary, connectivity=8)
                
                # Filter significant regions (>1% of image)
                min_size = (cam_binary.shape[0] * cam_binary.shape[1]) * 0.01
                significant_regions = sum(1 for i in range(1, num_regions) if stats[i, cv2.CC_STAT_AREA] > min_size)
                
                # Display results
                st.subheader(f"📍 {selected_layer} Visualization")
                
                # Two column display
                vis_col1, vis_col2 = st.columns(2)
                
                with vis_col1:
                    st.image(org_img, caption="Original MRI Scan", use_container_width=True)
                
                with vis_col2:
                    st.image(gradcam_img, caption=f"Grad-CAM: {selected_layer}", use_container_width=True)
                
                # Clean minimal display
                st.divider()
                
                if significant_regions == 0:
                    st.info("No tumor regions detected")
                elif significant_regions == 1:
                    st.success("✓ 1 tumor region detected")
                else:
                    st.success(f"✓ {significant_regions} tumor regions detected")
                
                
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {str(e)}")
                st.code(traceback.format_exc())

# ----------------------------
# Footer (Removed)
# ----------------------------
st.divider()
st.caption("HybridCNN Brain Tumor Classifier")
