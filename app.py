import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Brain Tumor Classification System",
    layout="wide"
)

st.title("🧠 Brain Tumor Classification System")
st.caption("HybridCNN — ResNet50 + DenseNet121 with Grad-CAM")

# ============================
# Session State
# ============================
if "model" not in st.session_state:
    st.session_state.model = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "device" not in st.session_state:
    st.session_state.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

# ============================
# HybridCNN
# ============================
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        self.densenet = models.densenet121(weights=None)
        self.densenet.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(2048 + 1024, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        return self.head(torch.cat([f1, f2], dim=1))

# ============================
# Load Model
# ============================
def load_model(model_file, num_classes):
    checkpoint = torch.load(model_file, map_location=st.session_state.device)
    model = HybridCNN(num_classes)

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    else:
        model = checkpoint

    model.to(st.session_state.device)
    model.eval()
    return model

# ============================
# Transform
# ============================
def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# ============================
# Prediction
# ============================
@torch.no_grad()
def predict(model, image, transform):
    x = transform(image).unsqueeze(0).to(st.session_state.device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    return probs, x

# ============================
# Grad-CAM
# ============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None
        self.model = model

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, m, i, o):
        self.activations = o.detach()

    def _save_gradient(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, x, class_idx):
        x = x.clone().requires_grad_(True)
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.abs(self.gradients[0]).mean(dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], device=weights.device)

        for i, w in enumerate(weights):
            cam += w * self.activations[0][i]

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(img, cam, threshold):
    h, w = img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    cam = np.where(cam > threshold, cam, 0)

    heatmap = np.zeros_like(img)
    heatmap[:, :, 0] = (cam * 255).astype(np.uint8)

    alpha = np.expand_dims(cam * 0.5, axis=2)
    return np.clip(img * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)

# ============================
# Sidebar
# ============================
st.sidebar.header("Configuration")

model_file = st.sidebar.file_uploader("Model (.pth)", type=["pth", "pt"])
classes_file = st.sidebar.file_uploader("Classes (.txt)", type=["txt"])
img_size = st.sidebar.number_input("Image Size", 128, 512, 224, 32)

st.sidebar.divider()
st.sidebar.subheader("Grad-CAM Sensitivity")

cam_threshold = st.sidebar.slider(
    "Activation Threshold",
    min_value=0.2,
    max_value=0.6,
    value=0.35,
    step=0.05
)

# ============================
# Auto Load Model
# ============================
if model_file and classes_file and not st.session_state.model_loaded:
    classes = [c.strip() for c in classes_file.read().decode().splitlines() if c.strip()]
    st.session_state.class_names = classes
    st.session_state.model = load_model(model_file, len(classes))
    st.session_state.model_loaded = True

# ============================
# Main Content
# ============================
uploaded_image = st.file_uploader(
    "Upload Brain MRI Image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_image and st.session_state.model_loaded:
    image = Image.open(uploaded_image).convert("RGB")

    probs, input_tensor = predict(
        st.session_state.model,
        image,
        get_transform(img_size)
    )

    probs_np = probs.cpu().numpy()
    top5_idx = np.argsort(probs_np)[-5:][::-1]

    pred_idx = top5_idx[0]
    pred_class = st.session_state.class_names[pred_idx]
    confidence = probs_np[pred_idx]

    # ============================
    # TOP SECTION (UNCHANGED)
    # ============================
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Input MRI")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"### **{pred_class}**")
        st.progress(float(confidence))
        st.caption(f"Confidence: {confidence * 100:.2f}%")

        df = pd.DataFrame({
            "Class": [st.session_state.class_names[i] for i in top5_idx],
            "Probability (%)": [f"{probs_np[i] * 100:.2f}" for i in top5_idx]
        })
        st.table(df)

    # ============================
    # GRAD-CAM BELOW
    # ============================
    st.divider()
    st.subheader("Tumor Localization (Grad-CAM)")

    target_layer = st.session_state.model.resnet.layer4[-1].conv3
    cam = GradCAM(st.session_state.model, target_layer).generate(
        input_tensor, pred_idx
    )

    img_np = np.array(image.resize((img_size, img_size)))
    cam_img = overlay_cam(img_np, cam, cam_threshold)

    gc_col1, gc_col2 = st.columns(2)

    with gc_col1:
        st.image(img_np, caption="Original MRI", use_container_width=True)

    with gc_col2:
        st.image(cam_img, caption="Grad-CAM Overlay", use_container_width=True)
