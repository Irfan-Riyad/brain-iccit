import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import traceback

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Brain Tumor Classification & Localization")

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Session State Init
# -------------------------------------------------
for key in [
    "model",
    "class_names",
    "checkpoint_classes",
    "model_loaded",
    "last_prediction",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------------------------
# HybridCNN Model
# -------------------------------------------------
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
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        return self.head(torch.cat([f1, f2], dim=1))

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def load_classes(file):
    return [l.strip() for l in file.read().decode().splitlines() if l.strip()]

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

@torch.no_grad()
def predict(model, image, transform):
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    return probs, tensor

# -------------------------------------------------
# Grad-CAM
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.gradients = None
        self.activations = None
        layer.register_forward_hook(self._save_act)
        layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, x, class_idx):
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()

        grads = torch.abs(self.gradients[0])
        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], device=x.device)

        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = cam.cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

def overlay_heatmap(img, cam):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = np.zeros_like(img)
    heat[..., 0] = (cam * 255).astype(np.uint8)
    alpha = cam[..., None] * 0.5
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Model Configuration")

model_file = st.sidebar.file_uploader("Model (.pth)", ["pth", "pt"])
classes_file = st.sidebar.file_uploader("Classes (.txt)", ["txt"])
img_size = st.sidebar.selectbox("Input Image Size", [224, 256, 384], index=0)

# -------------------------------------------------
# Auto Model Load
# -------------------------------------------------
if model_file and classes_file:
    try:
        classes = load_classes(classes_file)
        checkpoint = torch.load(model_file, map_location=device)

        model = HybridCNN(len(classes))

        if isinstance(checkpoint, dict):
            model.load_state_dict(
                checkpoint.get("state_dict", checkpoint),
                strict=False
            )
            if "classes" in checkpoint:
                st.session_state.checkpoint_classes = checkpoint["classes"]

        model.to(device).eval()

        st.session_state.model = model
        st.session_state.class_names = classes
        st.session_state.model_loaded = True

        st.sidebar.success("Model loaded successfully")

    except Exception:
        st.sidebar.error("Model loading failed")
        st.sidebar.code(traceback.format_exc())

# -------------------------------------------------
# Main UI
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("MRI Image")
    uploaded_image = st.file_uploader(
        "Upload Brain MRI",
        ["jpg", "png", "jpeg", "bmp"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, use_container_width=True)

with col2:
    st.subheader("Prediction Results")

    if uploaded_image and st.session_state.model_loaded:
        transform = get_transform(img_size)
        probs, tensor = predict(
            st.session_state.model, image, transform
        )

        class_names = (
            st.session_state.checkpoint_classes
            or st.session_state.class_names
        )

        # ---------------- TOP-5 ----------------
        probs_np = probs.cpu().numpy()
        top5_idx = np.argsort(probs_np)[-5:][::-1]

        top5_df = pd.DataFrame({
            "Rank": range(1, 6),
            "Class": [class_names[i] for i in top5_idx],
            "Probability": probs_np[top5_idx],
            "Confidence (%)": [f"{p*100:.2f}" for p in probs_np[top5_idx]]
        })

        pred_idx = top5_idx[0]

        st.metric("Predicted Class", class_names[pred_idx])
        st.metric("Confidence", f"{probs_np[pred_idx]*100:.2f}%")

        st.subheader("Top-5 Predictions")
        st.table(top5_df)

        # Save safely for Grad-CAM
        st.session_state.last_prediction = {
            "tensor": tensor,
            "predicted_idx": pred_idx,
            "image": image
        }

# -------------------------------------------------
# Grad-CAM
# -------------------------------------------------
if isinstance(st.session_state.last_prediction, dict):
    st.divider()
    st.subheader("Tumor Localization (Grad-CAM)")

    pred = st.session_state.last_prediction
    tensor = pred["tensor"]
    idx = pred["predicted_idx"]
    image = pred["image"]

    cam_engine = GradCAM(
        st.session_state.model,
        st.session_state.model.resnet.layer4[-1].conv3
    )

    cam = cam_engine.generate(tensor, idx)
    img_np = np.array(image.resize((img_size, img_size)))
    cam_img = overlay_heatmap(img_np, cam)

    c1, c2 = st.columns(2)
    c1.image(img_np, caption="Original MRI", use_container_width=True)
    c2.image(cam_img, caption="Grad-CAM Tumor Regions", use_container_width=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption(f"Device: {device} | HybridCNN (ResNet50 + DenseNet121)")
