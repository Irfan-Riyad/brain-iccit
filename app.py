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
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Brain Tumor Classification & Localization")

# -------------------------------------------------
# Session State
# -------------------------------------------------
for key in [
    "model", "class_names", "model_loaded",
    "checkpoint_classes", "last_prediction"
]:
    if key not in st.session_state:
        st.session_state[key] = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Model Definition
# -------------------------------------------------
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        self.densenet = models.densenet121(weights=None)
        self.densenet.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 1024, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        return self.classifier(torch.cat([f1, f2], dim=1))

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
        )
    ])

@torch.no_grad()
def predict(model, image, transform):
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return idx, probs[idx].item(), probs.cpu(), tensor

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

        g = torch.abs(self.gradients[0])
        w = g.mean(dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], device=x.device)

        for i, wi in enumerate(w):
            cam += wi * self.activations[0, i]

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
img_size = st.sidebar.selectbox("Input Size", [224, 256, 384])

# -------------------------------------------------
# Auto Model Loading
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

    except Exception as e:
        st.sidebar.error("Model loading failed")
        st.sidebar.code(traceback.format_exc())

# -------------------------------------------------
# Main Layout
# -------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("MRI Scan")
    uploaded_image = st.file_uploader(
        "Upload MRI Image",
        ["jpg", "png", "jpeg", "bmp"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, use_container_width=True)

with right:
    st.subheader("Prediction")

    if uploaded_image and st.session_state.model_loaded:
        transform = get_transform(img_size)
        idx, conf, probs, tensor = predict(
            st.session_state.model, image, transform
        )

        names = (
            st.session_state.checkpoint_classes
            or st.session_state.class_names
        )

        st.metric("Predicted Class", names[idx])
        st.metric("Confidence", f"{conf*100:.2f}%")

        df = pd.DataFrame({
            "Class": names,
            "Probability": probs.numpy()
        }).sort_values("Probability", ascending=False).head(5)

        st.table(df)

        st.session_state.last_prediction = (tensor, idx, image)

# -------------------------------------------------
# Grad-CAM (Auto)
# -------------------------------------------------
if st.session_state.last_prediction:
    st.divider()
    st.subheader("Tumor Localization (Grad-CAM)")

    tensor, idx, image = st.session_state.last_prediction
    cam_gen = GradCAM(
        st.session_state.model,
        st.session_state.model.resnet.layer4[-1].conv3
    )

    cam = cam_gen.generate(tensor, idx)
    img_np = np.array(image.resize((img_size, img_size)))
    cam_img = overlay_heatmap(img_np, cam)

    c1, c2 = st.columns(2)
    c1.image(img_np, caption="Original MRI", use_container_width=True)
    c2.image(cam_img, caption="Tumor Localization", use_container_width=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption(f"Device: {device} | HybridCNN (ResNet50 + DenseNet121)")
