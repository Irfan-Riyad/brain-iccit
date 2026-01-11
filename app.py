import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classifier")
st.caption("HybridCNN (ResNet50 + DenseNet121)")

# =========================================================
# Session State
# =========================================================
if "model" not in st.session_state:
    st.session_state.model = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None
if "checkpoint_classes" not in st.session_state:
    st.session_state.checkpoint_classes = None
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []
if "device" not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Background Logger
# =========================================================
def log(msg):
    st.session_state.debug_logs.append(str(msg))

# =========================================================
# HybridCNN Model (Exact Architecture)
# =========================================================
class HybridCNN(nn.Module):
    def __init__(self, num_classes, hidden=1024, p=0.5):
        super().__init__()

        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        self.densenet = models.densenet121(weights=None)
        self.densenet.classifier = nn.Identity()

        feat_dim = 2048 + 1024

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        return self.head(torch.cat([f1, f2], dim=1))

# =========================================================
# Image Transform (Training-Matched)
# =========================================================
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# =========================================================
# Load Classes
# =========================================================
def load_classes(file):
    return [c.strip() for c in file.read().decode("utf-8").splitlines() if c.strip()]

# =========================================================
# Load Model (Silent, Safe)
# =========================================================
def load_model(model_file, num_classes):
    device = st.session_state.device
    checkpoint = torch.load(model_file, map_location=device)

    log(f"Checkpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        log(f"Checkpoint keys: {list(checkpoint.keys())}")

        if "classes" in checkpoint:
            st.session_state.checkpoint_classes = checkpoint["classes"]
            log(f"Checkpoint classes found: {len(checkpoint['classes'])}")

    model = HybridCNN(num_classes=num_classes)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            log("Loaded from state_dict")
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            log("Loaded from model_state_dict")
        else:
            model.load_state_dict(checkpoint)
            log("Loaded raw state dict")

    model.to(device).eval()
    log(f"Model output features: {model.head[-1].out_features}")
    return model

# =========================================================
# Prediction (Silent)
# =========================================================
@torch.no_grad()
def predict(model, image, transform):
    device = st.session_state.device

    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]

    log(f"Logits: {logits.cpu().numpy()}")
    log(f"Probabilities: {probs.cpu().numpy()}")

    conf, idx = torch.max(probs, 0)
    return idx.item(), conf.item(), probs.cpu().numpy()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Configuration")

model_file = st.sidebar.file_uploader("Model (.pth)", type=["pth"])
classes_file = st.sidebar.file_uploader("Classes (.txt)", type=["txt"])
img_size = st.sidebar.selectbox("Image Size", [224, 256], index=0)
dev_mode = st.sidebar.checkbox("Developer Mode", value=False)

if st.sidebar.button("Load Model", type="primary"):
    if model_file and classes_file:
        st.session_state.debug_logs.clear()
        class_names = load_classes(classes_file)
        model = load_model(model_file, len(class_names))
        st.session_state.model = model
        st.session_state.class_names = class_names
        st.sidebar.success("Model loaded successfully")
    else:
        st.sidebar.error("Upload model and classes file")

# =========================================================
# Main Interface
# =========================================================
uploaded_image = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_image and st.session_state.model:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("Predict", type="primary"):
        idx, conf, probs = predict(
            st.session_state.model,
            image,
            get_transform(img_size)
        )

        classes = st.session_state.checkpoint_classes or st.session_state.class_names

        st.success(f"Prediction: **{classes[idx]}**")
        st.metric("Confidence", f"{conf*100:.2f}%")

        with st.expander("Top-5 Predictions"):
            top5 = np.argsort(probs)[-5:][::-1]
            df = pd.DataFrame({
                "Class": [classes[i] for i in top5],
                "Probability (%)": [f"{probs[i]*100:.2f}" for i in top5]
            })
            st.table(df)

elif uploaded_image and not st.session_state.model:
    st.warning("Please load the model first")

# =========================================================
# Developer Debug Panel (Hidden)
# =========================================================
if dev_mode:
    with st.expander("🛠 Developer Debug Logs", expanded=False):
        for msg in st.session_state.debug_logs:
            st.code(msg)

st.caption("Inference Mode • Debugging Hidden by Default")
