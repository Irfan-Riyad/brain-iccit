import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Image Classification App", layout="centered")
st.title("PyTorch Image Classification")
st.write("Upload a model, define classes, and predict an image.")

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model Loader
# ----------------------------
def load_model(model_path, num_classes):
    """
    Modify this function if you used a custom architecture.
    Default: ResNet50 backbone
    """
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device)
    model.eval()
    return model

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image).unsqueeze(0)
    return image

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Model Configuration")

model_path = st.sidebar.text_input("Path to .pth model file")

class_names_input = st.sidebar.text_area(
    "Class Names (comma-separated)",
    placeholder="e.g. glioma, meningioma, pituitary, normal"
)

img_size = st.sidebar.number_input(
    "Input Image Size",
    min_value=64,
    max_value=512,
    value=224,
    step=32
)

# ----------------------------
# Image Upload
# ----------------------------
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    if not model_path:
        st.error("Please provide a model path.")
    elif not class_names_input:
        st.error("Please provide class names.")
    elif uploaded_image is None:
        st.error("Please upload an image.")
    else:
        class_names = [c.strip() for c in class_names_input.split(",")]
        num_classes = len(class_names)

        try:
            # Load model
            model = load_model(model_path, num_classes)

            # Load & preprocess image
            image = Image.open(uploaded_image).convert("RGB")
            input_tensor = preprocess_image(image, img_size).to(device)

            # Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

            predicted_label = class_names[predicted_class.item()]
            confidence = confidence.item() * 100

            # Display
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

        except Exception as e:
            st.error(f"Error: {e}")
