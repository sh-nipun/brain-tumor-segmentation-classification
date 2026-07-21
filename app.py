"""
Brain Tumor Multitask Network — Streamlit Deployment
======================================================
MRI ছবি আপলোড দাও -> মডেল রান হবে -> Segmentation + Classification result দেখাবে

চালানোর নিয়ম:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CLS_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = "multitask_best.pth"   # app.py এর পাশেই এই ফাইলটা রাখতে হবে


# ============================================
# Model Architecture (notebook থেকে হুবহু কপি করা — না মিললে state_dict লোড হবে না)
# ============================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class MultitaskNetwork(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Shared Encoder (ResNet34)
        resnet = tv_models.resnet34(weights=None)  # deployment এ pretrained weight লাগবে না, আমরা নিজেদের weight লোড করবো
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Segmentation Head
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ResBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ResBlock(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = ResBlock(32, 32)
        self.seg_final = nn.Conv2d(32, 1, 1)

        # Classification Head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        d0 = self.dec0(self.up0(d1))
        seg_out = self.seg_final(d0)

        cls_out = self.cls_head(e4)
        return seg_out, cls_out


# ============================================
# Model Load (একবার লোড হয়ে cache হয়ে থাকবে, বারবার লোড হবে না)
# ============================================
@st.cache_resource
def load_model():
    model = MultitaskNetwork(num_classes=4)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================
# Preprocessing + Prediction (training এর মতো হুবহু preprocessing)
# ============================================
def preprocess_image(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img_display = img.copy()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img_tensor, img_display


def predict(model, img_tensor):
    with torch.no_grad():
        seg_out, cls_out = model(img_tensor)

    seg_pred = (torch.sigmoid(seg_out) > 0.5).cpu().squeeze().numpy()
    cls_probs = torch.softmax(cls_out, dim=1).cpu().squeeze().numpy()
    cls_pred = int(np.argmax(cls_probs))
    return seg_pred, cls_pred, cls_probs


def overlay_mask(img_display, seg_pred):
    """Original image এর উপর tumor mask কে লাল রঙে overlay করে দেখায়"""
    mask_resized = cv2.resize(
        seg_pred.astype(np.uint8), (img_display.shape[1], img_display.shape[0])
    )
    overlay = img_display.copy()
    overlay[mask_resized > 0] = [255, 60, 60]
    blended = cv2.addWeighted(img_display, 0.6, overlay, 0.4, 0)
    return blended


# ============================================
# UI
# ============================================
st.title("Brain Tumor Detection — Multitask Network")


with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("**Architecture:** Shared ResNet34 Encoder + Segmentation Decoder + Classification Head")
    st.write(f"**Classes:** {', '.join(CLS_CLASSES)}")
    st.write(f"**Input Size:** {IMG_SIZE}x{IMG_SIZE}")
    st.write(f"**Device:** {DEVICE}")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    with st.spinner(""):
        try:
            model = load_model()
        except FileNotFoundError:
            st.error(
                f"❌ `{MODEL_PATH}` ফাইল পাওয়া যায়নি। "
                f"app.py যেই ফোল্ডারে আছে, সেই একই ফোল্ডারে `multitask_best.pth` কপি করো।"
            )
            st.stop()

        img_tensor, img_display = preprocess_image(pil_img)
        seg_pred, cls_pred, cls_probs = predict(model, img_tensor)
        blended = overlay_mask(img_display, seg_pred)
        tumor_pixels = int(seg_pred.sum())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input MRI")
        st.image(img_display, use_container_width=True)

    with col2:
        st.subheader("Tumor Segmentation")
        st.image(blended, use_container_width=True)
        if tumor_pixels > 0:
            st.warning(f"⚠️ Tumor Detected ({tumor_pixels} pixels)")
        else:
            st.success("✅ No Tumor Region Found")

    with col3:
        st.subheader("Classification")
        st.metric("Predicted Type", CLS_CLASSES[cls_pred].upper())
        st.metric("Confidence", f"{cls_probs[cls_pred]*100:.2f}%")

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#e74c3c' if i == cls_pred else '#3498db' for i in range(len(CLS_CLASSES))]
        ax.bar(CLS_CLASSES, cls_probs * 100, color=colors)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        for i, p in enumerate(cls_probs):
            ax.text(i, p * 100 + 1, f"{p*100:.1f}%", ha="center", fontsize=8)
        st.pyplot(fig)

    st.divider()
    st.subheader("📋 Summary")
    st.write(f"🔍 **Tumor Type:** {CLS_CLASSES[cls_pred].upper()}")
    st.write(f"📊 **Confidence:** {cls_probs[cls_pred]*100:.2f}%")
    st.write(f"🎯 **Tumor Area:** {tumor_pixels} pixels")
    st.write(f"⚠️ **Status:** {'Tumor Detected!' if tumor_pixels > 0 else 'No Tumor Found'}")
else:
    st.info("Please upload an MRI image to get started.")
