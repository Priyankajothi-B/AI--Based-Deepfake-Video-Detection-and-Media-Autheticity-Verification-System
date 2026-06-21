import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image as PILImage
import torch.nn as nn
from torchvision import models

# Path setup
sys.path.append("/content/gradcam/src")

from gradcam_utils import GradCAM, preprocess, apply_heatmap


# =========================
# LOAD MODEL
# =========================
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, 2)
    )

    model_path = "/content/gradcam/models/best_model-v3.pt"

    model.load_state_dict(
        torch.load(model_path, map_location="cpu"),
        strict=False
    )

    model.eval()
    print("✅ Model loaded!")
    return model


model = load_model()
gradcam = GradCAM(model)


# =========================
# GRAD-CAM GENERATION
# =========================
def generate_gradcam(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = PILImage.fromarray(frame_rgb)

    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor.requires_grad_(True)

    # Grad-CAM
    cam = gradcam.generate(img_tensor)

    result = apply_heatmap(frame_rgb, cam)

    output_path = "/content/gradcam/outputs/gradcam_output.jpg"
    cv2.imwrite(output_path, result)

    return output_path, cam


# =========================
# EXPLAINABILITY (DETAILED)
# =========================
def explain_heatmap(cam):

    h, w = cam.shape
    cam = cam / (cam.max() + 1e-8)

    eye_region = cam[0:int(h*0.4), int(w*0.2):int(w*0.8)]
    mouth_region = cam[int(h*0.6):h, int(w*0.2):int(w*0.8)]
    face_region = cam

    eye_score = float(np.mean(eye_region))
    mouth_score = float(np.mean(mouth_region))
    face_score = float(np.mean(face_region))

    explanation = []

    explanation.append("🧠 Grad-CAM Analysis Report:")

    # Eye analysis
    if eye_score > 0.6:
        explanation.append(
            "👁️ Eye Region: Strong attention detected. This suggests unnatural eye movement, "
            "inconsistent blinking, or identity manipulation often seen in deepfakes."
        )
    else:
        explanation.append(
            "👁️ Eye Region: Normal attention. No strong signs of eye-related manipulation."
        )

    # Mouth analysis
    if mouth_score > 0.6:
        explanation.append(
            "👄 Mouth Region: High attention detected. This may indicate lip-sync mismatch, "
            "speech misalignment, or facial animation artifacts."
        )
    else:
        explanation.append(
            "👄 Mouth Region: Normal attention. Lip movement appears consistent."
        )

    # Face analysis
    if face_score > 0.7:
        explanation.append(
            "🧑 Face Region: Strong overall activation. Possible synthetic face generation detected."
        )
    elif face_score < 0.3:
        explanation.append(
            "🧑 Face Region: Low activation. Likely a REAL and natural video."
        )
    else:
        explanation.append(
            "🧑 Face Region: Moderate activation. Some suspicious but not conclusive patterns."
        )

    explanation.append(
        "📌 Final Insight: Grad-CAM highlights regions influencing the model’s decision. "
        "Higher activation in eyes/mouth often correlates with deepfake artifacts."
    )

    return explanation
