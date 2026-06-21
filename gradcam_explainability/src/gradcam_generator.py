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
# GRAD-CAM + CONFIDENCE
# =========================
def generate_gradcam(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None, None, None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = PILImage.fromarray(frame_rgb)

    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor.requires_grad_(True)

    # Model prediction
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]

    real_conf = float(probs[0])
    fake_conf = float(probs[1])

    # Grad-CAM
    cam = gradcam.generate(img_tensor)

    result = apply_heatmap(frame_rgb, cam)

    output_path = "/content/gradcam/outputs/gradcam_output.jpg"
    cv2.imwrite(output_path, result)

    return output_path, cam, real_conf, fake_conf


# =========================
# EXPLAINABILITY (CONSISTENT WITH MODEL)
# =========================
def explain_heatmap(cam, real_conf, fake_conf):

    h, w = cam.shape
    cam = cam / (cam.max() + 1e-8)

    eye_region = cam[0:int(h*0.4), int(w*0.2):int(w*0.8)]
    mouth_region = cam[int(h*0.6):h, int(w*0.2):int(w*0.8)]
    face_region = cam

    eye_score = float(np.mean(eye_region))
    mouth_score = float(np.mean(mouth_region))
    face_score = float(np.mean(face_region))

    explanation = []

    # =========================
    # FINAL DECISION FIRST
    # =========================
    if fake_conf > real_conf:
        explanation.append("🚨 FINAL PREDICTION: FAKE VIDEO DETECTED")
    else:
        explanation.append("✅ FINAL PREDICTION: REAL VIDEO DETECTED")

    explanation.append("\n🧠 WHY THE MODEL DECIDED THIS:")

    # =========================
    # EYES
    # =========================
    if eye_score > 0.5:
        explanation.append(
            "👁️ Eye Region strongly influenced the decision. "
            "Model detected irregular eye patterns such as unnatural blinking, gaze shifts, "
            "or identity inconsistency often found in deepfakes."
        )
    else:
        explanation.append(
            "👁️ Eye Region had low influence. Eye movement appears consistent with natural behavior."
        )

    # =========================
    # MOUTH
    # =========================
    if mouth_score > 0.5:
        explanation.append(
            "👄 Mouth Region influenced prediction. Possible lip-sync mismatch or speech-to-face misalignment detected."
        )
    else:
        explanation.append(
            "👄 Mouth Region shows normal behavior with no strong manipulation signals."
        )

    # =========================
    # FACE
    # =========================
    if face_score > 0.6:
        explanation.append(
            "🧑 Full face structure strongly influenced decision. This indicates synthetic facial generation artifacts."
        )
    else:
        explanation.append(
            "🧑 Face structure contribution is moderate or low."
        )

    # =========================
    # FINAL NOTE
    # =========================
    explanation.append(
        "\n📌 NOTE: Grad-CAM shows WHICH regions influenced the model decision, "
        "not whether those regions are fake independently."
    )

    return explanation


# =========================
# RUN + DISPLAY OUTPUT
# =========================
from IPython.display import Image, display

video_path = list(video.keys())[0]

path, cam, real_conf, fake_conf = generate_gradcam(video_path)

print("\n📊 CONFIDENCE SCORES:")
print("Real:", real_conf)
print("Fake:", fake_conf)

print("\n🧠 EXPLANATION:\n")
exp = explain_heatmap(cam, real_conf, fake_conf)

for line in exp:
    print(line)

print("\n🖼️ OUTPUT IMAGE:")
display(Image(path))
