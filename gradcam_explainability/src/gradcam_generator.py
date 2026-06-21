import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models

# Add src folder to Python path
sys.path.append("/content/gradcam/src")

from gradcam_utils import GradCAM, preprocess, apply_heatmap


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


def generate_gradcam(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame_rgb)

    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor.requires_grad_(True)

    cam = gradcam.generate(img_tensor)

    result_frame = apply_heatmap(frame_rgb, cam)

    output_path = "/content/gradcam/outputs/gradcam_output.jpg"

    cv2.imwrite(output_path, result_frame)

    return output_path
