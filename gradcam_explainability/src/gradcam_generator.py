import cv2
import torch
import tempfile
import os

from PIL import Image
import torch.nn as nn
from torchvision import models

from gradcam_utils import GradCAM, preprocess, apply_heatmap


def load_model(model_path):

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            model.classifier[1].in_features,
            2
        )
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        ),
        strict=False
    )

    model.eval()

    return model


def generate_gradcam_from_upload(
    uploaded_file,
    model_path
):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(
        uploaded_file.read()
    )

    temp_file.close()

    model = load_model(model_path)

    gradcam = GradCAM(model)

    cap = cv2.VideoCapture(
        temp_file.name
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    middle = total_frames // 2

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle
    )

    ret, frame = cap.read()

    cap.release()

    os.unlink(temp_file.name)

    if not ret:
        return {
            "gradcam_frame": None
        }

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(
        frame_rgb
    )

    img_tensor = preprocess(
        img
    ).unsqueeze(0)

    cam = gradcam.generate(
        img_tensor
    )

    result = apply_heatmap(
        frame_rgb,
        cam
    )

    return {
        "gradcam_frame": result
    }