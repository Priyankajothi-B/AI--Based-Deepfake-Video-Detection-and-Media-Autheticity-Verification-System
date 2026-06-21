
pythonimport sys
import os
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(_file_))
from gradcam_utils import GradCAM, preprocess, apply_heatmap

def load_model():
    # Load EfficientNet B0
    # Same as Member 1
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            model.classifier[1].in_features, 2
        )
    )

    # Load model from gradcam/models/
    model_path = os.path.join(
        os.path.dirname(_file_),
        "../models/best_model-v3.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        ),
        strict=False
    )

    model.eval()
    print("✅ Model loaded!")
    return model

# Load model once globally
model = load_model()
gradcam = GradCAM(model)

def generate_gradcam(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Get middle frame
    middle_frame_idx = total_frames // 2
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame_idx
    )

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )
    img = Image.fromarray(frame_rgb)

    # Preprocess
    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor.requires_grad = True

    # Generate CAM
    cam = gradcam.generate(img_tensor)

    # Apply heatmap
    result_frame = apply_heatmap(frame_rgb, cam)

    # Save output
    output_dir = os.path.join(
        os.path.dirname(_file_),
        "../outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "gradcam_output.jpg"
    )
    cv2.imwrite(output_path, result_frame)

    return {
        "gradcam_image_path": output_path,
        "gradcam_frame": result_frame
    }

def generate_gradcam_from_upload(uploaded_file):
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Generate Grad-CAM
    result = generate_gradcam(temp_file.name)

    # Delete temp file
    os.unlink(temp_file.name)

    return result

# Test
if _name_ == "_main_":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print("\n⏳ Generating Grad-CAM...")
        result = generate_gradcam(video_path)
        if result:
            print("✅ Grad-CAM generated!")
            print(f"Saved: {result['gradcam_image_path']}")
        else:
            print("❌ Failed!")
    else:
        print("Please provide video path!")
        print("Example: python src/gradcam_generator.py path/to/video.mp4")
