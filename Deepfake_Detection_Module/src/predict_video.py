import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile
import os

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    frame_indices = np.linspace(
        0, total_frames - 1, 10, dtype=int
    )
    scores = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )
        img = Image.fromarray(frame_rgb)
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            fake_score = prob[0][1].item()
            scores.append(fake_score)
    cap.release()
    if scores:
        return sum(scores) / len(scores)
    return 0.0

def predict_uploaded_video(uploaded_file, model):
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    temp_file.write(uploaded_file.read())
    temp_file.close()
    score = predict_video(temp_file.name, model)
    os.unlink(temp_file.name)
    return score
