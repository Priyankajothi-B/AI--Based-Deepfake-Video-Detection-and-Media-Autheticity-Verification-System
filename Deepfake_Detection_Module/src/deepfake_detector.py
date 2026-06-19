import sys
import os
sys.path.append(os.path.dirname(__file__))

from load_model import load_model
from predict_video import predict_video
from predict_video import predict_uploaded_video

# Load model once
model = load_model()

def get_deepfake_result(video_path):
    score = predict_video(video_path, model)
    percentage = round(score * 100, 2)

    if percentage >= 70:
        verdict = "FAKE"
        risk = "HIGH RISK"
    elif percentage >= 40:
        verdict = "SUSPICIOUS"
        risk = "MEDIUM RISK"
    else:
        verdict = "REAL"
        risk = "LOW RISK"

    return {
        "deepfake_score": percentage,
        "verdict": verdict,
        "risk_level": risk
    }

def get_deepfake_result_from_upload(uploaded_file):
    score = predict_uploaded_video(
        uploaded_file, model
    )
    percentage = round(score * 100, 2)

    if percentage >= 70:
        verdict = "FAKE"
        risk = "HIGH RISK"
    elif percentage >= 40:
        verdict = "SUSPICIOUS"
        risk = "MEDIUM RISK"
    else:
        verdict = "REAL"
        risk = "LOW RISK"

    return {
        "deepfake_score": percentage,
        "verdict": verdict,
        "risk_level": risk
    }
