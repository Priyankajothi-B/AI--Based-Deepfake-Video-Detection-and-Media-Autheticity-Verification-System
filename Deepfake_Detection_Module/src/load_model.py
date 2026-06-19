import torch
import torch.nn as nn
from torchvision import models

def load_model():
    # EfficientNet B0 — same as their repo
    model = models.efficientnet_b0(weights=None)
    
    # 2-class classifier with dropout 0.4
    # (exactly as their README says)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            model.classifier[1].in_features, 2
        )
    )
    
    # Load their pretrained weights
    model.load_state_dict(
        torch.load(
            "models/best_model-v3.pt",
            map_location="cpu"
        ),
        strict=False
    )
    
    model.eval()
    print("✅ Model loaded successfully!")
    return model
