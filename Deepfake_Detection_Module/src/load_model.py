import torch
import torch.nn as nn
from torchvision import models
import os


def load_model():

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            model.classifier[1].in_features,
            2
        )
    )

    current_dir = os.path.dirname(__file__)

    model_path = os.path.join(
        current_dir,
        "..",
        "models",
        "best_model-v3.pt"
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