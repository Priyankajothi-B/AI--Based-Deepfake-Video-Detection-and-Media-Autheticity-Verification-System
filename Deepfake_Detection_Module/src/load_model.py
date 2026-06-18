from torchvision import models

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights="DEFAULT")

print("EfficientNet-B0 loaded successfully!")