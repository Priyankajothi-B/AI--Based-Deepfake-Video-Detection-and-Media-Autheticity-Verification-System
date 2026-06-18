from torchvision import models, transforms
from PIL import Image
import torch

# Load model
model = models.efficientnet_b0(weights="DEFAULT")
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load one frame
img = Image.open("frames/frame_0.jpg")

# Preprocess
img_tensor = preprocess(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(img_tensor)

print("Prediction successful!")
print("Output shape:", output.shape)