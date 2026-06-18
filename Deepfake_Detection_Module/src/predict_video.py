import os
from PIL import Image
from torchvision import models, transforms
import torch

# Load model
model = models.efficientnet_b0(weights="DEFAULT")
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

frame_folder = "frames"

frame_count = 0

for file in os.listdir(frame_folder):

    if file.endswith(".jpg"):

        img_path = os.path.join(frame_folder, file)

        img = Image.open(img_path)

        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        frame_count += 1

print("Frames processed:", frame_count)