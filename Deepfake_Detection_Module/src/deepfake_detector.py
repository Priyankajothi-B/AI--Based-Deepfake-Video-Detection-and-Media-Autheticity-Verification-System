import os
from PIL import Image
from torchvision import models, transforms
import torch

# Load EfficientNet-B0
model = models.efficientnet_b0(weights="DEFAULT")
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

frame_folder = "frames"

scores = []

for file in os.listdir(frame_folder):

    if file.endswith(".jpg"):

        img_path = os.path.join(frame_folder, file)

        img = Image.open(img_path)

        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        confidence = torch.max(torch.softmax(output, dim=1)).item()

        scores.append(confidence)

avg_score = sum(scores) / len(scores)

print("Average Confidence Score:", round(avg_score, 4))