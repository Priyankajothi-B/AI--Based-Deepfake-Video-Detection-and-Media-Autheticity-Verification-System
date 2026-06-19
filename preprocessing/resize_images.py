import cv2
import os

input_folder = "dataset/frames/real"

for file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, (128, 128))
        cv2.imwrite(img_path, img)

print("Resizing completed")
