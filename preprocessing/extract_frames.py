import cv2
import os

video_folder = "dataset/videos/real"
output_folder = "dataset/frames/real"

os.makedirs(output_folder, exist_ok=True)

for video in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_name = f"{video}_frame{count}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
        count += 1

    cap.release()

print("Frames extracted successfully")
