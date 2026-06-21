import cv2
import os

def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps > 0:
        duration = round(frame_count / fps, 2)
    else:
        duration = 0

    file_size = round(os.path.getsize(video_path) / (1024 * 1024), 2)

    metadata = {
        "file_name": os.path.basename(video_path),
        "file_size_mb": file_size,
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_seconds": duration
    }

    cap.release()
    return metadata
