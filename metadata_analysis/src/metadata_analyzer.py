import os
import tempfile

from extract_metadata import extract_metadata
from authenticity_check import check_authenticity


def analyze_video(video_path):

    metadata = extract_metadata(video_path)

    if metadata is None:
        return None

    authenticity = check_authenticity(metadata)

    result = {
        "file_name": metadata["file_name"],
        "file_size_mb": metadata["file_size_mb"],
        "frame_count": metadata["frame_count"],
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "duration_seconds": metadata["duration_seconds"],
        "authenticity_score": authenticity["authenticity_score"],
        "authenticity_verdict": authenticity["verdict"],
        "authenticity_status": authenticity["status"],
        "issues": authenticity["issues"]
    }

    return result


# 🔥 THIS IS USED BY MEMBER 4 (IMPORTANT)
def analyze_uploaded_video(uploaded_file):

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    result = analyze_video(temp_file.name)

    os.unlink(temp_file.name)

    return result
