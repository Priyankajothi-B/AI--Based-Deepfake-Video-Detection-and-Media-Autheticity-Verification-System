def check_authenticity(metadata):

    issues = []
    score = 100

    if metadata["fps"] <= 0:
        issues.append("Invalid FPS detected")
        score -= 30
    elif metadata["fps"] < 10:
        issues.append("Unusually low FPS")
        score -= 15

    if metadata["width"] <= 0 or metadata["height"] <= 0:
        issues.append("Invalid resolution")
        score -= 30
    elif metadata["width"] < 100 or metadata["height"] < 100:
        issues.append("Very low resolution")
        score -= 15

    if metadata["frame_count"] <= 0:
        issues.append("No frames detected")
        score -= 30

    if metadata["file_size_mb"] < 0.1:
        issues.append("Suspiciously small file size")
        score -= 20

    if metadata["duration_seconds"] < 1:
        issues.append("Very short video duration")
        score -= 10

    score = max(0, score)

    if score >= 70:
        verdict = "AUTHENTIC"
        status = "No tampering detected"
    elif score >= 40:
        verdict = "SUSPICIOUS"
        status = "Possible tampering detected"
    else:
        verdict = "TAMPERED"
        status = "Tampering signs found"

    return {
        "authenticity_score": score,
        "verdict": verdict,
        "status": status,
        "issues": issues
    }
