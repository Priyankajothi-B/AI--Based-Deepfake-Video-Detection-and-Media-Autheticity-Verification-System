import sys
import os
sys.path.append(os.path.dirname(__file__))

from deepfake_detector import get_deepfake_result

# Test REAL video
print("\n🎬 Testing REAL video...")
result1 = get_deepfake_result("videos/sample.mp4")
print(f"Score: {result1['deepfake_score']}%")
print(f"Verdict: {result1['verdict']}")
print(f"Risk: {result1['risk_level']}")

print("\n----------------------------")

# Test FAKE video
print("\n🎭 Testing FAKE video...")
result2 = get_deepfake_result("videos/download (1).mp4")
print(f"Score: {result2['deepfake_score']}%")
print(f"Verdict: {result2['verdict']}")
print(f"Risk: {result2['risk_level']}")
