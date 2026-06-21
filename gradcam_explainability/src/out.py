
video_path = list(video.keys())[0]

path, cam = generate_gradcam(video_path)

print("\n🧠 EXPLANATION:\n")
exp = explain_heatmap(cam)

for line in exp:
    print(line)

from IPython.display import Image, display

print("\n🖼️ OUTPUT IMAGE:")
display(Image(path))
