import torch
import numpy as np
import cv2
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.features[-1]

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor):
        output = self.model(img_tensor)
        fake_score = output[0][1]

        self.model.zero_grad()
        fake_score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1], activations.shape[2])

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.clamp(cam, min=0)

        cam = cam.numpy()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam


def apply_heatmap(frame, cam):
    cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    result = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.4, 0)

    return result
