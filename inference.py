"""Inference helpers for a trained CSRNet checkpoint."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .model import CSRNet

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path, device="cpu"):
    model = CSRNet().to(device)
    # Load only checkpoints you trust. weights_only avoids general pickle deserialization.
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.inference_mode()
def predict(model, image: Image.Image, device="cpu"):
    image = image.convert("RGB")
    x = TRANSFORM(image).unsqueeze(0).to(device)
    density = model(x)[0, 0].cpu().numpy()
    # Density-map sum is valid only for weights trained with count-preserving target maps.
    return density, float(density.sum())

def normalized_heatmap(density):
    values = np.maximum(density, 0)
    maximum = float(values.max()) if values.size else 0.0
    return np.uint8(values / maximum * 255) if maximum > 0 else np.zeros_like(values, dtype=np.uint8)
