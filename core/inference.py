from pathlib import Path
import torch
from torch import nn
from PIL import Image
from config.settings import MODEL_PATH, IMAGE_SIZE
from core.image_utils import image_to_array

CLASS_NAMES = ["defective", "normal"]
MODEL_VERSION = "brake-disc-cnn-v1"

class BrakeDiscCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def model_status():
    return {"available": MODEL_PATH.exists(), "path": str(MODEL_PATH), "version": MODEL_VERSION if MODEL_PATH.exists() else "not-trained"}

def predict(image):
    """Returns prediction only when a trained checkpoint exists; never invents a fallback result."""
    if not MODEL_PATH.exists():
        return {"model_available": False, "predicted_class": None, "confidence": None, "probabilities": {}, "model_version": "not-trained", "message": "No trained model checkpoint found. Train a model first; disposition is REVIEW REQUIRED."}
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        classes = checkpoint.get("class_names", CLASS_NAMES) if isinstance(checkpoint, dict) else CLASS_NAMES
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = BrakeDiscCNN(num_classes=len(classes))
        model.load_state_dict(state)
        model.eval()
        arr = image_to_array(image, IMAGE_SIZE)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        return {"model_available": True, "predicted_class": str(classes[idx]), "confidence": conf,
                "probabilities": {str(c): float(p) for c, p in zip(classes, probs.tolist())},
                "model_version": checkpoint.get("model_version", MODEL_VERSION) if isinstance(checkpoint, dict) else MODEL_VERSION,
                "message": "Prediction produced by the locally saved CNN checkpoint."}
    except Exception as exc:
        return {"model_available": False, "predicted_class": None, "confidence": None, "probabilities": {}, "model_version": "load-error", "message": f"Model could not be loaded safely: {exc}. Disposition is REVIEW REQUIRED."}
