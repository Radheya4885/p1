"""
Utility script to load the saved .pkl models and weights/biases.
"""
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Architecture definitions for unpickling
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def load_model_bundle(pkl_path="models/custom_cnn.pkl"):
    """Loads the complete model bundle including architecture, weights, classes, and metadata."""
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_weights_and_biases(pkl_path="models/custom_cnn_weights_and_biases.pkl"):
    """Loads only the weights and biases dictionary (requires no custom class definitions)."""
    with open(pkl_path, "rb") as f:
        wb = pickle.load(f)
    return wb


def predict_image(image_path, model_name="vgg16", device="cpu"):
    """Runs prediction on a single image using either 'vgg16' or 'custom_cnn'."""
    pkl_file = f"models/{model_name}.pkl"
    bundle = load_model_bundle(pkl_file)
    model = bundle["model"].to(device)
    model.eval()
    classes = bundle["classes"]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()

    return {
        "predicted_class": classes[idx],
        "confidence": float(probs[idx].item() * 100),
        "all_probabilities": {classes[i]: float(probs[i].item() * 100) for i in range(len(classes))}
    }


if __name__ == "__main__":
    print("Testing load_model.py...")
    cnn_bundle = load_model_bundle("models/custom_cnn.pkl")
    print(f"Loaded Custom CNN successfully. Classes: {cnn_bundle['classes']}")

    vgg_bundle = load_model_bundle("models/vgg16.pkl")
    print(f"Loaded VGG-16 successfully. Classes: {vgg_bundle['classes']}")

    wb = load_weights_and_biases("models/custom_cnn_weights_and_biases.pkl")
    print(f"Loaded Custom CNN weights & biases: {len(wb['weights'])} weights, {len(wb['biases'])} biases.")
