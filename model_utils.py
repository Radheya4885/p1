import os
import io
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image

CLASSES = ["HDPE", "LDPE", "OTHER", "PET", "PP", "PS", "PVC"]

POLYMER_INFO = {
    "PET": {
        "code": 1,
        "symbol": "♳",
        "exact_name": "Polyethylene Terephthalate",
        "short_name": "PET / PETE",
        "common_uses": "Water & soda bottles, food jars, polyester clothing",
        "recyclability": "Widely Recycled (High Demand)",
        "color": "#10B981"
    },
    "HDPE": {
        "code": 2,
        "symbol": "♴",
        "exact_name": "High-Density Polyethylene",
        "short_name": "HDPE",
        "common_uses": "Milk jugs, shampoo/detergent bottles, pipes",
        "recyclability": "Widely Recycled (High Demand)",
        "color": "#3B82F6"
    },
    "PVC": {
        "code": 3,
        "symbol": "♵",
        "exact_name": "Polyvinyl Chloride",
        "short_name": "PVC",
        "common_uses": "Plumbing pipes, cable insulation, window profiles",
        "recyclability": "Specialized Facilities Only",
        "color": "#EF4444"
    },
    "LDPE": {
        "code": 4,
        "symbol": "♶",
        "exact_name": "Low-Density Polyethylene",
        "short_name": "LDPE",
        "common_uses": "Shopping bags, plastic wraps, squeeze bottles",
        "recyclability": "Store Drop-off / Selected Curbside",
        "color": "#F59E0B"
    },
    "PP": {
        "code": 5,
        "symbol": "♷",
        "exact_name": "Polypropylene",
        "short_name": "PP",
        "common_uses": "Yogurt containers, bottle caps, food takeout containers",
        "recyclability": "Increasingly Recycled",
        "color": "#8B5CF6"
    },
    "PS": {
        "code": 6,
        "symbol": "♸",
        "exact_name": "Polystyrene",
        "short_name": "PS (Styrofoam)",
        "common_uses": "Disposable coffee cups, egg cartons, packing materials",
        "recyclability": "Rarely Recycled",
        "color": "#EC4899"
    },
    "OTHER": {
        "code": 7,
        "symbol": "♹",
        "exact_name": "Other Plastics & Multi-Layer Resins",
        "short_name": "OTHER (Acrylic, Nylon, Polycarbonate, ABS)",
        "common_uses": "Eyeglass lenses, multi-layer pouches, electronics",
        "recyclability": "Difficult / Special Programs Only",
        "color": "#6B7280"
    }
}


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


# Register CustomCNN & ConvBlock onto __main__ to prevent AttributeError: module '__main__' has no attribute 'CustomCNN'
main_module = sys.modules.get('__main__')
if main_module is not None:
    setattr(main_module, 'CustomCNN', CustomCNN)
    setattr(main_module, 'ConvBlock', ConvBlock)


class SafeModelUnpickler(pickle.Unpickler):
    """Custom unpickler that safely redirects __main__.CustomCNN and __main__.ConvBlock."""
    def find_class(self, module, name):
        if name == "CustomCNN":
            return CustomCNN
        if name == "ConvBlock":
            return ConvBlock
        return super().find_class(module, name)


def safe_pickle_load(data_bytes: bytes) -> Any:
    return SafeModelUnpickler(io.BytesIO(data_bytes)).load()


def build_vgg16(num_classes=7):
    model = vgg16(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


def load_model_from_file(file_path_or_bytes, filename: str = "") -> Tuple[nn.Module, list]:
    """
    Loads a model from a file path or in-memory byte buffer (for uploaded files).
    Supports .pkl, .pt, and state_dict bundles.
    """
    # Ensure __main__ has CustomCNN & ConvBlock registered
    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        setattr(main_mod, 'CustomCNN', CustomCNN)
        setattr(main_mod, 'ConvBlock', ConvBlock)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(file_path_or_bytes, (str, Path)):
        filename = filename or str(file_path_or_bytes)
        with open(file_path_or_bytes, "rb") as f:
            data_bytes = f.read()
    else:
        data_bytes = file_path_or_bytes.read() if hasattr(file_path_or_bytes, "read") else file_path_or_bytes

    classes = CLASSES
    model = None

    # 1. Try safe unpickler first
    try:
        obj = safe_pickle_load(data_bytes)
        if isinstance(obj, dict):
            classes = obj.get("classes", CLASSES)
            if "model" in obj and isinstance(obj["model"], nn.Module):
                model = obj["model"]
            elif "model_state_dict" in obj or "state_dict" in obj:
                state_dict = obj.get("model_state_dict", obj.get("state_dict"))
                if any("features.0.block" in k for k in state_dict.keys()):
                    model = CustomCNN(num_classes=len(classes))
                else:
                    model = build_vgg16(num_classes=len(classes))
                model.load_state_dict(state_dict)
            elif "weights" in obj and "biases" in obj:
                # Weights & biases dictionary
                state_dict = obj.get("state_dict", {})
                if any("features.0.block" in k for k in state_dict.keys()):
                    model = CustomCNN(num_classes=len(classes))
                else:
                    model = build_vgg16(num_classes=len(classes))
                model.load_state_dict(state_dict)
            else:
                raise ValueError("Unrecognized dictionary structure in .pkl")
        elif isinstance(obj, nn.Module):
            model = obj
    except Exception:
        model = None

    # 2. Fallback to torch.load (for .pt / .pth or torch serialization)
    if model is None:
        buffer = io.BytesIO(data_bytes)
        ckpt = torch.load(buffer, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            classes = ckpt.get("classes", CLASSES)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            if any("features.0.block" in k for k in state_dict.keys()):
                model = CustomCNN(num_classes=len(classes))
            else:
                model = build_vgg16(num_classes=len(classes))
            model.load_state_dict(state_dict)
        elif isinstance(ckpt, nn.Module):
            model = ckpt
        else:
            raise ValueError("Unable to load model checkpoint.")

    model = model.to(device)
    model.eval()
    return model, classes


def predict_polymer(model: nn.Module, image: Image.Image, classes: list = None) -> Dict[str, Any]:
    """Runs inference and returns full polymer prediction details."""
    if classes is None:
        classes = CLASSES

    device = next(model.parameters()).device
    transform = get_eval_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    idx = probs.argmax().item()
    pred_class = classes[idx]
    confidence = float(probs[idx].item() * 100)

    # All probabilities sorted descending
    all_probs = []
    for i, class_name in enumerate(classes):
        p_val = float(probs[i].item() * 100)
        p_info = POLYMER_INFO.get(class_name, {})
        all_probs.append({
            "class_name": class_name,
            "resin_code": p_info.get("code", i + 1),
            "symbol": p_info.get("symbol", ""),
            "exact_name": p_info.get("exact_name", class_name),
            "probability": p_val
        })
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    info = POLYMER_INFO.get(pred_class, {
        "code": idx + 1,
        "symbol": "♼",
        "exact_name": pred_class,
        "short_name": pred_class,
        "common_uses": "Commercial plastic",
        "recyclability": "Recyclable",
        "color": "#3B82F6"
    })

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "resin_code": info["code"],
        "symbol": info["symbol"],
        "exact_name": info["exact_name"],
        "short_name": info["short_name"],
        "common_uses": info["common_uses"],
        "recyclability": info["recyclability"],
        "color": info["color"],
        "probabilities": all_probs
    }
