import io
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from model_utils import (
    load_model_from_file,
    predict_polymer,
    CLASSES,
    POLYMER_INFO
)

app = FastAPI(
    title="PolymerCode AI Backend",
    description="FastAPI service for plastic resin code identification and exact polymer prediction.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model registry
LOADED_MODELS: Dict[str, Any] = {}

def get_or_load_model(model_name: str):
    """Retrieve pre-loaded model or load on first request."""
    key = model_name.lower().strip()
    if key in LOADED_MODELS:
        return LOADED_MODELS[key]

    if "custom" in key or "cnn" in key:
        path = Path("models/custom_cnn.pkl")
    else:
        path = Path("models/vgg16.pkl")

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found at {path}"
        )

    model, classes = load_model_from_file(path)
    LOADED_MODELS[key] = (model, classes)
    return model, classes


@app.on_event("startup")
async def startup_event():
    """Warm up default VGG-16 model on startup."""
    try:
        get_or_load_model("vgg16")
    except Exception as e:
        print(f"Warning: could not preload VGG-16: {e}")


@app.get("/", tags=["Status"])
def root():
    return {
        "service": "PolymerCode AI API",
        "status": "online",
        "docs_url": "/docs",
        "available_models": ["vgg16", "custom_cnn"]
    }


@app.get("/classes", tags=["Metadata"])
def get_classes():
    """Returns the 7 resin identification codes and exact chemical polymer names."""
    return {
        "classes": CLASSES,
        "polymers": POLYMER_INFO
    }


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Polymer resin code image (.jpg, .png, .webp)"),
    model_name: str = Form("vgg16", description="Model to use: 'vgg16' (recommended) or 'custom_cnn'")
):
    """Predict exact polymer name and resin code from an uploaded image."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {e}"
        )

    model, classes = get_or_load_model(model_name)
    prediction = predict_polymer(model, image, classes)

    return {
        "model_used": model_name,
        "filename": file.filename,
        "prediction": prediction
    }


@app.post("/predict-custom-model", tags=["Inference"])
async def predict_custom_model(
    image_file: UploadFile = File(..., description="Polymer resin code image"),
    model_file: UploadFile = File(..., description="Custom model file (.pkl or .pt)")
):
    """Predict using a custom uploaded model file."""
    try:
        img_bytes = await image_file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {e}")

    try:
        model_bytes = await model_file.read()
        model, classes = load_model_from_file(model_bytes, filename=model_file.filename)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to load custom model: {e}")

    prediction = predict_polymer(model, image, classes)
    return {
        "model_used": model_file.filename,
        "filename": image_file.filename,
        "prediction": prediction
    }
