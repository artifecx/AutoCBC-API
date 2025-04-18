import io
from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
from typing import Dict

from utils.models import load_bloodsmear_model, load_hemocytometer_model, load_classification_model
from utils.classification import classify_cell, get_classes

app = FastAPI(title="CBC Analysis API")

origins = [
    "https://localhost:7031",
    "https://autocbc.azurewebsites.net",
    "http://127.0.0.1:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load blood smear detection model once
bloodsmear_model = load_bloodsmear_model()
bloodsmear_model.to(device)

# Load hemocytometer detection model once
hemocytometer_model = load_hemocytometer_model()
hemocytometer_model.to(device)

# Preload classification models
CLASSIFIER_FILES = [
    "yolo11l-cls.pt",
    "yolo11x-cls.pt"
]
loaded_classifiers: Dict[str, torch.nn.Module] = {}


@app.post("/analyze-differential", summary="Analyze Blood Smear image")
async def analyze_differential(
    raw: bytes = Body(..., media_type="application/octet-stream"),
    conf_threshold: float = Query(0.1, ge=0.1, le=1.0),
    classification_model: str = Query("yolo11x-cls.pt")
):
    """
    Analyze a blood smear image to detect and classify blood cells.

    - **raw**: image bytes with content-type application/octet-stream
    - **conf_threshold**: YOLO detection confidence threshold (0.0–1.0)
    - **classification_model**: 'combined' or one of the model filenames
    """
    # Load image from raw bytes
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    image_np = np.array(image)

    # YOLO detection
    with torch.no_grad():
        results = bloodsmear_model(image_np, conf=conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    # Initialize counts and output
    classes = get_classes()
    class_counts = {cls.upper(): 0 for cls in classes}
    wbc_count = 0
    output = []

    # Classify each detected cell
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        cell_np = image_np[y1:y2, x1:x2]

        if classification_model in loaded_classifiers:
            model_inst = loaded_classifiers[classification_model]
        else:
            model_inst = load_classification_model(classification_model)
        cell_cls = classify_cell(model_inst, cell_np)

        output.append({"bbox": [x1, y1, x2, y2], "class": cell_cls})
        class_counts[cell_cls.upper()] += 1
        if cell_cls.lower() not in ("platelet", "rbc"):
            wbc_count += 1

    return {"results": output, "class_counts": class_counts, "wbc_count": wbc_count}


@app.post("/analyze-absolute", summary="Analyze Hemocytometer image")
async def analyze_absolute(
    raw: bytes = Body(..., media_type="application/octet-stream"),
    conf_threshold: float = Query(0.1, ge=0.1, le=1.0)
):
    """
    Analyze a hemocytometer image to detect cells (no classification).

    - **raw**: image bytes (application/octet-stream)
    - **conf_threshold**: YOLO detection confidence threshold (0.0–1.0)
    """
    # Load image
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    image_np = np.array(image)

    # Detect cells with the hemocytometer model
    with torch.no_grad():
        results = hemocytometer_model(image_np, conf=conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    # Build output list of bboxes
    output = []
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        output.append({"bbox": [x1, y1, x2, y2]})

    total_count = len(output)

    return {"results": output, "total_count": total_count}
