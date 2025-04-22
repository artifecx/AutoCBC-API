import io
import asyncio
from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.models import (
    load_bloodsmear_model,
    load_hemocytometer_model,
    load_classification_model
)
from utils.classification import classify_cell, get_classes

app = FastAPI(title="CBC Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:7031",
        "https://autocbc.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load detection models once
bloodsmear_model = load_bloodsmear_model().to(device)
hemocytometer_model = load_hemocytometer_model().to(device)

# ThreadPool for offloading blocking work
executor = ThreadPoolExecutor()

# Preload classification models
CLASSIFIER_FILES = [
    "yolo11l-cls.pt",
    "yolo11x-cls.pt"
]
loaded_classifiers: Dict[str, torch.nn.Module] = {}
model_lock = threading.Lock()


@app.on_event("startup")
def preload_classifiers():
    """
    Load all classification models at startup under a lock,
    so we never race or reload on demand.
    """
    for fname in CLASSIFIER_FILES:
        model = load_classification_model(fname).to(device)
        with model_lock:
            loaded_classifiers[fname] = model


async def detect_cells_async(model, image_np, conf_threshold):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model(image_np, conf=conf_threshold)
    )


async def classify_cell_async(model, cell_np):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: classify_cell(model, cell_np)
    )


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
    results = await detect_cells_async(bloodsmear_model, image_np, conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    # Initialize counts and output
    classes = get_classes()
    class_counts = {cls.upper(): 0 for cls in classes}
    wbc_count = 0
    output = []

    # Grab the classification model under lock
    with model_lock:
        if classification_model not in loaded_classifiers:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown classification model '{classification_model}'"
            )
        cls_model = loaded_classifiers[classification_model]

    # Classify each detected cell (each call offloaded)
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        cell_np = image_np[y1:y2, x1:x2]

        cell_cls = await classify_cell_async(cls_model, cell_np)
        cell_cls = cell_cls.upper()

        output.append({"bbox": [x1, y1, x2, y2], "class": cell_cls})
        class_counts[cell_cls] += 1
        if cell_cls.lower() not in ("platelet", "rbc"):
            wbc_count += 1

    return {
        "results": output,
        "class_counts": class_counts,
        "wbc_count": wbc_count
    }


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

    # YOLO detection offloaded
    results = await detect_cells_async(hemocytometer_model, image_np, conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    # Build output list of bboxes
    output = [{"bbox": [int(x1), int(y1), int(x2), int(y2)]}
              for x1, y1, x2, y2 in boxes]
    total_count = len(output)

    return {"results": output, "total_count": total_count}
