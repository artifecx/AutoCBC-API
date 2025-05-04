import asyncio
import datetime
import io
import json
import uuid
import threading
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import BackgroundTasks, FastAPI, Query, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
from typing import Dict, List

from utils.models import (
    load_bloodsmear_model,
    load_classification_model,
    load_hemocytometer_model,
    load_hemo_classification_model
)
from utils.classification import (
    classify_cell,
    classify_hemo_cell,
    get_classes,
    get_hemo_classes,
)

from utils.logging import LoggingMiddleware

INPUT_DIR = Path("data/inputs")
OUTPUT_DIR = Path("data/outputs")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CBC Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:7031",
        "https://autocbc.azurewebsites.net",
        "https://gradsync.org"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

device = "cuda" if torch.cuda.is_available() else "cpu"

bloodsmear_model = load_bloodsmear_model().to(device)
hemocytometer_model = load_hemocytometer_model().to(device)
executor = ThreadPoolExecutor()

CLASSIFIER_FILES = ["yolo11l-cls.pt", "yolo11x-cls.pt", "yolo11x-cls-hemo.pt"]
loaded_classifiers: Dict[str, torch.nn.Module] = {}
model_lock = threading.Lock()

def save_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

@app.on_event("startup")
def preload_classifiers():
    for fname in CLASSIFIER_FILES:
        if fname == "yolo11x-cls-hemo.pt":
            model = load_hemo_classification_model().to(device)
        else:
            model = load_classification_model(fname).to(device)
        with model_lock:
            loaded_classifiers[fname] = model

async def detect_cells_async(model, batch_images, conf_threshold):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model(batch_images, conf=conf_threshold)
    )

async def classify_cell_async(model, cell_np):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: classify_cell(model, cell_np)
    )

async def classify_hemo_cell_async(model, cell_np):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: classify_hemo_cell(model, cell_np)
    )

async def process_image(image_bytes, conf_threshold, classification_model):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    image_np = np.array(image)

    results = await detect_cells_async(bloodsmear_model, image_np, conf_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    class_counts = {cls.upper(): 0 for cls in get_classes()}
    wbc_count = 0
    output = []

    with model_lock:
        cls_model = loaded_classifiers[classification_model]

    classify_tasks = [classify_cell_async(cls_model, image_np[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes]
    cell_classes = await asyncio.gather(*classify_tasks)

    for bbox, cell_cls in zip(boxes, cell_classes):
        cell_cls = cell_cls.upper()
        output.append({"bbox": bbox.tolist(), "class": cell_cls})
        class_counts[cell_cls] += 1
        if cell_cls.lower() not in ("platelet", "rbc"):
            wbc_count += 1

    return {"results": output, "class_counts": class_counts, "wbc_count": wbc_count}

@app.post("/analyze-differentials", summary="Batch Analyze Blood Smear images")
async def analyze_differentials(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., media_type="application/octet-stream"),
    conf_threshold: float = Query(0.5, ge=0.1, le=1.0),
    classification_model: str = Query("yolo11l-cls.pt"),
    batch_size: int = Query(8, ge=1, le=8)
):
    classes = get_classes()
    total_class_counts = {cls.upper(): 0 for cls in classes}
    total_wbc_count = 0
    batch_results = []

    semaphore = asyncio.Semaphore(batch_size)

    async def process_with_semaphore(upload):
        async with semaphore:
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")

            raw = await upload.read()

            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")

            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
            uid = uuid.uuid4().hex
            filename = f"{timestamp}_{uid}.png"
            input_path = INPUT_DIR / filename
            output_path = OUTPUT_DIR / f"{filename}.json"

            background_tasks.add_task(save_bytes, input_path, raw)
            result = await process_image(raw, conf_threshold, classification_model)
            # background_tasks.add_task(save_json, output_path, result)

            return result

    tasks = [process_with_semaphore(upload) for upload in files]
    results = await asyncio.gather(*tasks)

    for single in results:
        for cls, cnt in single["class_counts"].items():
            total_class_counts[cls] += cnt
        total_wbc_count += single["wbc_count"]
        batch_results.append(single)

    return {
        "batch": batch_results,
        "total_class_counts": total_class_counts,
        "total_wbc_count": total_wbc_count
    }

@app.post("/analyze-absolutes", summary="Batch Analyze Hemocytometer images")
async def analyze_absolutes(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., media_type="application/octet-stream"),
    conf_threshold: float = Query(0.2, ge=0.1, le=1.0),
    classification_model: str = Query("yolo11x-cls-hemo.pt"),
    batch_size: int = Query(8, ge=1, le=8),
):
    """
    Batch‐analyze one or more hemocytometer images: detect cells,
    classify each crop with `classify_hemo_cell`, and return per‐image
    and overall counts.
    """
    classes = get_hemo_classes()
    total_class_counts = {cls: 0 for cls in classes}
    total_count = 0
    batch_results = []

    semaphore = asyncio.Semaphore(batch_size)

    async def process_with_semaphore(upload: UploadFile):
        async with semaphore:
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")

            raw = await upload.read()
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")

            # save input
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
            uid = uuid.uuid4().hex
            filename = f"{timestamp}_{uid}.png"
            input_path = INPUT_DIR / filename
            background_tasks.add_task(save_bytes, input_path, raw)

            # --- detect ---
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            image_np = np.array(image)
            results = await detect_cells_async(hemocytometer_model, image_np, conf_threshold)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            # --- classify each crop ---
            with model_lock:
                cls_model = loaded_classifiers[classification_model]

            classify_tasks = [
                classify_hemo_cell_async(cls_model, image_np[y1:y2, x1:x2])
                for x1, y1, x2, y2 in boxes
            ]
            cell_classes = await asyncio.gather(*classify_tasks)

            # --- build per‐image result and counts ---
            output = []
            class_counts = {cls: 0 for cls in classes}
            for (x1, y1, x2, y2), cell_cls in zip(boxes, cell_classes):
                cls_label = cell_cls.upper()
                output.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": cls_label
                })
                class_counts[cls_label] += 1

            return {
                "results": output,
                "class_counts": class_counts,
                "total_count": len(output)
            }

    # launch all files in parallel (respecting semaphore)
    tasks = [process_with_semaphore(f) for f in files]
    results = await asyncio.gather(*tasks)

    # aggregate
    for r in results:
        for cls, cnt in r["class_counts"].items():
            total_class_counts[cls] += cnt
        total_count += r["total_count"]
        batch_results.append(r)

    return {
        "batch": batch_results,
        "total_class_counts": total_class_counts,
        "total_count": total_count
    }
