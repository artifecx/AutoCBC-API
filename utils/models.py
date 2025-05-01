import os
import torch
from functools import lru_cache
from huggingface_hub import login, hf_hub_download
from ultralytics import YOLO

# Load environment variables (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Authenticate with Hugging Face Hub
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not set.")
login(token=token)


@lru_cache(maxsize=1)
def load_bloodsmear_model() -> YOLO:
    """Download and return the YOLO detection model for blood smear images."""
    yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename="yolo11n.pt")
    return YOLO(yolo_path)


@lru_cache(maxsize=None)
def load_classification_model(model_name: str) -> torch.nn.Module:
    """Download and return a YOLO classification model by filename."""
    yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename=model_name)
    return YOLO(yolo_path)


@lru_cache(maxsize=1)
def load_hemocytometer_model() -> YOLO:
    """Download and return the YOLO detection model for hemocytometer images."""
    yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename="yolo11n-hemo.pt")
    return YOLO(yolo_path)


@lru_cache(maxsize=1)
def load_hemo_classification_model() -> YOLO:
    """Download and return the YOLO classification model for hemocytometer images."""
    yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename="yolo11l-cls-hemo.pt")
    return YOLO(yolo_path)
