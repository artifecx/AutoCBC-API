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
def load_yolo_model() -> YOLO:
    """Download and return the YOLO detection model."""
    yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename="yolo11s.pt")
    return YOLO(yolo_path)


@lru_cache(maxsize=None)
def load_classification_model(model_name: str, device: str = 'cpu') -> torch.nn.Module:
    """
    Download and return a YOLO classification model by filename.
    """
    yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename=model_name)
    return YOLO(yolo_path)


def release_cuda_cache():
    """Clear CUDA cache if a GPU is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
