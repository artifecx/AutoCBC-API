import os
import torch
import torch.nn as nn
from functools import lru_cache
from huggingface_hub import login, hf_hub_download
from torchvision import models
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
    Download and return a classification model by filename.
    Supports YOLO-based classifiers and torchvision models.
    """
    # YOLO-based classification
    if model_name in ("yolo11l-cls.pt", "yolo11x-cls.pt"):
        yolo_path = hf_hub_download(repo_id="artifecx/CBCModels", filename=model_name)
        return YOLO(yolo_path)

    # Torchvision-based classification
    model = _init_model_by_name(model_name)
    model_path = hf_hub_download(repo_id="artifecx/CBCModels", filename=model_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def _init_model_by_name(model_name: str) -> torch.nn.Module:
    """Initialize an untrained torchvision model adapted for 7 classes."""
    if model_name == "densenet201.pth":
        m = models.densenet201()
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, 7)
    elif model_name == "efficientnet_v2_l.pth":
        m = models.efficientnet_v2_l()
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, 7)
    elif model_name == "inception_v3.pth":
        m = models.inception_v3()
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 7)
    elif model_name == "resnet152.pth":
        m = models.resnet152()
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 7)
    elif model_name == "resnext101_64x4d.pth":
        m = models.resnext101_64x4d()
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 7)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return m


def release_cuda_cache():
    """Clear CUDA cache if a GPU is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
