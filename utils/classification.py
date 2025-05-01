import numpy as np
import torch
from PIL import Image


def get_classes() -> list[str]:
    """Return the list of CBC cell classes."""
    return [
        'Basophil', 'Lymphocyte', 'Monocyte',
        'Neutrophil', 'Platelet', 'RBC', 'Eosinophil'
    ]


def get_hemo_classes() -> list[str]:
    """Return the list of Hemocytometer cell classes."""
    return [
        'UNKNOWN', 'RBC', 'WBC'
    ]


def classify_cell(
    model: torch.nn.Module,
    cell_crop: np.ndarray | Image.Image
) -> str:
    """
    Perform single-cell classification using the specified model.

    Returns the predicted class name.
    """
    # Convert numpy arrays to PIL
    if isinstance(cell_crop, np.ndarray):
        cell_crop = Image.fromarray(cell_crop)

    classes = get_classes()
    results = model(cell_crop)
    pred_idx = results[0].probs.top1

    return classes[pred_idx]


def classify_hemo_cell(
    model: torch.nn.Module,
    cell_crop: np.ndarray | Image.Image
) -> str:
    """
    Perform single-cell classification using the specified model.

    Returns the predicted class name.
    """
    # Convert numpy arrays to PIL
    if isinstance(cell_crop, np.ndarray):
        cell_crop = Image.fromarray(cell_crop)

    classes = get_hemo_classes()
    results = model(cell_crop)
    pred_idx = results[0].probs.top1

    return classes[pred_idx]
