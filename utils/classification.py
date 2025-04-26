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
        'WBC', 'RBC', 'UNKNOWN'
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

    if pred_idx == 1:       # L-WBC - WBC
        return classes[0]
    if pred_idx == 2:       # RBC - RBC
        return classes[1]
    if pred_idx == 3:       # WBC - WBC
        return classes[0]

    return classes[2]       # Unknown
