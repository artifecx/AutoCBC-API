import numpy as np
import torch
from collections import Counter
from PIL import Image
from torchvision import transforms

# Normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EFFICIENTV2_MEAN = [0.5, 0.5, 0.5]
EFFICIENTV2_STD = [0.5, 0.5, 0.5]


def get_classes() -> list[str]:
    """Return the list of CBC cell classes."""
    return [
        'Basophil', 'Lymphocyte', 'Monocyte',
        'Neutrophil', 'Platelet', 'RBC', 'Eosinophil'
    ]


def classify_cell(
    model_name: str,
    model: torch.nn.Module,
    cell_crop: np.ndarray | Image.Image
) -> str:
    """
    Perform single-cell classification using the specified model.

    - If using YOLO-based classifier ("yolo11l-cls.pt" or "yolo11x-cls.pt"),
      pass the PIL image directly and use model[cell_crop].
    - Otherwise, apply standard torchvision transforms and do a forward pass.

    Returns the predicted class name.
    """
    # Convert numpy arrays to PIL
    if isinstance(cell_crop, np.ndarray):
        cell_crop = Image.fromarray(cell_crop)

    classes = get_classes()
    # YOLO-based classification
    if model_name in ("yolo11l-cls.pt", "yolo11x-cls.pt"):
        results = model(cell_crop)
        pred_idx = results[0].probs.top1
        return classes[pred_idx]

    # Choose normalization based on model type
    is_efficientv2 = model_name.lower().startswith("efficientnet_v2")
    mean = EFFICIENTV2_MEAN if is_efficientv2 else IMAGENET_MEAN
    std = EFFICIENTV2_STD if is_efficientv2 else IMAGENET_STD

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = preprocess(cell_crop).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()

    return classes[pred_idx]


def ensemble_classification(
    models_dict: dict[str, torch.nn.Module],
    cell_crop: np.ndarray | Image.Image
) -> str:
    """
    Perform majority-vote classification across multiple models.

    Skips YOLO-based classifiers in the ensemble.
    """
    votes: list[str] = []
    for name, mdl in models_dict.items():
        if name in ("yolo11l-cls.pt", "yolo11x-cls.pt"):
            continue
        votes.append(classify_cell(name, mdl, cell_crop))

    if not votes:
        raise ValueError("No classification models available for ensemble.")

    majority, _ = Counter(votes).most_common(1)[0]
    return majority
