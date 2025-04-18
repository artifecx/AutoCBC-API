import cv2
import numpy as np
from typing import List, Dict, Any, Optional


def get_colors() -> Dict[str, tuple[int, int, int]]:
    """
    Return a mapping from cell class names to BGR color tuples.
    """
    return {
        "BASOPHIL": (255, 0, 0),      # Blue
        "EOSINOPHIL": (0, 255, 0),     # Green
        "LYMPHOCYTE": (0, 255, 255),   # Yellow
        "MONOCYTE": (255, 0, 255),     # Magenta
        "NEUTROPHIL": (0, 165, 255),   # Orange
        "PLATELET": (255, 255, 0),     # Cyan
        "RBC": (0, 0, 255),            # Red
    }


def draw_boxes(
    image: np.ndarray,
    results: List[Dict[str, Any]],
    draw_classes: bool = True,
    excluded_classes: Optional[List[str]] = None
) -> np.ndarray:
    """
    Draw bounding boxes and optional class labels on an RGB image.

    Args:
        image: Input image as an RGB numpy array.
        results: List of detection results. Each result must have:
                 - 'bbox': [x1, y1, x2, y2]
                 - 'class': class name or index
        draw_classes: Whether to annotate boxes with class names.
        excluded_classes: List of class names to skip drawing.

    Returns:
        Annotated image as an RGB numpy array.
    """
    if excluded_classes is None:
        excluded_classes = []

    # Convert RGB to BGR for OpenCV drawing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    colors = get_colors()

    for result in results:
        bbox = result.get("bbox", [])
        cls = str(result.get("class", "")).upper()
        if cls in excluded_classes or len(bbox) != 4:
            continue

        color = colors.get(cls, (0, 0, 0))
        x1, y1, x2, y2 = map(int, bbox)
        # Draw rectangle
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 1)
        # Draw class label
        if draw_classes:
            cv2.putText(
                image_bgr,
                cls,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

    # Convert back to RGB before returning
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
