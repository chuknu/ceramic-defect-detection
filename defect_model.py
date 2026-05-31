#!/usr/bin/env python3
"""Model wrapper for ceramic defect detection using Ultralytics YOLO."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


def select_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DefectModel:
    def __init__(self, model_path: str | Path = "yolov26n.pt", conf: float = 0.25, device: str = "auto"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.conf = float(conf)
        self.device = select_device(device)
        self.model = YOLO(str(self.model_path))
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def predict(self, frame: Any) -> tuple[Any, list[dict[str, Any]]]:
        results = self.model(frame, conf=self.conf, device=self.device)
        result = results[0]
        annotated_frame = result.plot()
        defects: list[dict[str, Any]] = []

        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            xyxy = (
                result.boxes.xyxy.cpu().numpy()
                if hasattr(result.boxes.xyxy, "cpu")
                else np.asarray(result.boxes.xyxy)
            )
            classes = (
                result.boxes.cls.cpu().numpy()
                if hasattr(result.boxes.cls, "cpu")
                else np.asarray(result.boxes.cls)
            )
            scores = (
                result.boxes.conf.cpu().numpy()
                if hasattr(result.boxes.conf, "cpu")
                else np.asarray(result.boxes.conf)
            )

            for bbox, class_id, score in zip(xyxy, classes, scores):
                class_id = int(class_id)
                defects.append(
                    {
                        "class_id": class_id,
                        "label": self.get_label(class_id),
                        "confidence": float(score),
                        "bbox": [float(value) for value in bbox],
                    }
                )

        return annotated_frame, defects

    def get_label(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"defect_{class_id}")
