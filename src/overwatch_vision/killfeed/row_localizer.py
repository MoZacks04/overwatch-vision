from __future__ import annotations

from pathlib import Path

import numpy as np

from overwatch_vision.models import Rect


class KillFeedRowLocalizer:
    """
    Optional learned first-stage object detector for the full kill-feed ROI.

    If the trained model is absent, this class returns no boxes and the
    existing color/geometry detector continues to work unchanged.
    """

    def __init__(self, config: dict):
        cfg = config.get("killfeed_localizer", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.min_confidence = float(cfg.get("min_confidence", 0.20))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.45))
        self.imgsz = int(cfg.get("imgsz", 416))
        self.max_det = int(cfg.get("max_det", 8))

        project_root = Path(__file__).resolve().parents[3]
        self.model_path = project_root / str(
            cfg.get(
                "model_path",
                "models/killfeed_row_localizer.pt",
            )
        )

        self._model = None
        self._attempted_load = False
        self._available = False

    @property
    def ready(self) -> bool:
        return self._available and self._model is not None

    def _load(self):
        if self._attempted_load:
            return

        self._attempted_load = True

        if not self.enabled or not self.model_path.exists():
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            self._available = True

            print(
                "[row-localizer] loaded learned full-ROI kill-feed detector "
                f"(threshold={self.min_confidence:.2f}, imgsz={self.imgsz})."
            )
        except Exception as exc:
            print(f"[row-localizer] unavailable: {exc}")

    def detect(self, image: np.ndarray):
        if image is None or image.size == 0:
            return []

        self._load()

        if not self.ready:
            return []

        try:
            results = self._model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.min_confidence,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            print(f"[row-localizer] inference failed: {exc}")
            return []

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()

        h, w = image.shape[:2]
        output = []

        for coords, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = [int(round(float(v))) for v in coords]

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            output.append(
                (
                    Rect(x1=x1, y1=y1, x2=x2, y2=y2),
                    float(confidence),
                )
            )

        output.sort(key=lambda item: item[0].y1)
        return output
