from __future__ import annotations

from pathlib import Path

import numpy as np

from overwatch_vision.models import Rect


class KillFeedHeroIconLocalizer:
    """
    Optional learned detector for the two primary hero portraits inside an
    already-localized kill-feed row.

    The model is intentionally separate from hero identity classification:
    this class answers only WHERE the portraits are. A later classifier
    answers WHICH heroes they show.
    """

    def __init__(self, config: dict):
        cfg = config.get(
            "hero_icon_localizer",
            {},
        )

        self.enabled = bool(
            cfg.get("enabled", True)
        )
        self.min_confidence = float(
            cfg.get(
                "min_confidence",
                0.20,
            )
        )
        self.iou_threshold = float(
            cfg.get(
                "iou_threshold",
                0.45,
            )
        )
        self.imgsz = int(
            cfg.get("imgsz", 640)
        )
        self.max_det = int(
            cfg.get("max_det", 4)
        )

        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.model_path = (
            project_root
            / str(
                cfg.get(
                    "model_path",
                    "models/killfeed_hero_icon_localizer.pt",
                )
            )
        )

        self._model = None
        self._attempted_load = False
        self._available = False

    @property
    def ready(self) -> bool:
        return (
            self._available
            and self._model is not None
        )

    def warmup(self) -> bool:
        self._load()
        return self.ready

    def _load(self):
        if self._attempted_load:
            return

        self._attempted_load = True

        if not self.enabled:
            return

        if not self.model_path.exists():
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(
                str(self.model_path)
            )
            self._available = True

            print(
                "[hero-icon-localizer] loaded learned "
                "kill-feed hero portrait detector "
                f"(threshold={self.min_confidence:.2f}, "
                f"imgsz={self.imgsz})."
            )
        except Exception as exc:
            print(
                "[hero-icon-localizer] unavailable: "
                f"{exc}"
            )

    def _convert_result(
        self,
        result,
        image,
    ):
        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if (
            boxes is None
            or len(boxes) == 0
        ):
            return []

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )
        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        h, w = image.shape[:2]
        output = []

        for coords, confidence in zip(
            xyxy,
            confidences,
        ):
            x1, y1, x2, y2 = [
                int(round(float(v)))
                for v in coords
            ]

            x1 = max(
                0,
                min(w - 1, x1),
            )
            y1 = max(
                0,
                min(h - 1, y1),
            )
            x2 = max(
                x1 + 1,
                min(w, x2),
            )
            y2 = max(
                y1 + 1,
                min(h, y2),
            )

            output.append(
                (
                    Rect(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    float(confidence),
                )
            )

        output.sort(
            key=lambda item: item[0].cx
        )

        return output

    def detect_batch(
        self,
        images: list[np.ndarray],
    ):
        self._load()

        output = [
            []
            for _ in images
        ]

        if not self.ready:
            return output

        valid_indices = []
        valid_images = []

        for index, image in enumerate(images):
            if (
                image is None
                or image.size == 0
            ):
                continue

            valid_indices.append(index)
            valid_images.append(image)

        if not valid_images:
            return output

        try:
            results = self._model.predict(
                source=valid_images,
                imgsz=self.imgsz,
                conf=self.min_confidence,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            print(
                "[hero-icon-localizer] inference failed: "
                f"{exc}"
            )
            return output

        for source_index, result in zip(
            valid_indices,
            results,
        ):
            output[source_index] = (
                self._convert_result(
                    result,
                    images[source_index],
                )
            )

        return output

    def detect(
        self,
        image: np.ndarray,
    ):
        return self.detect_batch(
            [image]
        )[0]
