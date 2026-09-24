from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class TrainedHeroClassifier:
    """
    Optional tiny CNN classifier trained specifically on Overwatch kill-feed
    hero portraits.

    The model is loaded only when local model files exist. This keeps the
    current template recognizer as a fallback while allowing a much stronger
    kill-feed-specific model once enough labeled icon crops have been collected.
    """

    def __init__(self, config: dict):
        cfg = config.get("hero_classifier", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.min_confidence = float(
            cfg.get("min_confidence", 0.86)
        )
        self.min_margin = float(
            cfg.get("min_margin", 0.14)
        )
        self.input_size = int(
            cfg.get("input_size", 64)
        )

        project_root = Path(__file__).resolve().parents[3]

        self.model_path = project_root / str(
            cfg.get(
                "model_path",
                "models/killfeed_hero_classifier.pt",
            )
        )
        self.labels_path = project_root / str(
            cfg.get(
                "labels_path",
                "models/killfeed_hero_classifier_labels.json",
            )
        )

        self._model = None
        self._labels: list[str] = []
        self._torch = None
        self._attempted_load = False

    @property
    def ready(self) -> bool:
        return (
            self._model is not None
            and bool(self._labels)
        )

    def _load(self):
        if self._attempted_load:
            return

        self._attempted_load = True

        if (
            not self.enabled
            or not self.model_path.exists()
            or not self.labels_path.exists()
        ):
            return

        try:
            import torch

            labels = json.loads(
                self.labels_path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(labels, list) or not labels:
                return

            model = torch.jit.load(
                str(self.model_path),
                map_location="cpu",
            )
            model.eval()

            self._torch = torch
            self._labels = [
                str(label)
                for label in labels
            ]
            self._model = model

            print(
                "[hero-classifier] loaded "
                f"{len(self._labels)} hero classes."
            )
        except Exception as exc:
            print(
                "[hero-classifier] unavailable: "
                f"{exc}"
            )

    def _prepare(self, image: np.ndarray):
        if image is None or image.size == 0:
            return None

        self._load()
        if not self.ready:
            return None

        square = image

        if square.ndim == 2:
            square = cv2.cvtColor(
                square,
                cv2.COLOR_GRAY2BGR,
            )

        if square.ndim == 3 and square.shape[2] == 4:
            square = square[:, :, :3]

        h, w = square.shape[:2]
        side = min(h, w)

        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)

        square = square[
            y1:y1 + side,
            x1:x1 + side,
        ]

        square = cv2.resize(
            square,
            (
                self.input_size,
                self.input_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        rgb = cv2.cvtColor(
            square,
            cv2.COLOR_BGR2RGB,
        )

        tensor = (
            self._torch.from_numpy(
                rgb.astype(np.float32) / 255.0
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        return tensor

    def recognize(
        self,
        image: np.ndarray,
    ) -> tuple[str | None, float]:
        tensor = self._prepare(image)

        if tensor is None:
            return None, 0.0

        with self._torch.no_grad():
            logits = self._model(tensor)
            probabilities = self._torch.softmax(
                logits,
                dim=1,
            )[0]

        values, indices = self._torch.topk(
            probabilities,
            k=min(2, probabilities.numel()),
        )

        best_score = float(values[0].item())
        best_index = int(indices[0].item())
        best_name = self._labels[best_index]

        second_score = (
            float(values[1].item())
            if len(values) > 1
            else 0.0
        )
        margin = best_score - second_score

        if (
            best_score < self.min_confidence
            or margin < self.min_margin
        ):
            return None, best_score

        return best_name, best_score
