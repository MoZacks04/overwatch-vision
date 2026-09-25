from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class KillFeedRowVerifier:
    """
    Optional learned second-stage check for color-detector row proposals.

    The existing red/blue detector still proposes candidate rows. When a local
    verifier model exists, this class rejects proposals that do not visually
    resemble a real Overwatch kill-feed row. If no model exists, every proposal
    passes through unchanged.
    """

    def __init__(self, config: dict):
        cfg = config.get("killfeed_row_verifier", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.min_real_probability = float(
            cfg.get("min_real_probability", 0.68)
        )
        self.input_width = int(cfg.get("input_width", 256))
        self.input_height = int(cfg.get("input_height", 64))

        project_root = Path(__file__).resolve().parents[3]

        self.model_path = project_root / str(
            cfg.get(
                "model_path",
                "models/killfeed_row_verifier.pt",
            )
        )
        self.labels_path = project_root / str(
            cfg.get(
                "labels_path",
                "models/killfeed_row_verifier_labels.json",
            )
        )

        self._model = None
        self._torch = None
        self._real_index = 1
        self._attempted_load = False

    @property
    def ready(self) -> bool:
        return self._model is not None and self._torch is not None

    def _load(self) -> None:
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

            metadata = json.loads(
                self.labels_path.read_text(encoding="utf-8")
            )

            classes = metadata
            if isinstance(metadata, dict):
                classes = metadata.get("classes", [])
                self.input_width = int(
                    metadata.get("input_width", self.input_width)
                )
                self.input_height = int(
                    metadata.get("input_height", self.input_height)
                )

            if not isinstance(classes, list) or "real" not in classes:
                raise ValueError(
                    "row verifier labels must contain a 'real' class"
                )

            self._real_index = classes.index("real")

            model = torch.jit.load(
                str(self.model_path),
                map_location="cpu",
            )
            model.eval()

            self._torch = torch
            self._model = model

            print(
                "[row-verifier] loaded learned kill-feed row verifier "
                f"(threshold={self.min_real_probability:.2f})."
            )

        except Exception as exc:
            print(f"[row-verifier] unavailable: {exc}")

    def _prepare(self, image: np.ndarray):
        if image is None or image.size == 0:
            return None

        self._load()

        if not self.ready:
            return None

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        h, w = image.shape[:2]

        if h <= 0 or w <= 0:
            return None

        scale = min(
            self.input_width / float(w),
            self.input_height / float(h),
        )

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_LINEAR
            ),
        )

        canvas = np.full(
            (
                self.input_height,
                self.input_width,
                3,
            ),
            24,
            dtype=np.uint8,
        )

        x1 = max(0, (self.input_width - new_w) // 2)
        y1 = max(0, (self.input_height - new_h) // 2)

        canvas[
            y1:y1 + new_h,
            x1:x1 + new_w,
        ] = resized

        rgb = cv2.cvtColor(
            canvas,
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

    def real_probability(
        self,
        image: np.ndarray,
    ) -> float | None:
        tensor = self._prepare(image)

        if tensor is None:
            return None

        with self._torch.no_grad():
            logits = self._model(tensor)
            probabilities = self._torch.softmax(
                logits,
                dim=1,
            )[0]

        return float(
            probabilities[self._real_index].item()
        )

    def accept(
        self,
        image: np.ndarray,
    ) -> tuple[bool, float | None]:
        probability = self.real_probability(image)

        # No trained verifier yet: preserve the current detector behavior.
        if probability is None:
            return True, None

        return (
            probability >= self.min_real_probability,
            probability,
        )
