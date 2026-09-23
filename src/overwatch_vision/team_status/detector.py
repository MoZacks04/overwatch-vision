from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from overwatch_vision.team_status.digit_recognizer import (
    DigitTemplateRecognizer,
)


@dataclass(slots=True)
class TeamStatusState:
    friendly_alive: int | None = None
    enemy_alive: int | None = None
    confidence: float = 0.0
    timestamp: float = 0.0


class TeamStatusDetector:
    """
    Cheap team-count reader.

    This no longer uses EasyOCR. The HUD crop is fingerprinted first and only
    re-read when it changes enough. Digits are recognized with lightweight
    OpenCV template matching.
    """

    def __init__(self, config: dict):
        cfg = config.get("team_status", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.max_players = int(cfg.get("max_players", 6))
        self.change_threshold = float(
            cfg.get("change_threshold", 0.020)
        )
        self.min_confidence = float(
            cfg.get("min_confidence", 0.30)
        )

        self.recognizer = DigitTemplateRecognizer(
            max_digit=self.max_players
        )

        self.state = TeamStatusState()
        self._last_signature = None

    @staticmethod
    def _signature(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        small = cv2.resize(
            gray,
            (64, 24),
            interpolation=cv2.INTER_AREA,
        )
        return small.astype(np.float32) / 255.0

    def _changed_enough(self, image: np.ndarray) -> bool:
        signature = self._signature(image)

        if self._last_signature is None:
            self._last_signature = signature
            return True

        difference = float(
            np.mean(
                np.abs(
                    signature - self._last_signature
                )
            )
        )

        if difference >= self.change_threshold:
            self._last_signature = signature
            return True

        return False

    def _read_side(
        self,
        image: np.ndarray,
    ) -> tuple[int | None, float]:
        value, confidence = self.recognizer.recognize(
            image
        )

        if confidence < self.min_confidence:
            return None, confidence

        if value is not None and not (
            0 <= value <= self.max_players
        ):
            return None, 0.0

        return value, confidence

    def process(
        self,
        image: np.ndarray,
        timestamp: float,
    ) -> TeamStatusState:
        if not self.enabled:
            return self.state

        if image is None or image.size == 0:
            return self.state

        if (
            self.state.timestamp > 0
            and not self._changed_enough(image)
        ):
            return self.state

        if self.state.timestamp <= 0:
            self._last_signature = self._signature(image)

        _, w = image.shape[:2]

        left = image[
            :,
            :max(1, int(w * 0.45)),
        ]
        right = image[
            :,
            int(w * 0.55):,
        ]

        friendly, friendly_conf = self._read_side(
            left
        )
        enemy, enemy_conf = self._read_side(
            right
        )

        if friendly is None and enemy is None:
            return self.state

        scores = [
            value
            for value in (
                friendly_conf,
                enemy_conf,
            )
            if value > 0
        ]

        if friendly is not None:
            self.state.friendly_alive = friendly

        if enemy is not None:
            self.state.enemy_alive = enemy

        self.state.confidence = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )
        self.state.timestamp = timestamp

        return self.state
