from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from overwatch_vision.ocr import OCRReader


@dataclass(slots=True)
class TeamStatusState:
    friendly_alive: int | None = None
    enemy_alive: int | None = None
    confidence: float = 0.0
    timestamp: float = 0.0


class TeamStatusDetector:
    def __init__(self, config: dict, ocr: OCRReader):
        cfg = config.get("team_status", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_seconds = float(
            cfg.get("read_interval_seconds", 0.40)
        )
        self.max_players = int(cfg.get("max_players", 6))
        self.ocr = ocr

        self.state = TeamStatusState()
        self._last_read = -1.0

    @staticmethod
    def _first_number(text: str | None) -> int | None:
        if not text:
            return None

        match = re.search(r"\d+", text)
        if not match:
            return None

        try:
            return int(match.group(0))
        except ValueError:
            return None

    def _read_side(self, image: np.ndarray) -> tuple[int | None, float]:
        prepared = self.ocr.prepare_digit_image(image)
        text, confidence = self.ocr.read(
            prepared,
            allowlist="0123456789",
        )
        value = self._first_number(text)

        if value is not None and not (0 <= value <= self.max_players):
            return None, 0.0

        return value, confidence

    def process(
        self,
        image: np.ndarray,
        timestamp: float,
    ) -> TeamStatusState:
        if not self.enabled:
            return self.state

        if (
            self._last_read >= 0
            and timestamp - self._last_read < self.interval_seconds
        ):
            return self.state

        self._last_read = timestamp

        if image is None or image.size == 0:
            return self.state

        _, w = image.shape[:2]

        left = image[:, : max(1, int(w * 0.43))]
        right = image[:, int(w * 0.57):]

        friendly, friendly_conf = self._read_side(left)
        enemy, enemy_conf = self._read_side(right)

        if friendly is None and enemy is None:
            return self.state

        confidence_values = [
            value
            for value in (friendly_conf, enemy_conf)
            if value > 0
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        if friendly is not None:
            self.state.friendly_alive = friendly

        if enemy is not None:
            self.state.enemy_alive = enemy

        self.state.confidence = confidence
        self.state.timestamp = timestamp
        return self.state
