from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from overwatch_vision.utils.image_ops import fingerprint_similarity


@dataclass(slots=True)
class RecentParsedEvent:
    timestamp: float
    identity_key: str | None
    fingerprint: np.ndarray | None


class ParsedEventDeduper:
    """
    Final duplicate guard after OCR/hero parsing.

    Identity is preferred over raw track IDs:
      1. killer player -> victim player
      2. killer hero -> victim hero
      3. visual row fingerprint when identity is unavailable

    This catches the common case where one visible kill-feed row is lost by
    temporal tracking and recreated as a second track a few seconds later.
    """

    def __init__(self, config: dict):
        cfg = config.get("event_dedupe", {})

        self.window_seconds = float(
            cfg.get("window_seconds", 10.0)
        )
        self.visual_similarity = float(
            cfg.get("visual_similarity", 0.86)
        )

        self.recent: list[RecentParsedEvent] = []

    @staticmethod
    def _clean(value: str | None) -> str:
        if not value:
            return ""

        return re.sub(
            r"[^a-z0-9]",
            "",
            value.lower(),
        )

    def _identity_key(self, event) -> str | None:
        killer_name = self._clean(event.killer_name)
        victim_name = self._clean(event.victim_name)

        if killer_name and victim_name:
            return (
                "players:"
                f"{killer_name}>{victim_name}:"
                f"{event.killer_team or '?'}>"
                f"{event.victim_team or '?'}"
            )

        killer_hero = self._clean(event.killer_hero)
        victim_hero = self._clean(event.victim_hero)

        if killer_hero and victim_hero:
            return (
                "heroes:"
                f"{killer_hero}>{victim_hero}:"
                f"{event.killer_team or '?'}>"
                f"{event.victim_team or '?'}"
            )

        return None

    def _prune(self, timestamp: float):
        cutoff = timestamp - self.window_seconds
        self.recent = [
            item
            for item in self.recent
            if item.timestamp >= cutoff
        ]

    def is_duplicate(self, event) -> bool:
        timestamp = float(event.timestamp)
        self._prune(timestamp)

        identity_key = self._identity_key(event)
        fingerprint = getattr(
            event,
            "visual_fingerprint",
            None,
        )

        for item in self.recent:
            # Strongest possible signal: same parsed player pair or hero pair.
            if (
                identity_key is not None
                and item.identity_key is not None
                and identity_key == item.identity_key
            ):
                return True

            # Only use the visual fallback when at least one of the events
            # lacks a strong parsed identity. This avoids suppressing two
            # different well-identified eliminations that just look similar.
            if (
                fingerprint is not None
                and item.fingerprint is not None
                and (
                    identity_key is None
                    or item.identity_key is None
                )
            ):
                similarity = fingerprint_similarity(
                    item.fingerprint,
                    fingerprint,
                )

                if similarity >= self.visual_similarity:
                    return True

        self.recent.append(
            RecentParsedEvent(
                timestamp=timestamp,
                identity_key=identity_key,
                fingerprint=(
                    fingerprint.copy()
                    if fingerprint is not None
                    else None
                ),
            )
        )

        return False

    def reset(self):
        self.recent.clear()
