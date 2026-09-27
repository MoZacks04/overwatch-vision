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

    Parsed identity helps confirm a duplicate, but identity alone is never
    enough: legitimate eliminations can repeat the same hero/player pair.
    The visible row fingerprint must also closely match within the time window.

    This catches the common case where one visible kill-feed row is lost by
    temporal tracking and recreated as a second track a few seconds later
    without suppressing a different row that happens to involve the same hero.
    """

    def __init__(self, config: dict):
        cfg = config.get("event_dedupe", {})

        self.window_seconds = float(
            cfg.get("window_seconds", 10.0)
        )
        self.visual_similarity = float(
            cfg.get("visual_similarity", 0.96)
        )
        self.identity_visual_similarity = float(
            cfg.get(
                "identity_visual_similarity",
                0.90,
            )
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
            similarity = None

            if (
                fingerprint is not None
                and item.fingerprint is not None
            ):
                similarity = fingerprint_similarity(
                    item.fingerprint,
                    fingerprint,
                )

            # Identity alone is NOT enough to call something a duplicate.
            # Two legitimate eliminations can involve the same heroes/players
            # within five seconds. Require the row pixels to agree as well.
            if (
                identity_key is not None
                and item.identity_key is not None
                and identity_key == item.identity_key
                and similarity is not None
                and similarity
                >= self.identity_visual_similarity
            ):
                return True

            # With incomplete OCR/hero identity, only suppress an almost
            # identical visual row. This is intentionally strict so a fresh
            # kill entering as another exits is not discarded.
            if (
                (
                    identity_key is None
                    or item.identity_key is None
                )
                and similarity is not None
                and similarity
                >= self.visual_similarity
            ):
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
