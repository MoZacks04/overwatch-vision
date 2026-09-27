from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from overwatch_vision.utils.image_ops import fingerprint_similarity


@dataclass(slots=True)
class RecentParsedEvent:
    timestamp: float
    hero_key: str | None
    player_key: str | None
    fingerprint: np.ndarray | None


class ParsedEventDeduper:
    """
    Final duplicate guard after OCR/hero parsing.

    Parsed identity is the primary duplicate signal inside the short
    dedupe window. The same visible elimination can be recreated as several
    track IDs while the row animates/fades, so requiring the row fingerprint
    to remain almost identical is too strict.

    Hero-pair identity is stored independently from OCR player names so a row
    still dedupes when one parse reads a player name and another parse misses it.
    Visual similarity remains a fallback for rows whose hero identity is not
    available yet.
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

    def _player_key(self, event) -> str | None:
        killer_name = self._clean(event.killer_name)
        victim_name = self._clean(event.victim_name)

        if not killer_name or not victim_name:
            return None

        return (
            f"{killer_name}>{victim_name}:"
            f"{event.killer_team or '?'}>"
            f"{event.victim_team or '?'}"
        )

    def _hero_key(self, event) -> str | None:
        killer_hero = self._clean(event.killer_hero)
        victim_hero = self._clean(event.victim_hero)

        if not killer_hero or not victim_hero:
            return None

        return (
            f"{killer_hero}>{victim_hero}:"
            f"{event.killer_team or '?'}>"
            f"{event.victim_team or '?'}"
        )

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

        hero_key = self._hero_key(event)
        player_key = self._player_key(event)
        fingerprint = getattr(
            event,
            "visual_fingerprint",
            None,
        )

        for item in self.recent:
            # Strong parsed identity wins inside the short dedupe window.
            # This intentionally does NOT depend on the row fingerprint because
            # the same feed entry changes appearance as it slides and fades.
            if (
                hero_key is not None
                and item.hero_key is not None
                and hero_key == item.hero_key
            ):
                return True

            if (
                player_key is not None
                and item.player_key is not None
                and player_key == item.player_key
            ):
                return True

            # If either event lacks parsed identity, fall back to an almost
            # identical visual match.
            if (
                fingerprint is not None
                and item.fingerprint is not None
                and (
                    hero_key is None
                    or item.hero_key is None
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
                hero_key=hero_key,
                player_key=player_key,
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
