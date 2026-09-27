from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class DeadIdentity:
    dead_until: float
    hero: str | None
    player_name: str | None
    team: str | None


class EliminationStateTracker:
    """
    Stateful guard for spoken elimination calls.

    Once a victim is announced as eliminated, keep that identity "dead" until
    the configured respawn interval has elapsed. This catches duplicate
    kill-feed rows that survive the shorter visual/event dedupe windows.

    The state is intentionally simple for now, but it also gives future
    coaching models a place to query who is currently considered eliminated.
    """

    def __init__(self, config: dict):
        cfg = config.get("elimination_state", {})

        self.enabled = bool(
            cfg.get("enabled", True)
        )
        self.default_respawn_seconds = max(
            0.0,
            float(
                cfg.get(
                    "default_respawn_seconds",
                    10.0,
                )
            ),
        )
        self.use_player_name = bool(
            cfg.get("use_player_name", True)
        )
        self.use_hero_fallback = bool(
            cfg.get("use_hero_fallback", True)
        )

        self._dead_by_key: dict[str, DeadIdentity] = {}

    @staticmethod
    def _clean(value: str | None) -> str:
        if not value:
            return ""

        return re.sub(
            r"[^a-z0-9]",
            "",
            value.lower(),
        )

    def _victim_keys(self, event) -> list[str]:
        keys: list[str] = []
        team = self._clean(
            getattr(
                event,
                "victim_team",
                None,
            )
        ) or "?"

        if self.use_player_name:
            player = self._clean(
                getattr(
                    event,
                    "victim_name",
                    None,
                )
            )

            if player:
                keys.append(
                    f"player:{team}:{player}"
                )

        if self.use_hero_fallback:
            hero = self._clean(
                getattr(
                    event,
                    "victim_hero",
                    None,
                )
            )

            if hero:
                keys.append(
                    f"hero:{team}:{hero}"
                )

        return keys

    def _prune(self, timestamp: float):
        expired = [
            key
            for key, state in self._dead_by_key.items()
            if state.dead_until <= timestamp
        ]

        for key in expired:
            self._dead_by_key.pop(
                key,
                None,
            )

    def remaining_seconds(
        self,
        event,
        timestamp: float,
    ) -> float:
        if not self.enabled:
            return 0.0

        self._prune(timestamp)

        remaining = 0.0

        for key in self._victim_keys(event):
            state = self._dead_by_key.get(
                key
            )

            if state is None:
                continue

            remaining = max(
                remaining,
                state.dead_until - timestamp,
            )

        return max(
            0.0,
            remaining,
        )

    def can_announce(
        self,
        event,
        timestamp: float,
    ) -> bool:
        return (
            self.remaining_seconds(
                event,
                timestamp,
            )
            <= 0.0
        )

    def record_elimination(
        self,
        event,
        timestamp: float,
    ):
        if not self.enabled:
            return

        keys = self._victim_keys(event)

        if not keys:
            return

        dead_until = (
            timestamp
            + self.default_respawn_seconds
        )

        state = DeadIdentity(
            dead_until=dead_until,
            hero=getattr(
                event,
                "victim_hero",
                None,
            ),
            player_name=getattr(
                event,
                "victim_name",
                None,
            ),
            team=getattr(
                event,
                "victim_team",
                None,
            ),
        )

        for key in keys:
            existing = self._dead_by_key.get(
                key
            )

            if (
                existing is None
                or dead_until
                > existing.dead_until
            ):
                self._dead_by_key[key] = state

    def active_dead(self, timestamp: float):
        """
        Return currently dead identities for future game-state/coaching logic.
        """
        self._prune(timestamp)

        unique = {}
        for state in self._dead_by_key.values():
            identity = (
                state.team,
                state.player_name,
                state.hero,
                state.dead_until,
            )
            unique[identity] = state

        return list(
            unique.values()
        )

    def reset(self):
        self._dead_by_key.clear()
