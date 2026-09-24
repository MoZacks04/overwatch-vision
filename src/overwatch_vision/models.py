from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(slots=True)
class GameFrame:
    image: np.ndarray
    timestamp: float
    width: int
    height: int


@dataclass(slots=True)
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(slots=True)
class KillFeedRow:
    bbox_roi: Rect
    bbox_game: Rect
    crop: np.ndarray
    normalized: np.ndarray
    fingerprint: np.ndarray
    component_boxes_local: list[Rect] = field(default_factory=list)
    component_teams_local: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass(slots=True)
class KillFeedTrack:
    track_id: int
    row: KillFeedRow
    first_seen: float
    last_seen: float
    age_frames: int = 1
    missing_frames: int = 0
    confirmed: bool = False
    emitted: bool = False


@dataclass(slots=True)
class KillFeedEvent:
    event_type: str
    track_id: int
    timestamp: float
    confidence: float

    killer_name: Optional[str] = None
    victim_name: Optional[str] = None
    killer_hero: Optional[str] = None
    victim_hero: Optional[str] = None
    killer_team: Optional[str] = None
    victim_team: Optional[str] = None

    ability: Optional[str] = None
    critical: Optional[bool] = None

    parse_confidence: float = 0.0
    killer_hero_confidence: float = 0.0
    victim_hero_confidence: float = 0.0
    killer_name_confidence: float = 0.0
    victim_name_confidence: float = 0.0
