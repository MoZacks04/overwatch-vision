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

    # Explicit geometry for the two nameplates chosen as this row. Keeping
    # these separate from generic color components prevents background color
    # blobs from expanding the row or shifting hero crops.
    killer_panel_local: Optional[Rect] = None
    victim_panel_local: Optional[Rect] = None
    killer_team_hint: Optional[str] = None
    victim_team_hint: Optional[str] = None

    # Exact portrait boxes used by BOTH debug drawing and parsing. This keeps
    # the yellow boxes and recognizer input perfectly aligned.
    killer_hero_box_local: Optional[Rect] = None
    victim_hero_box_local: Optional[Rect] = None

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
    row_history: list[KillFeedRow] = field(default_factory=list)


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

    # Snapshot of the normalized row's lightweight visual fingerprint.
    # Used after parsing as a last-resort duplicate key when names/heroes
    # are not available.
    visual_fingerprint: Optional[np.ndarray] = None
