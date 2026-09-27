from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np

from overwatch_vision.utils.image_ops import (
    fingerprint_similarity,
    grayscale_fingerprint,
)


@dataclass(slots=True)
class RecentHardSample:
    timestamp: float
    side: str
    fingerprint: np.ndarray


class HeroHardExampleCollector:
    """
    Mine difficult live hero portraits while the user plays.

    The collector saves at most one representative crop per killer/victim side
    for each parsed kill-feed event. It prefers crops where the trained CNN has
    low confidence or a small top-1/top-2 margin. A small deterministic sample
    of confident crops is also kept so confidently-wrong predictions can be
    discovered during review.

    Nothing here changes runtime recognition. It only writes review material
    under datasets/killfeed_hard_examples/.
    """

    def __init__(self, config: dict):
        cfg = config.get("hard_example_collection", {})

        self.enabled = bool(
            cfg.get("enabled", True)
        )
        self.max_confidence = float(
            cfg.get(
                "uncertain_below_confidence",
                0.88,
            )
        )
        self.max_margin = float(
            cfg.get(
                "uncertain_below_margin",
                0.15,
            )
        )
        self.confident_sample_every = max(
            0,
            int(
                cfg.get(
                    "confident_sample_every",
                    25,
                )
            ),
        )
        self.recent_seconds = max(
            0.0,
            float(
                cfg.get(
                    "recent_dedupe_seconds",
                    12.0,
                )
            ),
        )
        self.recent_similarity = float(
            cfg.get(
                "recent_dedupe_similarity",
                0.97,
            )
        )
        self.fingerprint_size = max(
            8,
            int(
                cfg.get(
                    "fingerprint_size",
                    24,
                )
            ),
        )

        project_root = Path(__file__).resolve().parents[3]
        root_relative = str(
            cfg.get(
                "directory",
                "datasets/killfeed_hard_examples",
            )
        )

        self.root = project_root / root_relative
        self.pending_dir = self.root / "pending_identity"
        self.reviewed_dir = self.root / "reviewed_identity"
        self.localizer_failure_dir = (
            self.root / "localizer_failures"
        )

        self._recent: list[RecentHardSample] = []
        self._candidate_counter = 0

    def _ensure_dirs(self):
        self.pending_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.reviewed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.localizer_failure_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _prune_recent(self, timestamp: float):
        cutoff = (
            timestamp
            - self.recent_seconds
        )
        self._recent = [
            item
            for item in self._recent
            if item.timestamp >= cutoff
        ]

    def _is_recent_duplicate(
        self,
        side: str,
        fingerprint: np.ndarray,
        timestamp: float,
    ) -> bool:
        self._prune_recent(timestamp)

        for item in self._recent:
            if item.side != side:
                continue

            similarity = fingerprint_similarity(
                item.fingerprint,
                fingerprint,
            )

            if (
                similarity
                >= self.recent_similarity
            ):
                return True

        return False

    def _remember(
        self,
        side: str,
        fingerprint: np.ndarray,
        timestamp: float,
    ):
        self._recent.append(
            RecentHardSample(
                timestamp=timestamp,
                side=side,
                fingerprint=fingerprint.copy(),
            )
        )
        self._prune_recent(timestamp)

    @staticmethod
    def _rect_payload(rect):
        if rect is None:
            return None

        return {
            "x1": int(rect.x1),
            "y1": int(rect.y1),
            "x2": int(rect.x2),
            "y2": int(rect.y2),
        }

    @staticmethod
    def _safe_write_json(
        path: Path,
        payload: dict,
    ):
        temp = path.with_suffix(
            path.suffix + ".tmp"
        )
        temp.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp.replace(path)

    def _should_collect(
        self,
        confidence: float,
        margin: float,
    ) -> tuple[bool, str]:
        if (
            confidence
            < self.max_confidence
        ):
            return True, "low_confidence"

        if margin < self.max_margin:
            return True, "low_margin"

        self._candidate_counter += 1

        if (
            self.confident_sample_every > 0
            and self._candidate_counter
            % self.confident_sample_every
            == 0
        ):
            return True, "confident_audit"

        return False, ""

    def _save_identity_sample(
        self,
        *,
        event,
        row,
        crop: np.ndarray,
        side: str,
        team: str | None,
        hero_box,
        prediction,
        reason: str,
        source_frame_index: int,
        source_frame_count: int,
    ):
        if crop is None or crop.size == 0:
            return

        now = time.time()
        fingerprint = grayscale_fingerprint(
            crop,
            size=self.fingerprint_size,
        )

        if self._is_recent_duplicate(
            side,
            fingerprint,
            now,
        ):
            return

        self._ensure_dirs()

        sample_id = (
            f"{int(now * 1000)}"
            f"_track{int(event.track_id):04d}"
            f"_{side}"
        )

        portrait_name = (
            f"{sample_id}_portrait.png"
        )
        row_name = (
            f"{sample_id}_row.png"
        )
        metadata_name = (
            f"{sample_id}.json"
        )

        portrait_path = (
            self.pending_dir
            / portrait_name
        )
        row_path = (
            self.pending_dir
            / row_name
        )
        metadata_path = (
            self.pending_dir
            / metadata_name
        )

        cv2.imwrite(
            str(portrait_path),
            crop,
        )
        cv2.imwrite(
            str(row_path),
            row.crop,
        )

        (
            best_name,
            best_score,
            second_name,
            second_score,
            margin,
        ) = prediction

        payload = {
            "sample_id": sample_id,
            "created_unix": now,
            "event_timestamp": float(
                event.timestamp
            ),
            "track_id": int(
                event.track_id
            ),
            "side": side,
            "team": team,
            "reason": reason,
            "prediction": {
                "best": best_name,
                "best_confidence": float(
                    best_score
                ),
                "second": second_name,
                "second_confidence": float(
                    second_score
                ),
                "margin": float(
                    margin
                ),
            },
            "portrait_image": portrait_name,
            "row_image": row_name,
            "hero_box_local": (
                self._rect_payload(
                    hero_box
                )
            ),
            "row_bbox_game": (
                self._rect_payload(
                    row.bbox_game
                )
            ),
            "source_frame_index": int(
                source_frame_index
            ),
            "source_frame_count": int(
                source_frame_count
            ),
            "consensus_result": {
                "killer_hero": (
                    event.killer_hero
                ),
                "victim_hero": (
                    event.victim_hero
                ),
            },
        }

        self._safe_write_json(
            metadata_path,
            payload,
        )
        self._remember(
            side,
            fingerprint,
            now,
        )

    def _save_missing_box(
        self,
        *,
        event,
        row,
        missing_side: str,
    ):
        self._ensure_dirs()

        now = time.time()
        sample_id = (
            f"{int(now * 1000)}"
            f"_track{int(event.track_id):04d}"
            f"_{missing_side}_missing_box"
        )

        row_name = (
            f"{sample_id}_row.png"
        )
        metadata_name = (
            f"{sample_id}.json"
        )

        cv2.imwrite(
            str(
                self.localizer_failure_dir
                / row_name
            ),
            row.crop,
        )

        self._safe_write_json(
            self.localizer_failure_dir
            / metadata_name,
            {
                "sample_id": sample_id,
                "created_unix": now,
                "event_timestamp": float(
                    event.timestamp
                ),
                "track_id": int(
                    event.track_id
                ),
                "reason": "missing_hero_box",
                "missing_side": missing_side,
                "row_image": row_name,
                "row_bbox_game": (
                    self._rect_payload(
                        row.bbox_game
                    )
                ),
            },
        )

    def collect(
        self,
        event,
        rows,
        parser,
    ):
        if (
            not self.enabled
            or not rows
        ):
            return

        classifier = (
            parser.hero_recognizer
            .trained_classifier
        )

        # Force the optional model to load before checking readiness.
        classifier._load()

        if not classifier.ready:
            return

        rows = list(rows)
        frame_count = len(rows)

        for side in (
            "killer",
            "victim",
        ):
            candidates = []

            for frame_index, row in enumerate(
                rows
            ):
                if side == "killer":
                    box = (
                        row.killer_hero_box_local
                    )
                else:
                    box = (
                        row.victim_hero_box_local
                    )

                if box is None:
                    continue

                killer_crop, victim_crop = (
                    parser._hero_crops_for_row(
                        row
                    )
                )
                crop = (
                    killer_crop
                    if side == "killer"
                    else victim_crop
                )

                if (
                    crop is None
                    or crop.size == 0
                ):
                    continue

                prediction = (
                    classifier.predict_details(
                        crop
                    )
                )

                if prediction[0] is None:
                    continue

                (
                    _best_name,
                    best_score,
                    _second_name,
                    _second_score,
                    margin,
                ) = prediction

                should_collect, reason = (
                    self._should_collect(
                        best_score,
                        margin,
                    )
                )

                if not should_collect:
                    continue

                # Lowest margin is usually the most informative ambiguous
                # example; confidence breaks ties.
                hardness = (
                    margin,
                    best_score,
                )
                candidates.append(
                    (
                        hardness,
                        row,
                        crop,
                        box,
                        prediction,
                        reason,
                        frame_index,
                    )
                )

            if candidates:
                candidates.sort(
                    key=lambda item: item[0]
                )
                (
                    _hardness,
                    row,
                    crop,
                    box,
                    prediction,
                    reason,
                    frame_index,
                ) = candidates[0]

                team = (
                    row.killer_team_hint
                    if side == "killer"
                    else row.victim_team_hint
                )

                self._save_identity_sample(
                    event=event,
                    row=row,
                    crop=crop,
                    side=side,
                    team=team,
                    hero_box=box,
                    prediction=prediction,
                    reason=reason,
                    source_frame_index=frame_index,
                    source_frame_count=frame_count,
                )
                continue

            # If there was no usable portrait box in any consensus frame,
            # preserve one row for later localizer review.
            has_any_box = any(
                (
                    row.killer_hero_box_local
                    if side == "killer"
                    else row.victim_hero_box_local
                )
                is not None
                for row in rows
            )

            if not has_any_box:
                self._save_missing_box(
                    event=event,
                    row=rows[-1],
                    missing_side=side,
                )
