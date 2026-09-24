from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from overwatch_vision.models import KillFeedEvent, KillFeedTrack
from overwatch_vision.utils.image_ops import fingerprint_similarity


@dataclass(slots=True)
class RecentEmission:
    timestamp: float
    fingerprint: np.ndarray
    dhash: np.ndarray


class KillFeedTracker:
    def __init__(self, config):
        cfg = config["tracking"]

        self.confirmation_frames = int(cfg["confirmation_frames"])
        self.expire_after = int(cfg["missing_frames_before_expire"])
        self.max_vertical_shift_fraction = float(
            cfg["max_vertical_shift_fraction"]
        )
        self.match_threshold = float(
            cfg["fingerprint_match_threshold"]
        )

        # A row can briefly fade/animate enough to lose its original track
        # and then be recreated as a new track. Keep a short visual history
        # of emitted rows so the same visible kill-feed entry is not spoken
        # twice.
        self.dedupe_seconds = float(
            cfg.get("dedupe_seconds", 3.0)
        )
        self.dedupe_similarity = float(
            cfg.get("dedupe_similarity", 0.86)
        )
        self.dedupe_hash_similarity = float(
            cfg.get("dedupe_hash_similarity", 0.90)
        )

        self.tracks = []
        self.next_track_id = 1
        self.recent_emissions: list[RecentEmission] = []

    def reset(self):
        self.tracks.clear()
        self.recent_emissions.clear()
        self.next_track_id = 1

    def _match_score(self, old, new, roi_height):
        visual = fingerprint_similarity(
            old.fingerprint,
            new.fingerprint,
        )

        vertical_distance = abs(
            old.bbox_roi.cy - new.bbox_roi.cy
        )
        max_shift = max(
            1.0,
            roi_height * self.max_vertical_shift_fraction,
        )

        vertical = 1.0 - min(
            1.0,
            vertical_distance / max_shift,
        )

        return 0.85 * visual + 0.15 * vertical

    def _prune_recent(self, timestamp: float):
        cutoff = timestamp - self.dedupe_seconds
        self.recent_emissions = [
            item
            for item in self.recent_emissions
            if item.timestamp >= cutoff
        ]

    @staticmethod
    def _dhash(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros(128, dtype=np.bool_)

        if image.ndim == 3:
            # Avoid another OpenCV dependency here; simple channel mean is
            # sufficient for a perceptual difference hash.
            gray = image.mean(axis=2)
        else:
            gray = image

        # Nearest-neighbour index sampling keeps this tiny and deterministic.
        ys = np.linspace(
            0,
            gray.shape[0] - 1,
            8,
        ).astype(np.int32)
        xs = np.linspace(
            0,
            gray.shape[1] - 1,
            17,
        ).astype(np.int32)
        small = gray[np.ix_(ys, xs)]
        return (small[:, 1:] > small[:, :-1]).reshape(-1)

    def _is_recent_duplicate(self, row, timestamp: float) -> bool:
        self._prune_recent(timestamp)

        row_hash = self._dhash(row.normalized)

        for item in self.recent_emissions:
            similarity = fingerprint_similarity(
                item.fingerprint,
                row.fingerprint,
            )
            hash_similarity = float(
                np.mean(item.dhash == row_hash)
            )

            if (
                similarity >= self.dedupe_similarity
                or hash_similarity
                >= self.dedupe_hash_similarity
            ):
                return True

        return False

    def _remember_emission(self, row, timestamp: float):
        self.recent_emissions.append(
            RecentEmission(
                timestamp=timestamp,
                fingerprint=row.fingerprint.copy(),
                dhash=self._dhash(row.normalized),
            )
        )
        self._prune_recent(timestamp)

    def update(self, rows, timestamp, roi_height):
        events = []

        unmatched_row_indices = set(
            range(len(rows))
        )
        unmatched_track_indices = set(
            range(len(self.tracks))
        )

        candidate_pairs = []

        for ti, track in enumerate(self.tracks):
            for ri, row in enumerate(rows):
                score = self._match_score(
                    track.row,
                    row,
                    roi_height,
                )

                if score >= self.match_threshold:
                    candidate_pairs.append(
                        (score, ti, ri)
                    )

        candidate_pairs.sort(reverse=True)

        for score, ti, ri in candidate_pairs:
            if ti not in unmatched_track_indices:
                continue
            if ri not in unmatched_row_indices:
                continue

            track = self.tracks[ti]
            track.row = rows[ri]
            track.last_seen = timestamp
            track.age_frames += 1
            track.missing_frames = 0

            unmatched_track_indices.remove(ti)
            unmatched_row_indices.remove(ri)

            if (
                not track.confirmed
                and track.age_frames
                >= self.confirmation_frames
            ):
                track.confirmed = True

            if track.confirmed and not track.emitted:
                # Mark this track as handled either way. If it visually
                # duplicates a row emitted moments ago, silently suppress it.
                track.emitted = True

                if self._is_recent_duplicate(
                    track.row,
                    timestamp,
                ):
                    continue

                self._remember_emission(
                    track.row,
                    timestamp,
                )

                events.append(
                    KillFeedEvent(
                        event_type="new_row",
                        track_id=track.track_id,
                        timestamp=timestamp,
                        confidence=track.row.score,
                    )
                )

        for ti in unmatched_track_indices:
            self.tracks[ti].missing_frames += 1

        for ri in unmatched_row_indices:
            self.tracks.append(
                KillFeedTrack(
                    track_id=self.next_track_id,
                    row=rows[ri],
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
            )
            self.next_track_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.missing_frames
            <= self.expire_after
        ]

        return events
