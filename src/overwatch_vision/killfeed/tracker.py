from overwatch_vision.models import KillFeedEvent, KillFeedTrack
from overwatch_vision.utils.image_ops import fingerprint_similarity


class KillFeedTracker:
    def __init__(self, config):
        cfg = config["tracking"]

        self.confirmation_frames = int(cfg["confirmation_frames"])
        self.expire_after = int(cfg["missing_frames_before_expire"])
        self.max_vertical_shift_fraction = float(cfg["max_vertical_shift_fraction"])
        self.match_threshold = float(cfg["fingerprint_match_threshold"])

        self.tracks = []
        self.next_track_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1

    def _match_score(self, old, new, roi_height):
        visual = fingerprint_similarity(old.fingerprint, new.fingerprint)

        vertical_distance = abs(old.bbox_roi.cy - new.bbox_roi.cy)
        max_shift = max(1.0, roi_height * self.max_vertical_shift_fraction)

        vertical = 1.0 - min(1.0, vertical_distance / max_shift)

        return 0.85 * visual + 0.15 * vertical

    def update(self, rows, timestamp, roi_height):
        events = []

        unmatched_row_indices = set(range(len(rows)))
        unmatched_track_indices = set(range(len(self.tracks)))

        candidate_pairs = []

        for ti, track in enumerate(self.tracks):
            for ri, row in enumerate(rows):
                score = self._match_score(track.row, row, roi_height)

                if score >= self.match_threshold:
                    candidate_pairs.append((score, ti, ri))

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

            if not track.confirmed and track.age_frames >= self.confirmation_frames:
                track.confirmed = True

            if track.confirmed and not track.emitted:
                track.emitted = True
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
            if track.missing_frames <= self.expire_after
        ]

        return events
