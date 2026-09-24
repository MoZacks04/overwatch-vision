from overwatch_vision.models import KillFeedRow, Rect
from overwatch_vision.killfeed.row_detector import KillFeedRowDetector
from overwatch_vision.killfeed.row_normalizer import KillFeedRowNormalizer
from overwatch_vision.killfeed.tracker import KillFeedTracker
from overwatch_vision.utils.geometry import translate_rect
from overwatch_vision.utils.image_ops import grayscale_fingerprint


class KillFeedDetector:
    """
    Lightweight real-time kill-feed detector.

    This class deliberately does not perform OCR or hero recognition. It only
    finds rows, tracks them, and emits raw new-row events. Expensive parsing is
    handled by AsyncKillFeedParser on a background thread.
    """

    def __init__(self, config):
        self.config = config
        kcfg = config["killfeed"]

        self.row_detector = KillFeedRowDetector(config)
        self.normalizer = KillFeedRowNormalizer(
            width=int(kcfg["normalized_row_width"]),
            height=int(kcfg["normalized_row_height"]),
        )
        self.tracker = KillFeedTracker(config)

        self.last_debug = {
            "mask": None,
            "components": [],
            "rows": [],
            "events": [],
        }

    def reset(self):
        self.tracker.reset()
        self.last_debug = {
            "mask": None,
            "components": [],
            "rows": [],
            "events": [],
        }

    @staticmethod
    def _component_to_local(component, row_box):
        return Rect(
            x1=max(0, component.x1 - row_box.x1),
            y1=max(0, component.y1 - row_box.y1),
            x2=max(0, component.x2 - row_box.x1),
            y2=max(0, component.y2 - row_box.y1),
        )

    def get_track_row(self, track_id):
        for track in self.tracker.tracks:
            if track.track_id == track_id:
                return track.row
        return None

    def process(self, roi_image, roi_rect, timestamp):
        candidates, mask, components = self.row_detector.detect(
            roi_image
        )

        fingerprint_size = int(
            self.config["tracking"]["fingerprint_size"]
        )

        rows = []

        for candidate in candidates:
            box = candidate.bbox

            crop = roi_image[
                box.y1:box.y2,
                box.x1:box.x2,
            ]

            normalized = self.normalizer.normalize(crop)

            fingerprint = grayscale_fingerprint(
                normalized,
                size=fingerprint_size,
            )

            local_components = [
                self._component_to_local(component, box)
                for component in candidate.component_boxes
            ]
            local_teams = list(
                candidate.component_teams
            )

            rows.append(
                KillFeedRow(
                    bbox_roi=box,
                    bbox_game=translate_rect(
                        box,
                        roi_rect.x1,
                        roi_rect.y1,
                    ),
                    crop=crop,
                    normalized=normalized,
                    fingerprint=fingerprint,
                    component_boxes_local=local_components,
                    component_teams_local=local_teams,
                    score=candidate.score,
                )
            )

        events = self.tracker.update(
            rows=rows,
            timestamp=timestamp,
            roi_height=roi_image.shape[0],
        )

        self.last_debug = {
            "mask": mask,
            "components": components,
            "rows": rows,
            "events": events,
        }

        return rows, events
