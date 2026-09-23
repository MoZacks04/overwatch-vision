from overwatch_vision.models import KillFeedRow
from overwatch_vision.killfeed.row_detector import KillFeedRowDetector
from overwatch_vision.killfeed.row_normalizer import KillFeedRowNormalizer
from overwatch_vision.killfeed.tracker import KillFeedTracker
from overwatch_vision.utils.geometry import translate_rect
from overwatch_vision.utils.image_ops import grayscale_fingerprint


class KillFeedDetector:
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
        }

    def reset(self):
        self.tracker.reset()

    def process(self, roi_image, roi_rect, timestamp):
        candidates, mask, components = self.row_detector.detect(roi_image)

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
        }

        return rows, events
