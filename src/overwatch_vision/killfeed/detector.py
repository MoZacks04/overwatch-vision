from overwatch_vision.models import KillFeedRow, Rect
from overwatch_vision.killfeed.parser import KillFeedParser
from overwatch_vision.killfeed.row_detector import KillFeedRowDetector
from overwatch_vision.killfeed.row_normalizer import KillFeedRowNormalizer
from overwatch_vision.killfeed.tracker import KillFeedTracker
from overwatch_vision.utils.geometry import translate_rect
from overwatch_vision.utils.image_ops import grayscale_fingerprint


class KillFeedDetector:
    def __init__(self, config, ocr):
        self.config = config
        kcfg = config["killfeed"]

        self.row_detector = KillFeedRowDetector(config)
        self.normalizer = KillFeedRowNormalizer(
            width=int(kcfg["normalized_row_width"]),
            height=int(kcfg["normalized_row_height"]),
        )
        self.tracker = KillFeedTracker(config)
        self.parser = KillFeedParser(config, ocr)
        self.parser.warmup_async()

        self.last_debug = {
            "mask": None,
            "components": [],
            "rows": [],
            "events": [],
        }

    def reset(self):
        self.tracker.reset()

    @staticmethod
    def _component_to_local(component, row_box):
        return Rect(
            x1=max(0, component.x1 - row_box.x1),
            y1=max(0, component.y1 - row_box.y1),
            x2=max(0, component.x2 - row_box.x1),
            y2=max(0, component.y2 - row_box.y1),
        )

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
                    score=candidate.score,
                )
            )

        events = self.tracker.update(
            rows=rows,
            timestamp=timestamp,
            roi_height=roi_image.shape[0],
        )

        tracks_by_id = {
            track.track_id: track
            for track in self.tracker.tracks
        }

        for event in events:
            track = tracks_by_id.get(event.track_id)
            if track is None:
                continue

            parsed = self.parser.parse(track.row)

            event.killer_name = parsed.killer_name
            event.victim_name = parsed.victim_name
            event.killer_hero = parsed.killer_hero
            event.victim_hero = parsed.victim_hero
            event.killer_team = parsed.killer_team
            event.victim_team = parsed.victim_team
            event.ability = parsed.ability
            event.critical = parsed.critical

            event.parse_confidence = parsed.confidence
            event.killer_hero_confidence = (
                parsed.killer_hero_confidence
            )
            event.victim_hero_confidence = (
                parsed.victim_hero_confidence
            )
            event.killer_name_confidence = (
                parsed.killer_name_confidence
            )
            event.victim_name_confidence = (
                parsed.victim_name_confidence
            )

        self.last_debug = {
            "mask": mask,
            "components": components,
            "rows": rows,
            "events": events,
        }

        return rows, events
