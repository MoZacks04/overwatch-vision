import cv2
import numpy as np

from overwatch_vision.models import KillFeedRow, Rect
from overwatch_vision.killfeed.row_detector import KillFeedRowDetector
from overwatch_vision.killfeed.row_localizer import KillFeedRowLocalizer
from overwatch_vision.killfeed.row_normalizer import KillFeedRowNormalizer
from overwatch_vision.killfeed.row_verifier import KillFeedRowVerifier
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
        self.row_localizer = KillFeedRowLocalizer(config)
        self.row_localizer.warmup()
        self.row_verifier = KillFeedRowVerifier(config)
        self.normalizer = KillFeedRowNormalizer(
            width=int(kcfg["normalized_row_width"]),
            height=int(kcfg["normalized_row_height"]),
        )
        self.tracker = KillFeedTracker(config)

        self.last_debug = {
            "mask": None,
            "components": [],
            "proposals": [],
            "localizer_proposals": [],
            "verifier_rejections": [],
            "rows": [],
            "events": [],
        }

    def reset(self):
        self.tracker.reset()
        self.last_debug = {
            "mask": None,
            "components": [],
            "proposals": [],
            "localizer_proposals": [],
            "verifier_rejections": [],
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

    @staticmethod
    def _clip_rect(rect, width, height):
        return Rect(
            x1=max(0, min(width, rect.x1)),
            y1=max(0, min(height, rect.y1)),
            x2=max(0, min(width, rect.x2)),
            y2=max(0, min(height, rect.y2)),
        )

    @staticmethod
    def _portrait_texture_score(patch):
        if (
            patch is None
            or patch.size == 0
            or patch.shape[0] < 4
            or patch.shape[1] < 4
        ):
            return -1.0

        gray = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2GRAY,
        )

        texture = min(
            1.0,
            float(np.std(gray)) / 62.0,
        )

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
        edge_energy = np.sqrt(
            gx * gx + gy * gy
        )
        edge_score = min(
            1.0,
            float(np.mean(edge_energy)) / 72.0,
        )

        return (
            0.62 * texture
            + 0.38 * edge_score
        )

    def _refine_hero_box(
        self,
        row_image,
        base_box,
    ):
        """
        Nudge a geometric hero box a few pixels toward the most portrait-like
        local patch. The search is intentionally small so it fixes contour
        boundary drift without jumping onto the action icon or nearby scenery.
        """
        cfg = self.config.get(
            "killfeed_parse",
            {},
        )

        if not bool(
            cfg.get(
                "refine_hero_boxes",
                True,
            )
        ):
            return base_box

        h, w = row_image.shape[:2]
        size = max(
            4,
            min(
                base_box.width,
                base_box.height,
            ),
        )

        search_x = max(
            1,
            int(
                round(
                    size
                    * float(
                        cfg.get(
                            "hero_refine_search_x_rows",
                            0.24,
                        )
                    )
                )
            ),
        )
        search_y = max(
            1,
            int(
                round(
                    size
                    * float(
                        cfg.get(
                            "hero_refine_search_y_rows",
                            0.10,
                        )
                    )
                )
            ),
        )
        step = max(
            1,
            int(
                round(
                    size
                    * float(
                        cfg.get(
                            "hero_refine_step_rows",
                            0.06,
                        )
                    )
                )
            ),
        )

        best_box = base_box
        best_score = -1.0

        for dy in range(
            -search_y,
            search_y + 1,
            step,
        ):
            for dx in range(
                -search_x,
                search_x + 1,
                step,
            ):
                x1 = base_box.x1 + dx
                y1 = base_box.y1 + dy
                x2 = x1 + size
                y2 = y1 + size

                if (
                    x1 < 0
                    or y1 < 0
                    or x2 > w
                    or y2 > h
                ):
                    continue

                patch = row_image[
                    y1:y2,
                    x1:x2,
                ]
                visual = self._portrait_texture_score(
                    patch
                )

                distance = (
                    abs(dx) / max(1.0, search_x)
                    + abs(dy) / max(1.0, search_y)
                ) / 2.0

                # Stay close to the geometrically expected square unless a
                # nearby patch is clearly more portrait-like.
                score = (
                    visual
                    - 0.10 * distance
                )

                if score > best_score:
                    best_score = score
                    best_box = Rect(
                        x1,
                        y1,
                        x2,
                        y2,
                    )

        return best_box

    def _hero_boxes_from_panels(
        self,
        killer_panel,
        victim_panel,
        row_width,
        row_height,
        row_image=None,
    ):
        kcfg = self.config.get("killfeed_parse", {})

        reference_h = max(
            1.0,
            (
                killer_panel.height
                + victim_panel.height
            ) / 2.0,
        )

        size = max(
            4,
            int(
                round(
                    reference_h
                    * float(
                        kcfg.get(
                            "hero_icon_size_rows",
                            1.08,
                        )
                    )
                )
            ),
        )

        inner_offset = int(
            round(
                reference_h
                * float(
                    kcfg.get(
                        "hero_inner_edge_offset_rows",
                        0.04,
                    )
                )
            )
        )

        killer_cy = int(round(killer_panel.cy))
        victim_cy = int(round(victim_panel.cy))

        killer_x2 = killer_panel.x2 + inner_offset
        killer_x1 = killer_x2 - size

        victim_x1 = victim_panel.x1 - inner_offset
        victim_x2 = victim_x1 + size

        killer_y1 = killer_cy - size // 2
        victim_y1 = victim_cy - size // 2

        killer_box = self._clip_rect(
            Rect(
                killer_x1,
                killer_y1,
                killer_x2,
                killer_y1 + size,
            ),
            row_width,
            row_height,
        )

        victim_box = self._clip_rect(
            Rect(
                victim_x1,
                victim_y1,
                victim_x2,
                victim_y1 + size,
            ),
            row_width,
            row_height,
        )

        if row_image is not None:
            killer_box = self._refine_hero_box(
                row_image,
                killer_box,
            )
            victim_box = self._refine_hero_box(
                row_image,
                victim_box,
            )

        return killer_box, victim_box

    @staticmethod
    def _iou(a: Rect, b: Rect) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection <= 0:
            return 0.0

        union = (
            a.width * a.height
            + b.width * b.height
            - intersection
        )
        return intersection / max(1.0, float(union))

    def _localizer_panel_geometry(self, crop):
        """
        Recover red/blue nameplate geometry inside an already-localized row.

        Unlike the first-stage proposal detector, this does not apply global
        row geometry gates. Once the learned localizer says the crop is a row,
        we only need the strongest red and blue panel-like regions so the
        existing hero/name parser can keep working.
        """
        if crop is None or crop.size == 0:
            return None

        red_mask, blue_mask, _ = self.row_detector._panel_masks(crop)
        h, w = crop.shape[:2]

        def best_panel(mask):
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            best = None
            best_score = -1.0

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area <= 0:
                    continue

                x, y, bw, bh = cv2.boundingRect(contour)

                if bh < max(4, int(round(h * 0.34))):
                    continue
                if bw < max(8, int(round(w * 0.08))):
                    continue

                fill = area / max(1.0, float(bw * bh))
                center_bonus = 1.0 - min(
                    1.0,
                    abs((y + bh / 2.0) - h / 2.0)
                    / max(1.0, h / 2.0),
                )
                score = (
                    area
                    * (0.65 + 0.35 * fill)
                    * (0.80 + 0.20 * center_bonus)
                )

                if score > best_score:
                    best_score = score
                    best = Rect(
                        x1=x,
                        y1=y,
                        x2=x + bw,
                        y2=y + bh,
                    )

            return best

        red_box = best_panel(red_mask)
        blue_box = best_panel(blue_mask)

        if red_box is None or blue_box is None:
            return None

        if red_box.cx <= blue_box.cx:
            return (
                red_box,
                blue_box,
                "red",
                "blue",
            )

        return (
            blue_box,
            red_box,
            "blue",
            "red",
        )

    def _localized_row(
        self,
        roi_image,
        roi_rect,
        box,
        confidence,
        fingerprint_size,
    ):
        crop = roi_image[
            box.y1:box.y2,
            box.x1:box.x2,
        ]

        if crop is None or crop.size == 0:
            return None

        normalized = self.normalizer.normalize(crop)
        fingerprint = grayscale_fingerprint(
            normalized,
            size=fingerprint_size,
        )

        component_boxes = []
        component_teams = []
        killer_panel = None
        victim_panel = None
        killer_team = None
        victim_team = None
        killer_hero_box = None
        victim_hero_box = None

        geometry = self._localizer_panel_geometry(crop)

        if geometry is not None:
            (
                killer_panel,
                victim_panel,
                killer_team,
                victim_team,
            ) = geometry

            component_boxes = [
                killer_panel,
                victim_panel,
            ]
            component_teams = [
                killer_team,
                victim_team,
            ]

            (
                killer_hero_box,
                victim_hero_box,
            ) = self._hero_boxes_from_panels(
                killer_panel,
                victim_panel,
                crop.shape[1],
                crop.shape[0],
                crop,
            )

        return KillFeedRow(
            bbox_roi=box,
            bbox_game=translate_rect(
                box,
                roi_rect.x1,
                roi_rect.y1,
            ),
            crop=crop,
            normalized=normalized,
            fingerprint=fingerprint,
            component_boxes_local=component_boxes,
            component_teams_local=component_teams,
            killer_panel_local=killer_panel,
            victim_panel_local=victim_panel,
            killer_team_hint=killer_team,
            victim_team_hint=victim_team,
            killer_hero_box_local=killer_hero_box,
            victim_hero_box_local=victim_hero_box,
            score=float(confidence),
        )

    def get_track_row(self, track_id):
        for track in self.tracker.tracks:
            if track.track_id == track_id:
                return track.row
        return None

    def get_track_rows(self, track_id, limit=5):
        for track in self.tracker.tracks:
            if track.track_id == track_id:
                history = list(track.row_history)
                if not history:
                    history = [track.row]
                return history[-max(1, int(limit)):]
        return []

    def process(self, roi_image, roi_rect, timestamp):
        candidates, mask, components = self.row_detector.detect(
            roi_image
        )
        localizer_detections = self.row_localizer.detect(
            roi_image
        )

        fingerprint_size = int(
            self.config["tracking"]["fingerprint_size"]
        )

        rows = []
        verifier_rejections = []

        for candidate in candidates:
            box = candidate.bbox

            crop = roi_image[
                box.y1:box.y2,
                box.x1:box.x2,
            ]

            verified, verifier_probability = (
                self.row_verifier.accept(crop)
            )
            if not verified:
                verifier_rejections.append(
                    {
                        "bbox_roi": box,
                        "probability": verifier_probability,
                    }
                )
                continue

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

            killer_panel_local = self._component_to_local(
                candidate.killer_panel,
                box,
            )
            victim_panel_local = self._component_to_local(
                candidate.victim_panel,
                box,
            )

            killer_hero_box, victim_hero_box = (
                self._hero_boxes_from_panels(
                    killer_panel_local,
                    victim_panel_local,
                    crop.shape[1],
                    crop.shape[0],
                    crop,
                )
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
                    killer_panel_local=killer_panel_local,
                    victim_panel_local=victim_panel_local,
                    killer_team_hint=candidate.killer_team,
                    victim_team_hint=candidate.victim_team,
                    killer_hero_box_local=killer_hero_box,
                    victim_hero_box_local=victim_hero_box,
                    score=(
                        candidate.score
                        if verifier_probability is None
                        else (
                            0.35 * candidate.score
                            + 0.65 * verifier_probability
                        )
                    ),
                )
            )

        # Learned full-ROI detections are allowed to rescue rows that the
        # color/geometry proposal path missed. If a learned box overlaps an
        # already accepted legacy row, keep the richer legacy row metadata.
        for localizer_box, localizer_confidence in localizer_detections:
            if any(
                self._iou(localizer_box, row.bbox_roi) >= 0.45
                for row in rows
            ):
                continue

            learned_row = self._localized_row(
                roi_image=roi_image,
                roi_rect=roi_rect,
                box=localizer_box,
                confidence=localizer_confidence,
                fingerprint_size=fingerprint_size,
            )

            if learned_row is not None:
                rows.append(learned_row)

        # Do not display a verifier rejection if the learned localizer rescued
        # the same visual row.
        verifier_rejections = [
            item
            for item in verifier_rejections
            if not any(
                self._iou(
                    item["bbox_roi"],
                    row.bbox_roi,
                )
                >= 0.45
                for row in rows
            )
        ]

        events = self.tracker.update(
            rows=rows,
            timestamp=timestamp,
            roi_height=roi_image.shape[0],
        )

        self.last_debug = {
            "mask": mask,
            "components": components,
            "proposals": [
                candidate.bbox
                for candidate in candidates
            ],
            "localizer_proposals": [
                box
                for box, _ in localizer_detections
            ],
            "verifier_rejections": verifier_rejections,
            "rows": rows,
            "events": events,
        }

        return rows, events
