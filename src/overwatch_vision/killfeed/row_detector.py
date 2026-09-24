from dataclasses import dataclass

import cv2
import numpy as np

from overwatch_vision.models import Rect


@dataclass(slots=True)
class PanelComponent:
    bbox: Rect
    team: str


@dataclass(slots=True)
class RowCandidate:
    bbox: Rect
    component_boxes: list[Rect]
    component_teams: list[str]
    score: float


class KillFeedRowDetector:
    """
    Detect kill-feed rows from the colored Overwatch nameplates.

    Red and blue masks are kept separate all the way through detection so
    downstream parsing does not have to guess a panel's team from hero art or
    text pixels.
    """

    def __init__(self, config):
        self.cfg = config["killfeed"]

    def _panel_masks(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        colors = self.cfg["color_panels"]

        red = colors["red"]
        red_a = cv2.inRange(
            hsv,
            np.array(
                [
                    int(red["hue_low_1"]),
                    int(red["min_saturation"]),
                    int(red["min_value"]),
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    int(red["hue_high_1"]),
                    255,
                    255,
                ],
                dtype=np.uint8,
            ),
        )
        red_b = cv2.inRange(
            hsv,
            np.array(
                [
                    int(red["hue_low_2"]),
                    int(red["min_saturation"]),
                    int(red["min_value"]),
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    int(red["hue_high_2"]),
                    255,
                    255,
                ],
                dtype=np.uint8,
            ),
        )
        red_mask = cv2.bitwise_or(red_a, red_b)

        blue = colors["blue"]
        blue_mask = cv2.inRange(
            hsv,
            np.array(
                [
                    int(blue["hue_low"]),
                    int(blue["min_saturation"]),
                    int(blue["min_value"]),
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    int(blue["hue_high"]),
                    255,
                    255,
                ],
                dtype=np.uint8,
            ),
        )

        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((5, 3), np.uint8)

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel_open,
        )
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_OPEN,
            kernel_open,
        )
        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
        )

        combined = cv2.bitwise_or(
            red_mask,
            blue_mask,
        )
        return red_mask, blue_mask, combined

    def _find_panel_components(self, mask, team):
        h, w = mask.shape[:2]
        roi_area = float(h * w)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = roi_area * float(
            self.cfg["min_panel_area_fraction"]
        )
        max_area = roi_area * float(
            self.cfg["max_panel_area_fraction"]
        )
        min_w = w * float(
            self.cfg["min_panel_width_fraction"]
        )
        min_h = h * float(
            self.cfg["min_panel_height_fraction"]
        )
        max_h = h * float(
            self.cfg["max_panel_height_fraction"]
        )
        min_aspect = float(
            self.cfg["min_panel_aspect_ratio"]
        )
        min_fill = float(
            self.cfg["min_panel_fill_ratio"]
        )

        components = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if not (min_area <= area <= max_area):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            if bw < min_w or not (min_h <= bh <= max_h):
                continue

            aspect = bw / max(1.0, float(bh))
            if aspect < min_aspect:
                continue

            fill_ratio = area / max(1.0, float(bw * bh))
            if fill_ratio < min_fill:
                continue

            components.append(
                PanelComponent(
                    bbox=Rect(
                        x1=x,
                        y1=y,
                        x2=x + bw,
                        y2=y + bh,
                    ),
                    team=team,
                )
            )

        return components

    def _group_components(self, components, roi_height):
        tolerance = roi_height * float(
            self.cfg["row_group_y_tolerance_fraction"]
        )

        groups = []

        for component in components:
            best_group = None
            best_distance = float("inf")

            for group in groups:
                mean_y = sum(
                    item.bbox.cy
                    for item in group
                ) / len(group)

                distance = abs(
                    component.bbox.cy - mean_y
                )

                if (
                    distance <= tolerance
                    and distance < best_distance
                ):
                    best_group = group
                    best_distance = distance

            if best_group is None:
                groups.append([component])
            else:
                best_group.append(component)

        return groups

    def detect(self, roi):
        if roi.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8), []

        h, w = roi.shape[:2]

        red_mask, blue_mask, combined = self._panel_masks(roi)

        components = (
            self._find_panel_components(red_mask, "red")
            + self._find_panel_components(blue_mask, "blue")
        )
        components.sort(key=lambda item: item.bbox.cy)

        groups = self._group_components(
            components,
            h,
        )

        candidates = []

        min_components = int(
            self.cfg.get("min_components_per_row", 2)
        )
        min_row_width = w * float(
            self.cfg["min_row_width_fraction"]
        )
        max_right_gap = w * float(
            self.cfg["max_right_gap_fraction"]
        )

        for group in groups:
            if len(group) < min_components:
                continue

            group.sort(key=lambda item: item.bbox.cx)
            boxes = [
                item.bbox
                for item in group
            ]
            teams = [
                item.team
                for item in group
            ]

            x1 = min(r.x1 for r in boxes)
            y1 = min(r.y1 for r in boxes)
            x2 = max(r.x2 for r in boxes)
            y2 = max(r.y2 for r in boxes)

            pad_x = max(4, int(0.015 * w))
            pad_y = max(2, int(0.025 * h))

            row = Rect(
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(w, x2 + pad_x),
                min(h, y2 + pad_y),
            )

            if row.width < min_row_width:
                continue

            right_gap = w - row.x2
            if right_gap > max_right_gap:
                continue

            right_anchor_score = 1.0 - min(
                1.0,
                right_gap / max(1.0, max_right_gap),
            )

            width_score = min(
                1.0,
                row.width / max(1.0, w * 0.45),
            )

            component_score = min(
                1.0,
                len(group) / 2.0,
            )

            score = (
                0.45 * right_anchor_score
                + 0.30 * width_score
                + 0.25 * component_score
            )

            candidates.append(
                RowCandidate(
                    bbox=row,
                    component_boxes=boxes,
                    component_teams=teams,
                    score=score,
                )
            )

        candidates.sort(
            key=lambda c: c.bbox.y1
        )

        return (
            candidates,
            combined,
            [item.bbox for item in components],
        )
