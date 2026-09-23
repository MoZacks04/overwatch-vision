from dataclasses import dataclass

import cv2
import numpy as np

from overwatch_vision.models import Rect


@dataclass(slots=True)
class RowCandidate:
    bbox: Rect
    component_boxes: list
    score: float


class KillFeedRowDetector:
    def __init__(self, config):
        self.cfg = config["killfeed"]

    def _hud_mask(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        min_s = int(self.cfg["min_saturation"])
        min_v = int(self.cfg["min_value"])

        mask = cv2.inRange(
            hsv,
            np.array([0, min_s, min_v], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )

        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((5, 3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        return mask

    def detect(self, roi):
        if roi.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8), []

        h, w = roi.shape[:2]
        roi_area = float(h * w)

        mask = self._hud_mask(roi)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = roi_area * float(self.cfg["min_component_area_fraction"])
        max_area = roi_area * float(self.cfg["max_component_area_fraction"])

        components = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if not (min_area <= area <= max_area):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            if bw < 4 or bh < 4:
                continue

            components.append(Rect(x, y, x + bw, y + bh))

        components.sort(key=lambda r: r.cy)

        tolerance = h * float(self.cfg["row_group_y_tolerance_fraction"])
        groups = []

        for component in components:
            best_group = None
            best_distance = float("inf")

            for group in groups:
                mean_y = sum(r.cy for r in group) / len(group)
                distance = abs(component.cy - mean_y)

                if distance <= tolerance and distance < best_distance:
                    best_group = group
                    best_distance = distance

            if best_group is None:
                groups.append([component])
            else:
                best_group.append(component)

        candidates = []

        min_h = h * float(self.cfg["min_row_height_fraction"])
        max_h = h * float(self.cfg["max_row_height_fraction"])
        min_w = w * float(self.cfg["min_row_width_fraction"])

        for group in groups:
            x1 = min(r.x1 for r in group)
            y1 = min(r.y1 for r in group)
            x2 = max(r.x2 for r in group)
            y2 = max(r.y2 for r in group)

            pad_x = max(3, int(0.01 * w))
            pad_y = max(2, int(0.01 * h))

            row = Rect(
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(w, x2 + pad_x),
                min(h, y2 + pad_y),
            )

            if not (min_h <= row.height <= max_h):
                continue

            if row.width < min_w:
                continue

            right_anchor_score = 1.0 - min(
                1.0,
                max(0.0, (w - row.x2) / max(1.0, w * 0.30)),
            )

            component_score = min(1.0, len(group) / 4.0)

            score = 0.65 * right_anchor_score + 0.35 * component_score

            if score >= 0.40:
                candidates.append(
                    RowCandidate(
                        bbox=row,
                        component_boxes=group,
                        score=score,
                    )
                )

        candidates.sort(key=lambda c: c.bbox.y1)

        return candidates, mask, components
