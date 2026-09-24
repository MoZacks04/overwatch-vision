from dataclasses import dataclass

import cv2
import numpy as np

from overwatch_vision.models import Rect


@dataclass(slots=True)
class PanelComponent:
    bbox: Rect
    team: str
    fill_ratio: float


@dataclass(slots=True)
class RowCandidate:
    bbox: Rect
    component_boxes: list[Rect]
    component_teams: list[str]
    killer_panel: Rect
    victim_panel: Rect
    killer_team: str
    victim_team: str
    score: float


class KillFeedRowDetector:
    """
    Detect kill-feed rows as a PAIR of opposite-team nameplates.

    Earlier versions grouped every red/blue component at a similar Y value.
    That let red/blue map geometry get absorbed into the same group and pull
    the green row box far to the left. This detector only accepts one red and
    one blue HUD panel that:
      - line up vertically,
      - have similar heights,
      - sit a plausible distance apart,
      - form a plausible total row width,
      - and end near the right edge of the kill-feed ROI.

    Only the winning two panels define the row rectangle.
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

        # Small horizontal closing reconnects team-color regions interrupted
        # by bright text/portrait details, but is intentionally too small to
        # bridge separate map objects.
        close_w = max(
            3,
            int(
                round(
                    roi.shape[1]
                    * float(
                        self.cfg.get(
                            "panel_close_width_fraction",
                            0.006,
                        )
                    )
                )
            ),
        )
        close_h = max(2, int(round(roi.shape[0] * 0.015)))

        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones(
            (close_h, close_w),
            np.uint8,
        )

        def clean(mask):
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel_open,
            )
            return cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel_close,
            )

        red_mask = clean(red_mask)
        blue_mask = clean(blue_mask)
        combined = cv2.bitwise_or(red_mask, blue_mask)

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
        max_w = w * float(
            self.cfg.get(
                "max_panel_width_fraction",
                0.48,
            )
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
        max_aspect = float(
            self.cfg.get(
                "max_panel_aspect_ratio",
                12.0,
            )
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

            if not (min_w <= bw <= max_w):
                continue

            if not (min_h <= bh <= max_h):
                continue

            aspect = bw / max(1.0, float(bh))
            if not (min_aspect <= aspect <= max_aspect):
                continue

            fill_ratio = area / max(
                1.0,
                float(bw * bh),
            )

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
                    fill_ratio=fill_ratio,
                )
            )

        return components

    @staticmethod
    def _vertical_overlap(a: Rect, b: Rect) -> float:
        overlap = max(
            0,
            min(a.y2, b.y2) - max(a.y1, b.y1),
        )
        return overlap / max(
            1.0,
            float(min(a.height, b.height)),
        )

    def _pair_candidate(self, a, b, roi_w, roi_h):
        if a.team == b.team:
            return None

        left, right = (
            (a, b)
            if a.bbox.cx <= b.bbox.cx
            else (b, a)
        )

        # The two nameplates should be distinct horizontal objects.
        if right.bbox.cx <= left.bbox.cx:
            return None

        avg_h = (
            left.bbox.height + right.bbox.height
        ) / 2.0

        if avg_h <= 1:
            return None

        height_ratio = (
            min(left.bbox.height, right.bbox.height)
            / max(
                1.0,
                float(
                    max(
                        left.bbox.height,
                        right.bbox.height,
                    )
                ),
            )
        )

        min_height_ratio = float(
            self.cfg.get(
                "pair_min_height_ratio",
                0.68,
            )
        )
        if height_ratio < min_height_ratio:
            return None

        y_delta_rows = abs(
            left.bbox.cy - right.bbox.cy
        ) / avg_h

        max_y_delta_rows = float(
            self.cfg.get(
                "pair_max_center_y_delta_rows",
                0.42,
            )
        )
        if y_delta_rows > max_y_delta_rows:
            return None

        overlap = self._vertical_overlap(
            left.bbox,
            right.bbox,
        )
        min_overlap = float(
            self.cfg.get(
                "pair_min_vertical_overlap",
                0.58,
            )
        )
        if overlap < min_overlap:
            return None

        gap = right.bbox.x1 - left.bbox.x2
        gap_rows = gap / avg_h

        min_gap_rows = float(
            self.cfg.get(
                "pair_min_gap_rows",
                -0.30,
            )
        )
        max_gap_rows = float(
            self.cfg.get(
                "pair_max_gap_rows",
                3.40,
            )
        )

        if not (
            min_gap_rows
            <= gap_rows
            <= max_gap_rows
        ):
            return None

        row_x1 = min(
            left.bbox.x1,
            right.bbox.x1,
        )
        row_x2 = max(
            left.bbox.x2,
            right.bbox.x2,
        )
        row_width = row_x2 - row_x1

        min_row_width = roi_w * float(
            self.cfg["min_row_width_fraction"]
        )
        max_row_width = roi_w * float(
            self.cfg.get(
                "max_row_width_fraction",
                0.82,
            )
        )

        if not (
            min_row_width
            <= row_width
            <= max_row_width
        ):
            return None

        right_gap = roi_w - row_x2
        max_right_gap = roi_w * float(
            self.cfg["max_right_gap_fraction"]
        )

        if right_gap > max_right_gap:
            return None

        # Favor the geometry that real kill-feed rows have: near-perfect Y
        # alignment, similar panel heights, tight right anchoring, and dense
        # rectangular team-color panels.
        right_anchor_score = 1.0 - min(
            1.0,
            right_gap / max(1.0, max_right_gap),
        )
        y_score = 1.0 - min(
            1.0,
            y_delta_rows / max(
                0.01,
                max_y_delta_rows,
            ),
        )
        gap_score = 1.0 - min(
            1.0,
            abs(gap_rows - 1.10) / 2.60,
        )
        fill_score = min(
            1.0,
            (
                left.fill_ratio
                + right.fill_ratio
            ) / 1.45,
        )

        score = (
            0.34 * right_anchor_score
            + 0.24 * y_score
            + 0.18 * height_ratio
            + 0.12 * gap_score
            + 0.12 * fill_score
        )

        min_pair_score = float(
            self.cfg.get(
                "min_pair_score",
                0.55,
            )
        )
        if score < min_pair_score:
            return None

        pad_x = max(
            2,
            int(
                round(
                    avg_h
                    * float(
                        self.cfg.get(
                            "row_padding_x_rows",
                            0.08,
                        )
                    )
                )
            ),
        )
        pad_y = max(
            1,
            int(
                round(
                    avg_h
                    * float(
                        self.cfg.get(
                            "row_padding_y_rows",
                            0.08,
                        )
                    )
                )
            ),
        )

        row = Rect(
            max(0, row_x1 - pad_x),
            max(
                0,
                min(
                    left.bbox.y1,
                    right.bbox.y1,
                ) - pad_y,
            ),
            min(
                roi_w,
                row_x2 + pad_x,
            ),
            min(
                roi_h,
                max(
                    left.bbox.y2,
                    right.bbox.y2,
                ) + pad_y,
            ),
        )

        return RowCandidate(
            bbox=row,
            component_boxes=[
                left.bbox,
                right.bbox,
            ],
            component_teams=[
                left.team,
                right.team,
            ],
            killer_panel=left.bbox,
            victim_panel=right.bbox,
            killer_team=left.team,
            victim_team=right.team,
            score=score,
        )

    def _select_non_overlapping_rows(self, candidates):
        # When a map object creates a second possible pair at the same Y,
        # retain only the best-scoring interpretation of that row.
        candidates = sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        )

        selected = []

        center_separation_rows = float(
            self.cfg.get(
                "row_suppression_center_distance_rows",
                0.62,
            )
        )

        for candidate in candidates:
            candidate_h = max(
                1.0,
                float(candidate.bbox.height),
            )

            conflict = False

            for existing in selected:
                existing_h = max(
                    1.0,
                    float(existing.bbox.height),
                )
                distance = abs(
                    candidate.bbox.cy
                    - existing.bbox.cy
                )
                threshold = (
                    min(candidate_h, existing_h)
                    * center_separation_rows
                )

                if distance < threshold:
                    conflict = True
                    break

            if not conflict:
                selected.append(candidate)

        selected.sort(
            key=lambda item: item.bbox.y1
        )
        return selected

    def detect(self, roi):
        if roi.size == 0:
            return (
                [],
                np.zeros((1, 1), dtype=np.uint8),
                [],
            )

        h, w = roi.shape[:2]

        red_mask, blue_mask, combined = (
            self._panel_masks(roi)
        )

        red_components = self._find_panel_components(
            red_mask,
            "red",
        )
        blue_components = self._find_panel_components(
            blue_mask,
            "blue",
        )

        candidates = []

        for red_component in red_components:
            for blue_component in blue_components:
                candidate = self._pair_candidate(
                    red_component,
                    blue_component,
                    w,
                    h,
                )

                if candidate is not None:
                    candidates.append(candidate)

        candidates = self._select_non_overlapping_rows(
            candidates
        )

        debug_components = [
            item.bbox
            for item in (
                red_components
                + blue_components
            )
        ]

        return (
            candidates,
            combined,
            debug_components,
        )
