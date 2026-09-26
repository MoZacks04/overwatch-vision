from __future__ import annotations

import time

import cv2
import numpy as np


class DebugView:
    def __init__(self, config: dict):
        self.cfg = config["debug"]
        self.show_details = bool(self.cfg.get("enabled", True))

        self.last_event_time = -1.0
        self.last_event_text = ""

    def toggle(self):
        self.show_details = not self.show_details

    def notify_event(self, text: str):
        self.last_event_time = time.monotonic()
        self.last_event_text = text

    def _hero_icon_boxes(self, row):
        """
        Return the exact hero portrait boxes used by parsing.

        New rows carry explicit killer/victim hero geometry from the detector.
        Keeping debug rendering tied to those same Rects means the yellow K/V
        boxes are a truthful preview of the recognizer input.
        """
        killer = getattr(
            row,
            "killer_hero_box_local",
            None,
        )
        victim = getattr(
            row,
            "victim_hero_box_local",
            None,
        )

        if killer is not None and victim is not None:
            return [
                (
                    "K",
                    (
                        killer.x1,
                        killer.y1,
                        killer.x2,
                        killer.y2,
                    ),
                ),
                (
                    "V",
                    (
                        victim.x1,
                        victim.y1,
                        victim.x2,
                        victim.y2,
                    ),
                ),
            ]

        # Compatibility fallback for tracks created before explicit geometry
        # was added.
        components = list(
            getattr(row, "component_boxes_local", [])
            or []
        )
        components.sort(key=lambda box: box.cx)

        if len(components) < 2:
            return []

        killer_panel = components[0]
        victim_panel = components[-1]

        size = max(
            1,
            int(
                round(
                    (
                        killer_panel.height
                        + victim_panel.height
                    )
                    / 2.0
                )
            ),
        )

        killer = (
            max(
                killer_panel.x1,
                killer_panel.x2 - size,
            ),
            killer_panel.y1,
            killer_panel.x2,
            killer_panel.y2,
        )
        victim = (
            victim_panel.x1,
            victim_panel.y1,
            min(
                victim_panel.x2,
                victim_panel.x1 + size,
            ),
            victim_panel.y2,
        )

        return [
            ("K", killer),
            ("V", victim),
        ]

    def _draw_hero_boxes(
        self,
        output,
        row,
        origin_x,
        origin_y,
        thickness=2,
    ):
        if not self.cfg.get(
            "draw_hero_icon_boxes",
            True,
        ):
            return

        for label, (x1, y1, x2, y2) in self._hero_icon_boxes(row):
            ax1 = origin_x + x1
            ay1 = origin_y + y1
            ax2 = origin_x + x2
            ay2 = origin_y + y2

            cv2.rectangle(
                output,
                (ax1, ay1),
                (ax2, ay2),
                (0, 255, 255),
                thickness,
            )
            cv2.putText(
                output,
                label,
                (ax1, max(14, ay1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _event_banner_active(self):
        seconds = float(
            self.cfg.get("show_event_banner_seconds", 2.25)
        )
        return (
            self.last_event_time >= 0
            and time.monotonic() - self.last_event_time <= seconds
        )

    @staticmethod
    def _status_text(team_status_state):
        if team_status_state is None:
            return "TEAM STATUS: reading..."

        friendly = team_status_state.friendly_alive
        enemy = team_status_state.enemy_alive

        if friendly is None and enemy is None:
            return "TEAM STATUS: reading..."

        friendly_text = "?" if friendly is None else str(friendly)
        enemy_text = "?" if enemy is None else str(enemy)
        return f"TEAM STATUS: {friendly_text} vs {enemy_text}"

    def draw_full_view(
        self,
        frame,
        roi_rect,
        detector_debug,
        active_tracks,
        fps,
        team_status_rect=None,
        team_status_state=None,
        parser_pending=0,
    ):
        output = frame.copy()

        if self.show_details and self.cfg.get(
            "draw_search_region",
            True,
        ):
            cv2.rectangle(
                output,
                (roi_rect.x1, roi_rect.y1),
                (roi_rect.x2, roi_rect.y2),
                (255, 255, 255),
                2,
            )

        if (
            self.show_details
            and team_status_rect is not None
            and self.cfg.get("draw_team_status_region", True)
        ):
            cv2.rectangle(
                output,
                (team_status_rect.x1, team_status_rect.y1),
                (team_status_rect.x2, team_status_rect.y2),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                output,
                self._status_text(team_status_state),
                (
                    max(10, team_status_rect.x1 - 90),
                    max(20, team_status_rect.y2 + 22),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.show_details and self.cfg.get(
            "draw_rows",
            True,
        ):
            for box in detector_debug.get(
                "localizer_proposals",
                [],
            ):
                x1 = roi_rect.x1 + box.x1
                y1 = roi_rect.y1 + box.y1
                x2 = roi_rect.x1 + box.x2
                y2 = roi_rect.y1 + box.y2

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    1,
                )

            for rejected in detector_debug.get(
                "verifier_rejections",
                [],
            ):
                box = rejected["bbox_roi"]
                probability = rejected.get("probability")

                x1 = roi_rect.x1 + box.x1
                y1 = roi_rect.y1 + box.y1
                x2 = roi_rect.x1 + box.x2
                y2 = roi_rect.y1 + box.y2

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 80, 255),
                    2,
                )

                label = (
                    "REJECT"
                    if probability is None
                    else f"REJECT {probability:.2f}"
                )
                cv2.putText(
                    output,
                    label,
                    (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (0, 80, 255),
                    1,
                    cv2.LINE_AA,
                )

            for row in detector_debug["rows"]:
                box = row.bbox_game

                cv2.rectangle(
                    output,
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    (0, 255, 0),
                    3,
                )

                self._draw_hero_boxes(
                    output,
                    row,
                    row.bbox_game.x1,
                    row.bbox_game.y1,
                    thickness=2,
                )

        # A row detector only runs at ~10 Hz, while the preview runs at
        # ~30 FPS. Draw K/V boxes from confirmed tracks as well so they do
        # not disappear on the two frames between detector updates.
        if self.show_details:
            for track in active_tracks:
                if not track.confirmed:
                    continue

                self._draw_hero_boxes(
                    output,
                    track.row,
                    track.row.bbox_game.x1,
                    track.row.bbox_game.y1,
                    thickness=2,
                )

        confirmed = sum(
            1 for track in active_tracks
            if track.confirmed
        )

        status = (
            f"FPS {fps:.1f} | "
            f"tracks {len(active_tracks)} | "
            f"confirmed {confirmed} | "
            f"parse queue {parser_pending}"
        )

        cv2.putText(
            output,
            status,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self._event_banner_active():
            banner_width = min(
                output.shape[1] - 30,
                max(520, len(self.last_event_text) * 12),
            )

            cv2.rectangle(
                output,
                (15, 50),
                (15 + banner_width, 100),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                output,
                self.last_event_text[:90],
                (28, 83),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        scale = float(
            self.cfg.get("full_view_scale", 0.55)
        )

        if scale != 1.0:
            output = cv2.resize(
                output,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        return output

    def draw_killfeed_monitor(
        self,
        roi_image,
        detector_debug,
        active_tracks,
        fps,
        parser_pending=0,
    ):
        output = roi_image.copy()

        if output.size == 0:
            return np.zeros(
                (200, 500, 3),
                dtype=np.uint8,
            )

        if self.show_details and self.cfg.get(
            "draw_rows",
            True,
        ):
            for box in detector_debug.get(
                "localizer_proposals",
                [],
            ):
                cv2.rectangle(
                    output,
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    (255, 0, 255),
                    1,
                )
                cv2.putText(
                    output,
                    "localizer",
                    (box.x1, max(18, box.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            for rejected in detector_debug.get(
                "verifier_rejections",
                [],
            ):
                box = rejected["bbox_roi"]
                probability = rejected.get("probability")

                cv2.rectangle(
                    output,
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    (0, 80, 255),
                    2,
                )

                label = (
                    "reject"
                    if probability is None
                    else f"reject {probability:.2f}"
                )
                cv2.putText(
                    output,
                    label,
                    (box.x1, max(18, box.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (0, 80, 255),
                    1,
                    cv2.LINE_AA,
                )

            for row in detector_debug["rows"]:
                box = row.bbox_roi

                cv2.rectangle(
                    output,
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    output,
                    f"candidate {row.score:.2f}",
                    (box.x1, max(18, box.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                self._draw_hero_boxes(
                    output,
                    row,
                    row.bbox_roi.x1,
                    row.bbox_roi.y1,
                    thickness=1,
                )

        for track in active_tracks:
            if not track.confirmed:
                continue

            box = track.row.bbox_roi

            cv2.rectangle(
                output,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (255, 255, 255),
                1,
            )

            cv2.putText(
                output,
                f"ID {track.track_id}",
                (
                    max(0, box.x2 - 70),
                    max(18, box.y2 - 4),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            self._draw_hero_boxes(
                output,
                track.row,
                track.row.bbox_roi.x1,
                track.row.bbox_roi.y1,
                thickness=1,
            )

        proposal_count = len(
            detector_debug.get("proposals", [])
        )
        localizer_count = len(
            detector_debug.get("localizer_proposals", [])
        )
        reject_count = len(
            detector_debug.get("verifier_rejections", [])
        )

        status = (
            f"Kill Feed Monitor | "
            f"FPS {fps:.1f} | "
            f"hsv {proposal_count} | "
            f"loc {localizer_count} | "
            f"pass {len(detector_debug['rows'])} | "
            f"reject {reject_count} | "
            f"parse {parser_pending}"
        )

        cv2.rectangle(
            output,
            (0, 0),
            (min(output.shape[1], 470), 30),
            (15, 15, 15),
            -1,
        )

        cv2.putText(
            output,
            status,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if self._event_banner_active():
            banner_y1 = max(
                32,
                output.shape[0] - 46,
            )
            cv2.rectangle(
                output,
                (0, banner_y1),
                (
                    output.shape[1],
                    output.shape[0],
                ),
                (15, 15, 15),
                -1,
            )

            cv2.putText(
                output,
                self.last_event_text[:72],
                (9, output.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        scale = float(
            self.cfg.get(
                "killfeed_monitor_scale",
                1.50,
            )
        )

        if scale != 1.0:
            output = cv2.resize(
                output,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

        return output
