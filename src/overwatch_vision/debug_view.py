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
        Return diagnostic killer/victim portrait boxes in row-local coords.

        These are intentionally visual diagnostics, not trusted parsing
        results. They let us verify the exact portrait geometry from real
        kill-feed examples before training/tuning hero recognition.
        """
        components = list(
            getattr(row, "component_boxes_local", [])
            or []
        )
        components.sort(key=lambda box: box.cx)

        if len(components) < 2:
            return []

        killer_panel = components[0]
        victim_panel = components[-1]

        scale = float(
            self.cfg.get("hero_icon_width_rows", 1.0)
        )

        killer_width = max(
            1,
            int(round(killer_panel.height * scale)),
        )
        victim_width = max(
            1,
            int(round(victim_panel.height * scale)),
        )

        killer = (
            max(killer_panel.x1, killer_panel.x2 - killer_width),
            killer_panel.y1,
            killer_panel.x2,
            killer_panel.y2,
        )
        victim = (
            victim_panel.x1,
            victim_panel.y1,
            min(victim_panel.x2, victim_panel.x1 + victim_width),
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

        status = (
            f"Kill Feed Monitor | "
            f"FPS {fps:.1f} | "
            f"rows {len(detector_debug['rows'])} | "
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
