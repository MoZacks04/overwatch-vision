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

    def _event_banner_active(self):
        seconds = float(
            self.cfg.get("show_event_banner_seconds", 1.25)
        )
        return (
            self.last_event_time >= 0
            and time.monotonic() - self.last_event_time <= seconds
        )

    def draw_full_view(
        self,
        frame,
        roi_rect,
        detector_debug,
        active_tracks,
        fps,
        team_status_rect=None,
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
                "TEAM STATUS",
                (
                    team_status_rect.x1,
                    max(18, team_status_rect.y1 - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
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

        confirmed = sum(
            1 for track in active_tracks
            if track.confirmed
        )

        status = (
            f"FPS {fps:.1f} | "
            f"tracks {len(active_tracks)} | "
            f"confirmed {confirmed}"
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
            cv2.rectangle(
                output,
                (15, 50),
                (385, 96),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                output,
                self.last_event_text,
                (28, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
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
    ):
        """
        Dedicated enlarged view of only the kill-feed search region.
        Bounding boxes are drawn in ROI coordinates.
        """
        output = roi_image.copy()

        # Make the upper-right region easier to inspect on dark maps.
        if output.size == 0:
            return np.zeros((200, 500, 3), dtype=np.uint8)

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

        # Draw confirmed tracked rows separately so we can inspect whether
        # temporal matching is working when the feed shifts vertically.
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
                (max(0, box.x2 - 70), max(18, box.y2 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status = (
            f"Kill Feed Monitor | "
            f"FPS {fps:.1f} | "
            f"rows {len(detector_debug['rows'])}"
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
            banner_y1 = max(32, output.shape[0] - 43)
            cv2.rectangle(
                output,
                (0, banner_y1),
                (min(output.shape[1], 300), output.shape[0]),
                (15, 15, 15),
                -1,
            )

            cv2.putText(
                output,
                self.last_event_text,
                (9, output.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        scale = float(
            self.cfg.get("killfeed_monitor_scale", 1.50)
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
