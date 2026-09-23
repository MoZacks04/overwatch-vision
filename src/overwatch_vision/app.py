from __future__ import annotations

from pathlib import Path
import time

import cv2
import yaml

from overwatch_vision.audio.announcer import AudioAnnouncer
from overwatch_vision.capture import OverwatchCapture
from overwatch_vision.regions import HUDRegionManager
from overwatch_vision.killfeed.detector import KillFeedDetector
from overwatch_vision.debug_view import DebugView


def _load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "settings.yaml"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def main():
    config = _load_config()

    capture = OverwatchCapture(config)
    regions = HUDRegionManager(config)
    killfeed = KillFeedDetector(config)
    debug = DebugView(config)
    audio = AudioAnnouncer(config)

    audio.start()

    target_fps = float(
        config["capture"].get("target_fps", 30)
    )
    target_dt = (
        1.0 / target_fps
        if target_fps > 0
        else 0.0
    )

    show_full_view = bool(
        config["debug"].get("show_full_view", True)
    )
    show_killfeed_monitor = bool(
        config["debug"].get("show_killfeed_monitor", True)
    )

    suppress_first_seconds = float(
        config.get("audio", {}).get(
            "suppress_first_seconds",
            2.0,
        )
    )

    session_start = time.monotonic()

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    print("Overwatch Vision started.")
    print("Q = quit | D = toggle debug | R = reset tracker")
    print(
        f"Audio announcements begin after "
        f"{suppress_first_seconds:.1f}s baseline period."
    )

    try:
        while True:
            loop_start = time.perf_counter()

            game_frame = capture.grab()

            region = regions.killfeed_search_region(
                game_frame.image
            )
            team_status = regions.team_status_region(
                game_frame.image
            )

            rows, events = killfeed.process(
                roi_image=region.image,
                roi_rect=region.rect,
                timestamp=game_frame.timestamp,
            )

            baseline_complete = (
                time.monotonic() - session_start
                >= suppress_first_seconds
            )

            for event in events:
                print(
                    f"[killfeed] NEW ROW "
                    f"track={event.track_id} "
                    f"confidence={event.confidence:.2f} "
                    f"time={event.timestamp:.3f}"
                )

                # Do not announce rows that were already on screen when the
                # application was first opened.
                if baseline_complete:
                    announcement = "Elimination detected"
                    debug.notify_event(announcement)
                    audio.announce_elimination(announcement)

            now = time.perf_counter()
            dt = max(1e-6, now - previous_time)
            previous_time = now

            instantaneous_fps = 1.0 / dt

            if smoothed_fps == 0.0:
                smoothed_fps = instantaneous_fps
            else:
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * instantaneous_fps
                )

            if show_full_view:
                full_preview = debug.draw_full_view(
                    frame=game_frame.image,
                    roi_rect=region.rect,
                    detector_debug=killfeed.last_debug,
                    active_tracks=killfeed.tracker.tracks,
                    fps=smoothed_fps,
                    team_status_rect=team_status.rect,
                )

                cv2.imshow(
                    "Overwatch Vision - Full View",
                    full_preview,
                )

            if show_killfeed_monitor:
                feed_preview = debug.draw_killfeed_monitor(
                    roi_image=region.image,
                    detector_debug=killfeed.last_debug,
                    active_tracks=killfeed.tracker.tracks,
                    fps=smoothed_fps,
                )

                cv2.imshow(
                    "Overwatch Vision - Kill Feed Monitor",
                    feed_preview,
                )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("d"), ord("D")):
                debug.toggle()

            if key in (ord("r"), ord("R")):
                killfeed.reset()
                session_start = time.monotonic()
                print("[killfeed] tracker reset")

            elapsed = time.perf_counter() - loop_start
            remaining = target_dt - elapsed

            if remaining > 0:
                time.sleep(remaining)

    finally:
        audio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
