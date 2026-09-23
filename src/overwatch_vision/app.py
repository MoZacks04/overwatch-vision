from __future__ import annotations

from pathlib import Path
import time

import cv2
import yaml

from overwatch_vision.audio.announcer import AudioAnnouncer
from overwatch_vision.capture import OverwatchCapture
from overwatch_vision.debug_view import DebugView
from overwatch_vision.killfeed.detector import KillFeedDetector
from overwatch_vision.ocr import OCRReader
from overwatch_vision.regions import HUDRegionManager
from overwatch_vision.team_status.detector import TeamStatusDetector


def _load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "settings.yaml"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def _person_label(
    hero: str | None,
    player_name: str | None,
    include_player_name: bool,
) -> str:
    hero_text = hero or "unknown hero"

    if include_player_name and player_name:
        return f"{hero_text} {player_name}"

    return hero_text


def _event_console_text(event) -> str:
    killer = _person_label(
        event.killer_hero,
        event.killer_name,
        include_player_name=True,
    )
    victim = _person_label(
        event.victim_hero,
        event.victim_name,
        include_player_name=True,
    )

    return (
        f"{killer} [{event.killer_team or '?'}] "
        f"-> {victim} [{event.victim_team or '?'}] "
        f"| parse={event.parse_confidence:.2f}"
    )


def _event_speech(
    event,
    friendly_team: str,
    speak_player_names: bool,
) -> str:
    killer = _person_label(
        event.killer_hero,
        event.killer_name,
        include_player_name=speak_player_names,
    )
    victim = _person_label(
        event.victim_hero,
        event.victim_name,
        include_player_name=speak_player_names,
    )

    if (
        event.victim_team == friendly_team
        and event.killer_team != friendly_team
    ):
        return f"Enemy {killer} eliminated your {victim}."

    if (
        event.killer_team == friendly_team
        and event.victim_team != friendly_team
    ):
        return f"Your {killer} eliminated enemy {victim}."

    if (
        event.killer_hero
        or event.victim_hero
        or event.killer_name
        or event.victim_name
    ):
        return f"{killer} eliminated {victim}."

    return "Elimination detected."


def main():
    config = _load_config()

    capture = OverwatchCapture(config)
    regions = HUDRegionManager(config)

    ocr = OCRReader(config)
    ocr.warmup_async()

    killfeed = KillFeedDetector(config, ocr)
    team_status_detector = TeamStatusDetector(config, ocr)

    debug = DebugView(config)
    audio = AudioAnnouncer(config)
    audio.start()

    audio_cfg = config.get("audio", {})
    if bool(audio_cfg.get("speak_startup_test", True)):
        audio.announce("Overwatch Vision audio ready.")

    parse_cfg = config.get("killfeed_parse", {})
    friendly_team = str(
        parse_cfg.get("friendly_team", "blue")
    ).lower()
    speak_player_names = bool(
        audio_cfg.get("speak_player_names", False)
    )

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
        config["debug"].get(
            "show_killfeed_monitor",
            True,
        )
    )

    suppress_first_seconds = float(
        audio_cfg.get(
            "suppress_first_seconds",
            2.0,
        )
    )

    session_start = time.monotonic()
    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    previous_team_status = (None, None)

    print("Overwatch Vision started.")
    print("Q = quit | D = toggle debug | R = reset tracker")
    print(
        "[audio] A startup phrase has been queued. "
        "If you do not hear it, check the terminal for an audio error."
    )
    print(
        "[vision] Hero references and OCR load in the background. "
        "The first run can take longer."
    )

    try:
        while True:
            loop_start = time.perf_counter()

            game_frame = capture.grab()

            region = regions.killfeed_search_region(
                game_frame.image
            )
            team_status_region = regions.team_status_region(
                game_frame.image
            )

            team_status_state = team_status_detector.process(
                team_status_region.image,
                game_frame.timestamp,
            )

            current_team_status = (
                team_status_state.friendly_alive,
                team_status_state.enemy_alive,
            )

            if (
                current_team_status != previous_team_status
                and any(
                    value is not None
                    for value in current_team_status
                )
            ):
                print(
                    "[team-status] "
                    f"friendly={current_team_status[0]} "
                    f"enemy={current_team_status[1]} "
                    f"confidence={team_status_state.confidence:.2f}"
                )
                previous_team_status = current_team_status

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
                console_text = _event_console_text(event)
                print(
                    f"[killfeed] track={event.track_id} "
                    f"{console_text}"
                )

                speech = _event_speech(
                    event,
                    friendly_team=friendly_team,
                    speak_player_names=speak_player_names,
                )
                debug.notify_event(speech)

                if baseline_complete:
                    audio.announce_elimination(speech)

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
                    team_status_rect=team_status_region.rect,
                    team_status_state=team_status_state,
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # If a headless OpenCV build slips through, cleanup should not
            # hide the original viewer error with a second traceback.
            pass


if __name__ == "__main__":
    main()
