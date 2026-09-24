from __future__ import annotations

from pathlib import Path
import time

import cv2
import yaml

from overwatch_vision.audio.announcer import AudioAnnouncer
from overwatch_vision.capture import OverwatchCapture
from overwatch_vision.debug_view import DebugView
from overwatch_vision.killfeed.async_parser import AsyncKillFeedParser
from overwatch_vision.killfeed.detector import KillFeedDetector
from overwatch_vision.killfeed.parser import KillFeedParser
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


def _spoken_identity(
    hero: str | None,
    player_name: str | None,
    include_player_name: bool,
    fallback: str,
) -> str:
    if hero and include_player_name and player_name:
        return f"{hero} {player_name}"

    if hero:
        return hero

    if include_player_name and player_name:
        return player_name

    return fallback


def _event_speech(
    event,
    friendly_team: str,
    speak_player_names: bool,
) -> str | None:
    # Only make an ally/enemy call when both colored nameplates were
    # confidently classified and they form a valid opposing-team pair.
    # If the color read is ambiguous, staying silent is better than saying
    # the elimination happened in the wrong direction.
    standard_team_pair = {
        event.killer_team,
        event.victim_team,
    } == {"red", "blue"}

    if not standard_team_pair:
        if event.killer_hero and event.victim_hero:
            killer = _spoken_identity(
                event.killer_hero,
                event.killer_name,
                include_player_name=speak_player_names,
                fallback="killer",
            )
            victim = _spoken_identity(
                event.victim_hero,
                event.victim_name,
                include_player_name=speak_player_names,
                fallback="victim",
            )
            return f"{killer} eliminated {victim}."

        return None

    killer_is_friendly = (
        event.killer_team == friendly_team
    )
    victim_is_friendly = (
        event.victim_team == friendly_team
    )

    killer = _spoken_identity(
        event.killer_hero,
        event.killer_name,
        include_player_name=speak_player_names,
        fallback="ally" if killer_is_friendly else "enemy",
    )
    victim = _spoken_identity(
        event.victim_hero,
        event.victim_name,
        include_player_name=speak_player_names,
        fallback="ally" if victim_is_friendly else "enemy",
    )

    if victim_is_friendly and not killer_is_friendly:
        if event.killer_hero and event.victim_hero:
            return f"Enemy {killer} eliminated your {victim}."
        if event.killer_hero:
            return f"Enemy {killer} eliminated your ally."
        if event.victim_hero:
            return f"Enemy eliminated your {victim}."
        return "Enemy eliminated your ally."

    if killer_is_friendly and not victim_is_friendly:
        if event.killer_hero and event.victim_hero:
            return f"Your {killer} eliminated enemy {victim}."
        if event.killer_hero:
            return f"Your {killer} eliminated an enemy."
        if event.victim_hero:
            return f"Your ally eliminated enemy {victim}."
        return "Your ally eliminated an enemy."

    return None

def main():
    config = _load_config()

    capture = OverwatchCapture(config)
    regions = HUDRegionManager(config)

    # EasyOCR remains available for arbitrary player handles, but it is no
    # longer called from the real-time capture loop.
    ocr = OCRReader(config)
    ocr.warmup_async()

    killfeed = KillFeedDetector(config)

    parser = KillFeedParser(config, ocr)
    parser_worker = AsyncKillFeedParser(
        config,
        parser,
    )
    parser_worker.start()

    # Team counts now use cheap digit templates, not EasyOCR.
    team_status_detector = TeamStatusDetector(config)

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

    capture_cfg = config.get("capture", {})
    target_fps = float(
        capture_cfg.get("target_fps", 30)
    )
    target_dt = (
        1.0 / target_fps
        if target_fps > 0
        else 0.0
    )

    perf_cfg = config.get("performance", {})
    killfeed_every_n_frames = max(
        1,
        int(
            perf_cfg.get(
                "killfeed_every_n_frames",
                3,
            )
        ),
    )
    team_status_every_n_frames = max(
        1,
        int(
            perf_cfg.get(
                "team_status_every_n_frames",
                15,
            )
        ),
    )
    parser_results_per_frame = max(
        1,
        int(
            perf_cfg.get(
                "parser_results_per_frame",
                8,
            )
        ),
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

    frame_index = 0
    previous_team_status = (None, None)
    team_status_state = team_status_detector.state

    print("Overwatch Vision started.")
    print("Q = quit | D = toggle debug | R = reset tracker")
    print(
        "[performance] capture target="
        f"{target_fps:.0f} FPS, kill-feed check every "
        f"{killfeed_every_n_frames} frames "
        f"(~{target_fps / killfeed_every_n_frames:.1f} Hz), "
        f"team-status check every {team_status_every_n_frames} frames."
    )
    print(
        "[audio] A startup phrase has been queued. "
        "If you do not hear it, check the terminal for an audio error."
    )
    print(
        "[vision] Username OCR and hero parsing run in the background. "
        "They no longer block capture."
    )

    try:
        while True:
            loop_start = time.perf_counter()
            frame_index += 1

            game_frame = capture.grab()

            region = regions.killfeed_search_region(
                game_frame.image
            )
            team_status_region = regions.team_status_region(
                game_frame.image
            )

            # Team status is intentionally sampled slowly. Its own detector
            # also skips recognition if the tiny HUD crop has not changed.
            if (
                frame_index % team_status_every_n_frames
                == 0
            ):
                team_status_state = (
                    team_status_detector.process(
                        team_status_region.image,
                        game_frame.timestamp,
                    )
                )

                current_team_status = (
                    team_status_state.friendly_alive,
                    team_status_state.enemy_alive,
                )

                if (
                    current_team_status
                    != previous_team_status
                    and any(
                        value is not None
                        for value in current_team_status
                    )
                ):
                    print(
                        "[team-status] "
                        f"friendly={current_team_status[0]} "
                        f"enemy={current_team_status[1]} "
                        f"confidence="
                        f"{team_status_state.confidence:.2f}"
                    )
                    previous_team_status = (
                        current_team_status
                    )

            # At 30 capture FPS and N=3 this performs kill-feed detection at
            # roughly 10 Hz, which is fast enough to catch persistent feed
            # rows while leaving plenty of CPU headroom.
            if (
                frame_index % killfeed_every_n_frames
                == 0
            ):
                _, raw_events = killfeed.process(
                    roi_image=region.image,
                    roi_rect=region.rect,
                    timestamp=game_frame.timestamp,
                )

                baseline_complete = (
                    time.monotonic() - session_start
                    >= suppress_first_seconds
                )

                for event in raw_events:
                    if not baseline_complete:
                        print(
                            "[killfeed] baseline row ignored "
                            f"track={event.track_id}"
                        )
                        continue

                    row = killfeed.get_track_row(
                        event.track_id
                    )

                    if row is None:
                        print(
                            "[killfeed] track disappeared before "
                            f"parse queue: {event.track_id}"
                        )
                        continue

                    parser_worker.submit(
                        event,
                        row,
                    )

            # Parsed results are drained without waiting. If EasyOCR takes
            # 300 ms, capture continues while the parser thread works.
            parsed_events = parser_worker.drain_results(
                limit=parser_results_per_frame
            )

            for event in parsed_events:
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

                if speech:
                    debug.notify_event(speech)
                    audio.announce_elimination(speech)

            now = time.perf_counter()
            dt = max(
                1e-6,
                now - previous_time,
            )
            previous_time = now

            instantaneous_fps = 1.0 / dt

            if smoothed_fps == 0.0:
                smoothed_fps = instantaneous_fps
            else:
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * instantaneous_fps
                )

            parser_pending = parser_worker.pending_count

            if show_full_view:
                full_preview = debug.draw_full_view(
                    frame=game_frame.image,
                    roi_rect=region.rect,
                    detector_debug=killfeed.last_debug,
                    active_tracks=killfeed.tracker.tracks,
                    fps=smoothed_fps,
                    team_status_rect=team_status_region.rect,
                    team_status_state=team_status_state,
                    parser_pending=parser_pending,
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
                    parser_pending=parser_pending,
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
        parser_worker.stop()
        audio.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
