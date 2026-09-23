from __future__ import annotations

from dataclasses import dataclass
import queue
import threading

from overwatch_vision.killfeed.parser import KillFeedParser
from overwatch_vision.models import KillFeedEvent, KillFeedRow, Rect


@dataclass(slots=True)
class ParseJob:
    event: KillFeedEvent
    row: KillFeedRow


class AsyncKillFeedParser:
    """
    Runs expensive kill-feed parsing away from the capture loop.

    Row detection stays responsive while username OCR and hero recognition
    happen on a dedicated worker thread. Jobs are intentionally kept in an
    unbounded queue because a missed elimination is worse than a short delay
    in announcing it.
    """

    def __init__(self, config: dict, parser: KillFeedParser):
        self.config = config
        self.parser = parser

        self._jobs: queue.Queue[ParseJob | None] = queue.Queue()
        self._results: queue.Queue[KillFeedEvent] = queue.Queue()

        self._thread: threading.Thread | None = None
        self._started = False

    @staticmethod
    def _copy_rect(rect: Rect) -> Rect:
        return Rect(
            x1=rect.x1,
            y1=rect.y1,
            x2=rect.x2,
            y2=rect.y2,
        )

    @classmethod
    def _snapshot_row(cls, row: KillFeedRow) -> KillFeedRow:
        return KillFeedRow(
            bbox_roi=cls._copy_rect(row.bbox_roi),
            bbox_game=cls._copy_rect(row.bbox_game),
            crop=row.crop.copy(),
            normalized=row.normalized.copy(),
            fingerprint=row.fingerprint.copy(),
            component_boxes_local=[
                cls._copy_rect(box)
                for box in row.component_boxes_local
            ],
            score=row.score,
        )

    def start(self):
        if self._started:
            return

        self.parser.warmup_async()

        self._started = True
        self._thread = threading.Thread(
            target=self._worker,
            name="killfeed-parser",
            daemon=True,
        )
        self._thread.start()

    def submit(self, event: KillFeedEvent, row: KillFeedRow):
        if not self._started:
            self.start()

        self._jobs.put(
            ParseJob(
                event=event,
                row=self._snapshot_row(row),
            )
        )

    @property
    def pending_count(self) -> int:
        return self._jobs.qsize()

    def drain_results(self, limit: int = 16) -> list[KillFeedEvent]:
        results = []

        for _ in range(max(1, limit)):
            try:
                results.append(
                    self._results.get_nowait()
                )
            except queue.Empty:
                break

        return results

    def stop(self):
        if not self._started:
            return

        self._jobs.put(None)

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._started = False

    def _apply_parse(self, event: KillFeedEvent, parsed):
        event.killer_name = parsed.killer_name
        event.victim_name = parsed.victim_name
        event.killer_hero = parsed.killer_hero
        event.victim_hero = parsed.victim_hero
        event.killer_team = parsed.killer_team
        event.victim_team = parsed.victim_team
        event.ability = parsed.ability
        event.critical = parsed.critical

        event.parse_confidence = parsed.confidence
        event.killer_hero_confidence = (
            parsed.killer_hero_confidence
        )
        event.victim_hero_confidence = (
            parsed.victim_hero_confidence
        )
        event.killer_name_confidence = (
            parsed.killer_name_confidence
        )
        event.victim_name_confidence = (
            parsed.victim_name_confidence
        )

    def _worker(self):
        while True:
            job = self._jobs.get()

            if job is None:
                break

            try:
                parsed = self.parser.parse(job.row)
                self._apply_parse(job.event, parsed)
            except Exception as exc:
                print(
                    "[killfeed-parser] parse failed for "
                    f"track {job.event.track_id}: {exc}"
                )

            self._results.put(job.event)
