from __future__ import annotations

import queue
import threading

import pyttsx3


class AudioAnnouncer:
    """
    Non-blocking text-to-speech announcer.

    Speech runs on a dedicated worker thread so screen capture and vision
    processing do not stall while Windows is speaking.
    """

    def __init__(self, config: dict):
        cfg = config.get("audio", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.default_phrase = str(
            cfg.get("phrase", "Elimination detected")
        )
        self.rate = int(cfg.get("rate", 190))
        self.volume = float(cfg.get("volume", 0.90))

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=16)
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self):
        if not self.enabled or self._started:
            return

        self._started = True
        self._thread = threading.Thread(
            target=self._worker,
            name="overwatch-audio-announcer",
            daemon=True,
        )
        self._thread.start()

    def announce_elimination(self, text: str | None = None):
        if not self.enabled:
            return

        phrase = text or self.default_phrase

        try:
            self._queue.put_nowait(phrase)
        except queue.Full:
            # Dropping speech is preferable to blocking the vision loop.
            pass

    def stop(self):
        if not self._started:
            return

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self._started = False

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty(
                "volume",
                max(0.0, min(1.0, self.volume)),
            )

            while True:
                item = self._queue.get()

                if item is None:
                    break

                engine.say(item)
                engine.runAndWait()

            engine.stop()

        except Exception as exc:
            # Audio failure should never take down computer vision.
            print(f"[audio] disabled after error: {exc}")
