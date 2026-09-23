from __future__ import annotations

import queue
import threading


class AudioAnnouncer:
    """
    Non-blocking Windows speech announcer.

    Windows SAPI via pywin32 is preferred because it is dependable from a
    worker thread. pyttsx3 remains as a fallback.
    """

    def __init__(self, config: dict):
        cfg = config.get("audio", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.default_phrase = str(
            cfg.get("phrase", "Elimination detected")
        )
        self.rate = int(cfg.get("rate", 190))
        self.volume = float(cfg.get("volume", 1.0))

        self._queue: queue.Queue[str | None] = queue.Queue(
            maxsize=32
        )
        self._thread: threading.Thread | None = None
        self._started = False
        self.backend = "not started"

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

    def announce(self, text: str):
        if not self.enabled or not text:
            return

        try:
            self._queue.put_nowait(str(text))
        except queue.Full:
            pass

    def announce_elimination(self, text: str | None = None):
        self.announce(text or self.default_phrase)

    def stop(self):
        if not self._started:
            return

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._started = False

    def _worker(self):
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            try:
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Volume = int(
                    max(0.0, min(1.0, self.volume)) * 100
                )

                sapi_rate = round((self.rate - 180) / 25)
                voice.Rate = max(-10, min(10, sapi_rate))

                self.backend = "Windows SAPI"
                print(f"[audio] backend ready: {self.backend}")

                while True:
                    item = self._queue.get()
                    if item is None:
                        break

                    voice.Speak(item)
            finally:
                pythoncom.CoUninitialize()

            return
        except Exception as exc:
            print(
                "[audio] Windows SAPI backend unavailable; "
                f"trying pyttsx3: {exc}"
            )

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty(
                "volume",
                max(0.0, min(1.0, self.volume)),
            )

            self.backend = "pyttsx3"
            print(f"[audio] backend ready: {self.backend}")

            while True:
                item = self._queue.get()
                if item is None:
                    break

                engine.say(item)
                engine.runAndWait()

            engine.stop()

        except Exception as exc:
            self.backend = "disabled"
            print(f"[audio] disabled after error: {exc}")
