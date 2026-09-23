from __future__ import annotations

import threading

import cv2
import numpy as np


class OCRReader:
    """Lazy EasyOCR wrapper shared by HUD readers."""

    def __init__(self, config: dict):
        cfg = config.get("ocr", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.languages = list(cfg.get("languages", ["en"]))
        self.gpu = bool(cfg.get("gpu", False))

        self._reader = None
        self._lock = threading.Lock()
        self._failed = False
        self._loading = False
        self._worker: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._reader is not None

    def warmup_async(self):
        if (
            not self.enabled
            or self._failed
            or self._reader is not None
            or self._loading
        ):
            return

        self._loading = True
        self._worker = threading.Thread(
            target=self._load_worker,
            name="overwatch-ocr-loader",
            daemon=True,
        )
        self._worker.start()

    def _load_worker(self):
        try:
            self._load_reader()
        finally:
            self._loading = False

    def _load_reader(self):
        if not self.enabled or self._failed:
            return None

        if self._reader is not None:
            return self._reader

        with self._lock:
            if self._reader is not None:
                return self._reader

            try:
                import easyocr

                print(
                    "[ocr] Loading EasyOCR. The first run may download "
                    "model files and can take a little while..."
                )
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    verbose=False,
                )
                print("[ocr] EasyOCR ready.")
            except Exception as exc:
                self._failed = True
                print(f"[ocr] unavailable: {exc}")
                return None

        return self._reader

    def _get_reader(self):
        if self._reader is not None:
            return self._reader

        if self._loading:
            return None

        return self._load_reader()

    @staticmethod
    def prepare_name_image(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((32, 128), dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        white_text = np.where(
            (value >= 145) & (saturation <= 150),
            255,
            0,
        ).astype(np.uint8)

        kernel = np.ones((2, 2), np.uint8)
        white_text = cv2.morphologyEx(
            white_text,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return cv2.resize(
            white_text,
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def prepare_digit_image(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((32, 64), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            None,
            fx=5.0,
            fy=5.0,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return threshold

    def read(
        self,
        image: np.ndarray,
        allowlist: str | None = None,
        paragraph: bool = False,
    ) -> tuple[str | None, float]:
        reader = self._get_reader()
        if reader is None or image is None or image.size == 0:
            return None, 0.0

        try:
            results = reader.readtext(
                image,
                detail=1,
                paragraph=paragraph,
                allowlist=allowlist,
                decoder="greedy",
            )
        except Exception as exc:
            print(f"[ocr] read failed: {exc}")
            return None, 0.0

        if not results:
            return None, 0.0

        pieces = []
        confidences = []

        for result in results:
            if len(result) < 3:
                continue

            text = str(result[1]).strip()
            confidence = float(result[2])

            if text:
                pieces.append(text)
                confidences.append(confidence)

        if not pieces:
            return None, 0.0

        return " ".join(pieces), sum(confidences) / len(confidences)
