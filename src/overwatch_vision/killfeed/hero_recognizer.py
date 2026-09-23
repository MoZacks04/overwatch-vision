from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np


class HeroRecognizer:
    """
    Reference-image recognizer for kill-feed hero portraits.

    Reference portraits are downloaded locally from OverFast's hero list and
    cached outside Git. Recognition compares a lightweight NumPy gradient
    descriptor plus grayscale appearance, so it works even on OpenCV builds
    that do not expose cv2.HOGDescriptor.
    """

    def __init__(self, config: dict):
        cfg = config.get("hero_recognition", {})

        self.enabled = bool(cfg.get("enabled", True))
        self.api_url = str(
            cfg.get(
                "heroes_api_url",
                "https://overfast-api.tekrop.fr/heroes",
            )
        )
        self.min_confidence = float(
            cfg.get("min_confidence", 0.46)
        )
        self.refresh_hours = float(
            cfg.get("refresh_hours", 24.0)
        )

        project_root = Path(__file__).resolve().parents[3]
        cache_relative = str(
            cfg.get("cache_directory", ".cache/hero_portraits")
        )
        self.cache_dir = project_root / cache_relative
        self.manifest_path = self.cache_dir / "heroes.json"

        self._templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._ready = False
        self._loading = False
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None


    def warmup_async(self):
        if not self.enabled or self._ready or self._loading:
            return

        self._loading = True
        self._worker = threading.Thread(
            target=self._load_or_refresh,
            name="hero-reference-loader",
            daemon=True,
        )
        self._worker.start()

    def _manifest_is_stale(self) -> bool:
        if not self.manifest_path.exists():
            return True

        age_seconds = time.time() - self.manifest_path.stat().st_mtime
        return age_seconds > self.refresh_hours * 3600.0

    @staticmethod
    def _download_bytes(url: str) -> bytes:
        request = Request(
            url,
            headers={"User-Agent": "OverwatchVision/1.0"},
        )
        with urlopen(request, timeout=12) as response:
            return response.read()

    def _fetch_hero_list(self) -> list[dict]:
        payload = json.loads(
            self._download_bytes(self.api_url).decode("utf-8")
        )

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ("results", "heroes", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

        raise ValueError("Unexpected hero API response shape")

    def _refresh_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        heroes = self._fetch_hero_list()

        usable = []
        download_jobs = []

        for hero in heroes:
            key = str(hero.get("key", "")).strip()
            name = str(hero.get("name", "")).strip()
            portrait = str(hero.get("portrait", "")).strip()

            if not key or not name or not portrait:
                continue

            filename = f"{key}.png"
            destination = self.cache_dir / filename

            usable.append(
                {
                    "key": key,
                    "name": name,
                    "portrait": portrait,
                    "file": filename,
                }
            )

            if not destination.exists():
                download_jobs.append(
                    (portrait, destination)
                )

        def download_one(job):
            url, destination = job
            try:
                destination.write_bytes(
                    self._download_bytes(url)
                )
                return True
            except Exception:
                return False

        if download_jobs:
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(download_one, download_jobs))

        self.manifest_path.write_text(
            json.dumps(
                {
                    "source": self.api_url,
                    "heroes": usable,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _square_center_crop(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        size = min(h, w)
        x1 = max(0, (w - size) // 2)
        y1 = max(0, (h - size) // 2)
        return image[y1:y1 + size, x1:x1 + size]

    def _descriptor(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if image is None or image.size == 0:
            return None

        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            rgb = image[:, :, :3].copy()
            background = np.full_like(rgb, 32)
            weight = alpha.astype(np.float32)[:, :, None] / 255.0
            image = (
                rgb.astype(np.float32) * weight
                + background.astype(np.float32) * (1.0 - weight)
            ).astype(np.uint8)

        square = self._square_center_crop(image)
        square = cv2.resize(
            square,
            (48, 48),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Lightweight HOG-like descriptor implemented with NumPy instead of
        # cv2.HOGDescriptor. Some Windows OpenCV installations expose the
        # core image functions but omit HOGDescriptor, which previously made
        # the whole application fail during startup.
        gray_float = gray.astype(np.float32) / 255.0
        grad_y, grad_x = np.gradient(gray_float)

        magnitude = np.sqrt(
            grad_x * grad_x + grad_y * grad_y
        )
        angle = (
            np.degrees(np.arctan2(grad_y, grad_x)) + 180.0
        ) % 180.0

        cell_size = 8
        bins = 9
        bin_width = 180.0 / bins
        descriptor_parts = []

        for y in range(0, 48, cell_size):
            for x in range(0, 48, cell_size):
                cell_mag = magnitude[
                    y:y + cell_size,
                    x:x + cell_size,
                ].reshape(-1)
                cell_angle = angle[
                    y:y + cell_size,
                    x:x + cell_size,
                ].reshape(-1)

                hist = np.zeros(bins, dtype=np.float32)

                indices = np.floor(
                    cell_angle / bin_width
                ).astype(np.int32)
                indices = np.clip(
                    indices,
                    0,
                    bins - 1,
                )

                for idx, weight in zip(indices, cell_mag):
                    hist[idx] += float(weight)

                hist_norm = float(np.linalg.norm(hist))
                if hist_norm > 0:
                    hist /= hist_norm

                descriptor_parts.append(hist)

        hog = np.concatenate(
            descriptor_parts
        ).astype(np.float32)

        hog_norm = float(np.linalg.norm(hog))
        if hog_norm > 0:
            hog /= hog_norm

        gray_vector = gray.reshape(-1).astype(np.float32)
        gray_vector -= float(gray_vector.mean())
        gray_norm = float(np.linalg.norm(gray_vector))
        if gray_norm > 0:
            gray_vector /= gray_norm

        return hog, gray_vector

    def _load_templates_from_cache(self):
        if not self.manifest_path.exists():
            return

        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        templates = {}

        for hero in manifest.get("heroes", []):
            name = str(hero.get("name", "")).strip()
            filename = str(hero.get("file", "")).strip()

            if not name or not filename:
                continue

            path = self.cache_dir / filename
            if not path.exists():
                continue

            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            descriptor = self._descriptor(image)

            if descriptor is not None:
                templates[name] = descriptor

        with self._lock:
            self._templates = templates
            self._ready = bool(templates)

    def _load_or_refresh(self):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            if self._manifest_is_stale():
                try:
                    print("[heroes] Refreshing hero reference cache...")
                    self._refresh_cache()
                except Exception as exc:
                    print(
                        "[heroes] Could not refresh references; "
                        f"using any existing cache: {exc}"
                    )

            self._load_templates_from_cache()

            if self._ready:
                print(
                    f"[heroes] {len(self._templates)} hero references ready."
                )
            else:
                print(
                    "[heroes] No hero references available yet. "
                    "Hero names will be reported as unknown."
                )
        finally:
            self._loading = False

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return -1.0
        return float(np.dot(a, b))

    def recognize(
        self,
        image: np.ndarray,
    ) -> tuple[str | None, float]:
        if not self.enabled:
            return None, 0.0

        if not self._ready:
            self.warmup_async()
            return None, 0.0

        descriptor = self._descriptor(image)
        if descriptor is None:
            return None, 0.0

        hog, gray = descriptor

        best_name = None
        best_score = -1.0

        with self._lock:
            templates = list(self._templates.items())

        for name, (ref_hog, ref_gray) in templates:
            hog_score = self._cosine(hog, ref_hog)
            gray_score = self._cosine(gray, ref_gray)

            hog_score = (hog_score + 1.0) / 2.0
            gray_score = (gray_score + 1.0) / 2.0

            score = 0.72 * hog_score + 0.28 * gray_score

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.min_confidence:
            return None, max(0.0, best_score)

        return best_name, best_score
