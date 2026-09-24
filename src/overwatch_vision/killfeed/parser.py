from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import cv2
import numpy as np

from overwatch_vision.killfeed.hero_recognizer import HeroRecognizer
from overwatch_vision.models import Rect
from overwatch_vision.ocr import OCRReader


@dataclass(slots=True)
class ParsedKillFeedRow:
    killer_name: str | None = None
    victim_name: str | None = None
    killer_hero: str | None = None
    victim_hero: str | None = None
    killer_team: str | None = None
    victim_team: str | None = None
    ability: str | None = None
    critical: bool | None = None
    confidence: float = 0.0
    killer_hero_confidence: float = 0.0
    victim_hero_confidence: float = 0.0
    killer_name_confidence: float = 0.0
    victim_name_confidence: float = 0.0


class KillFeedParser:
    """
    Parse a detected row into attacker/victim details.

    Normal Overwatch kill-feed rows are laid out left-to-right as attacker,
    action/ability, victim. The colored panels tell us team affiliation and
    also provide stable geometry for name OCR and hero portrait crops.
    """

    NAME_ALLOWLIST = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789_-"
    )

    def __init__(self, config: dict, ocr: OCRReader):
        self.config = config
        self.cfg = config.get("killfeed_parse", {})
        self.ocr = ocr
        self.hero_recognizer = HeroRecognizer(config)

        self.save_low_confidence = bool(
            self.cfg.get("save_low_confidence_crops", True)
        )
        self.low_confidence_threshold = float(
            self.cfg.get("low_confidence_threshold", 0.58)
        )
        self.min_name_confidence = float(
            self.cfg.get("min_name_confidence", 0.55)
        )
        self.capture_hero_samples = bool(
            self.cfg.get("capture_hero_samples", True)
        )

        project_root = Path(__file__).resolve().parents[3]
        relative = str(
            self.cfg.get(
                "review_directory",
                "debug_frames/killfeed_review",
            )
        )
        self.review_dir = project_root / relative
        self.hero_sample_dir = (
            project_root
            / str(
                self.cfg.get(
                    "hero_sample_directory",
                    "debug_frames/hero_samples",
                )
            )
        )

    def warmup_async(self):
        self.hero_recognizer.warmup_async()

    @staticmethod
    def _safe_crop(image: np.ndarray, rect: Rect) -> np.ndarray:
        h, w = image.shape[:2]

        x1 = max(0, min(w, rect.x1))
        y1 = max(0, min(h, rect.y1))
        x2 = max(0, min(w, rect.x2))
        y2 = max(0, min(h, rect.y2))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return image[y1:y2, x1:x2]

    @staticmethod
    def _clean_name(text: str | None) -> str | None:
        if not text:
            return None

        cleaned = re.sub(
            r"[^A-Za-z0-9_-]",
            "",
            text,
        ).strip()

        if len(cleaned) < 2:
            return None

        return cleaned

    @staticmethod
    def _team_from_panel(panel: np.ndarray) -> str | None:
        if panel is None or panel.size == 0:
            return None

        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        flat = hsv.reshape(-1, 3)

        keep = flat[
            (flat[:, 1] >= 75)
            & (flat[:, 2] >= 70)
        ]

        if keep.size == 0:
            return None

        hues = keep[:, 0].astype(np.float32)

        red_fraction = float(
            np.mean(
                (hues <= 10)
                | (hues >= 160)
            )
        )
        blue_fraction = float(
            np.mean(
                (hues >= 85)
                & (hues <= 115)
            )
        )

        if red_fraction > blue_fraction and red_fraction >= 0.15:
            return "red"

        if blue_fraction >= red_fraction and blue_fraction >= 0.15:
            return "blue"

        return None

    @staticmethod
    def _local_components(row):
        boxes = list(
            getattr(row, "component_boxes_local", [])
            or []
        )
        teams = list(
            getattr(row, "component_teams_local", [])
            or []
        )

        paired = list(zip(boxes, teams))
        paired.sort(key=lambda item: item[0].cx)

        if paired:
            return paired

        boxes.sort(key=lambda box: box.cx)
        return [
            (box, None)
            for box in boxes
        ]

    def _party_crops(
        self,
        row_image: np.ndarray,
        panel_box: Rect,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        panel = self._safe_crop(
            row_image,
            panel_box,
        )

        ph, pw = panel.shape[:2]
        if ph <= 1 or pw <= 1:
            empty = np.zeros((1, 1, 3), dtype=np.uint8)
            return panel, empty, empty

        # Overwatch's kill-feed layout is asymmetric:
        #
        #   killer name | killer portrait | action slot | victim portrait | victim name
        #
        # The colored contour for the killer side often extends through the
        # action slot, so "take the final square of the killer panel" actually
        # crops the weapon/action icon instead of the hero. Use row-height
        # geometry to step one icon-width left of the action slot.
        inset_y = max(0, int(ph * 0.05))
        y1 = inset_y
        y2 = max(y1 + 1, ph - inset_y)

        if side == "killer":
            start_rows = float(
                self.cfg.get(
                    "killer_hero_start_from_right_rows",
                    2.25,
                )
            )
            end_rows = float(
                self.cfg.get(
                    "killer_hero_end_from_right_rows",
                    1.00,
                )
            )

            hero_x1 = int(
                round(pw - start_rows * ph)
            )
            hero_x2 = int(
                round(pw - end_rows * ph)
            )

            hero_x1 = max(0, min(pw - 1, hero_x1))
            hero_x2 = max(
                hero_x1 + 1,
                min(pw, hero_x2),
            )

            hero = panel[
                y1:y2,
                hero_x1:hero_x2,
            ]
            name = panel[
                y1:y2,
                0:max(1, hero_x1),
            ]
        else:
            hero_rows = float(
                self.cfg.get(
                    "victim_hero_width_rows",
                    1.15,
                )
            )

            hero_x2 = int(
                round(hero_rows * ph)
            )
            hero_x2 = max(
                1,
                min(pw, hero_x2),
            )

            hero = panel[
                y1:y2,
                0:hero_x2,
            ]
            name = panel[
                y1:y2,
                hero_x2:pw,
            ]

        return panel, name, hero

    def _read_name(
        self,
        image: np.ndarray,
    ) -> tuple[str | None, float]:
        prepared = self.ocr.prepare_name_image(image)
        text, confidence = self.ocr.read(
            prepared,
            allowlist=self.NAME_ALLOWLIST,
        )

        if confidence < self.min_name_confidence:
            return None, confidence

        return self._clean_name(text), confidence

    @staticmethod
    def _safe_filename_text(value: str | None) -> str:
        if not value:
            return "unknown"

        cleaned = re.sub(
            r"[^A-Za-z0-9_-]",
            "",
            value,
        )
        return cleaned or "unknown"

    def _save_hero_samples(
        self,
        row,
        killer_crop: np.ndarray,
        victim_crop: np.ndarray,
        killer_team: str | None,
        victim_team: str | None,
        killer_name: str | None,
        victim_name: str | None,
    ):
        if not self.capture_hero_samples:
            return

        try:
            self.hero_sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            stamp = int(time.time() * 1000)

            cv2.imwrite(
                str(
                    self.hero_sample_dir
                    / f"{stamp}_row.png"
                ),
                row.crop,
            )

            cv2.imwrite(
                str(
                    self.hero_sample_dir
                    / (
                        f"{stamp}_killer_"
                        f"{killer_team or 'unknown'}_"
                        f"{self._safe_filename_text(killer_name)}.png"
                    )
                ),
                killer_crop,
            )

            cv2.imwrite(
                str(
                    self.hero_sample_dir
                    / (
                        f"{stamp}_victim_"
                        f"{victim_team or 'unknown'}_"
                        f"{self._safe_filename_text(victim_name)}.png"
                    )
                ),
                victim_crop,
            )
        except Exception:
            pass

    def _save_review_crop(
        self,
        row,
        parsed: ParsedKillFeedRow,
    ):
        if not self.save_low_confidence:
            return

        useful_scores = [
            parsed.killer_hero_confidence,
            parsed.victim_hero_confidence,
            parsed.killer_name_confidence,
            parsed.victim_name_confidence,
        ]
        useful_scores = [
            score for score in useful_scores
            if score > 0
        ]

        average = (
            sum(useful_scores) / len(useful_scores)
            if useful_scores
            else 0.0
        )

        if average >= self.low_confidence_threshold:
            return

        try:
            self.review_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            filename = (
                f"row_{int(time.time() * 1000)}"
                f"_{average:.2f}.png"
            )
            cv2.imwrite(
                str(self.review_dir / filename),
                row.crop,
            )
        except Exception:
            pass

    def parse(self, row) -> ParsedKillFeedRow:
        components = self._local_components(row)

        if len(components) < 2:
            return ParsedKillFeedRow(
                confidence=0.0,
            )

        killer_box, killer_team_hint = components[0]
        victim_box, victim_team_hint = components[-1]

        killer_panel, killer_name_crop, killer_hero_crop = (
            self._party_crops(
                row.crop,
                killer_box,
                "killer",
            )
        )
        victim_panel, victim_name_crop, victim_hero_crop = (
            self._party_crops(
                row.crop,
                victim_box,
                "victim",
            )
        )

        killer_team = (
            killer_team_hint
            or self._team_from_panel(killer_panel)
        )
        victim_team = (
            victim_team_hint
            or self._team_from_panel(victim_panel)
        )

        killer_name, killer_name_conf = self._read_name(
            killer_name_crop
        )
        victim_name, victim_name_conf = self._read_name(
            victim_name_crop
        )

        killer_hero, killer_hero_conf = (
            self.hero_recognizer.recognize(
                killer_hero_crop
            )
        )
        victim_hero, victim_hero_conf = (
            self.hero_recognizer.recognize(
                victim_hero_crop
            )
        )

        self._save_hero_samples(
            row,
            killer_hero_crop,
            victim_hero_crop,
            killer_team,
            victim_team,
            killer_name,
            victim_name,
        )

        confidence_values = [
            value
            for value in (
                killer_name_conf,
                victim_name_conf,
                killer_hero_conf,
                victim_hero_conf,
            )
            if value > 0
        ]

        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else row.score
        )

        parsed = ParsedKillFeedRow(
            killer_name=killer_name,
            victim_name=victim_name,
            killer_hero=killer_hero,
            victim_hero=victim_hero,
            killer_team=killer_team,
            victim_team=victim_team,
            ability=None,
            critical=None,
            confidence=confidence,
            killer_hero_confidence=killer_hero_conf,
            victim_hero_confidence=victim_hero_conf,
            killer_name_confidence=killer_name_conf,
            victim_name_confidence=victim_name_conf,
        )

        self._save_review_crop(row, parsed)
        return parsed
