from __future__ import annotations

import cv2
import numpy as np


class DigitTemplateRecognizer:
    """
    Very small 0-6 recognizer for the Overwatch alive-player counter.

    This avoids running a neural OCR model on the team-status HUD. It compares
    a normalized bright digit against a bank of synthetic OpenCV digit shapes.
    """

    def __init__(self, max_digit: int = 6):
        self.max_digit = int(max_digit)
        self.width = 32
        self.height = 48
        self.templates = self._build_templates()

    def _normalize_binary(self, binary: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 1 or h <= 2:
            return None

        glyph = binary[
            max(0, y - 1):min(binary.shape[0], y + h + 1),
            max(0, x - 1):min(binary.shape[1], x + w + 1),
        ]

        gh, gw = glyph.shape[:2]
        scale = min(
            (self.width - 6) / max(1, gw),
            (self.height - 6) / max(1, gh),
        )

        out_w = max(1, int(round(gw * scale)))
        out_h = max(1, int(round(gh * scale)))

        resized = cv2.resize(
            glyph,
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )

        canvas = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        x0 = (self.width - out_w) // 2
        y0 = (self.height - out_h) // 2

        canvas[
            y0:y0 + out_h,
            x0:x0 + out_w,
        ] = resized

        return canvas

    def _build_templates(self):
        templates = []

        fonts = (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        )

        for digit in range(self.max_digit + 1):
            for font in fonts:
                for scale in (1.4, 1.6, 1.8):
                    for thickness in (2, 3):
                        canvas = np.zeros(
                            (72, 56),
                            dtype=np.uint8,
                        )

                        text = str(digit)
                        (tw, th), _ = cv2.getTextSize(
                            text,
                            font,
                            scale,
                            thickness,
                        )

                        x = max(1, (56 - tw) // 2)
                        y = max(th + 1, (72 + th) // 2)

                        cv2.putText(
                            canvas,
                            text,
                            (x, y),
                            font,
                            scale,
                            255,
                            thickness,
                            cv2.LINE_AA,
                        )

                        _, binary = cv2.threshold(
                            canvas,
                            80,
                            255,
                            cv2.THRESH_BINARY,
                        )

                        normalized = self._normalize_binary(
                            binary
                        )
                        if normalized is not None:
                            templates.append(
                                (
                                    digit,
                                    normalized,
                                )
                            )

        return templates

    @staticmethod
    def _similarity(
        candidate: np.ndarray,
        template: np.ndarray,
    ) -> float:
        a = candidate.astype(np.float32) / 255.0
        b = template.astype(np.float32) / 255.0

        intersection = float(
            np.sum(np.minimum(a, b))
        )
        union = float(
            np.sum(np.maximum(a, b))
        )

        iou = (
            intersection / union
            if union > 0
            else 0.0
        )

        a_vec = a.reshape(-1)
        b_vec = b.reshape(-1)

        denom = float(
            np.linalg.norm(a_vec)
            * np.linalg.norm(b_vec)
        )

        cosine = (
            float(np.dot(a_vec, b_vec)) / denom
            if denom > 0
            else 0.0
        )

        return 0.55 * iou + 0.45 * cosine

    def _extract_candidate(
        self,
        image: np.ndarray,
    ) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # The counter digit is one of the tallest bright shapes in each half.
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        h, w = binary.shape[:2]
        candidates = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)

            if bh < h * 0.26:
                continue

            if bw < 2 or bw > w * 0.65:
                continue

            if bh > h * 0.95:
                continue

            area = cv2.contourArea(contour)
            candidates.append(
                (
                    area + bh * 4.0,
                    (x, y, bw, bh),
                )
            )

        if not candidates:
            return None

        _, (x, y, bw, bh) = max(
            candidates,
            key=lambda item: item[0],
        )

        glyph = binary[
            max(0, y - 2):min(h, y + bh + 2),
            max(0, x - 2):min(w, x + bw + 2),
        ]

        return self._normalize_binary(glyph)

    def recognize(
        self,
        image: np.ndarray,
    ) -> tuple[int | None, float]:
        candidate = self._extract_candidate(image)

        if candidate is None:
            return None, 0.0

        best_digit = None
        best_score = -1.0

        for digit, template in self.templates:
            score = self._similarity(
                candidate,
                template,
            )

            if score > best_score:
                best_score = score
                best_digit = digit

        if best_score < 0.30:
            return None, max(0.0, best_score)

        return best_digit, best_score
