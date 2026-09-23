import cv2
import numpy as np


class KillFeedRowNormalizer:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

    def normalize(self, row):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if row.size == 0:
            return canvas

        h, w = row.shape[:2]
        if h <= 0 or w <= 0:
            return canvas

        scale = self.height / float(h)
        resized_w = max(1, round(w * scale))

        if resized_w > self.width:
            scale = self.width / float(w)
            resized_w = self.width
            resized_h = max(1, round(h * scale))
        else:
            resized_h = self.height

        resized = cv2.resize(
            row,
            (resized_w, resized_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

        y = max(0, (self.height - resized_h) // 2)
        x = max(0, self.width - resized_w)

        canvas[y:y + resized_h, x:x + resized_w] = resized
        return canvas
