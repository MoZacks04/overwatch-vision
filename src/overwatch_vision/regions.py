from dataclasses import dataclass
import numpy as np

from overwatch_vision.utils.geometry import normalized_rect_to_pixels


@dataclass(slots=True)
class RegionCrop:
    image: np.ndarray
    rect: object


class HUDRegionManager:
    def __init__(self, config):
        self.config = config

    def killfeed_search_region(self, game_image):
        h, w = game_image.shape[:2]
        cfg = self.config["killfeed"]["search_region"]

        rect = normalized_rect_to_pixels(
            width=w,
            height=h,
            left=float(cfg["left"]),
            top=float(cfg["top"]),
            right=float(cfg["right"]),
            bottom=float(cfg["bottom"]),
        )

        crop = game_image[rect.y1:rect.y2, rect.x1:rect.x2]
        return RegionCrop(image=crop, rect=rect)
