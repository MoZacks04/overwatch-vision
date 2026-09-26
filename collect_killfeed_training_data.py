from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import yaml

from overwatch_vision.capture import OverwatchCapture
from overwatch_vision.killfeed.row_detector import KillFeedRowDetector
from overwatch_vision.regions import HUDRegionManager


OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_rows"
    / "raw"
)
CANDIDATE_DIR = (
    PROJECT_ROOT
    / "debug_frames"
    / "row_candidates"
)
NEGATIVE_ROI_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_rows"
    / "negative_rois"
)
ROW_VERIFIER_REAL_DIR = (
    PROJECT_ROOT
    / ".cache"
    / "killfeed_row_verifier"
    / "real"
)
ROW_VERIFIER_FALSE_DIR = (
    PROJECT_ROOT
    / ".cache"
    / "killfeed_row_verifier"
    / "false"
)


def load_config():
    path = PROJECT_ROOT / "config" / "settings.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_real_crop_shapes() -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []

    if not ROW_VERIFIER_REAL_DIR.exists():
        return shapes

    for path in ROW_VERIFIER_REAL_DIR.glob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            continue

        h, w = image.shape[:2]
        if h > 0 and w > 0:
            shapes.append((w, h))

    return shapes


def save_confirmed_negative(
    image,
    real_crop_shapes: list[tuple[int, int]],
    crops_per_roi: int = 4,
) -> None:
    """Save a user-confirmed empty kill-feed ROI and realistic false crops."""
    NEGATIVE_ROI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ROW_VERIFIER_FALSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = int(time.time() * 1000)
    roi_path = NEGATIVE_ROI_DIR / f"negative_roi_{stamp}.png"
    cv2.imwrite(str(roi_path), image)

    roi_h, roi_w = image.shape[:2]
    rng = random.Random(stamp)
    saved = 0

    for index in range(crops_per_roi):
        if real_crop_shapes:
            base_w, base_h = rng.choice(real_crop_shapes)
            scale = rng.uniform(0.92, 1.08)
            crop_w = int(round(base_w * scale))
            crop_h = int(round(base_h * scale))
        else:
            crop_w = int(round(roi_w * rng.uniform(0.42, 0.82)))
            crop_h = int(round(roi_h * rng.uniform(0.18, 0.32)))

        crop_w = max(24, min(crop_w, roi_w))
        crop_h = max(16, min(crop_h, roi_h))

        # Kill-feed rows are normally right-aligned, so make the negatives
        # resemble the locations the verifier will actually see.
        max_right_gap = max(0, int(round(roi_w * 0.10)))
        right_gap = rng.randint(0, max_right_gap) if max_right_gap else 0
        x2 = max(crop_w, roi_w - right_gap)
        x1 = max(0, x2 - crop_w)

        max_y = max(0, roi_h - crop_h)
        y1 = rng.randint(0, max_y) if max_y else 0
        y2 = y1 + crop_h

        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        path = ROW_VERIFIER_FALSE_DIR / (
            f"row_{stamp}_negative_{index}.png"
        )
        if cv2.imwrite(str(path), crop):
            saved += 1

    print(
        f"[negative] saved {roi_path.name} and "
        f"{saved} confirmed FALSE row-sized crop(s)"
    )


def save_candidate_crops(image, candidates, stamp: int) -> int:
    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0

    for index, item in enumerate(candidates):
        box = item.bbox

        crop = image[
            max(0, box.y1):max(0, box.y2),
            max(0, box.x1):max(0, box.x2),
        ]

        if crop is None or crop.size == 0:
            continue

        path = CANDIDATE_DIR / (
            f"candidate_{stamp}_{index}_{item.score:.2f}.png"
        )

        if cv2.imwrite(str(path), crop):
            saved += 1

    return saved


def save_sample(image, candidates):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = int(time.time() * 1000)
    image_path = OUTPUT_DIR / f"{stamp}.png"
    metadata_path = OUTPUT_DIR / f"{stamp}.json"

    cv2.imwrite(
        str(image_path),
        image,
    )

    payload = {
        "image": image_path.name,
        "weak_labels": [
            {
                "bbox": {
                    "x1": item.bbox.x1,
                    "y1": item.bbox.y1,
                    "x2": item.bbox.x2,
                    "y2": item.bbox.y2,
                },
                "killer_team": item.killer_team,
                "victim_team": item.victim_team,
                "score": item.score,
            }
            for item in candidates
        ],
    }

    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_count = save_candidate_crops(
        image,
        candidates,
        stamp,
    )

    print(
        f"[dataset] saved {image_path.name} "
        f"with {len(candidates)} weak row label(s); "
        f"{candidate_count} candidate crop(s)"
    )


def main():
    config = load_config()

    capture = OverwatchCapture(config)
    regions = HUDRegionManager(config)
    detector = KillFeedRowDetector(config)
    real_crop_shapes = load_real_crop_shapes()

    auto_save = False
    last_auto_save = 0.0
    auto_interval = 0.55

    print("Kill-feed dataset collector")
    print("S = save current ROI for the future localization dataset")
    print("N = CONFIRM no kill-feed row is visible; save FALSE negatives")
    print("A = toggle auto-save while rows are visible")
    print("Q = quit")
    print()
    print(
        "Saved full ROI images go to "
        "datasets/killfeed_rows/raw/"
    )
    print(
        "Every proposed row crop also goes to "
        "debug_frames/row_candidates/ for verifier labeling."
    )
    print(
        "N saves the empty ROI plus 4 row-sized background crops directly "
        "as confirmed FALSE verifier examples."
    )

    while True:
        frame = capture.grab()
        region = regions.killfeed_search_region(
            frame.image
        )

        candidates, _, _ = detector.detect(
            region.image
        )

        preview = region.image.copy()

        for candidate in candidates:
            box = candidate.bbox

            cv2.rectangle(
                preview,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                preview,
                f"{candidate.score:.2f}",
                (
                    box.x1,
                    max(16, box.y1 - 3),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        status = (
            "AUTO ON"
            if auto_save
            else "AUTO OFF"
        )
        cv2.putText(
            preview,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Kill Feed Dataset Collector",
            preview,
        )

        now = time.monotonic()

        if (
            auto_save
            and candidates
            and now - last_auto_save
            >= auto_interval
        ):
            save_sample(
                region.image,
                candidates,
            )
            last_auto_save = now

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key in (ord("a"), ord("A")):
            auto_save = not auto_save
            print(
                "[dataset] auto-save "
                + ("ON" if auto_save else "OFF")
            )

        if key in (ord("s"), ord("S")):
            save_sample(
                region.image,
                candidates,
            )

        if key in (ord("n"), ord("N")):
            save_confirmed_negative(
                region.image,
                real_crop_shapes,
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
