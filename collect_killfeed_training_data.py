from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import yaml

from overwatch_vision.capture import OverwatchCapture
from overwatch_vision.killfeed.row_detector import KillFeedRowDetector
from overwatch_vision.regions import HUDRegionManager


PROJECT_ROOT = Path(__file__).resolve().parent
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


def load_config():
    path = PROJECT_ROOT / "config" / "settings.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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

    auto_save = False
    last_auto_save = 0.0
    auto_interval = 0.55

    print("Kill-feed dataset collector")
    print("S = save current ROI")
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

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
