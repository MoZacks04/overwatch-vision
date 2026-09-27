from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "killfeed_hero_icons"
SOURCE_IMAGE_DIR = SOURCE_ROOT / "images"
SOURCE_ANNOTATION_DIR = SOURCE_ROOT / "annotations"

DATASET_ROOT = PROJECT_ROOT / "datasets" / "killfeed_hero_identity"
IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"
BY_CLASS_DIR = DATASET_ROOT / "by_class"

WINDOW_NAME = "Kill Feed Hero Identity Labeler"
CANVAS_W = 1100
CANVAS_H = 760


def safe_label(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9 ._-]", "", value)
    return value.strip()


def annotation_path(sample_id: str) -> Path:
    return ANNOTATION_DIR / f"{sample_id}.json"


def known_labels() -> list[str]:
    labels = set()

    if BY_CLASS_DIR.exists():
        for path in BY_CLASS_DIR.iterdir():
            if path.is_dir():
                labels.add(path.name)

    # Reuse spellings from the older template dataset when available.
    old_templates = PROJECT_ROOT / ".cache" / "killfeed_hero_templates"
    if old_templates.exists():
        for path in old_templates.iterdir():
            if path.is_dir():
                labels.add(path.name)

    return sorted(labels, key=str.casefold)


def canonicalize_label(value: str) -> str:
    cleaned = safe_label(value)

    if not cleaned:
        return ""

    folded = cleaned.casefold()
    labels = known_labels()

    for label in labels:
        if label.casefold() == folded:
            return label

    # If the typed text is a unique prefix of a label we've already used,
    # accept it. This makes repeated labeling much faster without guessing.
    matches = [
        label
        for label in labels
        if label.casefold().startswith(folded)
    ]

    if len(matches) == 1:
        return matches[0]

    return cleaned


def load_samples():
    samples = []

    if not SOURCE_ANNOTATION_DIR.exists():
        return samples

    for source_annotation in sorted(
        SOURCE_ANNOTATION_DIR.glob("*.json")
    ):
        try:
            payload = json.loads(
                source_annotation.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        image_name = str(payload.get("image", "")).strip()
        image_path = SOURCE_IMAGE_DIR / image_name

        if not image_name or not image_path.exists():
            continue

        boxes = []

        for item in payload.get("boxes", []):
            try:
                box = [
                    int(item["x1"]),
                    int(item["y1"]),
                    int(item["x2"]),
                    int(item["y2"]),
                ]
            except Exception:
                continue

            boxes.append(box)

        boxes.sort(
            key=lambda box: (
                box[0] + box[2]
            ) / 2.0
        )

        for icon_index, box in enumerate(boxes, start=1):
            role = "unknown"

            if len(boxes) >= 2:
                if icon_index == 1:
                    role = "killer"
                elif icon_index == len(boxes):
                    role = "victim"

            sample_id = (
                f"{Path(image_name).stem}"
                f"__icon{icon_index:02d}"
            )

            samples.append(
                {
                    "sample_id": sample_id,
                    "row_image_path": image_path,
                    "row_image_name": image_name,
                    "source_annotation": source_annotation.name,
                    "icon_index": icon_index,
                    "icon_count": len(boxes),
                    "role": role,
                    "box": box,
                }
            )

    return samples


def extract_crop(row_image: np.ndarray, box):
    h, w = row_image.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return row_image[y1:y2, x1:x2].copy()


def remove_other_class_copies(filename: str, keep_label: str):
    if not BY_CLASS_DIR.exists():
        return

    for directory in BY_CLASS_DIR.iterdir():
        if (
            not directory.is_dir()
            or directory.name == keep_label
        ):
            continue

        candidate = directory / filename

        if candidate.exists():
            candidate.unlink()


def save_sample(sample, crop, label: str):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    BY_CLASS_DIR.mkdir(parents=True, exist_ok=True)

    sample_id = sample["sample_id"]
    image_target = IMAGE_DIR / f"{sample_id}.png"
    cv2.imwrite(str(image_target), crop)

    destination_dir = BY_CLASS_DIR / label
    destination_dir.mkdir(parents=True, exist_ok=True)

    remove_other_class_copies(
        image_target.name,
        label,
    )
    shutil.copy2(
        image_target,
        destination_dir / image_target.name,
    )

    payload = {
        "image": image_target.name,
        "hero": label,
        "role": sample["role"],
        "source_row_image": sample["row_image_name"],
        "source_annotation": sample["source_annotation"],
        "source_icon_index": int(sample["icon_index"]),
        "source_icon_count": int(sample["icon_count"]),
        "source_box": {
            "x1": int(sample["box"][0]),
            "y1": int(sample["box"][1]),
            "x2": int(sample["box"][2]),
            "y2": int(sample["box"][3]),
        },
    }

    target = annotation_path(sample_id)
    temp = target.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temp.replace(target)


def mark_skipped(sample):
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": None,
        "hero": None,
        "skipped": True,
        "role": sample["role"],
        "source_row_image": sample["row_image_name"],
        "source_annotation": sample["source_annotation"],
        "source_icon_index": int(sample["icon_index"]),
        "source_icon_count": int(sample["icon_count"]),
        "source_box": {
            "x1": int(sample["box"][0]),
            "y1": int(sample["box"][1]),
            "x2": int(sample["box"][2]),
            "y2": int(sample["box"][3]),
        },
    }

    target = annotation_path(sample["sample_id"])
    temp = target.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temp.replace(target)


def make_preview(
    row_image,
    crop,
    sample,
    typed,
    index,
    total,
    last_label,
):
    canvas = np.full(
        (CANVAS_H, CANVAS_W, 3),
        24,
        dtype=np.uint8,
    )

    cv2.putText(
        canvas,
        (
            f"Hero {index}/{total} | "
            f"role={sample['role']} | "
            f"row icons={sample['icon_count']}"
        ),
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        sample["sample_id"][:105],
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    # Row context with the target icon highlighted.
    row_h, row_w = row_image.shape[:2]
    max_row_w = CANVAS_W - 80
    row_scale = min(
        2.2,
        max_row_w / max(1, row_w),
    )
    row_display_w = max(1, int(round(row_w * row_scale)))
    row_display_h = max(1, int(round(row_h * row_scale)))

    row_view = cv2.resize(
        row_image,
        (row_display_w, row_display_h),
        interpolation=cv2.INTER_NEAREST,
    )

    row_x = (CANVAS_W - row_display_w) // 2
    row_y = 100

    canvas[
        row_y:row_y + row_display_h,
        row_x:row_x + row_display_w,
    ] = row_view

    x1, y1, x2, y2 = sample["box"]

    cv2.rectangle(
        canvas,
        (
            row_x + int(round(x1 * row_scale)),
            row_y + int(round(y1 * row_scale)),
        ),
        (
            row_x + int(round(x2 * row_scale)),
            row_y + int(round(y2 * row_scale)),
        ),
        (0, 255, 255),
        2,
    )

    # Large exact crop below the row context.
    crop_h, crop_w = crop.shape[:2]
    max_crop_w = 360
    max_crop_h = 360

    crop_scale = min(
        max_crop_w / max(1, crop_w),
        max_crop_h / max(1, crop_h),
    )
    crop_scale = max(1.0, crop_scale)

    crop_display_w = max(
        1,
        int(round(crop_w * crop_scale)),
    )
    crop_display_h = max(
        1,
        int(round(crop_h * crop_scale)),
    )

    crop_view = cv2.resize(
        crop,
        (crop_display_w, crop_display_h),
        interpolation=cv2.INTER_NEAREST,
    )

    crop_x = (
        CANVAS_W - crop_display_w
    ) // 2
    crop_y = min(
        310,
        row_y + row_display_h + 45,
    )

    canvas[
        crop_y:crop_y + crop_display_h,
        crop_x:crop_x + crop_display_w,
    ] = crop_view

    cv2.rectangle(
        canvas,
        (crop_x, crop_y),
        (
            crop_x + crop_display_w,
            crop_y + crop_display_h,
        ),
        (0, 255, 255),
        2,
    )

    footer_y = CANVAS_H - 105
    cv2.line(
        canvas,
        (0, footer_y),
        (CANVAS_W, footer_y),
        (70, 70, 70),
        1,
    )

    cv2.putText(
        canvas,
        f"Hero: {typed}",
        (24, footer_y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    repeat_text = (
        last_label
        if last_label
        else "(none yet)"
    )

    cv2.putText(
        canvas,
        (
            "ENTER save | empty ENTER skip | BACKSPACE edit | "
            f"R repeat last [{repeat_text}] | ESC quit"
        ),
        (24, footer_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return canvas


def prompt_label(
    row_image,
    crop,
    sample,
    index,
    total,
    last_label,
):
    typed = ""

    while True:
        preview = make_preview(
            row_image,
            crop,
            sample,
            typed,
            index,
            total,
            last_label,
        )
        cv2.imshow(
            WINDOW_NAME,
            preview,
        )

        if (
            cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            )
            < 1
        ):
            return "quit", None

        key = cv2.waitKeyEx(40)

        if key == -1:
            continue

        code = key & 0xFF

        if code == 27:
            return "quit", None

        if (
            code in (ord("r"), ord("R"))
            and not typed
            and last_label
        ):
            return "label", last_label

        if code in (10, 13):
            if not typed.strip():
                return "skip", None

            label = canonicalize_label(
                typed
            )

            if label:
                return "label", label

            continue

        if code in (8, 127):
            typed = typed[:-1]
            continue

        if 32 <= code <= 126:
            char = chr(code)

            if (
                char == " "
                and typed
                and not typed.endswith(" ")
            ):
                typed += " "
            elif char != " ":
                typed = safe_label(
                    typed + char
                )


def dataset_totals():
    labeled = 0
    skipped = 0
    by_hero = {}

    if not ANNOTATION_DIR.exists():
        return labeled, skipped, by_hero

    for path in ANNOTATION_DIR.glob(
        "*.json"
    ):
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        hero = payload.get("hero")

        if hero:
            hero = str(hero)
            labeled += 1
            by_hero[hero] = (
                by_hero.get(hero, 0)
                + 1
            )
        elif payload.get("skipped"):
            skipped += 1

    return labeled, skipped, by_hero


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Name exact hero portrait crops generated from the saved "
            "kill-feed hero-icon bounding boxes."
        )
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Revisit samples previously marked skipped.",
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Show every icon again, including already labeled samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_samples()

    if not samples:
        print(
            "No hero-icon boxes found. Expected "
            "datasets/killfeed_hero_icons/annotations."
        )
        return

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    BY_CLASS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending = []

    for sample in samples:
        saved = annotation_path(
            sample["sample_id"]
        )

        if args.relabel_all:
            pending.append(sample)
            continue

        if not saved.exists():
            pending.append(sample)
            continue

        if args.include_skipped:
            try:
                payload = json.loads(
                    saved.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                payload = {}

            if payload.get("skipped"):
                pending.append(sample)

    labeled, skipped, by_hero = dataset_totals()

    print("Kill-feed hero identity labeler")
    print(f"Source icon boxes: {len(samples)}")
    print(f"Already labeled: {labeled}")
    print(f"Already skipped: {skipped}")
    print(f"Remaining this run: {len(pending)}")
    print()
    print(
        "Type the HERO NAME shown in the highlighted portrait and press Enter."
    )
    print(
        "The large crop is generated from the exact box you already drew."
    )
    print(
        "Empty Enter = skip if the hero genuinely cannot be identified."
    )
    print(
        "R with an empty input = repeat the previous hero label."
    )
    print(
        "Already-used hero names support unique-prefix completion."
    )
    print()

    if not pending:
        print("Nothing left to label.")
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        CANVAS_W,
        CANVAS_H,
    )

    last_label = None
    processed = 0

    try:
        for index, sample in enumerate(
            pending,
            start=1,
        ):
            row_image = cv2.imread(
                str(sample["row_image_path"]),
                cv2.IMREAD_COLOR,
            )

            if row_image is None:
                print(
                    "[ERROR] Could not read row: "
                    f"{sample['row_image_path']}"
                )
                continue

            crop = extract_crop(
                row_image,
                sample["box"],
            )

            if (
                crop is None
                or crop.size == 0
            ):
                print(
                    "[ERROR] Empty crop: "
                    f"{sample['sample_id']}"
                )
                continue

            action, label = prompt_label(
                row_image,
                crop,
                sample,
                index,
                len(pending),
                last_label,
            )

            if action == "quit":
                print(
                    "Stopped. Saved labels are preserved."
                )
                break

            if action == "skip":
                mark_skipped(sample)
                processed += 1
                continue

            assert label is not None

            save_sample(
                sample,
                crop,
                label,
            )
            last_label = label
            processed += 1

    finally:
        cv2.destroyAllWindows()

    labeled, skipped, by_hero = dataset_totals()

    print()
    print(
        f"Processed this run: {processed}"
    )
    print(
        f"Total labeled: {labeled}"
    )
    print(
        f"Total skipped: {skipped}"
    )
    print("Samples by hero:")

    for hero, count in sorted(
        by_hero.items(),
        key=lambda item: (
            -item[1],
            item[0].casefold(),
        ),
    ):
        print(
            f"  {hero:<20} {count}"
        )

    print()
    print(
        f"Dataset: {DATASET_ROOT}"
    )


if __name__ == "__main__":
    main()
