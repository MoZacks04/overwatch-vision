from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "killfeed_hero_icons"
IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"

CACHE_ROOT = PROJECT_ROOT / ".cache" / "killfeed_hero_icon_yolo"
MODEL_PATH = PROJECT_ROOT / "models" / "killfeed_hero_icon_localizer.pt"


def parse_timestamp(value: str):
    match = re.search(r"(\d{12,})", value)

    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_records():
    records = []

    if not ANNOTATION_DIR.exists():
        return records

    for annotation_path in sorted(ANNOTATION_DIR.glob("*.json")):
        try:
            payload = json.loads(
                annotation_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        image_name = str(
            payload.get("image", "")
        ).strip()
        image_path = IMAGE_DIR / image_name

        if (
            not image_name
            or not image_path.exists()
        ):
            continue

        width = int(
            payload.get("width", 0)
        )
        height = int(
            payload.get("height", 0)
        )

        if width <= 0 or height <= 0:
            continue

        clean_boxes = []

        for item in payload.get("boxes", []):
            try:
                x1 = float(item["x1"])
                y1 = float(item["y1"])
                x2 = float(item["x2"])
                y2 = float(item["y2"])
            except Exception:
                continue

            x1 = max(
                0.0,
                min(float(width), x1),
            )
            x2 = max(
                0.0,
                min(float(width), x2),
            )
            y1 = max(
                0.0,
                min(float(height), y1),
            )
            y2 = max(
                0.0,
                min(float(height), y2),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            clean_boxes.append(
                (x1, y1, x2, y2)
            )

        source_image = str(
            payload.get(
                "source_image",
                image_name,
            )
        )

        records.append(
            {
                "annotation_path": annotation_path,
                "image_path": image_path,
                "image_name": image_name,
                "width": width,
                "height": height,
                "boxes": clean_boxes,
                "timestamp": parse_timestamp(
                    source_image
                ),
                "positive": bool(clean_boxes),
            }
        )

    return records


def temporal_groups(records, max_gap_ms: int):
    stamped = []
    unstamped = []

    for record in records:
        if record["timestamp"] is None:
            unstamped.append(record)
        else:
            stamped.append(record)

    stamped.sort(
        key=lambda item: item["timestamp"]
    )

    groups = []
    current = []
    previous = None

    for record in stamped:
        stamp = record["timestamp"]

        if (
            current
            and previous is not None
            and stamp - previous > max_gap_ms
        ):
            groups.append(current)
            current = []

        current.append(record)
        previous = stamp

    if current:
        groups.append(current)

    for record in unstamped:
        groups.append([record])

    return groups


def split_records(
    records,
    validation_fraction: float,
    group_gap_ms: int,
):
    rng = random.Random(29)

    train = []
    validation = []

    for bucket in (
        [
            record
            for record in records
            if record["positive"]
        ],
        [
            record
            for record in records
            if not record["positive"]
        ],
    ):
        if not bucket:
            continue

        groups = temporal_groups(
            bucket,
            group_gap_ms,
        )
        rng.shuffle(groups)

        target_validation = max(
            1,
            int(
                round(
                    len(bucket)
                    * validation_fraction
                )
            ),
        )

        selected = []
        selected_count = 0

        while (
            groups
            and selected_count
            < target_validation
        ):
            group = groups.pop()
            selected.append(group)
            selected_count += len(group)

        for group in selected:
            validation.extend(group)

        for group in groups:
            train.extend(group)

    rng.shuffle(train)
    rng.shuffle(validation)

    return train, validation


def write_yolo_label(
    path: Path,
    record,
):
    width = float(record["width"])
    height = float(record["height"])

    lines = []

    for x1, y1, x2, y2 in record["boxes"]:
        cx = (
            (x1 + x2)
            / 2.0
            / width
        )
        cy = (
            (y1 + y2)
            / 2.0
            / height
        )
        bw = (
            (x2 - x1)
            / width
        )
        bh = (
            (y2 - y1)
            / height
        )

        lines.append(
            "0 "
            f"{cx:.8f} "
            f"{cy:.8f} "
            f"{bw:.8f} "
            f"{bh:.8f}"
        )

    path.write_text(
        "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )


def build_yolo_cache(
    train_records,
    validation_records,
):
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)

    for split in ("train", "val"):
        (
            CACHE_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        (
            CACHE_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    for split, records in (
        ("train", train_records),
        ("val", validation_records),
    ):
        for index, record in enumerate(records):
            stem = (
                f"{index:06d}_"
                f"{record['image_path'].stem}"
            )

            image_target = (
                CACHE_ROOT
                / "images"
                / split
                / f"{stem}.png"
            )
            label_target = (
                CACHE_ROOT
                / "labels"
                / split
                / f"{stem}.txt"
            )

            shutil.copy2(
                record["image_path"],
                image_target,
            )
            write_yolo_label(
                label_target,
                record,
            )

    dataset_yaml = (
        CACHE_ROOT
        / "dataset.yaml"
    )

    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(
                    CACHE_ROOT.resolve()
                ),
                "train": "images/train",
                "val": "images/val",
                "names": {
                    0: "hero_icon",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return dataset_yaml


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLO detector to localize the two primary hero portraits "
            "inside an already detected Overwatch kill-feed row."
        )
    )
    parser.add_argument(
        "--base-model",
        default="yolo11n.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--group-gap-ms",
        type=int,
        default=2500,
    )
    args = parser.parse_args()

    records = load_records()

    if not records:
        raise SystemExit(
            "No hero-icon localization annotations found. "
            "Run label_killfeed_hero_icons.py first."
        )

    positive = [
        record
        for record in records
        if record["positive"]
    ]
    empty = [
        record
        for record in records
        if not record["positive"]
    ]
    total_boxes = sum(
        len(record["boxes"])
        for record in records
    )

    print()
    print("Hero-icon localization dataset:")
    print(
        f"  labeled rows:     {len(records)}"
    )
    print(
        f"  rows with icons:  {len(positive)}"
    )
    print(
        f"  zero-icon rows:   {len(empty)}"
    )
    print(
        f"  hero boxes:       {total_boxes}"
    )

    if len(positive) < 50:
        raise SystemExit(
            "Need at least 50 labeled rows containing hero icons before "
            "training. For a useful first model, 200+ varied rows is better."
        )

    train_records, validation_records = (
        split_records(
            records,
            args.validation_fraction,
            args.group_gap_ms,
        )
    )

    if (
        not train_records
        or not validation_records
    ):
        raise SystemExit(
            "Could not make a non-empty train/validation split."
        )

    print(
        f"  train:            {len(train_records)}"
    )
    print(
        f"  validation:       {len(validation_records)}"
    )
    print()

    dataset_yaml = build_yolo_cache(
        train_records,
        validation_records,
    )

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "Ultralytics/PyTorch are required. "
            "Install requirements-training.txt first. "
            f"Current error: {exc}"
        )

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training on "
        + (
            "GPU"
            if device != "cpu"
            else "CPU"
        )
        + f" using {args.base_model}."
    )

    model = YOLO(
        args.base_model
    )

    results = model.train(
        data=str(dataset_yaml),
        epochs=max(1, args.epochs),
        imgsz=max(160, args.imgsz),
        batch=max(1, args.batch_size),
        device=device,
        workers=0,
        patience=18,
        project=str(
            CACHE_ROOT
            / "runs"
        ),
        name="hero_icon_localizer",
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(
        str(results.save_dir)
    )
    best = (
        save_dir
        / "weights"
        / "best.pt"
    )

    if not best.exists():
        raise SystemExit(
            f"Training finished but best.pt was not found at: {best}"
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        best,
        MODEL_PATH,
    )

    print()
    print("Training complete.")
    print(
        f"Best model copied to: {MODEL_PATH}"
    )
    print()
    print(
        "Do not delete datasets/killfeed_hero_icons. "
        "Add more corrected boxes later and rerun this trainer."
    )


if __name__ == "__main__":
    main()
