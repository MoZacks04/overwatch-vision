from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "killfeed_localization"
IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"
CACHE_ROOT = PROJECT_ROOT / ".cache" / "killfeed_localizer_yolo"
MODEL_PATH = PROJECT_ROOT / "models" / "killfeed_row_localizer.pt"


def parse_timestamp(name: str):
    match = re.search(r"(\d{12,})", name)
    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_records():
    records = []

    for annotation_path in sorted(ANNOTATION_DIR.glob("*.json")):
        try:
            payload = json.loads(
                annotation_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        image_name = str(payload.get("image", "")).strip()
        image_path = IMAGE_DIR / image_name

        if not image_name or not image_path.exists():
            continue

        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
        boxes = payload.get("boxes", [])

        if width <= 0 or height <= 0:
            continue

        clean_boxes = []

        for item in boxes:
            try:
                x1 = float(item["x1"])
                y1 = float(item["y1"])
                x2 = float(item["x2"])
                y2 = float(item["y2"])
            except Exception:
                continue

            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            clean_boxes.append((x1, y1, x2, y2))

        records.append(
            {
                "annotation_path": annotation_path,
                "image_path": image_path,
                "width": width,
                "height": height,
                "boxes": clean_boxes,
                "timestamp": parse_timestamp(image_name),
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

    stamped.sort(key=lambda item: item["timestamp"])

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


def split_records(records, validation_fraction: float, group_gap_ms: int):
    positive = [record for record in records if record["positive"]]
    negative = [record for record in records if not record["positive"]]

    train = []
    validation = []

    rng = random.Random(17)

    for bucket in (positive, negative):
        groups = temporal_groups(bucket, group_gap_ms)
        rng.shuffle(groups)

        if len(groups) >= 2:
            val_count = max(
                1,
                int(round(len(groups) * validation_fraction)),
            )
            val_count = min(val_count, len(groups) - 1)
        else:
            val_count = 0

        val_groups = groups[:val_count]
        train_groups = groups[val_count:]

        for group in train_groups:
            train.extend(group)

        for group in val_groups:
            validation.extend(group)

    rng.shuffle(train)
    rng.shuffle(validation)

    return train, validation


def write_yolo_record(record, split: str):
    image_dir = CACHE_ROOT / "images" / split
    label_dir = CACHE_ROOT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_target = image_dir / record["image_path"].name
    label_target = label_dir / f"{record['image_path'].stem}.txt"

    shutil.copy2(record["image_path"], image_target)

    width = float(record["width"])
    height = float(record["height"])

    lines = []

    for x1, y1, x2, y2 in record["boxes"]:
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        lines.append(
            f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
        )

    label_target.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def prepare_dataset(
    validation_fraction: float,
    group_gap_ms: int,
):
    records = load_records()

    positive_count = sum(1 for item in records if item["positive"])
    negative_count = len(records) - positive_count

    if positive_count < 50:
        raise SystemExit(
            "Need at least 50 labeled positive full-ROI images before "
            "training. 150-300+ is recommended for the first useful model."
        )

    if negative_count < 25:
        raise SystemExit(
            "Need at least 25 confirmed-empty full-ROI images before "
            "training. Import N-key negative ROIs first."
        )

    train, validation = split_records(
        records,
        validation_fraction=validation_fraction,
        group_gap_ms=group_gap_ms,
    )

    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)

    for record in train:
        write_yolo_record(record, "train")

    for record in validation:
        write_yolo_record(record, "val")

    data_yaml = CACHE_ROOT / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(CACHE_ROOT),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "killfeed_row"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Localization dataset:")
    print(f"  total ROI images: {len(records)}")
    print(f"  positive:         {positive_count}")
    print(f"  confirmed empty:  {negative_count}")
    print(f"  train:            {len(train)}")
    print(f"  validation:       {len(validation)}")
    print()

    return data_yaml


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a full-ROI object detector for Overwatch kill-feed rows."
        )
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--group-gap-ms", type=int, default=2500)
    parser.add_argument(
        "--base-model",
        default="yolo11n.pt",
        help=(
            "Ultralytics base model. The default downloads pretrained "
            "weights on first use."
        ),
    )
    args = parser.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "The localization trainer needs Ultralytics. Install the "
            "training requirements with:\n"
            ".\\.venv\\Scripts\\python.exe -m pip install -r "
            "requirements-training.txt\n"
            f"Current import error: {exc}"
        )

    data_yaml = prepare_dataset(
        validation_fraction=args.validation_fraction,
        group_gap_ms=args.group_gap_ms,
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    print(
        f"Training on {'GPU' if device == 0 else 'CPU'} using "
        f"{args.base_model}."
    )
    print(
        "The first run may download pretrained base weights. "
        "Training can take a while on CPU."
    )
    print()

    model = YOLO(args.base_model)

    results = model.train(
        data=str(data_yaml),
        epochs=max(1, args.epochs),
        imgsz=max(320, args.imgsz),
        batch=max(1, args.batch),
        device=device,
        workers=0,
        patience=18,
        project=str(CACHE_ROOT / "runs"),
        name="killfeed_row_localizer",
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(str(results.save_dir))
    best = save_dir / "weights" / "best.pt"

    if not best.exists():
        raise SystemExit(
            f"Training finished but best.pt was not found under {save_dir}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODEL_PATH)

    print()
    print("Training complete.")
    print(f"Best model copied to: {MODEL_PATH}")
    print()
    print(
        "Do not delete datasets/killfeed_localization. Add more labeled "
        "images later and rerun this trainer to improve the same model."
    )


if __name__ == "__main__":
    main()
