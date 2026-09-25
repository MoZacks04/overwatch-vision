from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / ".cache" / "killfeed_row_verifier"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "killfeed_row_verifier.pt"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "killfeed_row_verifier_labels.json"

CLASS_NAMES = ["false", "real"]


def timestamp_from_path(path: Path) -> int | None:
    match = re.search(r"row_(\d+)_", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_row_image(
    path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape for {path}")

    scale = min(
        width / float(w),
        height / float(h),
    )
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        ),
    )

    canvas = np.full(
        (height, width, 3),
        24,
        dtype=np.uint8,
    )

    x1 = max(0, (width - new_w) // 2)
    y1 = max(0, (height - new_h) // 2)

    canvas[
        y1:y1 + new_h,
        x1:x1 + new_w,
    ] = resized

    return canvas


def augment(image: np.ndarray) -> np.ndarray:
    out = image.copy()

    if random.random() < 0.85:
        out = out.astype(np.float32)
        contrast = random.uniform(0.82, 1.18)
        brightness = random.uniform(-18.0, 18.0)
        out = np.clip(
            out * contrast + brightness,
            0,
            255,
        ).astype(np.uint8)

    if random.random() < 0.65:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue_shift = random.randint(-4, 4)
        sat_scale = random.uniform(0.72, 1.28)

        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.float32) * sat_scale,
            0,
            255,
        ).astype(np.int16)

        out = cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2BGR,
        )

    if random.random() < 0.18:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = out.shape[:2]
    dx = random.randint(-5, 5)
    dy = random.randint(-2, 2)
    scale = random.uniform(0.96, 1.04)

    matrix = cv2.getRotationMatrix2D(
        (w / 2.0, h / 2.0),
        random.uniform(-1.0, 1.0),
        scale,
    )
    matrix[0, 2] += dx
    matrix[1, 2] += dy

    out = cv2.warpAffine(
        out,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if random.random() < 0.35:
        out = cv2.GaussianBlur(
            out,
            (3, 3),
            random.uniform(0.15, 0.75),
        )

    if random.random() < 0.25:
        noise = np.random.normal(
            0.0,
            random.uniform(1.0, 4.0),
            out.shape,
        )
        out = np.clip(
            out.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)

    return out


def discover_dataset(root: Path):
    paths_by_class: dict[str, list[Path]] = {}

    for name in CLASS_NAMES:
        directory = root / name
        paths = (
            sorted(path for path in directory.glob("*.png") if path.is_file())
            if directory.exists()
            else []
        )
        paths_by_class[name] = paths

    if not paths_by_class["real"] or not paths_by_class["false"]:
        raise SystemExit(
            "Need both REAL and FALSE labeled row crops. "
            "Run label_killfeed_rows.py first."
        )

    return paths_by_class


def temporal_groups(
    paths: list[Path],
    max_gap_ms: int,
) -> list[list[Path]]:
    stamped = []
    unstamped = []

    for path in paths:
        stamp = timestamp_from_path(path)
        if stamp is None:
            unstamped.append(path)
        else:
            stamped.append((stamp, path))

    stamped.sort(key=lambda item: item[0])

    groups: list[list[Path]] = []
    current: list[Path] = []
    previous_stamp: int | None = None

    for stamp, path in stamped:
        if (
            current
            and previous_stamp is not None
            and stamp - previous_stamp > max_gap_ms
        ):
            groups.append(current)
            current = []

        current.append(path)
        previous_stamp = stamp

    if current:
        groups.append(current)

    for path in unstamped:
        groups.append([path])

    return groups


def split_grouped(
    paths_by_class: dict[str, list[Path]],
    validation_fraction: float,
    group_gap_ms: int,
):
    train = []
    validation = []
    group_counts = {}

    for class_index, name in enumerate(CLASS_NAMES):
        groups = temporal_groups(
            paths_by_class[name],
            max_gap_ms=group_gap_ms,
        )
        group_counts[name] = len(groups)

        random.shuffle(groups)

        if len(groups) >= 2:
            val_group_count = max(
                1,
                int(round(len(groups) * validation_fraction)),
            )
            val_group_count = min(
                val_group_count,
                len(groups) - 1,
            )
        else:
            val_group_count = 0

        val_groups = groups[:val_group_count]
        train_groups = groups[val_group_count:]

        for group in train_groups:
            train.extend(
                (path, class_index)
                for path in group
            )

        for group in val_groups:
            validation.extend(
                (path, class_index)
                for path in group
            )

    random.shuffle(train)
    random.shuffle(validation)

    return train, validation, group_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary CNN that verifies whether a proposed crop is a "
            "real Overwatch kill-feed row."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--allow-small-negative-class",
        action="store_true",
        help=(
            "Allow training with fewer than 20 FALSE examples. "
            "Not recommended because the verifier can become overconfident."
        ),
    )
    parser.add_argument(
        "--group-gap-ms",
        type=int,
        default=2500,
        help=(
            "Adjacent captures closer than this are kept in the same "
            "train/validation group to reduce near-duplicate leakage."
        ),
    )
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required for training. "
            f"Current error: {exc}"
        )

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    paths_by_class = discover_dataset(args.data)

    print()
    print("Kill-feed row verifier dataset:")
    for name in CLASS_NAMES:
        print(
            f"  {name:<8} {len(paths_by_class[name]):>4} samples"
        )

    false_count = len(paths_by_class["false"])
    real_count = len(paths_by_class["real"])

    if false_count < 20 and not args.allow_small_negative_class:
        raise SystemExit(
            "Only "
            f"{false_count} FALSE examples are labeled. "
            "Do not train the verifier yet: collect and label at least "
            "20 real false-positive proposals (50+ is better). "
            "Use collect_killfeed_training_data.py with auto-save, then run "
            "label_killfeed_rows.py --source debug_frames\\row_candidates. "
            "If you intentionally want an experimental run anyway, pass "
            "--allow-small-negative-class."
        )

    if min(false_count, real_count) < 10:
        print(
            "WARNING: one class has fewer than 10 examples. "
            "The first model may be unstable."
        )

    train_items, validation_items, group_counts = split_grouped(
        paths_by_class,
        validation_fraction=args.validation_fraction,
        group_gap_ms=args.group_gap_ms,
    )

    print()
    print("Temporal groups:")
    for name in CLASS_NAMES:
        print(f"  {name:<8} {group_counts[name]:>4} groups")
    print(
        f"Train images: {len(train_items)} | "
        f"Validation images: {len(validation_items)}"
    )

    if not train_items:
        raise SystemExit("No training images available after splitting.")

    class RowDataset(Dataset):
        def __init__(self, items, training: bool):
            self.items = items
            self.training = training

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, label = self.items[index]
            image = load_row_image(
                path,
                args.input_width,
                args.input_height,
            )

            if self.training:
                image = augment(image)

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            tensor = (
                torch.from_numpy(
                    image.astype(np.float32) / 255.0
                )
                .permute(2, 0, 1)
            )

            return tensor, label

    class TinyRowVerifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(24, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(48, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(96, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.classifier = nn.Linear(128, 2)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    train_counts = [0, 0]
    for _, label in train_items:
        train_counts[label] += 1

    total_train = sum(train_counts)
    class_weights = []
    for count in train_counts:
        class_weights.append(
            total_train / max(1.0, 2.0 * count)
        )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
        )
    )

    train_loader = DataLoader(
        RowDataset(train_items, training=True),
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=0,
    )

    validation_loader = (
        DataLoader(
            RowDataset(validation_items, training=False),
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=0,
        )
        if validation_items
        else None
    )

    model = TinyRowVerifier()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0012,
        weight_decay=0.0007,
    )

    best_state = None
    best_balanced_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * images.shape[0]
            total_seen += images.shape[0]

        train_loss = total_loss / max(1, total_seen)

        val_accuracy = -1.0
        balanced_accuracy = -1.0
        recalls = [0.0, 0.0]
        confusion = [[0, 0], [0, 0]]

        if validation_loader is not None:
            model.eval()

            with torch.no_grad():
                for images, labels in validation_loader:
                    predictions = model(images).argmax(dim=1)

                    for truth, prediction in zip(
                        labels.tolist(),
                        predictions.tolist(),
                    ):
                        confusion[truth][prediction] += 1

            total = sum(sum(row) for row in confusion)
            correct = confusion[0][0] + confusion[1][1]
            val_accuracy = correct / max(1, total)

            for class_index in range(2):
                class_total = sum(confusion[class_index])
                recalls[class_index] = (
                    confusion[class_index][class_index]
                    / max(1, class_total)
                )

            balanced_accuracy = sum(recalls) / 2.0

            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):
            if validation_loader is None:
                print(
                    f"epoch {epoch:>3}/{args.epochs} "
                    f"loss={train_loss:.4f}"
                )
            else:
                print(
                    f"epoch {epoch:>3}/{args.epochs} "
                    f"loss={train_loss:.4f} "
                    f"val_acc={val_accuracy:.1%} "
                    f"balanced={balanced_accuracy:.1%} "
                    f"false_recall={recalls[0]:.1%} "
                    f"real_recall={recalls[1]:.1%}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(
        1,
        3,
        args.input_height,
        args.input_width,
    )

    traced = torch.jit.trace(model, example)
    traced.save(str(args.model))

    args.labels.write_text(
        json.dumps(
            {
                "classes": CLASS_NAMES,
                "input_width": args.input_width,
                "input_height": args.input_height,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Training complete.")
    print(f"Model:  {args.model}")
    print(f"Labels: {args.labels}")

    if best_balanced_accuracy >= 0:
        print(
            "Best held-out balanced accuracy: "
            f"{best_balanced_accuracy:.1%}"
        )

    print()
    print(
        "Restart Overwatch Vision. If the verifier model exists, proposed "
        "color-detector rows must also pass the learned row check."
    )


if __name__ == "__main__":
    main()
