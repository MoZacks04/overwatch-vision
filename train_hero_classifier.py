from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_hero_identity"
    / "by_class"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "killfeed_hero_classifier.pt"
)
DEFAULT_LABELS = (
    PROJECT_ROOT
    / "models"
    / "killfeed_hero_classifier_labels.json"
)
IDENTITY_ANNOTATION_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_hero_identity"
    / "annotations"
)
SOURCE_ROW_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_hero_icons"
    / "images"
)


@lru_cache(maxsize=4096)
def load_image(
    path_text: str,
    size: int,
    context_fraction: float,
) -> np.ndarray:
    """
    Rebuild each training crop from the original kill-feed row when metadata
    is available. This lets us add real surrounding pixels without asking the
    user to relabel anything.
    """
    path = Path(path_text)
    image = None

    annotation = (
        IDENTITY_ANNOTATION_DIR
        / f"{path.stem}.json"
    )

    if annotation.exists():
        try:
            payload = json.loads(
                annotation.read_text(
                    encoding="utf-8"
                )
            )
            source_name = str(
                payload.get(
                    "source_row_image",
                    "",
                )
            ).strip()
            box = payload.get(
                "source_box",
                {},
            )
            source_path = (
                SOURCE_ROW_DIR
                / source_name
            )
            row = cv2.imread(
                str(source_path),
                cv2.IMREAD_COLOR,
            )

            if (
                row is not None
                and all(
                    key in box
                    for key in (
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    )
                )
            ):
                h, w = row.shape[:2]
                x1 = int(box["x1"])
                y1 = int(box["y1"])
                x2 = int(box["x2"])
                y2 = int(box["y2"])

                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                pad_x = int(
                    round(
                        box_w
                        * context_fraction
                    )
                )
                pad_y = int(
                    round(
                        box_h
                        * context_fraction
                    )
                )

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                if x2 > x1 and y2 > y1:
                    image = row[
                        y1:y2,
                        x1:x2,
                    ].copy()
        except Exception:
            image = None

    if image is None:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

    if image is None:
        raise ValueError(
            f"Could not read {path}"
        )

    # Keep the whole portrait/context crop. Center-cropping effectively zoomed
    # in and made the classifier sensitive to small detector-box differences.
    h, w = image.shape[:2]
    side = max(h, w)

    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left

    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    return cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )


def augment(image: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)

    contrast = random.uniform(0.85, 1.15)
    brightness = random.uniform(-16.0, 16.0)
    out = out * contrast + brightness
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Team-color backgrounds can otherwise become an accidental shortcut.
    # Mild hue/saturation jitter keeps hero structure useful while making
    # red-vs-blue framing much less predictive.
    if random.random() < 0.70:
        hsv = cv2.cvtColor(
            out,
            cv2.COLOR_BGR2HSV,
        ).astype(np.float32)

        hue_shift = random.uniform(
            -4.0,
            4.0,
        )
        saturation_scale = random.uniform(
            0.60,
            1.35,
        )

        hsv[:, :, 0] = (
            hsv[:, :, 0]
            + hue_shift
        ) % 180.0
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1]
            * saturation_scale,
            0,
            255,
        )

        out = cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2BGR,
        )

    if random.random() < 0.45:
        sigma = random.uniform(0.2, 0.8)
        out = cv2.GaussianBlur(
            out,
            (3, 3),
            sigma,
        )

    h, w = out.shape[:2]
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    angle = random.uniform(-2.0, 2.0)
    scale = random.uniform(0.90, 1.06)

    matrix = cv2.getRotationMatrix2D(
        (w / 2.0, h / 2.0),
        angle,
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
    classes = []
    paths_by_class = {}

    if not root.exists():
        raise SystemExit(
            f"Dataset folder does not exist: {root}\n"
            "Run label_killfeed_hero_identities.py first."
        )

    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue

        images = sorted(
            path
            for path in directory.glob("*.png")
            if path.is_file()
        )

        if images:
            classes.append(directory.name)
            paths_by_class[directory.name] = images

    if len(classes) < 2:
        raise SystemExit(
            "Need labeled samples for at least two heroes."
        )

    return classes, paths_by_class


def split_paths(
    classes,
    paths_by_class,
    validation_fraction: float,
):
    train = []
    validation = []

    for class_index, name in enumerate(classes):
        paths = list(paths_by_class[name])
        random.shuffle(paths)

        if len(paths) >= 5:
            val_count = max(
                1,
                int(
                    round(
                        len(paths)
                        * validation_fraction
                    )
                ),
            )
        else:
            val_count = 0

        validation.extend(
            (path, class_index)
            for path in paths[:val_count]
        )
        train.extend(
            (path, class_index)
            for path in paths[val_count:]
        )

    return train, validation


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny CNN on labeled Overwatch kill-feed hero icons."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--context-fraction",
        type=float,
        default=0.12,
        help=(
            "Fraction of the labeled hero box to include as real surrounding "
            "context on each side."
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

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    classes, paths_by_class = discover_dataset(
        args.data
    )

    print()
    print(
        "Training crop context: "
        f"{float(args.context_fraction):.0%} per side "
        "(rebuilt from original row annotations when available)"
    )
    print("Labeled hero classes:")
    for name in classes:
        count = len(paths_by_class[name])
        warning = (
            "  <-- collect more"
            if count < 3
            else ""
        )
        print(
            f"  {name:<18} {count:>3} samples"
            f"{warning}"
        )

    train_items, validation_items = split_paths(
        classes,
        paths_by_class,
        args.validation_fraction,
    )

    if len(train_items) < len(classes):
        raise SystemExit(
            "Not enough training images after splitting."
        )

    class IconDataset(Dataset):
        def __init__(self, items, training):
            self.items = items
            self.training = training

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, label = self.items[index]
            image = load_image(
                str(path),
                args.input_size,
                float(
                    args.context_fraction
                ),
            ).copy()

            if self.training:
                image = augment(image)

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            tensor = (
                torch.from_numpy(
                    image.astype(np.float32)
                    / 255.0
                )
                .permute(2, 0, 1)
            )

            return tensor, label

    class TinyHeroCNN(nn.Module):
        def __init__(self, class_count):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(
                    3,
                    24,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(
                    24,
                    48,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(
                    48,
                    96,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(
                    96,
                    128,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(
                    (1, 1)
                ),
            )

            self.classifier = nn.Linear(
                128,
                class_count,
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    train_loader = DataLoader(
        IconDataset(
            train_items,
            training=True,
        ),
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=0,
    )

    validation_loader = (
        DataLoader(
            IconDataset(
                validation_items,
                training=False,
            ),
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=0,
        )
        if validation_items
        else None
    )

    model = TinyHeroCNN(
        len(classes)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0015,
        weight_decay=0.0005,
    )
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item())
                * images.shape[0]
            )
            total_seen += images.shape[0]

        train_loss = (
            total_loss / max(1, total_seen)
        )

        validation_accuracy = -1.0

        if validation_loader is not None:
            model.eval()
            correct = 0
            count = 0

            with torch.no_grad():
                for images, labels in validation_loader:
                    prediction = model(
                        images
                    ).argmax(dim=1)

                    correct += int(
                        (prediction == labels)
                        .sum()
                        .item()
                    )
                    count += labels.numel()

            validation_accuracy = (
                correct / max(1, count)
            )

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_state = {
                    key: value.detach().clone()
                    for key, value
                    in model.state_dict().items()
                }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):
            if validation_loader is not None:
                print(
                    f"epoch {epoch:>3}/{args.epochs} "
                    f"loss={train_loss:.4f} "
                    f"val_acc="
                    f"{validation_accuracy:.1%}"
                )
            else:
                print(
                    f"epoch {epoch:>3}/{args.epochs} "
                    f"loss={train_loss:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    args.model.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.labels.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    example = torch.zeros(
        1,
        3,
        args.input_size,
        args.input_size,
    )
    traced = torch.jit.trace(
        model,
        example,
    )
    traced.save(
        str(args.model)
    )

    args.labels.write_text(
        json.dumps(
            classes,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Training complete.")
    print(f"Model:  {args.model}")
    print(f"Labels: {args.labels}")

    if best_accuracy >= 0:
        print(
            "Best held-out accuracy: "
            f"{best_accuracy:.1%}"
        )

    print()
    print(
        "Restart Overwatch Vision. The runtime will automatically "
        "prefer this trained classifier over generic portrait matching."
    )


if __name__ == "__main__":
    main()
