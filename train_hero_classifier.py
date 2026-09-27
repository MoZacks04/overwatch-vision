from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random
import re

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

            # Live hard examples are already saved as the exact contextual
            # crop seen by the runtime classifier. Do not try to reconstruct
            # them from the original hand-labeled row dataset.
            row = None

            if (
                source_name
                and not bool(
                    payload.get(
                        "hard_example",
                        False,
                    )
                )
            ):
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
                and isinstance(
                    box,
                    dict,
                )
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


@lru_cache(maxsize=4096)
def sample_timestamp(path_text: str) -> int | None:
    """
    Recover the original gameplay-frame timestamp from the saved identity
    annotation/source row name. Nearby frames from the same kill-feed event
    must stay on the same side of the train/validation split.
    """
    path = Path(path_text)
    annotation = (
        IDENTITY_ANNOTATION_DIR
        / f"{path.stem}.json"
    )

    candidates = [path.stem]

    if annotation.exists():
        try:
            payload = json.loads(
                annotation.read_text(
                    encoding="utf-8"
                )
            )
            candidates.append(
                str(
                    payload.get(
                        "source_row_image",
                        "",
                    )
                )
            )
            candidates.append(
                str(
                    payload.get(
                        "source_annotation",
                        "",
                    )
                )
            )
        except Exception:
            pass

    for value in candidates:
        match = re.search(
            r"(\d{12,})",
            value,
        )

        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    return None


def temporal_groups(
    paths,
    max_gap_ms: int,
):
    stamped = []
    unstamped = []

    for path in paths:
        stamp = sample_timestamp(
            str(path)
        )

        if stamp is None:
            unstamped.append(path)
        else:
            stamped.append(
                (stamp, path)
            )

    stamped.sort(
        key=lambda item: item[0]
    )

    groups = []
    current = []
    previous = None

    for stamp, path in stamped:
        if (
            current
            and previous is not None
            and stamp - previous
            > max_gap_ms
        ):
            groups.append(current)
            current = []

        current.append(path)
        previous = stamp

    if current:
        groups.append(current)

    # Unknown timestamps cannot safely be grouped, so keep each as its own
    # group rather than pretending unrelated samples are one event.
    groups.extend(
        [[path]]
        for path in unstamped
    )

    return groups


def split_paths(
    classes,
    paths_by_class,
    validation_fraction: float,
    group_gap_ms: int,
):
    train = []
    validation = []
    group_counts = {}

    rng = random.Random(17)

    for class_index, name in enumerate(classes):
        paths = list(
            paths_by_class[name]
        )
        groups = temporal_groups(
            paths,
            group_gap_ms,
        )
        group_counts[name] = len(groups)

        rng.shuffle(groups)

        # Validation must contain whole temporal groups. Keep at least one
        # group for training. Classes represented by only one gameplay event
        # cannot honestly be validated yet.
        target_validation = int(
            round(
                len(paths)
                * validation_fraction
            )
        )

        selected_validation = []
        selected_count = 0

        while (
            len(groups) > 1
            and selected_count
            < target_validation
        ):
            group = groups.pop()
            selected_validation.append(
                group
            )
            selected_count += len(
                group
            )

        for group in selected_validation:
            validation.extend(
                (path, class_index)
                for path in group
            )

        for group in groups:
            train.extend(
                (path, class_index)
                for path in group
            )

    rng.shuffle(train)
    rng.shuffle(validation)

    return (
        train,
        validation,
        group_counts,
    )


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
        default=45,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=96,
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
        help=(
            "Samples captured within this time gap are treated as one "
            "kill-feed event and never split across train/validation."
        ),
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
        from torch.utils.data import (
            DataLoader,
            Dataset,
            WeightedRandomSampler,
        )
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

    (
        train_items,
        validation_items,
        group_counts,
    ) = split_paths(
        classes,
        paths_by_class,
        args.validation_fraction,
        max(
            1,
            int(args.group_gap_ms),
        ),
    )

    print()
    print(
        "Estimated independent temporal groups "
        "(more groups = more real visual variety):"
    )
    for name in classes:
        print(
            f"  {name:<18} "
            f"{group_counts[name]:>3} groups"
        )

    print()
    print(
        f"Grouped split: train={len(train_items)} "
        f"validation={len(validation_items)} "
        f"(gap={int(args.group_gap_ms)} ms)"
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

            def block(
                in_channels,
                out_channels,
            ):
                return nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        out_channels
                    ),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        out_channels
                    ),
                    nn.SiLU(inplace=True),
                    nn.MaxPool2d(2),
                )

            self.features = nn.Sequential(
                block(3, 32),
                block(32, 64),
                block(64, 128),
                block(128, 192),
                nn.AdaptiveAvgPool2d(
                    (1, 1)
                ),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.20),
                nn.Linear(
                    192,
                    class_count,
                ),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    train_class_counts = [
        0
        for _ in classes
    ]
    for _, label in train_items:
        train_class_counts[label] += 1

    sample_weights = [
        1.0
        / max(
            1,
            train_class_counts[label],
        )
        for _, label in train_items
    ]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(
            train_items
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        IconDataset(
            train_items,
            training=True,
        ),
        batch_size=max(1, args.batch_size),
        sampler=train_sampler,
        shuffle=False,
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
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05,
    )

    best_state = None
    best_accuracy = -1.0
    best_macro_accuracy = -1.0

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
        macro_accuracy = -1.0

        if validation_loader is not None:
            model.eval()
            correct = 0
            count = 0
            class_correct = [
                0
                for _ in classes
            ]
            class_total = [
                0
                for _ in classes
            ]

            with torch.no_grad():
                for images, labels in validation_loader:
                    prediction = model(
                        images
                    ).argmax(dim=1)

                    correct_mask = (
                        prediction == labels
                    )

                    correct += int(
                        correct_mask
                        .sum()
                        .item()
                    )
                    count += labels.numel()

                    for class_index in range(
                        len(classes)
                    ):
                        mask = (
                            labels
                            == class_index
                        )
                        class_total[
                            class_index
                        ] += int(
                            mask.sum().item()
                        )
                        class_correct[
                            class_index
                        ] += int(
                            (
                                correct_mask
                                & mask
                            )
                            .sum()
                            .item()
                        )

            validation_accuracy = (
                correct / max(1, count)
            )

            available_class_scores = [
                class_correct[index]
                / class_total[index]
                for index in range(
                    len(classes)
                )
                if class_total[index] > 0
            ]
            macro_accuracy = (
                sum(
                    available_class_scores
                )
                / len(
                    available_class_scores
                )
                if available_class_scores
                else -1.0
            )

            # Pick the checkpoint by macro accuracy so large classes cannot
            # hide poor performance on less common heroes.
            if (
                macro_accuracy
                > best_macro_accuracy
            ):
                best_macro_accuracy = (
                    macro_accuracy
                )
                best_accuracy = (
                    validation_accuracy
                )
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
                    f"{validation_accuracy:.1%} "
                    f"macro="
                    f"{macro_accuracy:.1%}"
                )
            else:
                print(
                    f"epoch {epoch:>3}/{args.epochs} "
                    f"loss={train_loss:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    # Detailed grouped validation report from the selected checkpoint.
    if validation_loader is not None:
        confusion = np.zeros(
            (
                len(classes),
                len(classes),
            ),
            dtype=np.int64,
        )

        with torch.no_grad():
            for images, labels in validation_loader:
                predictions = model(
                    images
                ).argmax(dim=1)

                for truth, prediction in zip(
                    labels.tolist(),
                    predictions.tolist(),
                ):
                    confusion[
                        int(truth),
                        int(prediction),
                    ] += 1

        print()
        print(
            "Per-hero grouped validation:"
        )

        for class_index, name in enumerate(
            classes
        ):
            total = int(
                confusion[
                    class_index
                ].sum()
            )

            if total <= 0:
                print(
                    f"  {name:<18} "
                    "no independent validation group"
                )
                continue

            correct = int(
                confusion[
                    class_index,
                    class_index,
                ]
            )
            accuracy = (
                correct / total
            )

            mistakes = []
            for other_index, count in enumerate(
                confusion[class_index]
            ):
                if (
                    other_index
                    == class_index
                    or count <= 0
                ):
                    continue

                mistakes.append(
                    (
                        int(count),
                        classes[
                            other_index
                        ],
                    )
                )

            mistakes.sort(
                reverse=True
            )
            confused_text = (
                ", ".join(
                    f"{other} x{count}"
                    for count, other
                    in mistakes[:3]
                )
                if mistakes
                else "-"
            )

            print(
                f"  {name:<18} "
                f"{correct:>3}/{total:<3} "
                f"{accuracy:>6.1%} | "
                f"confused: "
                f"{confused_text}"
            )

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
            "Best grouped held-out accuracy: "
            f"{best_accuracy:.1%}"
        )
        print(
            "Best grouped macro accuracy: "
            f"{best_macro_accuracy:.1%}"
        )

    print()
    print(
        "Restart Overwatch Vision. The runtime will automatically "
        "prefer this trained classifier over generic portrait matching."
    )


if __name__ == "__main__":
    main()
