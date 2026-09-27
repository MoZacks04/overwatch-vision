from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

HARD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_hard_examples"
)
PENDING_DIR = (
    HARD_ROOT
    / "pending_identity"
)
REVIEWED_DIR = (
    HARD_ROOT
    / "reviewed_identity"
)
LOCALIZER_FAILURE_DIR = (
    HARD_ROOT
    / "localizer_failures"
)
DISCARDED_DIR = (
    HARD_ROOT
    / "discarded"
)

IDENTITY_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "killfeed_hero_identity"
)
IDENTITY_IMAGE_DIR = (
    IDENTITY_ROOT
    / "images"
)
IDENTITY_ANNOTATION_DIR = (
    IDENTITY_ROOT
    / "annotations"
)
IDENTITY_BY_CLASS_DIR = (
    IDENTITY_ROOT
    / "by_class"
)

WINDOW_NAME = "Hero Hard Example Review"
CANVAS_W = 1180
CANVAS_H = 820


def safe_label(value: str) -> str:
    value = value.strip()
    value = re.sub(
        r"[^A-Za-z0-9 ._-]",
        "",
        value,
    )
    return value.strip()


def known_labels() -> list[str]:
    labels = set()

    if IDENTITY_BY_CLASS_DIR.exists():
        for path in IDENTITY_BY_CLASS_DIR.iterdir():
            if path.is_dir():
                labels.add(path.name)

    return sorted(
        labels,
        key=str.casefold,
    )


def canonicalize_label(value: str) -> str:
    cleaned = safe_label(value)

    if not cleaned:
        return ""

    folded = cleaned.casefold()
    labels = known_labels()

    for label in labels:
        if label.casefold() == folded:
            return label

    matches = [
        label
        for label in labels
        if label.casefold().startswith(
            folded
        )
    ]

    if len(matches) == 1:
        return matches[0]

    return cleaned


def load_pending():
    samples = []

    if not PENDING_DIR.exists():
        return samples

    for metadata_path in sorted(
        PENDING_DIR.glob("*.json")
    ):
        try:
            payload = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        portrait_name = str(
            payload.get(
                "portrait_image",
                "",
            )
        ).strip()
        row_name = str(
            payload.get(
                "row_image",
                "",
            )
        ).strip()

        portrait_path = (
            PENDING_DIR
            / portrait_name
        )
        row_path = (
            PENDING_DIR
            / row_name
        )

        if (
            not portrait_name
            or not portrait_path.exists()
        ):
            continue

        samples.append(
            {
                "metadata_path": metadata_path,
                "payload": payload,
                "portrait_path": portrait_path,
                "row_path": (
                    row_path
                    if row_path.exists()
                    else None
                ),
            }
        )

    return samples


def fit_image(
    image: np.ndarray,
    max_w: int,
    max_h: int,
    *,
    nearest: bool = False,
):
    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        return None

    scale = min(
        max_w / float(w),
        max_h / float(h),
    )
    scale = max(
        0.01,
        scale,
    )

    out_w = max(
        1,
        int(round(w * scale)),
    )
    out_h = max(
        1,
        int(round(h * scale)),
    )

    interpolation = (
        cv2.INTER_NEAREST
        if nearest
        else (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )
    )

    return cv2.resize(
        image,
        (out_w, out_h),
        interpolation=interpolation,
    )


def make_preview(
    sample,
    portrait,
    row,
    typed: str,
    index: int,
    total: int,
):
    payload = sample["payload"]
    prediction = payload.get(
        "prediction",
        {},
    )

    canvas = np.full(
        (
            CANVAS_H,
            CANVAS_W,
            3,
        ),
        24,
        dtype=np.uint8,
    )

    best = prediction.get(
        "best",
        "?",
    )
    best_conf = float(
        prediction.get(
            "best_confidence",
            0.0,
        )
    )
    second = prediction.get(
        "second",
        "?",
    )
    second_conf = float(
        prediction.get(
            "second_confidence",
            0.0,
        )
    )
    margin = float(
        prediction.get(
            "margin",
            0.0,
        )
    )

    cv2.putText(
        canvas,
        (
            f"Hard example {index}/{total} | "
            f"track={payload.get('track_id', '?')} | "
            f"{payload.get('side', '?')} | "
            f"team={payload.get('team', '?')} | "
            f"reason={payload.get('reason', '?')}"
        ),
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"Model: {best} {best_conf:.2f} | "
            f"2nd: {second} {second_conf:.2f} | "
            f"margin={margin:.2f}"
        ),
        (24, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    # Full row for context.
    if row is not None:
        row_view = fit_image(
            row,
            CANVAS_W - 100,
            185,
            nearest=True,
        )

        if row_view is not None:
            rh, rw = row_view.shape[:2]
            rx = (
                CANVAS_W - rw
            ) // 2
            ry = 100

            canvas[
                ry:ry + rh,
                rx:rx + rw,
            ] = row_view

            cv2.rectangle(
                canvas,
                (rx, ry),
                (
                    rx + rw,
                    ry + rh,
                ),
                (100, 100, 100),
                1,
            )

            hero_box = payload.get(
                "hero_box_local"
            )

            if (
                isinstance(
                    hero_box,
                    dict,
                )
                and all(
                    key in hero_box
                    for key in (
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    )
                )
            ):
                source_h, source_w = (
                    row.shape[:2]
                )
                scale_x = (
                    rw
                    / max(
                        1.0,
                        float(source_w),
                    )
                )
                scale_y = (
                    rh
                    / max(
                        1.0,
                        float(source_h),
                    )
                )

                x1 = (
                    rx
                    + int(
                        round(
                            float(
                                hero_box["x1"]
                            )
                            * scale_x
                        )
                    )
                )
                y1 = (
                    ry
                    + int(
                        round(
                            float(
                                hero_box["y1"]
                            )
                            * scale_y
                        )
                    )
                )
                x2 = (
                    rx
                    + int(
                        round(
                            float(
                                hero_box["x2"]
                            )
                            * scale_x
                        )
                    )
                )
                y2 = (
                    ry
                    + int(
                        round(
                            float(
                                hero_box["y2"]
                            )
                            * scale_y
                        )
                    )
                )

                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2,
                )

    # Exact runtime portrait crop.
    portrait_view = fit_image(
        portrait,
        430,
        390,
        nearest=True,
    )

    if portrait_view is not None:
        ph, pw = (
            portrait_view.shape[:2]
        )
        px = (
            CANVAS_W - pw
        ) // 2
        py = 320

        canvas[
            py:py + ph,
            px:px + pw,
        ] = portrait_view

        cv2.rectangle(
            canvas,
            (px, py),
            (
                px + pw,
                py + ph,
            ),
            (0, 255, 255),
            2,
        )

    footer_y = CANVAS_H - 118

    cv2.line(
        canvas,
        (0, footer_y),
        (CANVAS_W, footer_y),
        (70, 70, 70),
        1,
    )

    cv2.putText(
        canvas,
        f"Correct hero: {typed}",
        (24, footer_y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            "ENTER save label | TAB accept model guess | "
            "SPACE defer | [ bad box | X discard | ESC quit"
        ),
        (24, footer_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            "Short names are fine; keep each hero spelling consistent."
        ),
        (24, footer_y + 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    return canvas


def prompt_label(
    sample,
    portrait,
    row,
    index,
    total,
):
    typed = ""

    prediction = sample[
        "payload"
    ].get(
        "prediction",
        {},
    )
    best_guess = str(
        prediction.get(
            "best",
            "",
        )
        or ""
    )

    while True:
        preview = make_preview(
            sample,
            portrait,
            row,
            typed,
            index,
            total,
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

        if code == 9 and not typed:
            label = canonicalize_label(
                best_guess
            )

            if label:
                return "label", label

            continue

        if code == 32 and not typed:
            return "defer", None

        if (
            code == ord("[")
            and not typed
        ):
            return "bad_box", None

        if (
            code in (
                ord("x"),
                ord("X"),
            )
            and not typed
        ):
            return "discard", None

        if code in (10, 13):
            if not typed.strip():
                continue

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
                and not typed.endswith(
                    " "
                )
            ):
                typed += " "
            elif char != " ":
                typed = safe_label(
                    typed + char
                )


def archive_sample(
    sample,
    destination: Path,
):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        sample["metadata_path"],
        sample["portrait_path"],
    ]

    if sample["row_path"] is not None:
        paths.append(
            sample["row_path"]
        )

    for path in paths:
        if (
            path is None
            or not path.exists()
        ):
            continue

        target = (
            destination
            / path.name
        )

        if target.exists():
            target.unlink()

        shutil.move(
            str(path),
            str(target),
        )


def save_identity(
    sample,
    label: str,
):
    payload = dict(
        sample["payload"]
    )
    sample_id = str(
        payload["sample_id"]
    )

    IDENTITY_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    IDENTITY_ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    IDENTITY_BY_CLASS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_name = (
        f"{sample_id}.png"
    )
    image_target = (
        IDENTITY_IMAGE_DIR
        / image_name
    )

    shutil.copy2(
        sample["portrait_path"],
        image_target,
    )

    class_dir = (
        IDENTITY_BY_CLASS_DIR
        / label
    )
    class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A unique sample ID should exist in only one class.
    for directory in (
        IDENTITY_BY_CLASS_DIR.iterdir()
    ):
        if (
            not directory.is_dir()
            or directory.name == label
        ):
            continue

        duplicate = (
            directory
            / image_name
        )

        if duplicate.exists():
            duplicate.unlink()

    shutil.copy2(
        image_target,
        class_dir / image_name,
    )

    annotation = {
        "image": image_name,
        "hero": label,
        "role": payload.get(
            "side",
            "unknown",
        ),
        "hard_example": True,
        "source": "live_runtime",
        "created_unix": payload.get(
            "created_unix"
        ),
        "event_timestamp": payload.get(
            "event_timestamp"
        ),
        "track_id": payload.get(
            "track_id"
        ),
        "team": payload.get(
            "team"
        ),
        "collection_reason": payload.get(
            "reason"
        ),
        "original_prediction": payload.get(
            "prediction",
            {},
        ),
    }

    annotation_path = (
        IDENTITY_ANNOTATION_DIR
        / f"{sample_id}.json"
    )
    temp = annotation_path.with_suffix(
        ".tmp"
    )
    temp.write_text(
        json.dumps(
            annotation,
            indent=2,
        ),
        encoding="utf-8",
    )
    temp.replace(
        annotation_path
    )

    payload[
        "reviewed_label"
    ] = label
    payload[
        "reviewed_unix"
    ] = time.time()

    sample["metadata_path"].write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    archive_sample(
        sample,
        REVIEWED_DIR / label,
    )


def main():
    samples = load_pending()

    print(
        "Hero hard-example reviewer"
    )
    print(
        f"Pending identity samples: {len(samples)}"
    )
    print(
        "ENTER save | TAB accept model guess | SPACE defer | "
        "[ bad box | X discard | ESC quit"
    )
    print()

    if not samples:
        print(
            "Nothing to review. Play with Overwatch Vision running first."
        )
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

    labeled = 0
    deferred = 0
    bad_boxes = 0
    discarded = 0

    try:
        for index, sample in enumerate(
            samples,
            start=1,
        ):
            portrait = cv2.imread(
                str(
                    sample[
                        "portrait_path"
                    ]
                ),
                cv2.IMREAD_COLOR,
            )

            if portrait is None:
                continue

            row = None

            if sample["row_path"] is not None:
                row = cv2.imread(
                    str(
                        sample[
                            "row_path"
                        ]
                    ),
                    cv2.IMREAD_COLOR,
                )

            action, label = prompt_label(
                sample,
                portrait,
                row,
                index,
                len(samples),
            )

            if action == "quit":
                break

            if action == "defer":
                deferred += 1
                continue

            if action == "bad_box":
                archive_sample(
                    sample,
                    LOCALIZER_FAILURE_DIR,
                )
                bad_boxes += 1
                continue

            if action == "discard":
                archive_sample(
                    sample,
                    DISCARDED_DIR,
                )
                discarded += 1
                continue

            if action == "label":
                assert label is not None

                save_identity(
                    sample,
                    label,
                )
                labeled += 1

    finally:
        cv2.destroyAllWindows()

    remaining = len(
        load_pending()
    )

    print()
    print(
        f"Labeled into identity dataset: {labeled}"
    )
    print(
        f"Deferred: {deferred}"
    )
    print(
        f"Marked bad-box/localizer failures: {bad_boxes}"
    )
    print(
        f"Discarded: {discarded}"
    )
    print(
        f"Still pending: {remaining}"
    )
    print()
    print(
        "When you have reviewed enough new samples, retrain with:"
    )
    print(
        r".\.venv\Scripts\python.exe .\train_hero_classifier.py"
    )


if __name__ == "__main__":
    main()
