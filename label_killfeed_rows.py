from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "debug_frames" / "killfeed_review"
DATASET_ROOT = PROJECT_ROOT / ".cache" / "killfeed_row_verifier"
REAL_DIR = DATASET_ROOT / "real"
FALSE_DIR = DATASET_ROOT / "false"
STATE_PATH = PROJECT_ROOT / ".cache" / "killfeed_row_labeler_state.json"
WINDOW_NAME = "Kill Feed Row Labeler"


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"version": 1, "samples": {}}

    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: could not read {STATE_PATH}; starting fresh.")
        return {"version": 1, "samples": {}}

    if not isinstance(data, dict):
        return {"version": 1, "samples": {}}

    samples = data.get("samples")
    if not isinstance(samples, dict):
        samples = {}

    return {"version": 1, "samples": samples}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(STATE_PATH)


def bootstrap_from_dataset(state: dict) -> None:
    sample_state = state["samples"]
    changed = False

    for status, directory in (("real", REAL_DIR), ("false", FALSE_DIR)):
        if not directory.exists():
            continue

        for path in directory.glob("*.png"):
            if path.name in sample_state:
                continue
            sample_state[path.name] = {"status": status}
            changed = True

    if changed:
        save_state(state)


def move_label_copy(source: Path, status: str) -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FALSE_DIR.mkdir(parents=True, exist_ok=True)

    target_dir = REAL_DIR if status == "real" else FALSE_DIR
    other_dir = FALSE_DIR if status == "real" else REAL_DIR

    other_copy = other_dir / source.name
    if other_copy.exists():
        other_copy.unlink()

    shutil.copy2(source, target_dir / source.name)


def build_preview(
    image: np.ndarray,
    filename: str,
    current: int,
    total: int,
    real_count: int,
    false_count: int,
    skipped_count: int,
) -> np.ndarray:
    canvas_w = 1100
    canvas_h = 620
    header_h = 120
    footer_h = 155

    canvas = np.full(
        (canvas_h, canvas_w, 3),
        24,
        dtype=np.uint8,
    )

    cv2.putText(
        canvas,
        f"Sample {current}/{total}",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    short_name = filename
    if len(short_name) > 110:
        short_name = short_name[:107] + "..."

    cv2.putText(
        canvas,
        short_name,
        (24, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"Labeled so far: REAL {real_count}   FALSE {false_count}   "
            f"SKIPPED {skipped_count}"
        ),
        (24, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )

    image_area_y1 = header_h
    image_area_y2 = canvas_h - footer_h
    image_area_h = image_area_y2 - image_area_y1
    image_area_w = canvas_w - 80

    h, w = image.shape[:2]
    scale = min(
        image_area_w / max(1, w),
        image_area_h / max(1, h),
    )
    scale = max(1.0, min(5.0, scale))

    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    eh, ew = enlarged.shape[:2]
    x1 = max(20, (canvas_w - ew) // 2)
    y1 = image_area_y1 + max(0, (image_area_h - eh) // 2)
    x2 = min(canvas_w, x1 + ew)
    y2 = min(image_area_y2, y1 + eh)

    canvas[y1:y2, x1:x2] = enlarged[
        : y2 - y1,
        : x2 - x1,
    ]

    footer_y = canvas_h - footer_h
    cv2.line(
        canvas,
        (0, footer_y),
        (canvas_w, footer_y),
        (75, 75, 75),
        1,
    )

    cv2.putText(
        canvas,
        "R = REAL kill-feed row",
        (24, footer_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "F = FALSE detection",
        (385, footer_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "S = skip / unsure",
        (735, footer_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Q or ESC = quit   |   Progress is saved after every image.",
        (24, footer_y + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            "REAL means the crop contains one actual Overwatch kill-feed row. "
            "If it is mostly scenery/text/noise, mark FALSE."
        ),
        (24, footer_y + 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (175, 175, 175),
        1,
        cv2.LINE_AA,
    )

    return canvas


def prompt(
    image: np.ndarray,
    filename: str,
    current: int,
    total: int,
    real_count: int,
    false_count: int,
    skipped_count: int,
) -> str:
    preview = build_preview(
        image,
        filename,
        current,
        total,
        real_count,
        false_count,
        skipped_count,
    )
    cv2.imshow(WINDOW_NAME, preview)

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"

        key = cv2.waitKeyEx(50)
        if key == -1:
            continue

        code = key & 0xFF

        if code in (27, ord("q"), ord("Q")):
            return "quit"
        if code in (ord("r"), ord("R"), ord("y"), ord("Y")):
            return "real"
        if code in (ord("f"), ord("F"), ord("n"), ord("N")):
            return "false"
        if code in (ord("s"), ord("S"), 32):
            return "skipped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label cropped kill-feed row candidates as real rows or false "
            "detections."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Folder containing row_*.png review crops.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Revisit previously skipped samples.",
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Show every source sample again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source

    samples = sorted(
        path
        for path in source_dir.glob("*.png")
        if path.is_file()
    )

    if not samples:
        print(f"No PNG row crops found in: {source_dir}")
        print(
            "Expected the normal runtime review crops under "
            "debug_frames/killfeed_review."
        )
        return

    state = load_state()
    bootstrap_from_dataset(state)
    sample_state = state["samples"]

    if args.relabel_all:
        pending = samples
    else:
        pending = []
        for path in samples:
            info = sample_state.get(path.name)
            if info is None:
                pending.append(path)
                continue

            if (
                args.include_skipped
                and isinstance(info, dict)
                and info.get("status") == "skipped"
            ):
                pending.append(path)

    def count_status(value: str) -> int:
        return sum(
            1
            for info in sample_state.values()
            if isinstance(info, dict)
            and info.get("status") == value
        )

    print("Kill-feed row labeler")
    print(f"Source: {source_dir}")
    print(f"Found {len(samples)} row crops.")
    print(
        "Existing labels: "
        f"REAL={count_status('real')} "
        f"FALSE={count_status('false')} "
        f"SKIPPED={count_status('skipped')}"
    )
    print(f"Remaining in this run: {len(pending)}")
    print()
    print("R = real row | F = false detection | S = skip | Q/ESC = quit")
    print("Progress is saved after every image.")
    print()

    if not pending:
        print("Nothing left to label in this run.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 620)

    processed = 0

    try:
        for index, path in enumerate(pending, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"[ERROR] Could not load: {path}")
                sample_state[path.name] = {"status": "failed"}
                save_state(state)
                continue

            action = prompt(
                image,
                path.name,
                index,
                len(pending),
                count_status("real"),
                count_status("false"),
                count_status("skipped"),
            )

            if action == "quit":
                print("Stopped. Progress has been saved.")
                break

            if action == "skipped":
                sample_state[path.name] = {"status": "skipped"}
                save_state(state)
                processed += 1
                continue

            move_label_copy(path, action)
            sample_state[path.name] = {"status": action}
            save_state(state)
            processed += 1

    finally:
        cv2.destroyAllWindows()

    print()
    print(f"Processed this run: {processed}")
    print(
        "Dataset totals: "
        f"REAL={count_status('real')} "
        f"FALSE={count_status('false')} "
        f"SKIPPED={count_status('skipped')}"
    )
    print(f"Dataset folder: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
