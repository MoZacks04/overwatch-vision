from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "debug_frames" / "hero_samples"
TEMPLATE_DIR = PROJECT_ROOT / ".cache" / "killfeed_hero_templates"
STATE_PATH = PROJECT_ROOT / ".cache" / "hero_labeler_state.json"
WINDOW_NAME = "Hero Sample Labeler"


def safe_label(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9 ._-]", "", value)
    return value.strip()


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"version": 1, "samples": {}}

    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: could not read {STATE_PATH}; starting a fresh labeler state.")
        return {"version": 1, "samples": {}}

    if not isinstance(data, dict):
        return {"version": 1, "samples": {}}

    samples = data.get("samples")
    if not isinstance(samples, dict):
        samples = {}

    return {"version": 1, "samples": samples}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(STATE_PATH)


def existing_labels() -> list[str]:
    if not TEMPLATE_DIR.exists():
        return []
    return sorted(
        path.name
        for path in TEMPLATE_DIR.iterdir()
        if path.is_dir()
    )


def canonicalize_label(label: str) -> str:
    """Reuse the existing spelling/capitalization of a class when possible."""
    cleaned = safe_label(label)
    if not cleaned:
        return ""

    folded = cleaned.casefold()
    for existing in existing_labels():
        if existing.casefold() == folded:
            return existing
    return cleaned


def bootstrap_state_from_templates(state: dict) -> None:
    """Recognize samples labeled by older versions of this script."""
    sample_state = state["samples"]

    if not TEMPLATE_DIR.exists():
        return

    found: dict[str, list[str]] = {}
    for label_dir in sorted(path for path in TEMPLATE_DIR.iterdir() if path.is_dir()):
        for image_path in label_dir.glob("*.png"):
            found.setdefault(image_path.name, []).append(label_dir.name)

    changed = False
    for sample_name, labels in found.items():
        if sample_name in sample_state:
            continue

        if len(labels) > 1:
            print(
                f"Warning: {sample_name} exists under multiple labels: "
                + ", ".join(labels)
            )

        sample_state[sample_name] = {
            "status": "labeled",
            "label": labels[0],
        }
        changed = True

    if changed:
        save_state(state)


def remove_other_label_copies(sample_name: str, keep_label: str) -> None:
    if not TEMPLATE_DIR.exists():
        return

    for label_dir in TEMPLATE_DIR.iterdir():
        if not label_dir.is_dir() or label_dir.name == keep_label:
            continue

        old_copy = label_dir / sample_name
        if old_copy.exists():
            old_copy.unlink()


def make_preview(
    image: np.ndarray,
    filename: str,
    typed_label: str,
    current: int,
    total: int,
) -> np.ndarray:
    height, width = image.shape[:2]

    target_long_side = 620
    scale = max(4.0, min(12.0, target_long_side / max(height, width)))
    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    top = 105
    bottom = 90
    canvas = cv2.copyMakeBorder(
        enlarged,
        top,
        bottom,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )

    cv2.putText(
        canvas,
        f"Sample {current}/{total}",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        filename[:90],
        (20, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Type hero name, ENTER = save | empty ENTER = skip | ESC = quit",
        (20, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    prompt_y = canvas.shape[0] - 48
    cv2.putText(
        canvas,
        "Hero:",
        (20, prompt_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        typed_label,
        (90, prompt_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Backspace edits. Progress is saved after every image.",
        (20, canvas.shape[0] - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    return canvas


def prompt_for_label(
    image: np.ndarray,
    filename: str,
    current: int,
    total: int,
) -> tuple[str, str | None]:
    typed = ""

    while True:
        preview = make_preview(image, filename, typed, current, total)
        cv2.imshow(WINDOW_NAME, preview)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "quit", None

        key = cv2.waitKeyEx(50)
        if key == -1:
            continue

        code = key & 0xFF

        if code == 27:  # Escape
            return "quit", None

        if code in (10, 13):  # Enter
            if not typed.strip():
                return "skip", None

            label = canonicalize_label(typed)
            if label:
                return "label", label
            continue

        if code in (8, 127):  # Backspace/Delete
            typed = typed[:-1]
            continue

        if 32 <= code <= 126:
            char = chr(code)
            candidate = safe_label(typed + char)
            # safe_label strips trailing spaces, so preserve a single typed space here.
            if char == " " and typed and not typed.endswith(" "):
                typed += " "
            elif char != " ":
                typed = candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label saved Overwatch kill-feed hero portrait crops."
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Show samples previously marked skipped as well as new samples.",
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Show every sample again, including already labeled samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = sorted(
        path
        for path in SAMPLE_DIR.glob("*.png")
        if "_killer_" in path.name or "_victim_" in path.name
    )

    if not samples:
        print(
            "No hero samples found. Run Overwatch Vision first so "
            "debug_frames/hero_samples contains K/V portrait crops."
        )
        return

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    bootstrap_state_from_templates(state)
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
            if args.include_skipped and info.get("status") == "skipped":
                pending.append(path)

    labeled_count = sum(
        1 for info in sample_state.values()
        if isinstance(info, dict) and info.get("status") == "labeled"
    )
    skipped_count = sum(
        1 for info in sample_state.values()
        if isinstance(info, dict) and info.get("status") == "skipped"
    )

    print("Hero sample labeler")
    print(f"Found {len(samples)} samples.")
    print(f"Already labeled: {labeled_count} | skipped: {skipped_count}")
    print(f"Remaining in this run: {len(pending)}")
    print()
    print("Controls are inside the image window:")
    print("  type hero name + Enter = label")
    print("  Enter with an empty name = skip")
    print("  Esc = quit")
    print()
    print("State is saved after every image under .cache/hero_labeler_state.json.")
    print("Use --include-skipped later to revisit skipped samples.")
    print("Use --relabel-all if you ever want to review the entire dataset.")

    if not pending:
        print("Nothing left to label in this run.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    processed = 0
    try:
        for index, path in enumerate(pending, start=1):
            image = cv2.imread(str(path))
            if image is None:
                print(f"[ERROR] Could not load image: {path}")
                sample_state[path.name] = {"status": "failed"}
                save_state(state)
                continue

            print(f"[{index}/{len(pending)}] {path.name}")

            action, label = prompt_for_label(
                image,
                path.name,
                index,
                len(pending),
            )

            if action == "quit":
                print("Stopped. Your progress has been saved.")
                break

            if action == "skip":
                sample_state[path.name] = {"status": "skipped"}
                save_state(state)
                processed += 1
                print("Skipped.")
                continue

            assert label is not None
            destination_dir = TEMPLATE_DIR / label
            destination_dir.mkdir(parents=True, exist_ok=True)

            remove_other_label_copies(path.name, label)

            destination = destination_dir / path.name
            shutil.copy2(path, destination)

            sample_state[path.name] = {
                "status": "labeled",
                "label": label,
            }
            save_state(state)
            processed += 1

            print(f"Saved template: {destination}")

    finally:
        cv2.destroyAllWindows()

    remaining = sum(
        1 for path in samples
        if path.name not in sample_state
    )
    print()
    print(f"Processed this run: {processed}")
    print(f"Unseen samples remaining: {remaining}")


if __name__ == "__main__":
    main()
