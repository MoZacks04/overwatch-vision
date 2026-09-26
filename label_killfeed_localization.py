from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "killfeed_rows" / "raw"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "killfeed_localization"
IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"
WINDOW_NAME = "Kill Feed Localization Labeler"

CANVAS_W = 1280
CANVAS_H = 760
HEADER_H = 92
FOOTER_H = 92
MARGIN = 20


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalized_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = clamp(int(round(x1)), 0, width - 1)
    y1 = clamp(int(round(y1)), 0, height - 1)
    x2 = clamp(int(round(x2)), 1, width)
    y2 = clamp(int(round(y2)), 1, height)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    return [x1, y1, x2, y2]


def load_weak_boxes(image_path: Path, width: int, height: int):
    metadata_path = image_path.with_suffix(".json")

    if not metadata_path.exists():
        return []

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    boxes = []

    for item in payload.get("weak_labels", []):
        bbox = item.get("bbox", {})
        box = normalized_box(
            [
                bbox.get("x1", 0),
                bbox.get("y1", 0),
                bbox.get("x2", 0),
                bbox.get("y2", 0),
            ],
            width,
            height,
        )

        if box is not None:
            boxes.append(box)

    return boxes


def annotation_path_for(image_path: Path) -> Path:
    return ANNOTATION_DIR / f"{image_path.stem}.json"


def load_existing_annotation(image_path: Path):
    path = annotation_path_for(image_path)

    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    boxes = []

    for bbox in payload.get("boxes", []):
        try:
            boxes.append(
                [
                    int(bbox["x1"]),
                    int(bbox["y1"]),
                    int(bbox["x2"]),
                    int(bbox["y2"]),
                ]
            )
        except Exception:
            continue

    return boxes


def save_annotation(
    source_path: Path,
    image: np.ndarray,
    boxes,
    confirmed_empty: bool = False,
):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    image_target = IMAGE_DIR / source_path.name
    annotation_target = annotation_path_for(source_path)

    if source_path.resolve() != image_target.resolve():
        shutil.copy2(source_path, image_target)

    h, w = image.shape[:2]

    payload = {
        "image": image_target.name,
        "width": int(w),
        "height": int(h),
        "confirmed_empty": bool(confirmed_empty),
        "boxes": [
            {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "class": "killfeed_row",
            }
            for box in boxes
        ],
        "source": str(source_path),
    }

    temp = annotation_target.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temp.replace(annotation_target)


class BoxEditor:
    def __init__(self, image: np.ndarray, boxes):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.boxes = [list(box) for box in boxes]
        self.drag_start = None
        self.drag_current = None
        self.undo_stack = []

        available_w = CANVAS_W - 2 * MARGIN
        available_h = CANVAS_H - HEADER_H - FOOTER_H - 2 * MARGIN

        self.scale = min(
            available_w / max(1, self.w),
            available_h / max(1, self.h),
        )
        self.scale = min(2.5, self.scale)

        self.display_w = max(1, int(round(self.w * self.scale)))
        self.display_h = max(1, int(round(self.h * self.scale)))
        self.offset_x = (CANVAS_W - self.display_w) // 2
        self.offset_y = (
            HEADER_H
            + MARGIN
            + max(
                0,
                (
                    CANVAS_H
                    - HEADER_H
                    - FOOTER_H
                    - 2 * MARGIN
                    - self.display_h
                )
                // 2,
            )
        )

    def push_undo(self):
        self.undo_stack.append([list(box) for box in self.boxes])
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.boxes = self.undo_stack.pop()

    def clear(self):
        if self.boxes:
            self.push_undo()
            self.boxes = []

    def window_to_image(self, x: int, y: int):
        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale

        if not (0 <= ix < self.w and 0 <= iy < self.h):
            return None

        return (
            clamp(int(round(ix)), 0, self.w - 1),
            clamp(int(round(iy)), 0, self.h - 1),
        )

    def remove_at(self, point):
        px, py = point
        hits = []

        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = max(1, (x2 - x1) * (y2 - y1))
                hits.append((area, index))

        if not hits:
            return

        _, index = min(hits)
        self.push_undo()
        self.boxes.pop(index)

    def mouse(self, event, x, y, flags, param):
        point = self.window_to_image(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if point is not None:
                self.drag_start = point
                self.drag_current = point
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None and point is not None:
                self.drag_current = point
            return

        if event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is None:
                return

            end = point if point is not None else self.drag_current
            start = self.drag_start
            self.drag_start = None
            self.drag_current = None

            if end is None:
                return

            box = normalized_box(
                [start[0], start[1], end[0], end[1]],
                self.w,
                self.h,
            )

            if box is not None:
                self.push_undo()
                self.boxes.append(box)
            return

        if event == cv2.EVENT_RBUTTONDOWN and point is not None:
            self.remove_at(point)

    def _screen_box(self, box):
        x1, y1, x2, y2 = box
        return (
            int(round(self.offset_x + x1 * self.scale)),
            int(round(self.offset_y + y1 * self.scale)),
            int(round(self.offset_x + x2 * self.scale)),
            int(round(self.offset_y + y2 * self.scale)),
        )

    def render(self, filename: str, index: int, total: int):
        canvas = np.full(
            (CANVAS_H, CANVAS_W, 3),
            22,
            dtype=np.uint8,
        )

        resized = cv2.resize(
            self.image,
            (self.display_w, self.display_h),
            interpolation=(
                cv2.INTER_AREA
                if self.scale < 1.0
                else cv2.INTER_NEAREST
            ),
        )

        canvas[
            self.offset_y:self.offset_y + self.display_h,
            self.offset_x:self.offset_x + self.display_w,
        ] = resized

        for box_index, box in enumerate(self.boxes, start=1):
            x1, y1, x2, y2 = self._screen_box(box)
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                canvas,
                str(box_index),
                (x1 + 4, max(18, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.drag_start is not None and self.drag_current is not None:
            temp = normalized_box(
                [
                    self.drag_start[0],
                    self.drag_start[1],
                    self.drag_current[0],
                    self.drag_current[1],
                ],
                self.w,
                self.h,
            )
            if temp is not None:
                x1, y1, x2, y2 = self._screen_box(temp)
                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2,
                )

        cv2.putText(
            canvas,
            f"Image {index}/{total} | boxes {len(self.boxes)}",
            (22, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            filename[:110],
            (22, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        footer_y = CANVAS_H - FOOTER_H
        cv2.line(
            canvas,
            (0, footer_y),
            (CANVAS_W, footer_y),
            (70, 70, 70),
            1,
        )

        cv2.putText(
            canvas,
            "LEFT DRAG add box | RIGHT CLICK delete | Z undo | C clear",
            (22, footer_y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "ENTER/A save+next | S skip | Q/ESC quit",
            (22, footer_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        return canvas


def dataset_totals():
    annotated = 0
    positive = 0
    empty = 0
    row_boxes = 0

    if not ANNOTATION_DIR.exists():
        return annotated, positive, empty, row_boxes

    for path in ANNOTATION_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        annotated += 1
        boxes = payload.get("boxes", [])
        row_boxes += len(boxes)

        if boxes:
            positive += 1
        else:
            empty += 1

    return annotated, positive, empty, row_boxes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Label full kill-feed ROI screenshots with kill-feed row boxes."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Folder containing full kill-feed ROI PNG images.",
    )
    parser.add_argument(
        "--confirmed-empty",
        action="store_true",
        help=(
            "Import every image in --source as a confirmed empty ROI with "
            "zero boxes. Use only for ROIs captured with the N key."
        ),
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Show images that already have saved annotations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = args.source

    images = sorted(
        path
        for path in source_dir.glob("*.png")
        if path.is_file()
    )

    if not images:
        print(f"No PNG images found in: {source_dir}")
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.confirmed_empty:
        imported = 0
        skipped = 0

        for path in images:
            if (
                annotation_path_for(path).exists()
                and not args.relabel_all
            ):
                skipped += 1
                continue

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[ERROR] Could not read: {path}")
                continue

            save_annotation(
                path,
                image,
                boxes=[],
                confirmed_empty=True,
            )
            imported += 1

        print(
            f"Imported {imported} confirmed-empty ROI image(s); "
            f"skipped {skipped} already annotated."
        )
        totals = dataset_totals()
        print(
            "Dataset totals: "
            f"ROI={totals[0]} positive={totals[1]} "
            f"empty={totals[2]} row_boxes={totals[3]}"
        )
        print(f"Dataset: {DATASET_ROOT}")
        return

    pending = []

    for path in images:
        if annotation_path_for(path).exists() and not args.relabel_all:
            continue
        pending.append(path)

    print("Kill-feed localization labeler")
    print(f"Source: {source_dir}")
    print(f"Found: {len(images)} image(s)")
    print(f"Remaining: {len(pending)}")
    print()
    print("Draw one box around EACH real kill-feed row.")
    print("Do not box team-status HUD, scenery, or non-feed UI.")
    print("Weak boxes from the old detector are loaded when available.")
    print()

    if not pending:
        print("Nothing left to label.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    saved = 0
    skipped = 0

    try:
        for index, path in enumerate(pending, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"[ERROR] Could not read: {path}")
                continue

            h, w = image.shape[:2]

            existing = load_existing_annotation(path)
            if existing is not None:
                initial_boxes = existing
            else:
                initial_boxes = load_weak_boxes(path, w, h)

            editor = BoxEditor(image, initial_boxes)
            cv2.setMouseCallback(WINDOW_NAME, editor.mouse)

            action = None

            while action is None:
                if (
                    cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1
                ):
                    action = "quit"
                    break

                preview = editor.render(
                    path.name,
                    index,
                    len(pending),
                )
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKeyEx(30)
                if key == -1:
                    continue

                code = key & 0xFF

                if code in (27, ord("q"), ord("Q")):
                    action = "quit"
                elif code in (13, 10, ord("a"), ord("A")):
                    action = "save"
                elif code in (ord("s"), ord("S")):
                    action = "skip"
                elif code in (ord("c"), ord("C")):
                    editor.clear()
                elif code in (ord("z"), ord("Z")):
                    editor.undo()

            if action == "quit":
                print("Stopped. Saved annotations are preserved.")
                break

            if action == "skip":
                skipped += 1
                continue

            save_annotation(
                path,
                image,
                boxes=editor.boxes,
                confirmed_empty=(len(editor.boxes) == 0),
            )
            saved += 1

    finally:
        cv2.destroyAllWindows()

    print()
    print(f"Saved this run: {saved}")
    print(f"Skipped this run: {skipped}")
    totals = dataset_totals()
    print(
        "Dataset totals: "
        f"ROI={totals[0]} positive={totals[1]} "
        f"empty={totals[2]} row_boxes={totals[3]}"
    )
    print(f"Dataset: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
