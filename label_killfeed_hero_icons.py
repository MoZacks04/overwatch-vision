from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ROW_DATASET_ROOT = PROJECT_ROOT / "datasets" / "killfeed_localization"
ROW_IMAGE_DIR = ROW_DATASET_ROOT / "images"
ROW_ANNOTATION_DIR = ROW_DATASET_ROOT / "annotations"

DATASET_ROOT = PROJECT_ROOT / "datasets" / "killfeed_hero_icons"
IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"

WINDOW_NAME = "Kill Feed Hero Icon Labeler"
CANVAS_W = 1400
CANVAS_H = 620
HEADER_H = 92
FOOTER_H = 110
MARGIN = 24


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


def load_row_records():
    records = []

    if not ROW_ANNOTATION_DIR.exists():
        return records

    for annotation_path in sorted(ROW_ANNOTATION_DIR.glob("*.json")):
        try:
            payload = json.loads(
                annotation_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        image_name = str(payload.get("image", "")).strip()
        image_path = ROW_IMAGE_DIR / image_name

        if not image_name or not image_path.exists():
            continue

        boxes = []

        for item in payload.get("boxes", []):
            try:
                boxes.append(
                    [
                        int(item["x1"]),
                        int(item["y1"]),
                        int(item["x2"]),
                        int(item["y2"]),
                    ]
                )
            except Exception:
                continue

        boxes.sort(
            key=lambda box: (
                box[1],
                box[0],
            )
        )

        for row_index, box in enumerate(boxes, start=1):
            row_id = f"{Path(image_name).stem}__row{row_index:02d}"

            records.append(
                {
                    "row_id": row_id,
                    "source_image_path": image_path,
                    "source_image_name": image_name,
                    "source_annotation": annotation_path.name,
                    "source_row_index": row_index,
                    "source_row_bbox": box,
                }
            )

    return records


def extract_row(record):
    image = cv2.imread(
        str(record["source_image_path"]),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = record["source_row_bbox"]

    x1 = clamp(x1, 0, max(0, w - 1))
    y1 = clamp(y1, 0, max(0, h - 1))
    x2 = clamp(x2, x1 + 1, w)
    y2 = clamp(y2, y1 + 1, h)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop.copy()


def _best_panel(mask, row_height: int, row_width: int):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h < max(4, int(round(row_height * 0.30))):
            continue
        if w < max(8, int(round(row_width * 0.05))):
            continue

        fill = area / max(1.0, float(w * h))
        center_error = abs(
            (y + h / 2.0)
            - row_height / 2.0
        ) / max(1.0, row_height / 2.0)

        score = (
            area
            * (0.65 + 0.35 * fill)
            * (1.0 - 0.20 * min(1.0, center_error))
        )

        candidates.append(
            (
                score,
                [x, y, x + w, y + h],
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def suggested_icon_boxes(row_image):
    """
    Preload the old geometric K/V guesses as editable suggestions.

    These are only UI suggestions. They are never burned into the PNG and
    only become training labels if the user saves them.
    """
    if row_image is None or row_image.size == 0:
        return []

    h, w = row_image.shape[:2]

    if h < 6 or w < 16:
        return []

    hsv = cv2.cvtColor(
        row_image,
        cv2.COLOR_BGR2HSV,
    )
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    usable = (
        (sat >= 55)
        & (val >= 55)
    )

    red = np.where(
        usable
        & (
            (hue <= 13)
            | (hue >= 157)
        ),
        255,
        0,
    ).astype(np.uint8)

    blue = np.where(
        usable
        & (hue >= 82)
        & (hue <= 120),
        255,
        0,
    ).astype(np.uint8)

    kernel_w = max(
        3,
        int(round(w * 0.010)),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_w, 3),
    )

    red = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        kernel,
    )
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_CLOSE,
        kernel,
    )

    red_panel = _best_panel(red, h, w)
    blue_panel = _best_panel(blue, h, w)

    if red_panel is None or blue_panel is None:
        return []

    panels = sorted(
        [red_panel, blue_panel],
        key=lambda box: (box[0] + box[2]) / 2.0,
    )

    left, right = panels

    reference_h = max(
        5,
        int(
            round(
                (
                    (left[3] - left[1])
                    + (right[3] - right[1])
                )
                / 2.0
            )
        ),
    )
    size = max(
        5,
        int(round(reference_h * 1.08)),
    )

    left_cy = (left[1] + left[3]) / 2.0
    right_cy = (right[1] + right[3]) / 2.0

    left_box = normalized_box(
        [
            left[2] - size,
            left_cy - size / 2.0,
            left[2],
            left_cy + size / 2.0,
        ],
        w,
        h,
    )

    right_box = normalized_box(
        [
            right[0],
            right_cy - size / 2.0,
            right[0] + size,
            right_cy + size / 2.0,
        ],
        w,
        h,
    )

    return [
        box
        for box in (left_box, right_box)
        if box is not None
    ]


def annotation_path(row_id: str) -> Path:
    return ANNOTATION_DIR / f"{row_id}.json"


def image_path(row_id: str) -> Path:
    return IMAGE_DIR / f"{row_id}.png"


def load_existing(row_id: str):
    path = annotation_path(row_id)

    if not path.exists():
        return None

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None

    boxes = []

    for item in payload.get("boxes", []):
        try:
            boxes.append(
                [
                    int(item["x1"]),
                    int(item["y1"]),
                    int(item["x2"]),
                    int(item["y2"]),
                ]
            )
        except Exception:
            continue

    return boxes


def save_annotation(record, row_image, boxes):
    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_id = record["row_id"]
    target_image = image_path(row_id)
    target_annotation = annotation_path(row_id)

    cv2.imwrite(
        str(target_image),
        row_image,
    )

    h, w = row_image.shape[:2]

    payload = {
        "image": target_image.name,
        "width": int(w),
        "height": int(h),
        "confirmed_no_icons": len(boxes) == 0,
        "boxes": [
            {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "class": "hero_icon",
            }
            for box in boxes
        ],
        "source_image": record["source_image_name"],
        "source_annotation": record["source_annotation"],
        "source_row_index": int(record["source_row_index"]),
        "source_row_bbox": {
            "x1": int(record["source_row_bbox"][0]),
            "y1": int(record["source_row_bbox"][1]),
            "x2": int(record["source_row_bbox"][2]),
            "y2": int(record["source_row_bbox"][3]),
        },
    }

    temp = target_annotation.with_suffix(".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temp.replace(target_annotation)


class BoxEditor:
    def __init__(self, image, boxes):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.boxes = [
            list(box)
            for box in boxes
        ]
        self.drag_start = None
        self.drag_current = None
        self.undo_stack = []

        available_w = CANVAS_W - 2 * MARGIN
        available_h = (
            CANVAS_H
            - HEADER_H
            - FOOTER_H
            - 2 * MARGIN
        )

        self.scale = min(
            available_w / max(1, self.w),
            available_h / max(1, self.h),
        )

        # Kill-feed rows are short. Enlarge them aggressively enough that
        # small portrait boundaries are easy to place by hand.
        self.scale = min(8.0, self.scale)

        self.display_w = max(
            1,
            int(round(self.w * self.scale)),
        )
        self.display_h = max(
            1,
            int(round(self.h * self.scale)),
        )

        self.offset_x = (
            CANVAS_W - self.display_w
        ) // 2
        self.offset_y = (
            HEADER_H
            + MARGIN
            + max(
                0,
                (
                    available_h
                    - self.display_h
                )
                // 2,
            )
        )

    def push_undo(self):
        self.undo_stack.append(
            [
                list(box)
                for box in self.boxes
            ]
        )
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.boxes = self.undo_stack.pop()

    def clear(self):
        if self.boxes:
            self.push_undo()
            self.boxes = []

    def window_to_image(self, x, y):
        ix = (
            x - self.offset_x
        ) / self.scale
        iy = (
            y - self.offset_y
        ) / self.scale

        if not (
            0 <= ix < self.w
            and 0 <= iy < self.h
        ):
            return None

        return (
            clamp(
                int(round(ix)),
                0,
                self.w - 1,
            ),
            clamp(
                int(round(iy)),
                0,
                self.h - 1,
            ),
        )

    def remove_at(self, point):
        px, py = point
        hits = []

        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box

            if (
                x1 <= px <= x2
                and y1 <= py <= y2
            ):
                area = max(
                    1,
                    (x2 - x1)
                    * (y2 - y1),
                )
                hits.append(
                    (area, index)
                )

        if not hits:
            return

        _, index = min(hits)
        self.push_undo()
        self.boxes.pop(index)

    def mouse(self, event, x, y, flags, param):
        point = self.window_to_image(
            x,
            y,
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            if point is not None:
                self.drag_start = point
                self.drag_current = point
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if (
                self.drag_start is not None
                and point is not None
            ):
                self.drag_current = point
            return

        if event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is None:
                return

            end = (
                point
                if point is not None
                else self.drag_current
            )
            start = self.drag_start
            self.drag_start = None
            self.drag_current = None

            if end is None:
                return

            box = normalized_box(
                [
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                ],
                self.w,
                self.h,
            )

            if box is not None:
                self.push_undo()
                self.boxes.append(box)
            return

        if (
            event == cv2.EVENT_RBUTTONDOWN
            and point is not None
        ):
            self.remove_at(point)

    def _screen_box(self, box):
        x1, y1, x2, y2 = box

        return (
            int(
                round(
                    self.offset_x
                    + x1 * self.scale
                )
            ),
            int(
                round(
                    self.offset_y
                    + y1 * self.scale
                )
            ),
            int(
                round(
                    self.offset_x
                    + x2 * self.scale
                )
            ),
            int(
                round(
                    self.offset_y
                    + y2 * self.scale
                )
            ),
        )

    def render(
        self,
        row_id,
        index,
        total,
    ):
        canvas = np.full(
            (CANVAS_H, CANVAS_W, 3),
            22,
            dtype=np.uint8,
        )

        resized = cv2.resize(
            self.image,
            (
                self.display_w,
                self.display_h,
            ),
            interpolation=(
                cv2.INTER_AREA
                if self.scale < 1.0
                else cv2.INTER_NEAREST
            ),
        )

        canvas[
            self.offset_y:
            self.offset_y + self.display_h,
            self.offset_x:
            self.offset_x + self.display_w,
        ] = resized

        ordered = sorted(
            enumerate(self.boxes),
            key=lambda item: (
                item[1][0]
                + item[1][2]
            )
            / 2.0,
        )

        role_by_index = {}

        if len(ordered) >= 2:
            role_by_index[
                ordered[0][0]
            ] = "K"
            role_by_index[
                ordered[-1][0]
            ] = "V"

        for box_index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._screen_box(
                box
            )
            role = role_by_index.get(
                box_index,
                str(box_index + 1),
            )

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                canvas,
                role,
                (
                    x1 + 4,
                    max(
                        HEADER_H + 18,
                        y1 + 18,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if (
            self.drag_start is not None
            and self.drag_current is not None
        ):
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
                x1, y1, x2, y2 = (
                    self._screen_box(temp)
                )
                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2,
                )

        cv2.putText(
            canvas,
            (
                f"Row {index}/{total} | "
                f"hero boxes {len(self.boxes)}"
            ),
            (22, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            row_id[:120],
            (22, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        footer_y = (
            CANVAS_H - FOOTER_H
        )

        cv2.line(
            canvas,
            (0, footer_y),
            (CANVAS_W, footer_y),
            (70, 70, 70),
            1,
        )

        cv2.putText(
            canvas,
            (
                "Box ONLY the main killer + victim HERO PORTRAITS. "
                "Ignore ability icons, skulls, text and small assist icons."
            ),
            (22, footer_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                "LEFT DRAG add | RIGHT CLICK delete | "
                "Z undo | C clear"
            ),
            (22, footer_y + 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                "ENTER/A save+next | S skip | Q/ESC quit | "
                "with 2 boxes: left=K, right=V"
            ),
            (22, footer_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        return canvas


def dataset_totals():
    rows = 0
    rows_with_icons = 0
    empty = 0
    icon_boxes = 0

    if not ANNOTATION_DIR.exists():
        return (
            rows,
            rows_with_icons,
            empty,
            icon_boxes,
        )

    for path in ANNOTATION_DIR.glob(
        "*.json"
    ):
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        rows += 1
        boxes = payload.get(
            "boxes",
            [],
        )
        icon_boxes += len(boxes)

        if boxes:
            rows_with_icons += 1
        else:
            empty += 1

    return (
        rows,
        rows_with_icons,
        empty,
        icon_boxes,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract each labeled kill-feed row and draw boxes around the "
            "main killer/victim hero portraits."
        )
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help=(
            "Show rows that already have saved hero-icon annotations."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    records = load_row_records()

    if not records:
        print(
            "No labeled kill-feed rows found. "
            "Expected datasets/killfeed_localization first."
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

    pending = []

    for record in records:
        if (
            annotation_path(
                record["row_id"]
            ).exists()
            and not args.relabel_all
        ):
            continue

        pending.append(record)

    print("Kill-feed hero icon labeler")
    print(
        f"Source kill-feed rows: {len(records)}"
    )
    print(
        f"Remaining this run: {len(pending)}"
    )
    print()
    print(
        "Draw tight boxes around ONLY the two primary hero portraits."
    )
    print(
        "Do not box player text, action/ability icons, skulls, or assist icons."
    )
    print(
        "Existing green boxes are OLD-GEOMETRY SUGGESTIONS only. "
        "Correct/delete them whenever they are wrong."
    )
    print(
        "With two boxes, the left portrait is treated as killer (K) "
        "and the right portrait as victim (V)."
    )
    print(
        "If a real feed row genuinely has fewer than two primary portraits, "
        "label only what is actually visible."
    )
    print()

    if not pending:
        totals = dataset_totals()
        print("Nothing left to label.")
        print(
            "Dataset totals: "
            f"rows={totals[0]} "
            f"with_icons={totals[1]} "
            f"empty={totals[2]} "
            f"icon_boxes={totals[3]}"
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

    saved = 0
    skipped = 0

    try:
        for index, record in enumerate(
            pending,
            start=1,
        ):
            row_image = extract_row(
                record
            )

            if row_image is None:
                print(
                    "[ERROR] Could not extract "
                    f"{record['row_id']}"
                )
                continue

            existing = load_existing(
                record["row_id"]
            )

            if existing is not None:
                initial_boxes = existing
            else:
                initial_boxes = (
                    suggested_icon_boxes(
                        row_image
                    )
                )

            editor = BoxEditor(
                row_image,
                initial_boxes,
            )

            cv2.setMouseCallback(
                WINDOW_NAME,
                editor.mouse,
            )

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
                    record["row_id"],
                    index,
                    len(pending),
                )
                cv2.imshow(
                    WINDOW_NAME,
                    preview,
                )

                key = cv2.waitKeyEx(30)

                if key == -1:
                    continue

                code = key & 0xFF

                if code in (
                    27,
                    ord("q"),
                    ord("Q"),
                ):
                    action = "quit"
                elif code in (
                    13,
                    10,
                    ord("a"),
                    ord("A"),
                ):
                    action = "save"
                elif code in (
                    ord("s"),
                    ord("S"),
                ):
                    action = "skip"
                elif code in (
                    ord("c"),
                    ord("C"),
                ):
                    editor.clear()
                elif code in (
                    ord("z"),
                    ord("Z"),
                ):
                    editor.undo()

            if action == "quit":
                print(
                    "Stopped. Saved annotations are preserved."
                )
                break

            if action == "skip":
                skipped += 1
                continue

            save_annotation(
                record,
                row_image,
                editor.boxes,
            )
            saved += 1

    finally:
        cv2.destroyAllWindows()

    totals = dataset_totals()

    print()
    print(
        f"Saved this run: {saved}"
    )
    print(
        f"Skipped this run: {skipped}"
    )
    print(
        "Dataset totals: "
        f"rows={totals[0]} "
        f"with_icons={totals[1]} "
        f"empty={totals[2]} "
        f"icon_boxes={totals[3]}"
    )
    print(
        f"Dataset: {DATASET_ROOT}"
    )


if __name__ == "__main__":
    main()
