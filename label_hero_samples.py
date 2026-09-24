from __future__ import annotations

from pathlib import Path
import re
import shutil

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "debug_frames" / "hero_samples"
TEMPLATE_DIR = PROJECT_ROOT / ".cache" / "killfeed_hero_templates"


def safe_label(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9 ._-]", "", value)
    return value.strip()


def main():
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

    print("Hero sample labeler")
    print("Type a hero name, S to skip, or Q to quit.")
    print("The labeled templates stay local under .cache/.")

    for index, path in enumerate(samples, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue

        enlarged = cv2.resize(
            image,
            None,
            fx=5.0,
            fy=5.0,
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imshow(
            "Hero Sample - type label in terminal",
            enlarged,
        )
        cv2.waitKey(1)

        print()
        print(f"[{index}/{len(samples)}] {path.name}")
        label = input("Hero: ").strip()

        if label.lower() == "q":
            break

        if label.lower() in ("", "s", "skip"):
            continue

        label = safe_label(label)
        if not label:
            continue

        destination_dir = TEMPLATE_DIR / label
        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            destination_dir
            / f"{path.stem}.png"
        )
        shutil.copy2(
            path,
            destination,
        )

        print(f"Saved template: {destination}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
