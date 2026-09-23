from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from overwatch_vision.utils.geometry import normalized_rect_to_pixels


def test_normalized_rect_1080p():
    rect = normalized_rect_to_pixels(
        1920,
        1080,
        0.60,
        0.025,
        0.995,
        0.30,
    )

    assert rect.x1 == 1152
    assert rect.y1 == 27
    assert rect.x2 == 1910
    assert rect.y2 == 324


def test_normalized_rect_scales():
    a = normalized_rect_to_pixels(
        1920,
        1080,
        0.5,
        0.0,
        1.0,
        0.5,
    )

    b = normalized_rect_to_pixels(
        3840,
        2160,
        0.5,
        0.0,
        1.0,
        0.5,
    )

    assert b.x1 == a.x1 * 2
    assert b.x2 == a.x2 * 2
    assert b.y2 == a.y2 * 2
