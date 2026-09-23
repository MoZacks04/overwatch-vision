from overwatch_vision.models import Rect


def normalized_rect_to_pixels(width, height, left, top, right, bottom):
    left = max(0.0, min(1.0, left))
    top = max(0.0, min(1.0, top))
    right = max(left, min(1.0, right))
    bottom = max(top, min(1.0, bottom))

    return Rect(
        x1=round(left * width),
        y1=round(top * height),
        x2=round(right * width),
        y2=round(bottom * height),
    )


def translate_rect(rect, dx, dy):
    return Rect(
        rect.x1 + dx,
        rect.y1 + dy,
        rect.x2 + dx,
        rect.y2 + dy,
    )
