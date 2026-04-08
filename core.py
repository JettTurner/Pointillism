from __future__ import annotations
import math
from pathlib import Path

import cv2
import numpy as np
from pointillism import (
    ColorPalette,
    VectorField,
    compute_color_probabilities,
    color_select,
    randomized_grid,
    limit_size,
)


def auto_stroke_scale(img: np.ndarray, value: int) -> int:
    if value > 0:
        return value
    scale = int(math.ceil(max(img.shape) / 1000))
    return scale


def auto_gradient_radius(img: np.ndarray, value: int) -> int:
    if value > 0:
        return value
    radius = int(round(max(img.shape) / 50))
    return radius


def load_image(path: Path, limit: int = 0) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if limit > 0:
        img = limit_size(img, limit)
    return img


def build_palette(img: np.ndarray, size: int) -> ColorPalette:
    palette = ColorPalette.from_image(img, size)
    # optional extensions
    return palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])


def compute_gradient(img: np.ndarray, radius: int) -> VectorField:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient = VectorField.from_gradient(gray)
    gradient.smooth(radius)
    return gradient


def paint_image(
    img: np.ndarray,
    palette: ColorPalette,
    gradient: VectorField,
    stroke_scale: int,
) -> np.ndarray:
    """Main pointillism painting routine."""
    result = cv2.medianBlur(img, 11)
    grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
    batch_size = 10_000

    for start in range(0, len(grid), batch_size):
        batch = grid[start : start + batch_size]
        pixels = np.array([img[y, x] for y, x in batch])
        probs = compute_color_probabilities(pixels, palette, k=9)

        for i, (y, x) in enumerate(batch):
            color = color_select(probs[i], palette)
            angle = math.degrees(gradient.direction(y, x)) + 90
            length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
            color_int = tuple(int(c) for c in color)
            cv2.ellipse(
                result,
                (x, y),
                (length, stroke_scale),
                angle,
                0,
                360,
                color_int,
                -1,
                cv2.LINE_AA,
            )

    return result


def process_image(
    img_path: Path,
    palette_size: int = 20,
    stroke_scale: int = 0,
    gradient_radius: int = 0,
    limit_size_val: int = 0,
) -> tuple[np.ndarray, ColorPalette, VectorField]:
    img = load_image(img_path, limit_size_val)
    stroke_scale = auto_stroke_scale(img, stroke_scale)
    gradient_radius = auto_gradient_radius(img, gradient_radius)
    palette = build_palette(img, palette_size)
    gradient = compute_gradient(img, gradient_radius)
    result = paint_image(img, palette, gradient, stroke_scale)
    return result, palette, gradient