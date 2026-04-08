from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import limit_size, regulate


class ColorPalette:
    def __init__(self, colors: np.ndarray, base_len: int = 0):
        self.colors = np.asarray(colors, dtype=np.float32)
        self.base_len = base_len if base_len > 0 else len(self.colors)

    @staticmethod
    def from_image(
        img: np.ndarray,
        n: int,
        max_img_size: int = 200,
        n_init: int | str = "auto",
    ) -> "ColorPalette":
        """Extract dominant colors from an image using KMeans."""

        # Downscale for performance
        img_small = limit_size(img, max_img_size)

        pixels = img_small.reshape(-1, 3)

        clt = KMeans(
            n_clusters=n,
            n_init=n_init,  # modern sklearn
            random_state=42,  # reproducibility
        )
        clt.fit(pixels)

        return ColorPalette(clt.cluster_centers_)

    def extend(
        self,
        extensions: Iterable[Tuple[int, int, int]],
    ) -> "ColorPalette":
        """Extend palette with regulated variations."""

        base_colors = self.colors.astype(np.uint8).reshape(1, -1, 3)

        extended_sets: List[np.ndarray] = []
        for ext in extensions:
            regulated = regulate(base_colors, *ext).reshape(-1, 3)
            extended_sets.append(regulated)

        combined = np.vstack([self.colors.reshape(-1, 3), *extended_sets])

        return ColorPalette(combined, self.base_len)

    def to_image(self, cell_size: int = 80) -> np.ndarray:
        """Render palette as a grid image."""

        cols = self.base_len
        rows = math.ceil(len(self.colors) / cols)

        result = np.zeros(
            (rows * cell_size, cols * cell_size, 3),
            dtype=np.uint8,
        )

        for idx, color in enumerate(self.colors):
            y = idx // cols
            x = idx % cols

            color_uint8 = tuple(int(c) for c in color)

            cv2.rectangle(
                result,
                (x * cell_size, y * cell_size),
                ((x + 1) * cell_size, (y + 1) * cell_size),
                color_uint8,
                -1,
            )

        return result

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, item: int) -> np.ndarray:
        return self.colors[item]