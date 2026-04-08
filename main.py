from __future__ import annotations
import argparse
from pathlib import Path
import cv2

from core import process_image, limit_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pointillism image generator")
    parser.add_argument("img_path", type=str, help="Path to input image")
    parser.add_argument("--palette-size", type=int, default=20, help="Number of palette colors")
    parser.add_argument("--stroke-scale", type=int, default=0, help="Brush stroke scale (0 = auto)")
    parser.add_argument("--gradient-radius", type=int, default=0, help="Gradient smoothing radius (0 = auto)")
    parser.add_argument("--limit-image-size", type=int, default=0, help="Limit maximum image size")
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.img_path)

    try:
        result, palette, gradient = process_image(
            img_path,
            palette_size=args.palette_size,
            stroke_scale=args.stroke_scale,
            gradient_radius=args.gradient_radius,
            limit_size_val=args.limit_image_size,
        )

        output_path = img_path.with_name(f"{img_path.stem}_drawing.jpg")
        cv2.imwrite(str(output_path), result)
        print(f"Saved pointillism image to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()