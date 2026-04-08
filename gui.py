import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from threading import Thread
import time

import cv2
from PIL import Image, ImageTk
import numpy as np

from core import (
    ColorPalette,
    VectorField,
    randomized_grid,
    compute_color_probabilities,
    color_select,
    limit_size,
)


class PointillismGUIWithProgress:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pointillism Generator (Live Preview)")
        self.root.state("zoomed")

        # left control panel
        self.control_frame = tk.Frame(root, width=300, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # right preview panel
        self.preview_frame = tk.Frame(root, bg="black")
        self.preview_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Variables
        self.img_path: Path | None = None
        self.output_path: Path | None = None
        self.palette_size_var = tk.IntVar(value=20)
        self.stroke_scale_var = tk.IntVar(value=0)
        self.gradient_radius_var = tk.IntVar(value=0)

        # Controls
        tk.Label(self.control_frame, text="Pointillism Generator", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.control_frame, text="Select Image", command=self.select_image).pack(pady=5)
        self.img_label = tk.Label(self.control_frame, text="No image selected", wraplength=280)
        self.img_label.pack(pady=5)

        tk.Label(self.control_frame, text="Palette Size:").pack()
        tk.Entry(self.control_frame, textvariable=self.palette_size_var).pack()
        tk.Label(self.control_frame, text="Stroke Scale (0 = auto):").pack()
        tk.Entry(self.control_frame, textvariable=self.stroke_scale_var).pack()
        tk.Label(self.control_frame, text="Gradient Radius (0 = auto):").pack()
        tk.Entry(self.control_frame, textvariable=self.gradient_radius_var).pack()

        tk.Button(self.control_frame, text="Select Output File", command=self.select_output).pack(pady=10)
        self.output_label = tk.Label(self.control_frame, text="No output selected", wraplength=280)
        self.output_label.pack(pady=5)

        self.generate_button = tk.Button(self.control_frame, text="Generate Live", command=self.start_live_generation)
        self.generate_button.pack(pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.control_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)

        # Preview canvas
        self.canvas = tk.Label(self.preview_frame, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Keep the original image for preview
        self.original_img: np.ndarray | None = None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if file_path:
            self.img_path = Path(file_path)
            self.img_label.config(text=str(self.img_path))

            # Load and display original image
            img = cv2.imread(str(self.img_path))
            if img is not None:
                self.original_img = img
                self.update_preview(img)

    def select_output(self):
        file_path = filedialog.asksaveasfilename(
            title="Select Output File",
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
        )
        if file_path:
            self.output_path = Path(file_path)
            self.output_label.config(text=str(self.output_path))

    def start_live_generation(self):
        if self.img_path is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        if self.output_path is None:
            messagebox.showwarning("No output", "Please select an output file first.")
            return

        # disable buttons
        self.generate_button.config(state=tk.DISABLED)

        thread = Thread(target=self.live_paint)
        thread.start()

    def live_paint(self):
        try:
            img = cv2.imread(str(self.img_path))
            if img is None:
                raise ValueError("Failed to load image.")

            stroke_scale = self.stroke_scale_var.get() or int(round(max(img.shape) / 1000))
            gradient_radius = self.gradient_radius_var.get() or int(round(max(img.shape) / 50))

            # Palette
            palette = ColorPalette.from_image(img, self.palette_size_var.get())
            palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

            # Gradient
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gradient = VectorField.from_gradient(gray)
            gradient.smooth(gradient_radius)

            # Result canvas
            result = cv2.medianBlur(img, 11)
            grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
            batch_size = 5000

            total_batches = len(grid) / batch_size

            for batch_idx, start in enumerate(range(0, len(grid), batch_size)):
                batch = grid[start: start + batch_size]
                pixels = np.array([img[y, x] for y, x in batch])
                probs = compute_color_probabilities(pixels, palette, k=9)

                for i, (y, x) in enumerate(batch):
                    color = color_select(probs[i], palette)
                    angle = np.degrees(gradient.direction(y, x)) + 90
                    length = int(round(stroke_scale + stroke_scale * np.sqrt(gradient.magnitude(y, x))))
                    color_int = tuple(int(c) for c in color)
                    cv2.ellipse(result, (x, y), (length, stroke_scale), angle, 0, 360, color_int, -1, cv2.LINE_AA)

                # Update preview & progress
                self.update_preview(result)
                self.progress_var.set((batch_idx + 1) / total_batches * 100)

            cv2.imwrite(str(self.output_path), result)
            messagebox.showinfo("Done", f"Saved to: {self.output_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.generate_button.config(state=tk.NORMAL)

    def update_preview(self, img: np.ndarray):
        frame_w = self.preview_frame.winfo_width()
        frame_h = self.preview_frame.winfo_height()
        img_disp = limit_size(img, max(frame_w, frame_h))
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(im_pil)

        self.canvas.imgtk = imgtk
        self.canvas.config(image=imgtk)
        self.root.update_idletasks()
        time.sleep(0.01)


if __name__ == "__main__":
    root = tk.Tk()
    app = PointillismGUIWithProgress(root)
    root.mainloop()