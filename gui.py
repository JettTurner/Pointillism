import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2

from core import process_image, limit_size


class PointillismGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pointillism Generator")
        self.root.geometry("500x400")

        self.img_path: Path | None = None
        self.palette_size_var = tk.IntVar(value=20)
        self.stroke_scale_var = tk.IntVar(value=0)
        self.gradient_radius_var = tk.IntVar(value=0)

        tk.Label(root, text="Pointillism Generator", font=("Arial", 16)).pack(pady=10)
        tk.Button(root, text="Select Image", command=self.select_image).pack(pady=5)
        self.img_label = tk.Label(root, text="No image selected")
        self.img_label.pack(pady=5)

        tk.Label(root, text="Palette Size:").pack()
        tk.Entry(root, textvariable=self.palette_size_var).pack()
        tk.Label(root, text="Stroke Scale (0 = auto):").pack()
        tk.Entry(root, textvariable=self.stroke_scale_var).pack()
        tk.Label(root, text="Gradient Radius (0 = auto):").pack()
        tk.Entry(root, textvariable=self.gradient_radius_var).pack()

        tk.Button(root, text="Generate Pointillism", command=self.generate).pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if file_path:
            self.img_path = Path(file_path)
            self.img_label.config(text=str(self.img_path))

    def generate(self):
        if self.img_path is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        try:
            result, palette, gradient = process_image(
                self.img_path,
                self.palette_size_var.get(),
                self.stroke_scale_var.get(),
                self.gradient_radius_var.get(),
            )
            output_path = self.img_path.with_name(f"{self.img_path.stem}_drawing.jpg")
            cv2.imwrite(str(output_path), result)
            messagebox.showinfo("Done", f"Saved to: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = PointillismGUI(root)
    root.mainloop()