import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Scale, Button
from src.ImageSource import VideoImageSource

class VideoApp:
    def __init__(self, video_path):
        self.source = VideoImageSource(video_path)
        self.current_frame = self.makeImageSmall(self.source.get_image())
        self.last_click_min = None
        self.last_click_max = None
        self.video_filename = os.path.basename(video_path)

        cv2.namedWindow("Video Frame")
        cv2.setMouseCallback("Video Frame", self.on_mouse_click)

        self.root = tk.Tk()
        self.root.title("LAB Color Mask Control")

        LENGTH = 400

        self.explanation = Label(self.root, text="""
                                 Use arrow keys to navigate frames
                                 Click to sample LAB values, click update to set bounds based on the sample.
                                 A higher A value means more red, a lower one means more green.
                                 A higher B value means more yellow, a lower one means more blue.
                                 The L value is for lightness, 0 being black and 255 being white.
                                 """)

        self.l_min = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="L Min", length=LENGTH
        )
        self.l_max = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="L Max", length=LENGTH
        )
        self.a_min = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="A Min", length=LENGTH
        )
        self.a_max = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="A Max", length=LENGTH
        )
        self.b_min = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="B Min", length=LENGTH
        )
        self.b_max = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="B Max", length=LENGTH
        )
        self.region_size = Scale(
            self.root, from_=1, to=20, orient=tk.HORIZONTAL, label="Region Size", length=LENGTH
        )
        self.padding = Scale(
            self.root, from_=0, to=20, orient=tk.HORIZONTAL, label="Padding", length=LENGTH
        )
        self.erode_amount = Scale(
            self.root, from_=1, to=20, orient=tk.HORIZONTAL, label="Erode Amount", length=LENGTH
        )


        self.update_bounds_btn = Button(
            self.root, text="Update Bounds", command=self.update_bounds
        )
        self.save_btn = Button(self.root, text="Save", command=self.save_bounds)

        # Set all the mins to 255 and maxes to 0
        self.l_min.set(255)
        self.l_max.set(0)
        self.a_min.set(255)
        self.a_max.set(0)
        self.b_min.set(255)
        self.b_max.set(0)

        for widget in [
            self.explanation,
            self.l_min,
            self.l_max,
            self.a_min,
            self.a_max,
            self.b_min,
            self.b_max,
            self.update_bounds_btn,
            self.save_btn,
            self.region_size,
            self.padding,
            self.erode_amount,
        ]:
            widget.pack()

        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Left>", self.prev_frame)
        self.update_frame()
        self.root.mainloop()

    def update_frame(self):
        if self.current_frame is None:
            return

        lab_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2LAB)
        lower = np.array([self.l_min.get(), self.a_min.get(), self.b_min.get()])
        upper = np.array([self.l_max.get(), self.a_max.get(), self.b_max.get()])
        mask = cv2.inRange(lab_frame, lower, upper)
        
        # erode the mask
        erode_amount = self.erode_amount.get()
        kernel = np.ones((erode_amount, erode_amount), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)


        # Append mask to the side of the current frame for visualization
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_frame = np.vstack((self.current_frame, mask_colored))

        self.root.after(10, lambda: cv2.imshow("Video Frame", combined_frame))

        self.root.after(50, self.update_frame)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rs = self.region_size.get()
            
            x1 = max(0, x - rs)
            x2 = min(self.current_frame.shape[1], x + rs)
            y1 = max(0, y - rs)
            y2 = min(self.current_frame.shape[0], y + rs)
            
            region = self.current_frame[y1:y2, x1:x2]
            lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            min_val, max_val = lab_region.min(axis=(0, 1)), lab_region.max(axis=(0, 1))
            print(f"Clicked Region LAB Min: {min_val}, Max: {max_val}")

            self.last_click_min = min_val
            self.last_click_max = max_val

    def update_bounds(self, event=None):
        if self.last_click_min is not None and self.last_click_max is not None:
            padding_amount = self.padding.get()
            self.l_min.set(self.last_click_min[0] - padding_amount)
            self.l_max.set( self.last_click_max[0] + padding_amount)
            self.a_min.set(self.last_click_min[1] - padding_amount)
            self.a_max.set(self.last_click_max[1] + padding_amount)
            self.b_min.set( self.last_click_min[2] - padding_amount)
            self.b_max.set(self.last_click_max[2] + padding_amount)
            
    def save_bounds(self):
        output_filename = f"bounds_{self.video_filename}.txt"
        with open(output_filename, "w") as f:
            f.write(f"L Min: {self.l_min.get()}\n")
            f.write(f"L Max: {self.l_max.get()}\n")
            f.write(f"A Min: {self.a_min.get()}\n")
            f.write(f"A Max: {self.a_max.get()}\n")
            f.write(f"B Min: {self.b_min.get()}\n")
            f.write(f"B Max: {self.b_max.get()}\n")
            current_lower_bound = np.array([self.l_min.get(), self.a_min.get(), self.b_min.get()])
            current_upper_bound = np.array([self.l_max.get(), self.a_max.get(), self.b_max.get()])
   
        
        print(f"Bounds saved to {output_filename}")

    def next_frame(self, event=None):
        self.source.next_frame()
        self.current_frame = self.makeImageSmall(self.source.get_image(increment=False))

    def prev_frame(self, event=None):
        self.source.prev_frame()
        self.current_frame = self.makeImageSmall(self.source.get_image(increment=False))

    def makeImageSmall(self, image):
        scale = 0.4
        return cv2.resize(image, (0, 0), fx=scale, fy=scale)


if __name__ == "__main__":
    VideoApp("data/video/output.avi")
