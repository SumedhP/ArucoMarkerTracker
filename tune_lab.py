import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale
from tqdm import tqdm

class VideoImageSource:
    """ Retrieves images from a video file, one at a time. """
    def __init__(self, video_path: str):
        self.video = cv2.VideoCapture(video_path)
        assert self.video.isOpened(), f"Could not open video file {video_path}"
        self.frames = []
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Loading Video", unit="frame"):
            ret, frame = self.video.read()
            if not ret:
                break
            self.frames.append(frame)
        self.frame_number = 0
    
    def get_image(self):
        if self.frame_number >= len(self.frames):
            return None
        frame = self.frames[self.frame_number]
        return frame
    
    def next_frame(self):
        if self.frame_number < len(self.frames) - 1:
            self.frame_number += 1
    
    def prev_frame(self):
        if self.frame_number > 0:
            self.frame_number -= 1
    
    def reset(self):
        self.frame_number = 0

class VideoApp:
    def __init__(self, video_path):
        self.source = VideoImageSource(video_path)
        self.current_frame = self.source.get_image()
        
        cv2.namedWindow("Video Frame")
        cv2.setMouseCallback("Video Frame", self.on_mouse_click)

        self.root = tk.Tk()
        self.root.title("LAB Color Mask Control")
        
        self.l_min = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="L Min")
        self.l_max = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="L Max")
        self.a_min = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="A Min")
        self.a_max = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="A Max")
        self.b_min = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="B Min")
        self.b_max = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="B Max")
        
        for widget in [self.l_min, self.l_max, self.a_min, self.a_max, self.b_min, self.b_max]:
            widget.pack()
            widget.set(128)  # Default midpoint
        
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
        
        cv2.imshow("Video Frame", self.current_frame)
        cv2.imshow("LAB Mask", mask)
        
        self.root.after(50, self.update_frame)
    
    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            region_size = 5
            x1, x2 = max(0, x - region_size), min(self.current_frame.shape[1], x + region_size)
            y1, y2 = max(0, y - region_size), min(self.current_frame.shape[0], y + region_size)
            region = self.current_frame[y1:y2, x1:x2]
            lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            min_val, max_val = lab_region.min(axis=(0, 1)), lab_region.max(axis=(0, 1))
            print(f"Clicked Region LAB Min: {min_val}, Max: {max_val}")
    
    def next_frame(self, event=None):
        self.source.next_frame()
        self.current_frame = self.source.get_image()
    
    def prev_frame(self, event=None):
        self.source.prev_frame()
        self.current_frame = self.source.get_image()

if __name__ == "__main__":
    VideoApp("data/video/red_aruco_marker.mp4")  # Replace with the path to your video file
