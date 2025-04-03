from typing import List
import cv2
import cv2.typing as cvt
from tqdm import tqdm


class VideoImageSource:
    """
    Retrieves images from a video file, one at a time.
    """

    def __init__(self, video_path: str):
        """
        Args:
            video_path (str): Path to the video file
        """
        self.video = cv2.VideoCapture(video_path)
        assert self.video.isOpened(), f"Could not open video file {video_path}"

        self.frames: List[cvt.MatLike] = []

        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total_frames), desc="Loading Video", unit="frame"):
            ret, frame = self.video.read()
            if not ret:
                break
            self.frames.append(frame)

        self.frame_number = 0

    def get_image(self, increment=True) -> cvt.MatLike:
        if self.frame_number >= len(self.frames):
            return None

        frame = self.frames[self.frame_number]
        if increment:
            self.frame_number += 1
        return frame

    def next_frame(self):
        if self.frame_number < len(self.frames) - 1:
            self.frame_number += 1

    def prev_frame(self):
        if self.frame_number > 0:
            self.frame_number -= 1

    def reset(self):
        self.frame_number = 0
