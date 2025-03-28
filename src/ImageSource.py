import abc
from typing import List
import cv2
import cv2.typing as cvt
from tqdm import tqdm


class ImageSource(metaclass=abc.ABCMeta):
    """
    Base class to define a source for the images we've like to process
    """

    @abc.abstractmethod
    def get_image(self) -> cvt.MatLike:
        pass

    @abc.abstractmethod
    def reset():
        pass


class ImageListSource(ImageSource):
    """
    Takes in a list of images and returns them one by one
    """

    def __init__(self, images: List[str]):
        assert len(images) > 0

        self.images: List[cvt.MatLike] = []
        for image in images:
            self.images.append(cv2.imread(image))
        self.image_number = 0

    def get_image(self) -> cvt.MatLike:
        if self.image_number >= len(self.images):
            return None

        image = self.images[self.image_number]
        self.image_number += 1
        return image

    def reset(self):
        self.image_number = 0


class VideoImageSource(ImageSource):
    """
    Takes in a video file and returns the frames one by one
    """

    def __init__(self, video_path: str):
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

    def get_image(self) -> cvt.MatLike:
        if self.frame_number >= len(self.frames):
            return None

        frame = self.frames[self.frame_number]
        self.frame_number += 1
        return frame

    def reset(self):
        self.frame_number = 0
