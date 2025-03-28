import abc
from typing import List
import cv2
import cv2.typing as cvt
from tqdm import tqdm


class ImageSource(metaclass=abc.ABCMeta):
    """
    Abstract base class to define the interface for sources from which
    image frames can be retrieved.
    """

    @abc.abstractmethod
    def get_image(self) -> cvt.MatLike:
        """
        Get the next image from the source.

        Returns:
            cvt.MatLike: The next image in the sequence or None if no more images are available.
        """
        pass

    @abc.abstractmethod
    def reset():
        """
        Resets the source to the beginning of the sequence.
        """
        pass


class ImageListSource(ImageSource):
    """
    Retrieves images from a list of image file paths, one at a time.
    """

    def __init__(self, images: List[str]):
        """
        Args:
            images (List[str]): List of image file paths
        """
        assert len(images) > 0

        self.images: List[cvt.MatLike] = []
        # Preload all images into memory
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

    def get_image(self) -> cvt.MatLike:
        if self.frame_number >= len(self.frames):
            return None

        frame = self.frames[self.frame_number]
        self.frame_number += 1
        return frame

    def reset(self):
        self.frame_number = 0
