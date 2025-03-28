import numpy as np
import cv2
import cv2.typing as cvt
from typing import Tuple


def crop_top_bottom(
    image: cvt.MatLike, top: int, bottom: int
) -> Tuple[cvt.MatLike, int]:
    """
    Crop the top and bottom of the image.

    :param image: Input image.
    :param top: Number of pixels to remove from the top.
    :param bottom: Number of pixels to remove from the bottom.
    :return: Cropped image and the top crop value.
    """
    if top < 0 or bottom < 0:
        raise ValueError("Top and bottom crop values must be non-negative 🤦")

    height = image.shape[0]

    if top + bottom >= height:
        raise ValueError(
            "Cropping values are too large. The resulting image would be empty OwO 🫨"
        )

    return image[top : height - bottom, :], top


def crop_roi(image: cvt.MatLike, ROI: cvt.Rect) -> cvt.MatLike:
    """
    Crop the image to the region of interest.

    :param image: Input image.
    :param ROI: Region of interest.
    :return: Cropped image.
    """
    return image[ROI.y : ROI.y + ROI.height, ROI.x : ROI.x + ROI.width]
