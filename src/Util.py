import numpy as np
import cv2
import cv2.typing as cvt
from typing import Tuple, List


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
    x, y, width, height = ROI
    return image[y : y + height, x : x + width]


def update_roi(corners, horizontal_padding: int = 100, vertical_padding: int = 50):
    rectangles: List[cvt.RotatedRect] = [
        cv2.minAreaRect(corners[i]) for i in range(len(corners))
    ]

    largest_marker_corners = corners[np.argmax([max(r[1]) for r in rectangles])]
    roi = cv2.boundingRect(np.array(largest_marker_corners))

    # Expand the ROI by padding
    x, y, w, h = roi

    roi = (
        max(0, x - horizontal_padding),
        max(0, y - vertical_padding),
        w + horizontal_padding * 2,
        h + vertical_padding * 2,
    )
    return roi


def apply_color_threshold(image: cvt.MatLike, red_bound, blue_bound):
    """
    Apple a color threshold to an image using the LAB color space.

    :param image: Input image.
    :param red_bound: Threshold for red color.
    :param blue_bound: Threshold for blue color.
    :return: Binary mask of the image.
    """

    lab_space_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
