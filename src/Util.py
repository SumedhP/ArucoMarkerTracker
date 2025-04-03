import numpy as np
import cv2
import cv2.typing as cvt
from typing import Tuple
from line_profiler import profile


@profile
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
        raise ValueError("Top and bottom crop values must be non-negative")

    height = image.shape[0]

    if top + bottom >= height:
        raise ValueError(
            "Cropping values are too large. The resulting image would be empty"
        )

    return image[top : height - bottom, :], top


@profile
def crop_roi(image: cvt.MatLike, roi: cvt.Rect) -> cvt.MatLike:
    """
    Crop the image to the region of interest.

    :param image: Input image.
    :param roi: Region of interest.
    :return: Cropped image.
    """
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


@profile
def update_roi(
    corners, horizontal_padding: int = 100, vertical_padding: int = 50
) -> Tuple[int, int, int, int]:
    """
    Updates the region of interest (ROI) based on the largest detected marker.
    The ROI is expanded by the specified padding.

    :param corners: List of marker corner points.
    :param horizontal_padding: Horizontal padding in pixels.
    :param vertical_padding: Vertical padding in pixels.
    :return: Updated ROI as (x, y, width, height).
    """
    if len(corners) == 0:
        raise ValueError("No corners provided to update the ROI.")

    # Determine the rectangles for each set of corners
    rectangles = [cv2.minAreaRect(np.array(marker)) for marker in corners]

    # Find the largest rectangle based on its area
    largest_index = np.argmax([w * h for _, (w, h), _ in rectangles])
    largest_marker_corners = corners[largest_index]

    # Get the dimensions of the largest rectangle and expand it by the padding
    x, y, w, h = cv2.boundingRect(np.array(largest_marker_corners))

    roi = (
        max(0, x - horizontal_padding),
        max(0, y - vertical_padding),
        w + 2 * horizontal_padding,
        h + 2 * vertical_padding,
    )

    return roi


@profile
def apply_color_threshold(
    image: cvt.MatLike,
) -> Tuple[bool, int, int, cvt.MatLike, cvt.MatLike]:
    """
    Applies a color threshold to an image using the LAB color space.

    :param image: Input image.
    :return: A tuple containing:
        - A boolean indicating whether red or blue color was detected.
        - The top-left x and y coordinates of the detected region.
        - The cropped image (or None if no color detected).
        - The generated mask.
    """
    # LAB color space bounds for red and blue colors
    RED_BOUNDS = (
        np.array([94, 161, 137], dtype=np.uint8),
        np.array([191, 197, 166], dtype=np.uint8),
    )
    BLUE_BOUNDS = (
        np.array([62, 98, 65], dtype=np.uint8),
        np.array([179, 145, 114], dtype=np.uint8),
    )

    DOWNSCALING_FACTOR = 4
    PADDING = 50

    # Downscale the image and convert to LAB color space
    downscaled_image = image[::DOWNSCALING_FACTOR, ::DOWNSCALING_FACTOR, :]

    lab_image = cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2LAB)

    # Generate masks for red and blue colors in the image
    red_mask = cv2.inRange(lab_image, *RED_BOUNDS)
    blue_mask = cv2.inRange(lab_image, *BLUE_BOUNDS)
    mask = red_mask | blue_mask

    # If no colors are detected, return early
    if np.count_nonzero(mask) == 0:
        return False, 0, 0, None, mask

    def get_bounding_indices(
        arr: np.ndarray, scale: int, max_value: int
    ) -> Tuple[int, int]:
        """
        Computes the minimum and maximum indices of the bounding box in the mask,
        and scales them back up to original image size.
        """
        lower_bound = np.argmax(arr) * scale
        upper_bound = (len(arr) - np.argmax(arr[::-1])) * scale

        return max(lower_bound - PADDING, 0), min(upper_bound + PADDING, max_value)

    # Get the bounding box indices for the mask
    min_col, max_col = get_bounding_indices(
        np.any(mask, axis=0), DOWNSCALING_FACTOR, image.shape[1]
    )
    min_row, max_row = get_bounding_indices(
        np.any(mask, axis=1), DOWNSCALING_FACTOR, image.shape[0]
    )

    # Crop the image to the detected region
    cropped_image = image[min_row:max_row, min_col:max_col]

    return True, min_row, min_col, cropped_image, mask
