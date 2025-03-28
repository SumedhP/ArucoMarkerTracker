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


def apply_color_threshold(image: cvt.MatLike) -> Tuple[bool, int, int, cvt.MatLike]:
    """
    Apple a color threshold to an image using the LAB color space.

    :param image: Input image.
    :return: whether the image has red or blue color, the top left x and y coordinates of the detected color, and the cropped image.
    """
    
    RED_LOWER_BOUND = np.array([130, 200, 190], dtype=np.uint8)
    RED_UPPER_BOUND = np.array([140, 210, 200], dtype=np.uint8)
    
    BLUE_LOWER_BOUND = np.array([75, 200, 15], dtype=np.uint8)
    BLUE_UPPER_BOUND = np.array([85, 210, 30], dtype=np.uint8)
    
    IMAGE_STEP_SIZE = 4
    
    downsampled_image = image[::IMAGE_STEP_SIZE, ::IMAGE_STEP_SIZE, :]

    lab_space_image = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2LAB)
    # Apply the mask to the original image
    red_mask = cv2.inRange(lab_space_image, RED_LOWER_BOUND, RED_UPPER_BOUND)
    blue_mask = cv2.inRange(lab_space_image, BLUE_LOWER_BOUND, BLUE_UPPER_BOUND)
    mask = cv2.bitwise_or(red_mask, blue_mask)
    
    # Check if the mask is empty
    if np.count_nonzero(mask) == 0:
        return False, 0, 0, None
    
    def get_min_max_index(arr, scalar, maximum):
        PADDING = 50
        lower_bound: int = np.argmax(arr).astype(int) * scalar
        lower_bound = max(lower_bound - PADDING, 0)
        upper_bound: int = (len(arr) - np.argmax(arr[::-1]).astype(int)) * scalar
        upper_bound = min(upper_bound + PADDING, maximum)
        return lower_bound, upper_bound
    
    columns_in_mask = np.any(mask, axis=0)
    rows_in_mask = np.any(mask, axis=1)
    
    min_col, max_col = get_min_max_index(columns_in_mask, IMAGE_STEP_SIZE, image.shape[1])
    min_row, max_row = get_min_max_index(rows_in_mask, IMAGE_STEP_SIZE, image.shape[0])
    
    print(f"Min row: {min_row}, Min col: {min_col}")
    print(f"Max row: {max_row}, Max col: {max_col}")
    
    # Crop the image around this area
    cropped_image = image[min_row:max_row, min_col:max_col]
    
    
    return True, min_row, min_col, cropped_image, mask

    
