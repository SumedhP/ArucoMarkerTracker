from typing import List, Tuple
import numpy as np
import cv2

from dataclasses import dataclass


@dataclass
class MarkerROI:
    bottom_left_x1: int
    bottom_left_y1: int
    top_right_x2: int
    top_right_y2: int


class Detector:
    BOUNDING_BOX_MARGIN_VERTICAL = 0.5
    BOUNDING_BOX_MARGIN_HORIZONTAL = 1
    RESIZED_PIXEL_WIDTH = 100

    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.adaptiveThreshWinSizeMin = 10
        self.detector_params.adaptiveThreshWinSizeMax = 100
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.refine_params = cv2.aruco.RefineParameters()

        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, self.detector_params, self.refine_params
        )

        self.ROI: MarkerROI = None

    def _detect_ROI(self, image: np.ndarray) -> Tuple[List[List[int]], List[int]]:
        assert self.ROI is not None
        img_roi = image[
            self.ROI.bottom_left_y1 : self.ROI.top_right_y2,
            self.ROI.bottom_left_x1 : self.ROI.top_right_x2,
        ]

        scale_factor = self.RESIZED_PIXEL_WIDTH / img_roi.shape[1]

        img_roi = cv2.resize(img_roi, (0, 0), fx=scale_factor, fy=scale_factor)
        corners, ids, _ = self.detector.detectMarkers(img_roi)

        if len(corners) == 0:
            return None, None

        for corner in corners:
            corner[0] *= 1 / scale_factor
            corner[0][:, 0] += self.ROI.bottom_left_x1
            corner[0][:, 1] += self.ROI.bottom_left_y1

        return corners, ids

    def detect(self, image: np.ndarray, display: bool = False):
        corners, ids = None, None
        if self.ROI is not None:
            corners, ids = self._detect_ROI(image)

        if corners is None:
            # Either we didn't have an ROI or we didn't find any markers in the ROI
            # Scan the whole image
            corners, ids, _ = self.detector.detectMarkers(image)

        if len(corners) == 0:
            # No markers found
            self.ROI = None
            return None, None

        largest_marker_corners, largest_marker_id = self._get_best_candidate(corners, ids)

        # Set the ROI to be width * bounding margin
        min_x = int(min(largest_marker_corners[:, 0]))
        min_y = int(min(largest_marker_corners[:, 1]))
        max_x = int(max(largest_marker_corners[:, 0]))
        max_y = int(max(largest_marker_corners[:, 1]))

        roi_width = (max_x - min_x) * self.BOUNDING_BOX_MARGIN_HORIZONTAL
        roi_height = (max_y - min_y) * self.BOUNDING_BOX_MARGIN_VERTICAL
        roi_width = int(roi_width)
        roi_height = int(roi_height)

        new_roi_bottom_left_x = max(min_x - roi_width, 0)
        new_roi_bottom_left_y = max(min_y - roi_height, 0)
        new_roi_top_right_x = min(max_x + roi_width, image.shape[1])
        new_roi_top_right_y = min(max_y + roi_height, image.shape[0])

        self.ROI = MarkerROI(
            new_roi_bottom_left_x,
            new_roi_bottom_left_y,
            new_roi_top_right_x,
            new_roi_top_right_y,
        )

        # Draw the largest marker on the image
        if display:
            print(
                f"Marker ROI size: (%d, %d)"
                % (
                    new_roi_top_right_x - new_roi_bottom_left_x,
                    new_roi_top_right_y - new_roi_bottom_left_y,
                )
            )
            print(f"Marker size: (%d, %d)" % (max_x - min_x, max_y - min_y))

            cv2.rectangle(
                image,
                (int(min_x), int(min_y)),
                (int(max_x), int(max_y)),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                image,
                (int(new_roi_bottom_left_x), int(new_roi_bottom_left_y)),
                (int(new_roi_top_right_x), int(new_roi_top_right_y)),
                (255, 0, 0),
                2,
            )
            cv2.imshow("Detected Markers", image)
            cv2.waitKey(0)

        return largest_marker_corners, largest_marker_id

    def _get_size_of_corner(self, corner: List[List[int]]) -> int:
        corner: List[int] = corner[0]
        width: int = max(corner[:, 0]) - min(corner[:, 0])
        height: int = max(corner[:, 1]) - min(corner[:, 1])
        return width * height

    def _get_best_candidate(
        self, corners: List[List[int]], ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        marker_dict: dict[bytes, int] = {}
        for i in range(len(ids)):
            marker_dict[corners[i].tobytes()] = ids[i]

        # Sort the corners by size, largest to smallest
        sorted_corners:List[List[int]] = sorted(corners, key=self._get_size_of_corner, reverse=True)

        # Get the largest marker
        largest_marker_corners:List[int] = sorted_corners[0]
        largest_marker_id = marker_dict[largest_marker_corners.tobytes()]

        return largest_marker_corners[0], largest_marker_id


if __name__ == "__main__":
    detector = Detector()
    image = cv2.imread("image.png")

    import time

    ITERATIONS = 10
    start = time.time()

    for _ in range(ITERATIONS):
        detector.detect(image)
        detector.reset_roi()

    end = time.time()
    print("Average time: ", (end - start) / ITERATIONS * 1000, "ms")

    detector.detect(image)

    start = time.time()

    for _ in range(ITERATIONS):
        detector.detect(image)

    end = time.time()
    print("Average time: ", (end - start) / ITERATIONS * 1000, "ms")
    detector.detect(image, True)
