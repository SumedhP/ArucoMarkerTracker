# Need to showcase baseline detector, Aruco3 detector, Decimation
# Our stuff of : Image crop, Image ROI, Image HSV/Lab filtration
# Something that combines all 3 of these things

import abc
from typing import List

import cv2
from cv2.aruco import (
    DetectorParameters,
    ArucoDetector,
    RefineParameters,
    DICT_6X6_1000,
)
import cv2.typing as cvt
import numpy as np


class Detector(metaclass=abc.ABCMeta):
    def __init__(self):
        # Default detector configuration, using subpixel corner refinement
        self.detector_params = DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Dictionary of Marker IDs
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICT_6X6_1000)

        self.detector = ArucoDetector(
            self.dictionary, self.detector_params, RefineParameters()
        )

    def detectMarkers(self, image: cvt.MatLike):
        return self.detector.detectMarkers(image)

    def getName(self) -> str:
        return "Baseline Detector"
    
    def getAnnotatedFrame(self, image: cvt.MatLike, corners, ids, rejected):
        frame = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return frame


class Aruco3Detector(Detector):
    """
    Detector using the Aruco3 algorithm to dynamically speed up detection based on the size of the markers detected in the previous frame
    """

    def __init__(self):
        super().__init__()
        self.detector_params.useAruco3Detection = True
        self.detector.setDetectorParameters(self.detector_params)
        self.min_marker_size = 0

    def detectMarkers(self, image: cvt.MatLike):
        corners, ids, rejected = self.detector.detectMarkers(image)
        # Find the smallest marker size
        if len(corners) > 0:
            rectangles: List[cvt.RotatedRect] = [
                cv2.minAreaRect(corners[i]) for i in range(len(corners))
            ]
            self.min_marker_size = min([min(rectangle[1]) for rectangle in rectangles])

            # Update the minimum length being detected by Aruco3
            self.detector_params.minMarkerLengthRatioOriginalImg = (
                self.min_marker_size / max(image.shape[:2])
            )
            self.detector.setDetectorParameters(self.detector_params)
        else:
            # If no markers are found, reset the min marker size
            self.min_marker_size = 0
            self.detector_params.minMarkerLengthRatioOriginalImg = 0
            self.detector.setDetectorParameters(self.detector_params)

        return corners, ids, rejected

    def getName(self) -> str:
        return "Aruco3"

    def getMinMarkerSize(self) -> float:
        return self.min_marker_size

    def getAnnotatedFrame(self, image, corners, ids, rejected):
        frame = super().getAnnotatedFrame(image, corners, ids, rejected)

        # Add the current min marker size to the top left of the frame
        cv2.putText(
            frame,
            f"Min Marker Size: {self.min_marker_size:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        return frame
