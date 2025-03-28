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
from Util import crop_top_bottom


def list_detectors():
    return [Detector.getName()] + [cls.getName() for cls in Detector.__subclasses__()]


def get_detector(detector_name: str):
    if Detector.getName() == detector_name:
        return Detector()
    for cls in Detector.__subclasses__():
        if cls.getName() == detector_name:
            return cls()
    raise ValueError(f"Detector {detector_name} not found")


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

    @staticmethod
    def getName() -> str:
        return "BaselineDetector"

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
                self.min_marker_size * 0.5
            ) / max(image.shape[:2])

            self.detector.setDetectorParameters(self.detector_params)
        else:
            # If no markers are found, reset the min marker size
            self.min_marker_size = 0
            self.detector_params.minMarkerLengthRatioOriginalImg = 0
            self.detector.setDetectorParameters(self.detector_params)

        return corners, ids, rejected

    @staticmethod
    def getName() -> str:
        return "Aruco3"

    def getMinMarkerSize(self) -> float:
        return self.min_marker_size

    def getAnnotatedFrame(self, image, corners, ids, rejected):
        frame = super().getAnnotatedFrame(image, corners, ids, rejected)

        # Add the current min marker size to the top left of the frame
        cv2.putText(
            frame,
            f"Min Marker ratio: {self.detector_params.minMarkerLengthRatioOriginalImg:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        return frame


class AprilTagDetector(Detector):
    def __init__(self):
        super().__init__()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector_params.aprilTagQuadDecimate = 1.0
        self.detector.setDetectorParameters(self.detector_params)

    def setDecimation(self, decimation: float):
        assert decimation >= 1.0
        self.detector_params.aprilTagQuadDecimate = decimation
        self.detector.setDetectorParameters(self.detector_params)

    @staticmethod
    def getName() -> str:
        return "AprilTag"


class CroppedDetector(Detector):

    def __init__(self, top_crop: int = 150, bottom_crop: int = 150):
        super().__init__()
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def detectMarkers(self, image: cvt.MatLike):
        cropped_image, top = crop_top_bottom(image, self.top_crop, self.bottom_crop)
        corners, ids, rejected = super().detectMarkers(cropped_image)

        # Adjust the corners to the original image
        for corner in corners:
            corner += (0, top)

        return corners, ids, rejected
