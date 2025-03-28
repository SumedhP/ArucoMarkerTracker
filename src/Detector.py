import abc
from typing import List, Optional

import cv2
from cv2.aruco import (
    DetectorParameters,
    ArucoDetector,
    RefineParameters,
    DICT_6X6_1000,
)
import cv2.typing as cvt
import numpy as np
from .Util import *


def list_detectors():
    return [Detector.getName()] + [cls.getName() for cls in Detector.__subclasses__()]


def get_detector(detector_name: str):
    def stringEqualsIgnoreCase(string1: str, string2: str) -> str:
        return string1.lower().strip() == string2.lower().strip()

    if stringEqualsIgnoreCase(Detector.getName(), detector_name):
        return Detector()
    for cls in Detector.__subclasses__():
        if stringEqualsIgnoreCase(cls.getName(), detector_name):
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
        return "Baseline"

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
        self.setDecimation(1.0)

    def setDecimation(self, decimation: float):
        assert decimation >= 1.0
        self.detector_params.aprilTagQuadDecimate = decimation
        self.detector.setDetectorParameters(self.detector_params)

    @staticmethod
    def getName() -> str:
        return "AprilTag"


class CroppedDetector(Detector):
    def __init__(self, top_crop: int = 50, bottom_crop: int = 50):
        super().__init__()
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def detectMarkers(self, image: cvt.MatLike):
        cropped_image, top = crop_top_bottom(image, self.top_crop, self.bottom_crop)
        corners, ids, rejected = super().detectMarkers(cropped_image)

        # Adjust the corners to the original image to account for cropping
        for corner in corners:
            corner += (0, top)

        return corners, ids, rejected

    @staticmethod
    def getName() -> str:
        return "Cropped"


class ROIDetector(Detector):
    def __init__(self, resize: bool = True, resize_height: int = 50):
        super().__init__()
        self.roi: Optional[cvt.Rect] = None
        self.resize = resize
        self.resize_height = resize_height

    def detectMarkers(self, image):
        # If we have an ROI, first attempt to scan in that region. If no markers are found, scan the entire image
        corners, ids, rejected = None, None, None

        if self.roi is not None:
            roi_image = crop_roi(image, self.roi)

            if self.resize:
                scale = self.resize_height / roi_image.shape[0]
                roi_image = cv2.resize(roi_image, (0, 0), fx=scale, fy=scale)

            corners, ids, rejected = super().detectMarkers(roi_image)

            roi_x, roi_y, _, _ = self.roi
            for corner in corners:
                if self.resize:
                    corner /= scale

                corner += (roi_x, roi_y)

        if corners is None or len(corners) == 0:
            corners, ids, rejected = super().detectMarkers(image)

        if len(corners) > 0:
            self.roi = update_roi(corners)
        else:
            self.roi = None

        return corners, ids, rejected

    @staticmethod
    def getName() -> str:
        return "ROI"

    def getAnnotatedFrame(self, image, corners, ids, rejected):
        frame = super().getAnnotatedFrame(image, corners, ids, rejected)
        if self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame


class ColorDetector(Detector):
    def __init__(self):
        super().__init__()
    
    def detectMarkers(self, image):
        success, top, left, cropped, _ = apply_color_threshold(image)
        if not success:
            return None, None, None
        
        corners, ids, rejected = super().detectMarkers(cropped)
        for corner in corners:
            corner += (top, left)
        
        return corners, ids, rejected
    
    

    def getAnnotatedFrame(self, image, corners, ids, rejected):
        # Extend the size of the frame to also show the mask on the right side
        frame = super().getAnnotatedFrame(image, corners, ids, rejected)
        success, _, _, _, mask = apply_color_threshold(image)

        if success:
            # Normalize and convert mask to uint8
            mask = (mask > 0).astype(np.uint8) * 255  

            # Resize using nearest-neighbor interpolation
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert to 3-channel image
            mask = np.stack([mask] * 3, axis=-1)  

            # Create new frame with space for the mask
            new_frame = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
            new_frame[:, : image.shape[1]] = frame
            new_frame[:, image.shape[1] :] = mask
            frame = new_frame

        return frame

    @staticmethod
    def getName() -> str:
        return "Color"
