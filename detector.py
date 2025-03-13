from typing import List
import cv2
import time

from dataclasses import dataclass
@dataclass
class MarkerROI:
    bottom_left_x1: int
    bottom_left_y1: int
    top_right_x2: int
    top_right_y2: int
    
    def size(self) -> int:
        return (self.top_right_x2 - self.bottom_left_x1) * (self.top_right_y2 - self.bottom_left_y1)

class Detector():
    BOUNDING_BOX_MARGIN_VERTICAL = 0.5
    BOUNDING_BOX_MARGIN_HORIZONTAL = 1
    
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.adaptiveThreshWinSizeMin = 10
        self.detector_params.adaptiveThreshWinSizeMax = 100
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.refine_params = cv2.aruco.RefineParameters()
        
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params, self.refine_params)
        
        self.ROI : MarkerROI = None
        
    def _get_size_of_corner(self, corner) -> int:
        corner = corner[0]
        # print(corner)
        min_X = min(corner[:, 0])
        min_Y = min(corner[:, 1])
        max_X = max(corner[:, 0])
        max_Y = max(corner[:, 1])
        # print(min_X, min_Y, max_X, max_Y)
        return (max_X - min_X) * (max_Y - min_Y)
    
    def reset_roi(self):
        self.ROI = None
    
    def detect(self, image, display=False):
        resize_time = 0
        
        if self.ROI is not None:
            img_roi = image[self.ROI.bottom_left_y1:self.ROI.top_right_y2, self.ROI.bottom_left_x1:self.ROI.top_right_x2]
            
            scale_factor = 100 / img_roi.shape[1]
            
            import time
            start = time.time()
            img_roi = cv2.resize(img_roi, (0, 0), fx=scale_factor, fy=scale_factor)
            end = time.time()
            resize_time = (end - start) * 1000
            
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(img_roi) # Scan the ROI
            
            if (len(corners) == 0):
                print("No markers detected in ROI")
                self.reset_roi()
                corners, ids, rejectedImgPoints = self.detector.detectMarkers(image) # Scan the whole image
            else:
                # print("OMG CHAT WE SAW SMTH IN THE ROI")
                # Adjust corners by the ROI
                for corner in corners:
                    corner[0] *= 1/scale_factor
                    corner[0][:, 0] += self.ROI.bottom_left_x1
                    corner[0][:, 1] += self.ROI.bottom_left_y1
                    
        else:
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(image) # Scan the whole image
        
        if(len(corners) == 0):
            # print("No markers detected")
            self.reset_roi()
            return None, None, None
        
        # At this point, if we have a result, find the largest marker
        # Put corner and id into a Dictionary
        marker_dict = {}
        for i in range(len(ids)):
            marker_dict[corners[i].tobytes()] = ids[i]
        
        # Sort the corners by size, largest to smallest
        sorted_corners = sorted(corners, key=self._get_size_of_corner, reverse=True)
        # Get the largest marker
        largest_marker_corners = sorted_corners[0]
        largest_marker_id = marker_dict[largest_marker_corners.tobytes()]
        
        # Set the ROI to be width * bounding margin
        min_x = int(min(largest_marker_corners[0][:, 0]))
        min_y = int(min(largest_marker_corners[0][:, 1]))
        max_x = int(max(largest_marker_corners[0][:, 0]))
        max_y = int(max(largest_marker_corners[0][:, 1]))
        
        # print("The largest marker was at: ", min_x, min_y, max_x, max_y)
        
        roi_width = (max_x - min_x) * self.BOUNDING_BOX_MARGIN_HORIZONTAL
        roi_height = (max_y - min_y) * self.BOUNDING_BOX_MARGIN_VERTICAL 
        roi_width = int(roi_width)
        roi_height = int(roi_height)
        # print("Roi width: ", roi_width, "Roi height: ", roi_height)
        
        new_roi_bottom_left_x = max(min_x - roi_width, 0)
        new_roi_bottom_left_y = max(min_y - roi_height, 0)
        new_roi_top_right_x = min(max_x + roi_width, image.shape[1])
        new_roi_top_right_y = min(max_y + roi_height, image.shape[0])
        
        self.ROI = MarkerROI(new_roi_bottom_left_x, new_roi_bottom_left_y, new_roi_top_right_x, new_roi_top_right_y)
        
        # Draw the largest marker on the image
        if display:
            print(f"Marker ROI size: (%d, %d)" % (new_roi_top_right_x - new_roi_bottom_left_x, new_roi_top_right_y - new_roi_bottom_left_y))
            print(f"Marker size: (%d, %d)" % (max_x - min_x, max_y - min_y))
            
            cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
            cv2.rectangle(image, (int(new_roi_bottom_left_x), int(new_roi_bottom_left_y)), (int(new_roi_top_right_x), int(new_roi_top_right_y)), (255, 0, 0), 2)
            cv2.imshow("Detected Markers", image)
            cv2.waitKey(0)
        
        return largest_marker_corners, largest_marker_id, resize_time
        
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
