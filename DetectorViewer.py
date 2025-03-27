from src.ImageSource import VideoImageSource
from src.Detector import Detector, Aruco3Detector
import cv2

feed = VideoImageSource("data/video/video1.avi")

print("Start")

while True:
    frame = feed.get_image()
    if frame is None:
        break
    
    detector = Detector()
    corners, ids, rejected = detector.detectMarkers(frame)
    frame = detector.getAnnotatedFrame(frame, corners, ids, rejected)
    
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

print("End")
    
    
