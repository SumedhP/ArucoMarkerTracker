from videofeed import Videofeed
import cv2
from detector import Detector

video = Videofeed("video3.avi")
detector = Detector()

times = []
import time

while True:
    frame = video.getFrame()
    if frame is None:
        break
    
    start_time = time.time()
    corners, id = detector.detect(frame)
    end_time = time.time()
    
    times.append((end_time - start_time)* 1000)
    print("Time taken: ", (end_time - start_time) * 1000, "ms ")
    
    if(corners is not None):
        cv2.rectangle(frame, (int(corners[0][0]), int(corners[0][1])), (int(corners[2][0]), int(corners[2][1])), (0, 255, 0), 2)
    
    cv2.imshow("Detected Markers", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

import matplotlib.pyplot as plt
plt.plot(times)
plt.ylabel('Time (ms)')
plt.xlabel('Frame')
plt.show()
plt.savefig("time.png")
    