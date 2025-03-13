from videofeed import Videofeed
import cv2
from detector import Detector

video = Videofeed("video4.avi")
detector = Detector()

OUTPUT_FILE = "corners_roi.csv"
with open(OUTPUT_FILE, "w") as f:
    f.write("x1,y1,x2,y2\n")

while True:
    frame = video.getFrame()
    if frame is None:
        break
    
    corners, id = detector.detect(frame)
    
    if corners is not None:
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{corners[0][0]}, {corners[0][1]}, {corners[2][0]}, {corners[2][1]}\n")
    
print("Done")

OUTPUT_FILE = "corners_raw.csv"
video = Videofeed("video4.avi")
with open(OUTPUT_FILE, "w") as f:
    f.write("x1,y1,x2,y2\n")

while True:
    frame = video.getFrame()
    if frame is None:
        break
    
    corners, id = detector.detect(frame, resize=False)
    
    if corners is not None:
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{corners[0][0]}, {corners[0][1]}, {corners[2][0]}, {corners[2][1]}\n")

print("Done")


csv1 = "corners_raw.csv"
csv2 = "corners_roi.csv"
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data1 = pd.read_csv(csv1)
data2 = pd.read_csv(csv2)

# Print out the possible headers for each file
print(data1.columns)
print(data2.columns)

plt.plot(data1["x1"], label="Raw")
plt.plot(data2["x1"], label="ROI")
plt.title("x1")
plt.legend()
# Set size to 1080p
plt.gcf().set_size_inches(1920/100, 1080/100)
plt.savefig("150px_resize.jpg")
plt.show()
