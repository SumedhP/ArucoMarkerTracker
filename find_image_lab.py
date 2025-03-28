from src.ImageSource import VideoImageSource
import cv2
import numpy as np
# Open up a video and allow the user to select a region on interest. In there, convert it to LAB and return the lowest and highest values of the L channel

feed = VideoImageSource("data/video/red_aruco_marker.mp4")

feed.reset()

frame = feed.get_image()
if frame is None:
    raise ValueError("No image found in the video feed.")

# # Display the frame to the user and allow them to select a region of interest (ROI)
# roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
# cv2.destroyWindow("Select ROI")
# if roi is None:
#     raise ValueError("No ROI selected.")

# # Crop the image to the selected ROI
# x, y, w, h = roi
# roi_image = frame[y:y+h, x:x+w]

# # Convert the cropped image to LAB color space
# lab_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2Lab)

# l = lab_image[:, :, 0]  # L channel
# a = lab_image[:, :, 1]  # A channel
# b = lab_image[:, :, 2]  # B channel

# min_l = np.min(l)
# max_l = np.max(l)

# min_a = np.min(a)
# max_a = np.max(a)

# min_b = np.min(b)
# max_b = np.max(b)
# print(f"Min L: {min_l}, Max L: {max_l}")
# print(f"Min A: {min_a}, Max A: {max_a}")
# print(f"Min B: {min_b}, Max B: {max_b}")

red_min_l = 130
red_max_l = 140
red_min_a = 200
red_max_a = 210
red_min_b = 190
red_max_b = 200

blue_min_l = 75
blue_max_l = 85
blue_min_a = 200
blue_max_a = 210
blue_min_b = 15
blue_max_b = 30

# Create bitmask
# lower_bound = np.array([min_l, min_a, min_b], dtype=np.uint8)
# upper_bound = np.array([max_l, max_a, max_b], dtype=np.uint8)

red_lower_bound = np.array([red_min_l, red_min_a, red_min_b], dtype=np.uint8)
red_upper_bound = np.array([red_max_l, red_max_a, red_max_b], dtype=np.uint8)

blue_lower_bound = np.array([blue_min_l, blue_min_a, blue_min_b], dtype=np.uint8)
blue_upper_bound = np.array([blue_max_l, blue_max_a, blue_max_b], dtype=np.uint8)


# Apply the mask to the original image
lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
# mask = cv2.inRange(lab_image, lower_bound, upper_bound)
blue_mask = cv2.inRange(lab_image, blue_lower_bound, blue_upper_bound)
red_mask = cv2.inRange(lab_image, red_lower_bound, red_upper_bound)
mask = cv2.bitwise_or(blue_mask, red_mask)

masked_image = cv2.bitwise_and(frame, frame, mask=mask)

# Display the masked image and the original image
cv2.imshow("Masked Image", masked_image)
cv2.imshow("Original Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now apply this mask over all remaining images in the feed and show the masked image

while True:
    frame = feed.get_image()
    if frame is None:
        break

    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    # mask = cv2.inRange(lab_image, lower_bound, upper_bound)
    blue_mask = cv2.inRange(lab_image, blue_lower_bound, blue_upper_bound)
    red_mask = cv2.inRange(lab_image, red_lower_bound, red_upper_bound)
    mask = cv2.bitwise_or(blue_mask, red_mask)

    # Dilate mask by a lot
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Masked Image", masked_image)
    cv2.imshow("Original Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
