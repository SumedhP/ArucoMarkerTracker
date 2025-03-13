import cv2

# Define a class that takes in a video input and gets the frames from it. Have a method to get the next frame

class Videofeed:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.frames = []
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            self.frames.append(frame)
        self.frame_number = 0
    
    def getFrame(self):
        if(self.frame_number >= len(self.frames)):
            return None
        frame = self.frames[self.frame_number]
        self.frame_number += 1
        return frame

