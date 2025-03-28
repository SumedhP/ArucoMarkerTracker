import argparse
from src.ImageSource import VideoImageSource
from src.Detector import list_detectors, get_detector
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Visualize marker detection with a specified detector."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/video/video1.avi",
        help="Path to the video file.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="BaselineDetector",
        help=f"Choose the detector to use. Options are: {list_detectors()}",
    )
    args = parser.parse_args()

    detector = get_detector(args.detector)
    print(f"Starting detection using {detector.getName()}...")

    feed = VideoImageSource(args.video)

    while True:
        frame = feed.get_image()
        if frame is None:
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        frame = detector.getAnnotatedFrame(frame, corners, ids, rejected)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
