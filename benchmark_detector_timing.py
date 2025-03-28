import csv
from src.Detector import *
from src.ImageSource import VideoImageSource
import time
import matplotlib.pyplot as plt
import pandas as pd

BENCHMARK_FILE = "detector_timings.csv"


def timingFunction(detector: Detector, feed: VideoImageSource):
    times_ms = []
    feed.reset()
    print(f"Benchmarking detection using {detector.getName()}")
    while True:
        frame = feed.get_image()
        if frame is None:
            break

        start_time = time.time_ns()
        detector.detectMarkers(frame)
        times_ms.append((time.time_ns() - start_time) / 1e6)

    return times_ms


def storeResults(times_ms, detector_name):
    # Append the results to the benchmark file with frame numbers
    with open(BENCHMARK_FILE, "a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n", delimiter=",")
        for frame_number, time_ms in enumerate(times_ms):
            writer.writerow([detector_name, frame_number, time_ms])


def takeBenchmarks():
    detector = Detector()
    feed = VideoImageSource("data/video/video1.avi")

    # Write initial header to the benchmark file
    with open(BENCHMARK_FILE, "w") as f:
        writer = csv.writer(f, lineterminator="\n", delimiter=",")
        writer.writerow(["Detector", "Frame", "Time (ms)"])

    base_detector_times = timingFunction(detector, feed)
    storeResults(base_detector_times, detector.getName())

    # Aruco3
    detector = Aruco3Detector()
    aruco_times = timingFunction(detector, feed)
    storeResults(aruco_times, detector.getName())

    # Cropped
    detector = CroppedDetector()
    cropped_times = timingFunction(detector, feed)
    storeResults(cropped_times, detector.getName())

    # ROI
    detector = ROIDetector()
    roi_times = timingFunction(detector, feed)
    storeResults(roi_times, detector.getName())

    # ROI no resize
    # detector = ROIDetector(resize=False)
    # roi_no_resize_times = timingFunction(detector, feed)
    # storeResults(roi_no_resize_times, detector.getName() + " No Resize")

    # AprilTag with 2x decimation
    # detector = AprilTagDetector()
    # detector.setDecimation(2)
    # april_tag_decimation_times = timingFunction(detector, feed)
    # storeResults(april_tag_decimation_times, "April tag 2x Decimation")
    
    # Color Detector
    detector = ColorDetector()
    color_times = timingFunction(detector, feed)
    storeResults(color_times, detector.getName())


def displayResults():
    df = pd.read_csv(BENCHMARK_FILE)
    df = df.groupby("Detector")

    plt.figure(figsize=(18, 9), dpi=100)

    for detector_name, group in df:
        # Smooth out the data
        group["Time (ms)"] = group["Time (ms)"].rolling(window=20).mean()
        plt.plot(group["Frame"], group["Time (ms)"], label=detector_name)

    plt.xlabel("Frame")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.savefig("detector_timings.png")
    plt.show()


def main():
    takeBenchmarks()

    displayResults()


if __name__ == "__main__":
    main()
