import csv
from src.Detector import *
from src.ImageSource import VideoImageSource
import time
import matplotlib.pyplot as plt
import pandas as pd

BENCHMARK_FILE = "detector_timings.csv"
VIDEO_FILE = "data/video/video1.avi"


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
    feed = VideoImageSource(VIDEO_FILE)

    # Write initial header to the benchmark file
    with open(BENCHMARK_FILE, "w") as f:
        writer = csv.writer(f, lineterminator="\n", delimiter=",")
        writer.writerow(["Detector", "Frame", "Time (ms)"])
        
    def benchmark(detector, name = None):
        detector_times = timingFunction(detector, feed)
        storeResults(detector_times, name if name != None else detector.getName())

    # benchmark(Detector())
    
    # benchmark(Aruco3Detector())
    
    # benchmark(CroppedDetector())
    
    benchmark(ROIDetector())
    
    benchmark(ROIDetector(resize=False), "ROI Detector No Resize")

    # benchmark(AprilTagDetector(), "April Tag 1x Decimation")
    
    # benchmark(AprilTagDetector(decimation=2), "April Tag 2x Decimation")
    
    # benchmark(AprilTagDetector(decimation=3), "April Tag 3x Decimation")
    
    # benchmark(ColorDetector())


def displayResults():
    df = pd.read_csv(BENCHMARK_FILE)
    df = df.groupby("Detector")

    plt.figure(figsize=(18, 9), dpi=100)

    for detector_name, group in df:
        # Smooth out the data
        group["Time (ms)"] = group["Time (ms)"].rolling(window=30).mean()
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
