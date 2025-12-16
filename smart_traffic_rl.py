# smart_traffic_rl.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort import Sort
import math
import requests

# -----------------------------
# VIDEO PATHS
# -----------------------------
VIDEO_PATHS = {
    "N": "north.mp4",
    "E": "east.mp4",
    "S": "south.mp4",
    "W": "west.mp4"
}

LANES = ["N", "E", "S", "W"]

# -----------------------------
# Initialize YOLOv8
# -----------------------------
yolo_model = YOLO("yolov8n.pt")

# -----------------------------
# Initialize SORT trackers
# -----------------------------
trackers = {lane: Sort() for lane in LANES}

# -----------------------------
# Traffic parameters
# -----------------------------
PARAMS = {
    "S": 1.8,
    "min_green": 8,
    "max_green": 45,
    "safety_buffer": 3,
    "cycle_target": 60,
    "max_wait_threshold": 60,
    "alpha": 1.0,
    "beta": 0.6
}

# -----------------------------
# Count vehicles (YOLO + SORT)
# -----------------------------
def count_vehicles(video_path, lane):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return 0

    results = yolo_model(frame, verbose=False)[0]
    detections = []

    for box, cls, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf
    ):
        if int(cls) in [2, 5, 7] and conf > 0.4:
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append([x1, y1, x2, y2, conf.cpu().numpy()])

    tracked = trackers[lane].update(np.array(detections))
    return len(tracked)

# -----------------------------
# Heuristic decision (baseline)
# -----------------------------
def choose_phase_and_green(queues, last_served):
    alpha = PARAMS["alpha"]
    beta = PARAMS["beta"]

    scores = {
        d: alpha * queues[d] + beta * last_served[d]
        for d in LANES
    }
    chosen = max(scores, key=scores.get)

    q = queues[chosen]
    clear_t = q / PARAMS["S"] if PARAMS["S"] > 0 else 0
    green = math.ceil(clear_t) + PARAMS["safety_buffer"]

    green = max(PARAMS["min_green"], min(PARAMS["max_green"], green))
    return chosen, green

# -----------------------------
# Send data to Flask website
# -----------------------------
def send_to_website(counts, phase, green):
    try:
        requests.post(
            "http://127.0.0.1:5000/update",
            json={
                "N": counts["N"],
                "E": counts["E"],
                "S": counts["S"],
                "W": counts["W"],
                "green_lane": phase,
                "green_time": green
            },
            timeout=1
        )
    except Exception as e:
        print("Website update failed:", e)

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    last_served = {lane: 0 for lane in LANES}

    while True:
        # 1. Count vehicles
        counts = {lane: count_vehicles(VIDEO_PATHS[lane], lane) for lane in LANES}

        # 2. Decide signal
        phase, green = choose_phase_and_green(counts, last_served)

        for lane in LANES:
            last_served[lane] += green
        last_served[phase] = 0

        # 3. Console output
        print(f"[DECISION] Green: {phase} for {green}s")
        print(f"Counts: {counts}")
        print("-" * 50)

        # 4. Send to website
        send_to_website(counts, phase, green)

        # 5. Wait for green duration
        time.sleep(green)
