# smart_traffic_rl.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort import Sort
import math

# -----------------------------
# VIDEO PATHS (replace with your videos)
# -----------------------------
VIDEO_PATHS = {
    "N": "north.mp4",
    "E": "east.mp4",
    "S": "south.mp4",
    "W": "west.mp4"
}

# -----------------------------
# Initialize YOLOv8 model
# -----------------------------
model = YOLO("yolov8n.pt")  # YOLOv8 nano (lightweight)

# -----------------------------
# Initialize SORT tracker for each lane
# -----------------------------
LANES = ["N", "E", "S", "W"]
trackers = {lane: Sort() for lane in LANES}

# -----------------------------
# Traffic parameters
# -----------------------------
PARAMS = {
    "S": 1.8,  # vehicles/sec clearance rate
    "min_green": 8,
    "max_green": 45,
    "safety_buffer": 3,
    "cycle_target": 60,
    "max_wait_threshold": 60,
    "alpha": 1.0,
    "beta": 0.6
}

# -----------------------------
# Count vehicles in a frame using YOLO + SORT
# -----------------------------
def count_vehicles(video_path, lane):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return 0
    # YOLO detection
    results = model(frame)[0]
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        # COCO classes: car=2, bus=5, truck=7
        if int(cls) in [2, 5, 7]:
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append([x1, y1, x2, y2, conf.cpu().numpy()])
    tracked = trackers[lane].update(np.array(detections))
    cap.release()
    return len(tracked)

# -----------------------------
# Heuristic function to choose next green phase
# -----------------------------
def choose_phase_and_green(queues, last_served, params):
    S = params["S"]
    min_green = params["min_green"]
    max_green = params["max_green"]
    safety_buffer = params["safety_buffer"]
    cycle_target = params["cycle_target"]
    max_wait_threshold = params["max_wait_threshold"]
    alpha = params["alpha"]
    beta = params["beta"]

    # Starvation check
    for d in LANES:
        if last_served.get(d,0) >= max_wait_threshold and queues.get(d,0) > 0:
            chosen = d
            break
    else:
        # Score-based selection
        scores = {d: alpha*queues.get(d,0) + beta*last_served.get(d,0) for d in LANES}
        chosen = max(scores, key=scores.get)

    q = queues.get(chosen,0)
    clear_t = q / S if S>0 else 0
    G = math.ceil(clear_t) + safety_buffer

    # Proportional fallback
    total_q = sum(queues.values())
    if total_q > 0 and q < 2:
        prop = q / max(1, total_q)
        G = max(1, int(round(prop*cycle_target)))

    G = max(min_green, min(max_green, G))
    return chosen, G

# -----------------------------
# Main loop: count vehicles and decide phase
# -----------------------------
if __name__ == "__main__":
    last_served = {lane: 0 for lane in LANES}

    while True:
        counts = {lane: count_vehicles(VIDEO_PATHS[lane], lane) for lane in LANES}
        total = sum(counts.values())
        density = {lane: counts[lane]/max(1,total) for lane in LANES}
        clearing = {lane: counts[lane]/PARAMS["S"] for lane in LANES}

        # Choose next phase
        phase, green = choose_phase_and_green(counts, last_served, PARAMS)
        last_served[phase] = 0
        for lane in LANES:
            if lane != phase:
                last_served[lane] += green

        # Print
        print(f"Next Phase: {phase}, Green: {green}s")
        print(f"Counts: {counts}, Density: {density}, Clearing: {clearing}")
        print("-"*50)

        time.sleep(10)  # update every 10 seconds
