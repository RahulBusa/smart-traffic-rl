import cv2
import numpy as np
import time
from ultralytics import YOLO
from stable_baselines3 import PPO


# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "ppo_sumo_traffic"

VIDEOS = {
    "N": "north.mp4",
    "E": "east.mp4",
    "S": "south.mp4",
    "W": "west.mp4"
}

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


# -------------------------------
# Phase → Direction labels
# (visual-level, not physical change)
# -------------------------------
PHASE_TO_DIR = {
    0: ["NORTH-SOUTH", "SOUTH-NORTH"],
    1: ["EAST-WEST", "WEST-EAST"],
    2: ["NORTH ONLY"],
    3: ["SOUTH ONLY"]
}

PHASE_ORDER = [0, 1, 2, 3]
MAX_GREEN_TIME = 12  # seconds


# -------------------------------
# Vehicle counting
# -------------------------------
def count_vehicles(yolo, frame):
    results = yolo(frame, verbose=False)[0]
    if results.boxes is None:
        return 0
    return sum(
        1 for box in results.boxes
        if int(box.cls[0]) in VEHICLE_CLASSES
    )


# -------------------------------
# Draw traffic lights
# -------------------------------
def draw_traffic_lights(frame, phase):
    colors = {d: (0, 0, 255) for d in ["N", "E", "S", "W"]}

    if phase == 0:        # North-South
        colors["N"] = colors["S"] = (0, 255, 0)
    elif phase == 1:      # East-West
        colors["E"] = colors["W"] = (0, 255, 0)
    elif phase == 2:      # North only
        colors["N"] = (0, 255, 0)
    elif phase == 3:      # South only
        colors["S"] = (0, 255, 0)

    cv2.circle(frame, (60, 150), 18, colors["N"], -1)
    cv2.circle(frame, (120, 150), 18, colors["E"], -1)
    cv2.circle(frame, (180, 150), 18, colors["S"], -1)
    cv2.circle(frame, (240, 150), 18, colors["W"], -1)

    cv2.putText(
        frame, "N   E   S   W",
        (40, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Loading PPO RL model...")
    model = PPO.load(MODEL_PATH)

    print("Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")

    caps = {d: cv2.VideoCapture(p) for d, p in VIDEOS.items()}

    current_phase = 0
    last_switch_time = time.time()

    # direction toggle for display only
    dir_toggle = 0

    print("Running final traffic controller with directional flow display...")

    while True:
        queues = []
        frames = {}

        # Read frames and count vehicles
        for d, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                return
            frames[d] = frame
            queues.append(count_vehicles(yolo, frame))

        # RL state
        state = np.concatenate([
            np.array(queues) / 50.0,
            np.zeros(4)
        ]).astype(np.float32)

        # RL suggests NEXT phase
        rl_phase, _ = model.predict(state, deterministic=True)
        rl_phase = int(rl_phase)

        elapsed = time.time() - last_switch_time
        green_left = max(0, int(MAX_GREEN_TIME - elapsed))

        # -------- PHASE SWITCH ONLY WHEN TIMER ENDS --------
        if green_left == 0:
            if rl_phase == current_phase:
                idx = PHASE_ORDER.index(current_phase)
                next_phase = PHASE_ORDER[(idx + 1) % len(PHASE_ORDER)]
                status = "FORCED FAIRNESS SWITCH"
            else:
                next_phase = rl_phase
                status = "RL PHASE SWITCH"

            current_phase = next_phase
            last_switch_time = time.time()
            dir_toggle += 1   # change visual direction only
        else:
            status = "STABLE GREEN"

        # Select direction label
        labels = PHASE_TO_DIR[current_phase]
        if len(labels) > 1:
            active_dir = labels[dir_toggle % len(labels)]
        else:
            active_dir = labels[0]

        # Console output
        print(
            f"Queues: {queues} | "
            f"Active: {active_dir} | "
            f"Green Left: {green_left}s | "
            f"Status: {status}"
        )

        # Display
        for frame in frames.values():
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), (0, 0, 0), -1)

            cv2.putText(
                frame, f"GREEN: {active_dir}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 255, 255), 3
            )

            cv2.putText(
                frame, f"GREEN TIME LEFT: {green_left} sec",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"STATUS: {status}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (200, 200, 200), 2
            )

            draw_traffic_lights(frame, current_phase)

        for d, frame in frames.items():
            cv2.imshow(f"Traffic {d}", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
