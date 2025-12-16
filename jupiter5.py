import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
import time

# ------------------------
# YOLOv8 Setup
# ------------------------
yolo_model = YOLO("yolov8n.pt")

# ------------------------
# DeepSORT Setup
# ------------------------
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# ------------------------
# Global Vehicle Registry
# ------------------------
global_vehicle_registry = {}  # {global_id: {"embedding": vec, "hist": hist, "last_seen": timestamp}}
global_id_counter = 1
TIME_GAP = 10  # seconds max time gap for same vehicle across cameras

# ------------------------
# Vehicle Counters
# ------------------------
last_print_time = time.time()
interval = 10  # seconds
camera_vehicle_counts = {1: set(), 2: set(), 3: set()}
global_vehicle_set = set()

# ------------------------
# Helper Functions
# ------------------------
def compute_color_histogram(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((512,))
    hist = cv2.calcHist([crop], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def assign_global_id(new_embedding, new_hist):
    global global_id_counter, global_vehicle_registry
    now = time.time()
    
    for gid, data in global_vehicle_registry.items():
        time_diff = now - data["last_seen"]
        if time_diff > TIME_GAP:
            continue  # ignore old sightings
        
        # Similarity by embedding + color histogram
        sim_embed = cosine_similarity([new_embedding], [data["embedding"]])[0][0]
        sim_hist = cv2.compareHist(new_hist.astype('float32'), data["hist"].astype('float32'), cv2.HISTCMP_CORREL)
        if sim_embed > 0.75 and sim_hist > 0.6:  # threshold can be tuned
            global_vehicle_registry[gid]["last_seen"] = now
            return gid

    # New vehicle
    global_vehicle_registry[global_id_counter] = {"embedding": new_embedding, "hist": new_hist, "last_seen": now}
    global_id_counter += 1
    return global_id_counter - 1

# ------------------------
# Process Frame
# ------------------------
def process_frame(frame, cam_id):
    results = yolo_model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if cls in [2,3,5,7] and conf > 0.4:
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        local_id = t.track_id
        ltrb = t.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Embedding
        embedding = t.features[-1] if len(t.features) > 0 else np.random.rand(128)
        # Color histogram
        hist = compute_color_histogram(frame, (x1,y1,x2,y2))

        # Assign global ID
        global_id = assign_global_id(embedding, hist)

        # Update counts
        camera_vehicle_counts[cam_id].add(global_id)
        global_vehicle_set.add(global_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"Cam{cam_id} → GID {global_id}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255),2)
    return frame

# ------------------------
# Multi-Camera Loop
# ------------------------
video_paths = {1:"tiny.mp4", 2:"tiny.mp4", 3:"tiny.mp4"}
caps = {cid: cv2.VideoCapture(path) for cid, path in video_paths.items()}

while True:
    frames = {}
    for cid, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = process_frame(frame, cid)
        frames[cid] = frame

    if frames:
        combined = cv2.hconcat([f for f in frames.values()])
        cv2.imshow("Multi-Camera Traffic Monitoring", combined)

    # Print counts every interval
    if time.time() - last_print_time >= interval:
        print("===== Vehicle Count (last 10s) =====")
        for cid, vehicles in camera_vehicle_counts.items():
            print(f"Camera {cid}: {len(vehicles)} vehicles")
        print(f"Global Unique Vehicles: {len(global_vehicle_set)}\n")
        camera_vehicle_counts = {1:set(),2:set(),3:set()}
        last_print_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
