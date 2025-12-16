# controller_api.py
from flask import Flask, request, jsonify
import math, threading, time, logging

app = Flask(__name__)

# -------------------------
# Parameters (tune as needed)
# -------------------------
PARAMS = {
    "S": 1.8,
    "min_green": 6,
    "max_green": 45,
    "safety_buffer": 2,
    "cycle_target": 60,
    "max_wait_threshold": 120,
    "alpha": 1.0,
    "beta": 0.6
}

# -------------------------
# Internal controller state
# -------------------------
last_served = {"N": 0, "E": 0, "S": 0, "W": 0}   # seconds since last serve (controller-managed)
lock = threading.Lock()
last_chosen = None
repeat_count = 0
MAX_REPEAT = 2   # when same chosen > MAX_REPEAT times, force rotation

# -------------------------
# Helper utilities
# -------------------------
def _age_last_served(seconds):
    """Increase last_served timers by seconds (called when we serve for 'green' seconds)."""
    with lock:
        for d in last_served:
            last_served[d] += seconds

def local_heuristic(queues):
    """
    Compute a score for each direction and choose the best.
    Returns: chosen_dir, green_seconds, scores_dict
    """
    # priority scores = alpha * queue + beta * waiting_time
    scores = {d: PARAMS["alpha"] * queues.get(d,0) + PARAMS["beta"] * last_served.get(d,0) for d in ["N","E","S","W"]}
    # choose best
    chosen = max(scores, key=scores.get)
    q = queues.get(chosen, 0)
    clear_t = q / PARAMS["S"] if PARAMS["S"] > 0 else 0
    G = math.ceil(clear_t) + PARAMS["safety_buffer"]

    total_q = sum(queues.values())
    if total_q > 0 and q < 2:
        # proportional fallback
        prop = q / max(1, total_q)
        G = max(1, int(round(prop * PARAMS["cycle_target"])))
    G = max(PARAMS["min_green"], min(PARAMS["max_green"], G))
    return chosen, int(G), scores

# -------------------------
# /decide endpoint
# -------------------------
@app.route("/decide", methods=["POST"])
def decide():
    global last_chosen, repeat_count
    data = request.get_json(force=True)
    queues = data.get("counts", {"N":0,"E":0,"S":0,"W":0})

    # compute heuristic decision (or RL if you plug later)
    chosen, green, scores = local_heuristic(queues)

    # rotation guard to avoid repeated serving
    if last_chosen == chosen:
        repeat_count += 1
    else:
        repeat_count = 0

    if repeat_count > MAX_REPEAT:
        # force choose next-best direction (that is not last_chosen)
        sorted_dirs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for d, sc in sorted_dirs:
            if d != last_chosen:
                chosen = d
                green = max(PARAMS["min_green"], int(round(green // 2)))
                app.logger.info(f"Rotation guard triggered: forcing {chosen} for {green}s")
                break
        repeat_count = 0

    # update last_served by green (we will serve chosen for 'green' seconds)
    _age_last_served(green)
    with lock:
        last_served[chosen] = 0

    last_chosen = chosen

    # Log decision concisely (matches format you requested)
    app.logger.info(f"Decision made: {chosen} for {green}s | queues={queues} | scores={scores}")

    return jsonify({"phase": chosen, "green": int(green)})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000)
