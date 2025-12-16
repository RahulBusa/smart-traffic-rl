"""
controller_api.py

Flask API exposing /decide endpoint.
POST JSON: {"counts":{"N":12,"E":8,"S":20,"W":5}, "last_served":{"N":10,...}}
Response: {"phase":"S", "green":15, "amber":3, "all_red":1}
"""

from flask import Flask, request, jsonify
import math

app = Flask(__name__)

# -----------------------------
# Heuristic function
# -----------------------------
def choose_phase_and_green(queues, last_served, params):
    S = params.get("S", 1.8)
    min_green = params.get("min_green", 8)
    max_green = params.get("max_green", 45)
    safety_buffer = params.get("safety_buffer", 3)
    cycle_target = params.get("cycle_target", 60)
    max_wait_threshold = params.get("max_wait_threshold", 60)
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 0.6)

    # 1) Starvation check
    for d in ["N","E","S","W"]:
        if last_served.get(d,0) >= max_wait_threshold and queues.get(d,0) > 0:
            chosen = d
            break
    else:
        # 2) Score-based selection
        scores = {d: alpha*queues.get(d,0) + beta*last_served.get(d,0) for d in ["N","E","S","W"]}
        chosen = max(scores, key=scores.get)

    # 3) Clearance time
    q = queues.get(chosen,0)
    clear_t = q / S if S>0 else 0
    G = math.ceil(clear_t) + safety_buffer

    # 4) Proportional fallback
    total_q = sum(queues.values())
    if total_q > 0 and q < 2:
        prop = q / max(1, total_q)
        G = max(1, int(round(prop*cycle_target)))

    # Clamp to min/max
    G = max(min_green, min(max_green, G))
    return chosen, G

# -----------------------------
# Default parameters
# -----------------------------
PARAMS = {
    "S": 1.8,
    "min_green": 8,
    "max_green": 45,
    "safety_buffer": 3,
    "cycle_target": 60,
    "max_wait_threshold": 60,
    "alpha": 1.0,
    "beta": 0.6,
    "amber": 3,
    "all_red": 1
}

# -----------------------------
# Root route (for browser check)
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return "Controller API is running! Use POST /decide to get phase decisions."

# -----------------------------
# API endpoint
# -----------------------------
@app.route("/decide", methods=["POST"])
def decide():
    try:
        data = request.get_json(force=True)
        counts = data.get("counts", {"N":0,"E":0,"S":0,"W":0})
        last_served = data.get("last_served", {"N":0,"E":0,"S":0,"W":0})
        phase, green = choose_phase_and_green(counts, last_served, PARAMS)
        resp = {
            "phase": phase,
            "green": int(green),
            "amber": PARAMS["amber"],
            "all_red": PARAMS["all_red"]
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Run Flask server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
