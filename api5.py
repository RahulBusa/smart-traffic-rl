from flask import Flask, request, jsonify
import math

app = Flask(__name__)

latest_data = {"counts": {"N":0,"E":0,"S":0,"W":0}, "decision": {}}

# Existing /decide endpoint
@app.route("/decide", methods=["POST"])
def decide():
    global latest_data
    data = request.get_json()
    counts = data.get("counts", {})
    last_served = data.get("last_served", {})
    phase, green = choose_phase_and_green(counts, last_served, PARAMS)

    resp = {
        "phase": phase,
        "green": green,
        "amber": PARAMS["amber"],
        "all_red": PARAMS["all_red"]
    }

    latest_data = {"counts": counts, "decision": resp}
    return jsonify(resp)

# New endpoint for website
@app.route("/status", methods=["GET"])
def status():
    return jsonify(latest_data)
