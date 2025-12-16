# sumo_bridge.py
import traci
import time
import requests

# -------------------------
# SUMO command (use sumo-gui to see GUI)
# -------------------------
SUMO_CMD = ["sumo-gui", "-c", "4way.sumocfg"]
# SUMO_CMD = ["sumo", "-c", "4way.sumocfg"]  # headless if desired

# -------------------------
# lane ids (edge_laneIndex)
# adjust if your net has different names
# -------------------------
LANES_N = ["NtoC_0", "NtoC_1"]
LANES_E = ["EtoC_0", "EtoC_1"]
LANES_S = ["StoC_0", "StoC_1"]
LANES_W = ["WtoC_0", "WtoC_1"]

TLS_ID = "C"

# -------------------------
# Helpers
# -------------------------
def dir_to_phase(chosen):
    # tl.add.xml uses phase 0 = NS green, 1 = all-red, 2 = EW green, 3 = all-red
    if chosen in ("N", "S"):
        return 0
    if chosen in ("E", "W"):
        return 2
    return 0

def get_lane_counts():
    return {
        "N": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_N),
        "E": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_E),
        "S": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_S),
        "W": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_W),
    }

def request_decision(counts):
    url = "http://127.0.0.1:5000/decide"
    try:
        r = requests.post(url, json={"counts": counts}, timeout=1.0)
        if r.status_code == 200:
            return r.json()
        else:
            print("Controller API error status:", r.status_code)
    except Exception as e:
        print("Controller request exception:", e)
    # fallback: choose max queue
    maxd = max(counts, key=counts.get)
    return {"phase": maxd, "green": 10}

# -------------------------
# Main loop
# -------------------------
def main():
    traci.start(SUMO_CMD)
    print("SUMO started via TraCI")

    green_active = False
    green_end_time = -1
    all_red_duration = 1.0   # seconds gap between greens
    all_red_active = False
    all_red_end = -1
    current_phase_idx = None

    step = 0
    while step < 200000:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        counts = get_lane_counts()

        # Always print latest counts (like previous output)
        if step % 1 == 0:
            # print counts once per sim-second approx
            print(f"[t={sim_time:.1f}] counts: {counts}")

        # If in all-red gap, wait until it ends
        if all_red_active:
            if sim_time >= all_red_end:
                all_red_active = False
                print(f"[t={sim_time:.1f}] All-red ended; ready for next decision.")
            else:
                step += 1
                time.sleep(0.01)
                continue

        # If no green active or green expired -> get new decision
        if (not green_active) or (sim_time >= green_end_time):
            # If green just expired -> trigger all-red
            if green_active and sim_time >= green_end_time:
                print(f"[t={sim_time:.1f}] Green ended for phase {current_phase_idx}. Entering all-red for {all_red_duration}s.")
                try:
                    traci.trafficlight.setPhase(TLS_ID, 1)  # set to all-red phase index (1)
                    traci.trafficlight.setPhaseDuration(TLS_ID, all_red_duration)
                except Exception:
                    pass
                all_red_active = True
                all_red_end = sim_time + all_red_duration
                green_active = False
                step += 1
                time.sleep(0.01)
                continue

            # Ask controller for next decision
            decision = request_decision(counts)
            chosen = decision.get("phase")
            green = int(decision.get("green", 10))
            phase_idx = dir_to_phase(chosen)

            # Apply phase & duration
            traci.trafficlight.setPhase(TLS_ID, phase_idx)
            traci.trafficlight.setPhaseDuration(TLS_ID, green)

            green_active = True
            green_end_time = sim_time + green
            current_phase_idx = phase_idx

            # Print START log (with counts)
            print(f"[t={sim_time:.1f}] START: serve {chosen} for {green}s -> phase {phase_idx}. counts={counts}")

        else:
            # green active -> remaining countdown logging occasionally
            remaining = int(round(green_end_time - sim_time))
            # print each second
            if step % int(max(1, round(1.0 / max(traci.simulation.getDeltaT(), 0.1)))) == 0:
                print(f"[t={sim_time:.1f}] GREEN active (phase {current_phase_idx}), remaining ≈ {remaining}s | counts={counts}")

        step += 1
        time.sleep(0.01)

    traci.close()

if __name__ == "__main__":
    main()
