# rl_controller.py
# RL DQN agent that controls SUMO traffic lights via TraCI.
# Save as project-folder/sumo/rl_controller.py
import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci

# ----------------------------
# CONFIG
# ----------------------------
SUMO_CMD = ["sumo-gui", "-c", "4way.sumocfg"]   # or use "sumo" for headless
TLS_ID = "C"
LANES_N = ["NtoC_0", "NtoC_1"]
LANES_E = ["EtoC_0", "EtoC_1"]
LANES_S = ["StoC_0", "StoC_1"]
LANES_W = ["WtoC_0", "WtoC_1"]
STEP_SLEEP = 0.0     # small pause per sim step to allow GUI to update (0.0 fast, 0.01 smooth)
DT = 1.0             # agent decision timestep (seconds)

# Action space: (direction, green seconds)
DIRECTIONS = ["N", "E", "S", "W"]
GREEN_CHOICES = [6, 10, 15, 20]   # discrete green durations
ACTIONS = [(d, g) for d in DIRECTIONS for g in GREEN_CHOICES]
N_ACTIONS = len(ACTIONS)

# RL hyperparams
STATE_SIZE = 4
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 20000
MIN_REPLAY_SIZE = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000  # steps
TARGET_UPDATE = 1000  # steps
MAX_TRAIN_STEPS = 80000

MODEL_SAVE_PATH = "dqn_sumo_model.pth"

# ----------------------------
# NETWORK / DQN
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# Helpers: SUMO ↔ State/Action/Reward
# ----------------------------
def read_lane_counts():
    """Return dict of queues for N,E,S,W (integers)."""
    return {
        "N": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_N),
        "E": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_E),
        "S": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_S),
        "W": sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES_W),
    }

def state_from_counts(counts):
    """Return numpy array [N, E, S, W] normalized (optional)."""
    arr = np.array([counts[d] for d in ["N","E","S","W"]], dtype=np.float32)
    # simple normalization to avoid large values — divide by (1 + max observed)
    return arr  # you can divide by a constant if desired

def apply_action(action_idx):
    """Set SUMO trafficlight phase for chosen direction and return green duration."""
    direction, green = ACTIONS[action_idx]
    # map to phase index in tl.add.xml: 0=NS green, 2=EW green
    if direction in ("N","S"):
        phase_index = 0
    else:
        phase_index = 2
    traci.trafficlight.setPhase(TLS_ID, phase_index)
    traci.trafficlight.setPhaseDuration(TLS_ID, green)
    return direction, green

def accumulate_reward_during_green(green_seconds):
    """
    Step SUMO for green_seconds (in simulation seconds) and compute reward.
    Reward = negative total queue length integrated over the green period.
    Also returns served vehicles count during the period (for diagnostics).
    """
    total_reward = 0.0
    served = 0
    steps = max(1, int(round(green_seconds / DT)))  # number of agent steps (here DT=1s)
    for _ in range(steps):
        traci.simulationStep()
        if STEP_SLEEP:
            time.sleep(STEP_SLEEP)
        counts = read_lane_counts()
        total_reward += -sum(counts.values())  # negative total queue
    return total_reward

# ----------------------------
# DQN TRAIN / EVAL
# ----------------------------
def select_action(state, policy_net, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * max(0.0, (1 - steps_done / EPS_DECAY))
    if random.random() < eps_threshold:
        return random.randrange(N_ACTIONS)
    else:
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            qvals = policy_net(state_v).cpu().numpy()[0]
            return int(np.argmax(qvals))

def optimize_model(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return
    trans = replay_buffer.sample(BATCH_SIZE)
    states = torch.tensor(np.array(trans.state), dtype=torch.float32)
    actions = torch.tensor(trans.action, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(trans.reward, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(trans.next_state), dtype=torch.float32)
    dones = torch.tensor(trans.done, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * GAMMA * next_q

    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----------------------------
# MAIN: train or run
# ----------------------------
def train_or_run(train=True, n_steps=20000):
    device = torch.device("cpu")
    policy_net = QNetwork(STATE_SIZE, N_ACTIONS).to(device)
    target_net = QNetwork(STATE_SIZE, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(BUFFER_CAPACITY)

    # Start SUMO
    traci.start(SUMO_CMD)
    print("SUMO started for RL training.")

    # Populate initial replay buffer with random actions (to avoid zero-length)
    steps = 0
    # reset simulation time by letting SUMO run a bit
    for _ in range(5):
        traci.simulationStep()

    # initial state
    counts = read_lane_counts()
    state = state_from_counts(counts)

    # Pre-fill with random transitions
    print("Populating replay buffer with random interactions...")
    while len(replay) < MIN_REPLAY_SIZE:
        a = random.randrange(N_ACTIONS)
        _, green = ACTIONS[a]
        # apply action, accumulate reward over green
        direction, g = apply_action(a)
        reward = accumulate_reward_during_green(g)
        next_counts = read_lane_counts()
        next_state = state_from_counts(next_counts)
        done = False  # episodes are long; we will not set terminal
        replay.push(state, a, reward, next_state, done)
        state = next_state
        # occasionally step a bit to allow new traffic arrivals
        steps += 1
        if steps % 100 == 0:
            print(f"buffer size: {len(replay)}")

    print("Starting training loop...")
    steps_done = 0
    episode = 0
    total_rewards = []

    while steps_done < (n_steps if train else 10000):
        episode += 1
        # get current state (fresh)
        counts = read_lane_counts()
        state = state_from_counts(counts)

        # choose action
        action_idx = select_action(state, policy_net, steps_done)
        direction, green = apply_action(action_idx)

        # step through green and collect reward
        reward = accumulate_reward_during_green(green)
        next_counts = read_lane_counts()
        next_state = state_from_counts(next_counts)
        done = False

        # store transition
        replay.push(state, action_idx, reward, next_state, done)

        # optimize
        optimize_model(policy_net, target_net, optimizer, replay)

        # periodic target update
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

        if steps_done % 50 == 0:
            print(f"[step {steps_done}] reward={reward:.1f}, counts={next_counts}, eps={EPS_END + (EPS_START - EPS_END) * max(0.0, (1 - steps_done / EPS_DECAY)):.3f}")

        if steps_done % 500 == 0:
            torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
            print("Model checkpoint saved.")

        if not train:
            # if running in eval, limit cycles
            if steps_done > 5000:
                break

    # final save
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
    print("Training finished. Model saved to", MODEL_SAVE_PATH)
    traci.close()

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train DQN online against SUMO")
    parser.add_argument("--steps", type=int, default=20000, help="Number of agent interaction steps for training")
    args = parser.parse_args()

    if args.train:
        print("Starting training mode.")
        train_or_run(train=True, n_steps=args.steps)
    else:
        print("Running pretrained policy (if exists).")
        # load model and run inference loop
        if not os.path.exists(MODEL_SAVE_PATH):
            print("Model not found. Run with --train to train first.")
        else:
            # load and run policy in eval mode
            policy_net = QNetwork(STATE_SIZE, N_ACTIONS)
            policy_net.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
            policy_net.eval()
            # start SUMO
            traci.start(SUMO_CMD)
            print("SUMO started for evaluation.")
            # run loop
            for _ in range(2000):
                traci.simulationStep()
                counts = read_lane_counts()
                state = state_from_counts(counts)
                with torch.no_grad():
                    qvals = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]
                    action_idx = int(np.argmax(qvals))
                direction, green = apply_action(action_idx)
                print(f"Eval choose {direction} for {green}s | counts={counts}")
                # step green duration
                for _ in range(int(round(green / DT))):
                    traci.simulationStep()
                    time.sleep(STEP_SLEEP)
            traci.close()
