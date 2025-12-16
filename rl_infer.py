# rl_infer.py
# Lightweight RL inference wrapper. Loads the trained DQN and returns decisions.
import os
import numpy as np
import torch
import torch.nn as nn

# Action space must match training
DIRECTIONS = ["N","E","S","W"]
GREEN_CHOICES = [6, 10, 15, 20]
ACTIONS = [(d,g) for d in DIRECTIONS for g in GREEN_CHOICES]

MODEL_PATH = "dqn_sumo_model.pth"   # produced by rl training script

# -------------------------
# QNetwork architecture (must match training code)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size=4, n_actions=len(ACTIONS), hidden=128):
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

# -------------------------
# load policy
# -------------------------
_policy = None

def load_policy(model_path=MODEL_PATH):
    global _policy
    if _policy is not None:
        return _policy
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RL model not found at '{model_path}'. Train first or place model there.")
    net = QNetwork()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    _policy = net
    return _policy

# -------------------------
# public API
# -------------------------
def rl_decide(counts):
    """
    counts: dict {"N":int,"E":int,"S":int,"W":int}
    returns: {"phase": direction, "green": duration}
    """
    # ensure model loaded
    net = load_policy()
    state = np.array([counts.get("N",0), counts.get("E",0), counts.get("S",0), counts.get("W",0)], dtype=np.float32)
    with torch.no_grad():
        qvals = net(torch.tensor(state).unsqueeze(0).float()).numpy()[0]
    action_idx = int(np.argmax(qvals))
    direction, green = ACTIONS[action_idx]
    return {"phase": direction, "green": int(green)}
