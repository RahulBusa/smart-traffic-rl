import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.queues = None
        self.waits = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.queues = np.random.randint(0, 30, size=4)
        self.waits = np.zeros(4)

        return self._state(), {}

    def step(self, action):
        cleared = min(self.queues[action], 5)
        self.queues[action] -= cleared

        self.waits += 1
        self.waits[action] = 0

        reward = -(self.queues.sum() + 0.5 * self.queues.max())

        terminated = False     # no terminal state yet
        truncated = False      # no time limit yet

        return self._state(), reward, terminated, truncated, {}

    def _state(self):
        return np.concatenate([
            self.queues / 50.0,
            self.waits / 50.0
        ]).astype(np.float32)


# ---- TRAIN PPO ----
env = TrafficEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_traffic")
