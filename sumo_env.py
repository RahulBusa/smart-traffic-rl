import gymnasium as gym
import numpy as np
import traci


class TrafficEnv(gym.Env):
    def __init__(self, sumocfg_path):
        super().__init__()

        self.sumocfg_path = sumocfg_path

        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.tls_id = None
        self.in_lanes = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        traci.start([
            "sumo",
            "-c", self.sumocfg_path,
            "--start",
            "--quit-on-end"
        ])

        # Auto-detect traffic light
        self.tls_id = traci.trafficlight.getIDList()[0]

        # Auto-detect incoming lanes
        links = traci.trafficlight.getControlledLinks(self.tls_id)
        self.in_lanes = list({l[0][0] for l in links})

        return self._state(), {}

    def step(self, action):
        traci.trafficlight.setPhase(self.tls_id, int(action))

        for _ in range(10):
            traci.simulationStep()

        state = self._state()
        reward = self._reward()

        terminated = traci.simulation.getMinExpectedNumber() == 0
        truncated = False

        return state, reward, terminated, truncated, {}

    def _state(self):
        queues = []
        waits = []

        for lane in self.in_lanes[:4]:
            queues.append(traci.lane.getLastStepVehicleNumber(lane))
            waits.append(traci.lane.getWaitingTime(lane))

        while len(queues) < 4:
            queues.append(0)
            waits.append(0)

        return np.concatenate([
            np.array(queues) / 50.0,
            np.array(waits) / 100.0
        ]).astype(np.float32)

    def _reward(self):
        return -sum(traci.lane.getWaitingTime(l) for l in self.in_lanes)

    def close(self):
        traci.close()
