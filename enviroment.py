import numpy as np
import gymnasium as gym
from scipy.ndimage import rotate

class SpiderEnv(gym.Env):
    def __init__(self, target_init_pos=None, map_shape=8):
        super().__init__()
        self.map_shape = map_shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(map_shape, map_shape), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(12)

        # comandos: (dx, dy, dtheta)
        self.commands = {
            0: (0.0, 0.0, np.deg2rad(10)),
            1: (0.0, 0.0, -np.deg2rad(10)),
            2: (0.1, 0.0, 0.0),
            3: (0.05, 0.0, 0.0),
            4: (0.1, 0.0, 0.0),
            5: (-0.05, 0.0, 0.0),
            6: (-0.1, 0.0, 0.0),
            7: (0.15, 0.0, 0.0),
            8: (0.2, 0.0, 0.0),
            9: (0.15, 0.0, 0.0),
            10:(-0.1, 0.0, 0.0),
            11:(-0.2, 0.0, 0.0)
        }

        if target_init_pos is None:
            self.target_pos = np.random.uniform(-1, 1, size=2)
        else:
            self.target_pos = np.array(target_init_pos, dtype=np.float32)

        self.theta = 0.0
        self.state = np.zeros((map_shape, map_shape), dtype=np.float32)

    def reset(self, *, options=None):
        self.theta = 0.0
        if options and "target_init_pos" in options:
            self.target_pos = np.array(options["target_init_pos"], dtype=np.float32)
        else:
            self.target_pos = np.random.uniform(-1, 1, size=2)
        self._update_grid()
        return self.state, {}

    def _update_grid(self):
        """
        Grilla centrada en el robot (0,0) con target rotado según theta
        """
        grid = np.zeros((self.map_shape, self.map_shape), dtype=np.float32)
        cx, cy = self.map_shape // 2, self.map_shape // 2

        # rotar target según theta (robot siempre en 0)
        R = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])
        t = np.array(self.target_pos)
        t_new = R @ (t - np.array([dx, dy]))
        return t_new

    def step(self, action):
        dx, dy, dtheta = self.commands[action]

        # mover target relativo al robot
        R = np.array([
            [np.cos(-dtheta), -np.sin(-dtheta)],
            [np.sin(-dtheta),  np.cos(-dtheta)]
        ])
        self.target_pos = R @ (self.target_pos - np.array([dx, dy]))

        # actualizar theta y grilla
        self.theta += dtheta
        self._update_grid()

        # distancia al target
        distance = np.linalg.norm(self.target_pos)
        done = distance < 0.05
        reward = -distance

        obs = self.state
        info = {"distance": distance, "theta": self.theta}
        return obs, reward, done, False, info

    def render(self):
        print("Grilla:")
        print(self.state)
