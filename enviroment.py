import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class SpiderEnv(gym.Env):
    def __init__(self, target_init_pos=None, map_shape=4, success_radius=0.08, max_steps=200,render_mode=None):
        self.observation_space = gym.spaces.Box(low=0, high=1,shape=(map_shape,map_shape), dtype=np.float32)
        self.commands = {
            0: (227,[]), #Pivot Left
            1: 228, #Pivot Right
            2: 251, #FwdSteer Left
            3: (252,[0.0773,0.0,np.deg2rad(4.44)]), #Fwd
            4: 253, #FwdSteer Right
            5: 256, #BwdSteer Left
            6: 258, #BwdSteer Right
            7: 261, #FastFwdSteer Left
            8: 262, #FastFwd
            9: 263, #FastFwdSteer Right
            10:(266,[-0.0588,0.0,0.0]), #Bwd
            11:267, #FastBwd
        }
        self.action_space = gym.spaces.Discrete(len(self.commands))

        if target_init_pos is None:
            self.target_pos = np.random.uniform(-1, 1, size=2)
        else:
            self.target_pos = target_init_pos
        self.step_count = 0
        self.success_radius = float(success_radius)
        self.max_steps = int(max_steps)
        self.fig, self.ax = None, None

    def reset(self, options: dict = None):
        self.step_count = 0
        if options is None or options.get('target_init_pos') is None:
            self.target_pos = np.random.uniform(-1, 1, size=2)
        else:
            self.target_pos = np.array(options['target_init_pos'], dtype=np.float64)

        obs = self.target_pos.astype(np.float32)
        info = {}
        return obs, info

    def get_obs(self):
        pass

    def calc_new_target(self, theta, dx, dy):
        R = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])
        t = np.array(self.target_pos)
        t_new = R @ (t - np.array([dx, dy]))
        return t_new

    def step(self, action):
        command, movement = self.commands[action]
        dx, dy, dtheta = movement
        self.target_pos = self.calc_new_target(dtheta, dx, dy)
        self.step_count += 1

        dist = float(np.linalg.norm(self.target_pos))
        terminated = dist <= self.success_radius
        truncated = self.step_count >= self.max_steps
        reward = -dist

        obs = self.target_pos.astype(np.float32)
        info = {"comando": command, "target": self.target_pos}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_title("SpiderEnv - Target movement")
            self.robot_dot, = self.ax.plot(0, 0, 'bo', label='Robot (0,0)')
            self.target_dot, = self.ax.plot([], [], 'ro', label='Target')
            self.range_circle = plt.Circle((0, 0), self.success_radius, color='g', fill=False, linestyle='--')
            self.ax.add_artist(self.range_circle)
            self.ax.legend()
            plt.ion()
            plt.show()

        # Actualizar posición del target
        self.target_dot.set_data(self.target_pos[0], self.target_pos[1])
        self.ax.set_title(f"Step {self.step_count} | Target=({self.target_pos[0]:.2f},{self.target_pos[1]:.2f})")
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

env = SpiderEnv(render_mode="human")
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(3)
    print(info)
