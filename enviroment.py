# import numpy as np
# import gymnasium as gym

# class SpiderEnv(gym.Env):
#     def __init__(self, target_init_pos=None, map_shape=4, render_mode=None):
#         self.observation_space = gym.spaces.Box(low=0, high=1,shape=(map_shape,map_shape), dtype=np.float32)
#         self.commands = {
#             0: (227,[]), #Pivot Left
#             1: 228, #Pivot Right
#             2: 251, #FwdSteer Left
#             3: (252,[0.0773,0.0,4.44]), #Fwd
#             4: 253, #FwdSteer Right
#             5: 256, #BwdSteer Left
#             6: 258, #BwdSteer Right
#             7: 261, #FastFwdSteer Left
#             8: 262, #FastFwd
#             9: 263, #FastFwdSteer Right
#             10:(266,[-0.0588,0.0,0.0]), #Bwd
#             11:267, #FastBwd
#         }
#         self.action_space = gym.spaces.Discrete(len(self.commands))

#         if target_init_pos is None:
#             sigma = 3.0 
#             self.target_pos = self.robot_pos + np.random.randn(2) * sigma
#         else:
#             self.target_pos = target_init_pos

#         self.state = np.zeros((map_shape, map_shape), dtype=np.float32)

#     def reset(self, options: dict):
#         if options['target_init_pos'] is None:
#             sigma = 3.0 
#             self.target_pos = self.robot_pos + np.random.randn(2) * sigma
#         else:
#             self.target_pos = options['target_init_pos']

#         self.state = np.zeros((options['map_shape'], options['map_shape']), dtype=np.float32)
#         obs = self.state
#         info = {}
#         return obs,info

    
#     def get_obs(self):
#         return np.array(self.robot_pos[0],self.robot_pos[1],self.robot_pos[2],
#                         self.target_pos[0],self.target_pos[1], dtype=np.float32),{}

#     def rotar_obs_calc_new_target(self, theta,dx,dy):
#         R = np.array([
#             [np.cos(theta), -np.sin(theta)],
#             [np.sin(theta),  np.cos(theta)]
#         ])
#         obs_rot = self.observation_space @ R.T

#         t = np.array(self.target_pos)
#         t_new = R @ (t - np.array([dx, dy]))
#         return obs_rot, t_new


#     def step(self, action):
#         command,movement = self.commands[action]
#         dx = movement[0]
#         dy = movement[1]
#         dtheta = movement[2]
#         rot_space,t_new = self.rotar_obs_calc_new_target(dtheta,dx,dy)        
        
#         self.state = rot_space
#         self.target_pos = t_new

        
#         # tx = [dx-self]
#         # ty = 
#         # self.robot_pos = [dx,dy,dtheta]
#         # self.observation_space = 
    
#     # def get_distance_to_target(self):

        

#     # def render(self):
        

#     # def close(self):
#     #     pass


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
            [np.cos(-self.theta), -np.sin(-self.theta)],
            [np.sin(-self.theta),  np.cos(-self.theta)]
        ])
        target_rot = R @ self.target_pos

        # convertir a índices de grilla
        tx = int(cx + target_rot[0] * (self.map_shape // 2))
        ty = int(cy + target_rot[1] * (self.map_shape // 2))

        if 0 <= tx < self.map_shape and 0 <= ty < self.map_shape:
            grid[ty, tx] = 1.0

        self.state = rotate(grid, 0, reshape=False, order=1, mode='nearest')  # opcional, ya está en theta relativo

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
