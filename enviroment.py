import numpy as np
import gymnasium as gym

class SpiderEnv(gym.Env):
    def __init__(self, robot_init_pos, target_init_pos=None, render_mode=None):
        high = np.array([8, 8, np.pi, 8, 8], dtype=np.float32)
        low  = np.array([-8, -8, -np.pi, -8, -8], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(12)
        self.robot_pos = robot_init_pos

        if target_init_pos is None:
            sigma = 3.0 
            self.target_pos = self.robot_pos + np.random.randn(2) * sigma
        else:
            self.target_pos = target_init_pos
        
        self.commands = {
            0: (227,[]), #Pivot Left
            1: 228, #Pivot Right
            2: 251, #FwdSteer Left
            3: (252,[0.0773,0.0,4.44]), #Fwd
            4: 253, #FwdSteer Right
            5: 256, #BwdSteer Left
            6: 258, #BwdSteer Right
            7: 261, #FastFwdSteer Left
            8: 262, #FastFwd
            9: 263, #FastFwdSteer Right
            10:(266,[-0.0588,0.0,0.0]), #Bwd
            11:267, #FastBwd
        }

    def reset(self, robot_init_pos=None, target_init_pos=None):
        if robot_init_pos is None:
            self.robot_pos = self.robot_pos
        else:
            self.robot_pos = robot_init_pos

        if target_init_pos is None:
            sigma = 3.0 
            self.target_pos = self.robot_pos + np.random.randn(2) * sigma
        else:
            self.target_pos = target_init_pos

    
    def get_obs(self):
        return np.array(self.robot_pos[0],self.robot_pos[1],self.robot_pos[2],
                        self.target_pos[0],self.target_pos[1], dtype=np.float32),{}

        

    def step(self, action):
        command = self.commands[action]
        
        

    # def render(self):
        

    def close(self):
        pass


