import numpy as np
import gymnasium as gym

class SpiderEnv(gym.Env):
    def __init__(self, target_init_pos=None, render_mode=None):
        self.observation_space = gym.spaces.Box(low=0, high=1,shape=(4,4), dtype=np.float32)
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
        self.action_space = gym.spaces.Discrete(len(self.commands))
        self.robot_pos = [0,0,0]

        if target_init_pos is None:
            sigma = 3.0 
            self.target_pos = self.robot_pos + np.random.randn(2) * sigma
        else:
            self.target_pos = target_init_pos
        

    def reset(self, options: dict):
        if options['robot_init_pos'] is None:
            self.robot_pos = self.robot_pos
        else:
            self.robot_pos = [0,0,0]

        if options['target_init_pos'] is None:
            sigma = 3.0 
            self.target_pos = self.robot_pos + np.random.randn(2) * sigma
        else:
            self.target_pos = options['target_init_pos']

    
    def get_obs(self):
        return np.array(self.robot_pos[0],self.robot_pos[1],self.robot_pos[2],
                        self.target_pos[0],self.target_pos[1], dtype=np.float32),{}

        

    def step(self, action):
        command,movement = self.commands[action]
        dx = self.robot_pos[0] + movement[0]
        dy = self.robot_pos[1] + movement[1]
        dtheta = self.robot_pos[2] + movement[2]
        self.robot_pos = [dx,dy,dtheta]


        # tx = [dx-self]
        # ty = 
        # self.robot_pos = [dx,dy,dtheta]
        # self.observation_space = 
    
    # def get_distance_to_target(self):

        

    # def render(self):
        

    # def close(self):
    #     pass


