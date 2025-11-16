import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
from gymnasium.envs.registration import register

class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(self, target_init_pos=None, map_shape=4, success_radius=0.2, max_steps=200,render_mode=None):
        self.world_size = map_shape
        half = map_shape / 2.0
        self.observation_space = gym.spaces.Box(
            low=np.array([-half, -half], dtype=np.float32),
            high=np.array([ half,  half], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        self.commands = {
            0: (227,[0.0667,-0.0052,np.deg2rad(32.72)]), #Pivot Left
            1: (228,[0.0667,0.0052,np.deg2rad(32.72)]), #Pivot Right
            2: (251,[0.0776,-0.0074,np.deg2rad(10.94)]), #FwdSteer Left
            3: (252,[0.0773,0.0,np.deg2rad(4.44)]), #Fwd
            4: (253,[0.0748,0.0064,np.deg2rad(9.79)]), #FwdSteer Right
            5: (256,[-0.0603,0.0025,np.deg2rad(5.81)]), #BwdSteer Left
            6: (258,[-0.0641,0.002,0.03175]), #BwdSteer Right
            # 7: (261,[]), #FastFwdSteer Left
            7: (262,[0.0893,0.0,0.0]), #FastFwd
            # 9: 263, #FastFwdSteer Right
            8:(266,[-0.0588,0.0,0.0]), #Bwd
            9:(267,[-0.715, 0.0, 0.0]), #FastBwd
        }
        self.action_space = gym.spaces.Discrete(len(self.commands))

        if target_init_pos is None:
            self.target_pos = np.random.uniform(-1, 1, size=2)
        else:
            self.target_pos = target_init_pos
        self.step_count = 0
        self.success_radius = float(success_radius)
        self.max_steps = int(max_steps)
        
        self.render_mode = render_mode
        self.window_size = 600 
        self.screen = None
        self.clock = None

        self.scale = (self.window_size * 0.4) / (self.world_size / 2.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0

        half = 4.0 / 2.0

        if options is None or options.get('target_init_pos') is None:
            self.target_pos = np.random.uniform(-half, half, size=2)
        else:
            self.target_pos = np.array(options['target_init_pos'], dtype=np.float64)

        obs = self.target_pos.astype(np.float32)
        info = {}
        return obs, info

    def get_obs(self):
        return self.target_pos.astype(np.float32)

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
        cosine_dist = float(self.target_pos[0] / dist) 
        reward = -dist + 0.5 * cosine_dist

        obs = self.target_pos.astype(np.float32)
        info = {"comando": command, "target": self.target_pos}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def _world_to_screen(self, x, y):
        cx = self.window_size // 2
        cy = self.window_size // 2
        sx = int(cx + x * self.scale)
        sy = int(cy - y * self.scale) 
        return sx, sy

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("SpiderEnv - Pygame render")
            self.clock = pygame.time.Clock()

        # manejar eventos de cierre de ventana
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # fondo
        self.screen.fill((30, 30, 30))

        # dibujar cuadrado del mundo (4m x 4m)
        half = self.world_size / 2.0
        top_left = self._world_to_screen(-half, half)
        bottom_right = self._world_to_screen(half, -half)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        pygame.draw.rect(self.screen, (80, 80, 80), (top_left[0], top_left[1], width, height), 2)

        # robot en el centro
        robot_x, robot_y = self._world_to_screen(0.0, 0.0)
        pygame.draw.circle(self.screen, (0, 0, 255), (robot_x, robot_y), 8)

        # círculo de radio de éxito
        # lo aproximo como círculo en pixeles
        radius_px = int(self.success_radius * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (robot_x, robot_y), radius_px, 1)

        # target
        tx, ty = self.target_pos
        target_px = self._world_to_screen(tx, ty)
        pygame.draw.circle(self.screen, (255, 0, 0), target_px, 6)

        # línea robot -> target
        pygame.draw.line(self.screen, (200, 200, 0), (robot_x, robot_y), target_px, 1)

        # texto con info
        font = pygame.font.SysFont(None, 20)
        dist = float(np.linalg.norm(self.target_pos))
        text_surface = font.render(
            f"Step: {self.step_count}  Dist: {dist:.2f} m",
            True,
            (255, 255, 255)
        )
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None