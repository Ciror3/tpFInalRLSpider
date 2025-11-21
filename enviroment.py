# import numpy as np
# import gymnasium as gym
# import matplotlib.pyplot as plt
# import pygame
# from gymnasium.envs.registration import register

# class SpiderEnv(gym.Env):
#     metadata = {"render_modes": ["human", None], "render_fps": 30}

#     def __init__(self, target_init_pos=None, map_shape=4, success_radius=0.2, max_steps=200, render_mode=None):
#         self.world_size = map_shape
#         half = map_shape / 2.0
        
#         # ### NUEVO: Memoria ###
#         # Ahora la observacion es de tamaño 3: [target_x, target_y, last_action_norm]
#         # last_action_norm será un valor entre 0 y 1.
#         self.observation_space = gym.spaces.Box(
#             low=np.array([-half, -half, 0.0], dtype=np.float32),
#             high=np.array([ half,  half, 1.0], dtype=np.float32),
#             shape=(3,), 
#             dtype=np.float32
#         )

#         self.commands = {
#                     0: (227,[0.0667, -0.0052, np.deg2rad(32.72)]),  # Pivot Left (OK)
#                     1: (228,[0.0667,  0.0052, -np.deg2rad(32.72)]), # Pivot Right (CORREGIDO: Signo menos)
                    
#                     2: (251,[0.0776, -0.0074, np.deg2rad(10.94)]),  # FwdSteer Left (OK)
#                     # Nota: Tu "Fwd" (3) tiene giro positivo, o sea, el robot "tira" a la izquierda naturalmente.
#                     # Si eso es un dato real del robot, déjalo así.
#                     3: (252,[0.0773,  0.0,    np.deg2rad(4.44)]),   # Fwd (Drift a la izquierda natural)
                    
#                     4: (253,[0.0748,  0.0064, -np.deg2rad(9.79)]),  # FwdSteer Right (CORREGIDO: Signo menos)
                    
#                     5: (256,[-0.0603, 0.0025, np.deg2rad(5.81)]),   # BwdSteer Left (OK - asumiendo que al ir atrás la cola va a la izq)
                    
#                     # OJO: BwdSteer Right también necesita negativo si queremos simetría
#                     6: (258,[-0.0641, 0.002,  -0.03175]),           # BwdSteer Right (CORREGIDO: Signo menos)
                    
#                     7: (262,[0.0893,  0.0,    0.0]),                # FastFwd (OK)
#                     8: (266,[-0.0588, 0.0,    0.0]),                # Bwd (OK)
#                     9: (267,[-0.715,  0.0,    0.0]),                # FastBwd (OK)
#                 }
#         self.action_space = gym.spaces.Discrete(len(self.commands))

#         if target_init_pos is None:
#             self.target_pos = np.random.uniform(-1, 1, size=2)
#         else:
#             self.target_pos = target_init_pos
            
#         self.step_count = 0
#         self.success_radius = float(success_radius)
#         self.max_steps = int(max_steps)
        
#         # ### NUEVO: Inicializamos la variable de memoria
#         self.last_action = 0 
        
#         self.render_mode = render_mode
#         self.window_size = 600 
#         self.screen = None
#         self.clock = None

#         self.scale = (self.window_size * 0.4) / (self.world_size / 2.0)

#     def reset(self, *, seed: int | None = None, options: dict | None = None):
#         super().reset(seed=seed)
#         self.step_count = 0

#         half = self.world_size / 2.0 # Corregido hardcode de 4.0 a self.world_size

#         if options is None or options.get('target_init_pos') is None:
#             self.target_pos = np.random.uniform(-half, half, size=2)
#         else:
#             self.target_pos = np.array(options['target_init_pos'], dtype=np.float64)

#         # ### NUEVO: Al reiniciar, reseteamos la memoria (asumimos acción 0 o neutral)
#         self.last_action = 0
#         obs = self.get_obs() # Usamos la funcion auxiliar para armar el vector de 3
        
#         info = {}
#         return obs, info

#     def get_obs(self):
#         # ### NUEVO: Construimos vector [x, y, last_action_normalizada]
#         # Normalizamos la accion dividiendo por el maximo (9) para que quede entre 0 y 1
#         action_norm = float(self.last_action) / 9.0
#         return np.array([self.target_pos[0], self.target_pos[1], action_norm], dtype=np.float32)

#     def calc_new_target(self, theta, dx, dy):
#         R = np.array([
#             [np.cos(-theta), -np.sin(-theta)],
#             [np.sin(-theta),  np.cos(-theta)]
#         ])
#         t = np.array(self.target_pos)
#         t_new = R @ (t - np.array([dx, dy]))
#         return t_new

#     def step(self, action):
#         command, movement = self.commands[action]
#         dx, dy, dtheta = movement
#         self.target_pos = self.calc_new_target(dtheta, dx, dy)
#         self.step_count += 1

#         dist = float(np.linalg.norm(self.target_pos))
#         terminated = dist <= self.success_radius
#         truncated = self.step_count >= self.max_steps
#         cosine_dist = float(self.target_pos[0] / dist) 
        
#         # Reward Base
#         reward = -dist + 0.5 * cosine_dist
        
#         if action != self.last_action:
#             reward -= 0.1  # Penalización sutil pero acumulativa
        
#         # Actualizamos la memoria para el siguiente frame
#         self.last_action = action

#         obs = self.get_obs() # Obtenemos la nueva observación de 3 valores
#         info = {"comando": command, "target": self.target_pos}

#         if self.render_mode == "human":
#             self.render()

#         return obs, reward, terminated, truncated, info
    
#     def _world_to_screen(self, x, y):
#         cx = self.window_size // 2
#         cy = self.window_size // 2
#         sx = int(cx + x * self.scale)
#         sy = int(cy - y * self.scale) 
#         return sx, sy

#     def render(self):
#         if self.render_mode != "human":
#             return

#         if self.screen is None:
#             pygame.init()
#             self.screen = pygame.display.set_mode((self.window_size, self.window_size))
#             pygame.display.set_caption("SpiderEnv - Pygame render")
#             self.clock = pygame.time.Clock()

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self.close()

#         self.screen.fill((30, 30, 30))

#         half = self.world_size / 2.0
#         top_left = self._world_to_screen(-half, half)
#         bottom_right = self._world_to_screen(half, -half)
#         width = bottom_right[0] - top_left[0]
#         height = bottom_right[1] - top_left[1]
#         pygame.draw.rect(self.screen, (80, 80, 80), (top_left[0], top_left[1], width, height), 2)

#         robot_x, robot_y = self._world_to_screen(0.0, 0.0)
#         pygame.draw.circle(self.screen, (0, 0, 255), (robot_x, robot_y), 8)

#         radius_px = int(self.success_radius * self.scale)
#         pygame.draw.circle(self.screen, (0, 255, 0), (robot_x, robot_y), radius_px, 1)

#         tx, ty = self.target_pos
#         target_px = self._world_to_screen(tx, ty)
#         pygame.draw.circle(self.screen, (255, 0, 0), target_px, 6)

#         pygame.draw.line(self.screen, (200, 200, 0), (robot_x, robot_y), target_px, 1)

#         font = pygame.font.SysFont(None, 20)
#         dist = float(np.linalg.norm(self.target_pos))
        
#         # Info en pantalla
#         text_surface = font.render(
#             f"Step: {self.step_count}  Dist: {dist:.2f} m  LastAct: {self.last_action}",
#             True,
#             (255, 255, 255)
#         )
#         self.screen.blit(text_surface, (10, 10))

#         pygame.display.flip()
#         self.clock.tick(self.metadata["render_fps"])

#     def close(self):
#         if self.screen is not None:
#             pygame.display.quit()
#             pygame.quit()
#             self.screen = None
#             self.clock = None


import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
from gymnasium.envs.registration import register

class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 10}

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
                    0: (227,[0.0667, -0.0052, np.deg2rad(32.72)]),  # Pivot Left (OK)
                    1: (228,[0.0667,  0.0052, -np.deg2rad(32.72)]), # Pivot Right (CORREGIDO: Signo menos)
                    
                    2: (251,[0.0776, -0.0074, np.deg2rad(10.94)]),  # FwdSteer Left (OK)
                    # Nota: Tu "Fwd" (3) tiene giro positivo, o sea, el robot "tira" a la izquierda naturalmente.
                    # Si eso es un dato real del robot, déjalo así.
                    3: (252,[0.0773,  0.0,    np.deg2rad(4.44)]),   # Fwd (Drift a la izquierda natural)
                    
                    4: (253,[0.0748,  0.0064, -np.deg2rad(9.79)]),  # FwdSteer Right (CORREGIDO: Signo menos)
                    
                    5: (256,[-0.0603, 0.0025, np.deg2rad(5.81)]),   # BwdSteer Left (OK - asumiendo que al ir atrás la cola va a la izq)
                    
                    # OJO: BwdSteer Right también necesita negativo si queremos simetría
                    6: (258,[-0.0641, 0.002,  -0.03175]),           # BwdSteer Right (CORREGIDO: Signo menos)
                    
                    7: (262,[0.0893,  0.0,    0.0]),                # FastFwd (OK)
                    8: (266,[-0.0588, 0.0,    0.0]),                # Bwd (OK)
                    9: (267,[-0.715,  0.0,    0.0]),                # FastBwd (OK)
        }
        self.action_space = gym.spaces.Discrete(len(self.commands))

        if target_init_pos is None:
            self.target_pos = np.random.uniform(-1, 1, size=2)
        else:
            self.target_pos = target_init_pos
        self.step_count = 0
        self.success_radius = float(success_radius)
        self.max_steps = int(max_steps)
        # Distancia inicial al objetivo (escalar) para poder calcular delta en el reward
        self.last_distance = float(np.linalg.norm(self.target_pos))
        
        self.render_mode = render_mode
        self.window_size = 600 
        self.screen = None
        self.clock = None

        self.scale = (self.window_size * 0.4) / (self.world_size / 2.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0

        half = self.world_size / 2.0

        if options is None or options.get('target_init_pos') is None:
            self.target_pos = np.random.uniform(-half, half, size=2)
        else:
            self.target_pos = np.array(options['target_init_pos'], dtype=np.float64)

        # Actualizamos la distancia previa para que el reward use el valor correcto desde el primer paso
        self.last_distance = float(np.linalg.norm(self.target_pos))
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
        # cosine_dist = float(self.target_pos[0] / dist) 

        

        reward = self.last_distance - dist - 0.1 # recompensa por acercarse
        self.last_distance = dist

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
