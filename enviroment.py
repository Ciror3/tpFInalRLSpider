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
import pygame

class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 5}

    def __init__(self,target_init_pos=None,map_shape_x=7,map_shape_y=8,success_radius=0.2,max_steps=200,render_mode=None):
        # tamaños del mundo en cada eje
        self.world_size_x = float(map_shape_x)
        self.world_size_y = float(map_shape_y)
        # para escalado en pantalla sigo usando un único "world_size"
        self.world_size = max(self.world_size_x, self.world_size_y)

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0

        self.observation_space = gym.spaces.Box(
            low=np.array([-half_x, -half_y], dtype=np.float32),
            high=np.array([ half_x,  half_y], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.commands = {
            0: (227,[0.00589, -0.00468, -np.deg2rad(31.2)]),  # Pivot Left 
            1: (228,[0.00002, 0.00490, np.deg2rad(27.3)]), # Pivot Right 

            2: (251,[0.07877, -0.01080, -np.deg2rad(10.2283)]),  # FwdSteer Left (OK)
            3: (252,[0.07160, -0.00625,    np.deg2rad(1.6188)]),   # Fwd (Drift a la izquierda natural)
            4: (253,[0.06613, 0.01076, np.deg2rad(7.8102)]),  # FwdSteer Right 

            5: (256,[-0.05835, 0.00319, -np.deg2rad(4.1511)]),   # BwdSteer Left (OK)
            6: (258,[-0.06480, 0.00246,  np.deg2rad(10.2672)]),           # BwdSteer Right 

            7: (262,[0.10923, -0.02321, np.deg2rad(3.3947)]),                # FastFwd (OK)
            8: (266,[-0.05726, 0.00327, -np.deg2rad(3.7919)]),                # Bwd (OK)
            9: (267,[-0.08296, -0.00177, -np.deg2rad(4.8641)]),                # FastBwd (OK)
            10: (261,[0.07163, -0.01131, -np.deg2rad(2.1642)]),  # FastFwdSteer Left
            11: (263,[0.07658, -0.00776, np.deg2rad(7.7978)]),  # FastFwdSteer Right
        }


# Pivot_Left, 227, 5, 0.00589, -0.00468, 31.2
# Pivot_Right, 228, 5, 0.00002, 0.00490, -27.3
# FwdSteer_Left, 251, 5, 0.07877, -0.01080, 10.2283
# Fwd, 252, 5, 0.07160, -0.00625, 1.6188
# FwdSteer_Right, 253, 5, 0.06613, 0.01076, -7.8102
# BwdSteer_Left, 256, 5, -0.05835, 0.00319, -4.1511
# BwdSteer_Right, 258, 5, -0.06480, 0.00246, 10.2672
# FastFwd, 262, 5, 0.10923, -0.02321, -3.3947
# Bwd, 266, 5, -0.05726, 0.00327, -3.7919
# FastBwd, 267, 5, -0.08296, -0.00177, 4.8641
# FastFwdSteer_Left, 261, 5, 0.07163, -0.01131, 2.1642
# FastFwdSteer_Right, 263, 5, 0.07658, -0.00776, -7.7978
        self.action_space = gym.spaces.Discrete(len(self.commands))

        if target_init_pos is None:
            self.target_pos = np.array(
                [
                    np.random.uniform(-half_x, half_x),
                    np.random.uniform(-half_y, half_y),
                ],
                dtype=np.float64,
            )
        else:
            self.target_pos = np.array(target_init_pos, dtype=np.float64)

        self.step_count = 0
        self.success_radius = float(success_radius)
        self.max_steps = int(max_steps)
        self.last_distance = float(np.linalg.norm(self.target_pos))
        self.last_angle = self.angle_misalignment(self.target_pos)

        self.render_mode = render_mode
        self.window_size = 600
        self.screen = None
        self.clock = None

        self.scale = (self.window_size * 0.4) / (self.world_size / 2.0)
        self.noise_ratio = 0.05  # 5% de ruido gaussiano relativo al movimiento
        # Config de recompensas
        self.step_cost = 0.08
        self.distance_scale = self.world_size * 1.2  # Escala el delta de distancia
        self.orientation_weight = 0.12
        self.success_bonus = 4.0
        self.backtrack_penalty = 0.1

    def angle_misalignment(self, target_vec):
        d = np.linalg.norm(target_vec)
        if d < 1e-6:
            return 0.0
        angle = np.arctan2(target_vec[1], target_vec[0]) 
        a = abs(angle)             
        misalignment = min(a, abs(np.pi - a)) 
        return misalignment

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0

        if options is None or options.get("target_init_pos") is None:
            self.target_pos = np.array(
                [
                    np.random.uniform(-half_x, half_x),
                    np.random.uniform(-half_y, half_y),
                ],
                dtype=np.float64,
            )
        else:
            self.target_pos = np.array(options["target_init_pos"], dtype=np.float64)

        self.last_distance = float(np.linalg.norm(self.target_pos))
        # Refrescamos la orientación inicial para que el shaping use el valor actual
        self.last_angle = self.angle_misalignment(self.target_pos)
        obs = self.target_pos.astype(np.float32)
        info = {}
        return obs, info

    def get_obs(self):
        return self.target_pos.astype(np.float32)

    def calc_new_target(self, theta, dx, dy):
        R = np.array(
            [
                [np.cos(-theta), -np.sin(-theta)],
                [np.sin(-theta),  np.cos(-theta)],
            ]
        )
        t = np.array(self.target_pos)
        t_new = R @ (t - np.array([dx, dy]))
        return t_new

    def _apply_movement_noise(self, movement):
        movement = np.asarray(movement, dtype=np.float64)
        sigma = np.abs(movement) * self.noise_ratio
        noise = np.random.normal(loc=0.0, scale=sigma)
        return movement + noise

    def step(self, action):
        command, movement = self.commands[action]
        dx, dy, dtheta = self._apply_movement_noise(movement)
        self.target_pos = self.calc_new_target(dtheta, dx, dy)
        self.step_count += 1

        dist = float(np.linalg.norm(self.target_pos))
        terminated = dist <= self.success_radius
        truncated = self.step_count >= self.max_steps

        new_angle = self.angle_misalignment(self.target_pos)
        ori_improvement = self.last_angle - new_angle
        ori_improvement /= (np.pi / 2.0)

        reward = 0.0

        # 1) Progreso en distancia (escalado por tamaño de mapa)
        distance_delta = self.last_distance - dist
        reward += distance_delta * self.distance_scale

        # 2) Coste base por paso
        reward -= self.step_cost

        # 3) Orientación solo si hubo avance, ponderado por distancia actual
        if distance_delta > 0:
            far_scale = min(1.0, dist / (self.world_size / 2.0)) if self.world_size > 0 else 0.0
            reward += self.orientation_weight * ori_improvement * far_scale

        # 4) Penalización extra si te alejaste
        if distance_delta < 0:
            reward -= self.backtrack_penalty

        # 5) Bonus por éxito
        if terminated:
            reward += self.success_bonus

        self.last_distance = dist
        self.last_angle = new_angle

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((30, 30, 30))

        # rectángulo del mundo, ahora usando world_size_x/y
        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0
        top_left = self._world_to_screen(-half_x, half_y)
        bottom_right = self._world_to_screen(half_x, -half_y)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        pygame.draw.rect(
            self.screen, (80, 80, 80),
            (top_left[0], top_left[1], width, height),
            2
        )

        robot_x, robot_y = self._world_to_screen(0.0, 0.0)
        pygame.draw.circle(self.screen, (0, 0, 255), (robot_x, robot_y), 8)

        radius_px = int(self.success_radius * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (robot_x, robot_y), radius_px, 1)

        tx, ty = self.target_pos
        target_px = self._world_to_screen(tx, ty)
        pygame.draw.circle(self.screen, (255, 0, 0), target_px, 6)

        pygame.draw.line(self.screen, (200, 200, 0), (robot_x, robot_y), target_px, 1)

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
