import numpy as np
import gymnasium as gym
import pygame


class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 5}

    def __init__(
        self,
        target_init_pos=None,
        map_shape_x=7,
        map_shape_y=8,
        success_radius=0.2,
        max_steps=200,
        render_mode=None,
        num_obstacles=1,
        obstacle_radius_range=(0.25, 0.35),
        obstacle_clearance=0.6,
        obs_max_obstacles=1,
        robot_radius=0.08,
        collision_penalty=5.0,
        repulse_weight=0.3,
    ):
        self.world_size_x = float(map_shape_x)
        self.world_size_y = float(map_shape_y)
        self.world_size = max(self.world_size_x, self.world_size_y)
        self.half_extents = np.array([self.world_size_x / 2.0, self.world_size_y / 2.0], dtype=np.float32)

        self.obs_max_obstacles = int(obs_max_obstacles)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    np.array([-1.0, -1.0], dtype=np.float32),  # target normalizado
                    np.full(2 * self.obs_max_obstacles, -1.0, dtype=np.float32),  # obstáculos normalizados
                    np.zeros(self.obs_max_obstacles, dtype=np.float32),  # máscara de presencia
                ]
            ),
            high=np.concatenate(
                [
                    np.array([1.0, 1.0], dtype=np.float32),
                    np.full(2 * self.obs_max_obstacles, 1.0, dtype=np.float32),
                    np.ones(self.obs_max_obstacles, dtype=np.float32),
                ]
            ),
            shape=(2 + 2 * self.obs_max_obstacles + self.obs_max_obstacles,),
            dtype=np.float32,
        )

        self.commands = {
            0: (227, [0.00589, -0.00468, -np.deg2rad(31.2)]),  # Pivot Left
            1: (228, [0.00002, 0.00490, np.deg2rad(27.3)]),  # Pivot Right
            2: (251, [0.07877, -0.01080, -np.deg2rad(10.2283)]),  # FwdSteer Left
            3: (252, [0.07160, -0.00625, np.deg2rad(1.6188)]),  # Fwd
            4: (253, [0.06613, 0.01076, np.deg2rad(7.8102)]),  # FwdSteer Right
            5: (256, [-0.05835, 0.00319, -np.deg2rad(4.1511)]),  # BwdSteer Left
            6: (258, [-0.06480, 0.00246, np.deg2rad(10.2672)]),  # BwdSteer Right
            7: (262, [0.10923, -0.02321, np.deg2rad(3.3947)]),  # FastFwd
            8: (266, [-0.05726, 0.00327, -np.deg2rad(3.7919)]),  # Bwd
            9: (267, [-0.08296, -0.00177, -np.deg2rad(4.8641)]),  # FastBwd
            10: (261, [0.07163, -0.01131, -np.deg2rad(2.1642)]),  # FastFwdSteer Left
            11: (263, [0.07658, -0.00776, np.deg2rad(7.7978)]),  # FastFwdSteer Right
        }
        self.action_space = gym.spaces.Discrete(len(self.commands))

        self.num_obstacles = int(num_obstacles)
        self.obstacle_radius_range = tuple(obstacle_radius_range)
        self.obstacle_clearance = float(obstacle_clearance)
        self.robot_radius = float(robot_radius)
        self.collision_penalty = float(collision_penalty)
        self.repulse_weight = float(repulse_weight)
        self.obstacles = np.zeros((0, 3), dtype=np.float64)

        if target_init_pos is None:
            self.target_pos = np.array(
                [
                    np.random.uniform(-self.half_extents[0], self.half_extents[0]),
                    np.random.uniform(-self.half_extents[1], self.half_extents[1]),
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
        self.noise_ratio = 0.05
        self.step_cost = 0.08
        self.distance_scale = self.world_size * 1.2
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

        self.obstacles = self._sample_obstacles(self.target_pos)
        self.last_distance = float(np.linalg.norm(self.target_pos))
        self.last_angle = self.angle_misalignment(self.target_pos)
        obs = self.get_obs()
        info = {}
        return obs, info

    def get_obs(self):
        target_norm = (self.target_pos / self.half_extents).astype(np.float32)
        obs_coords, obs_mask = self._obstacle_obs()
        return np.concatenate([target_norm, obs_coords, obs_mask], axis=0)

    def calc_new_target(self, theta, dx, dy):
        rotation = np.array(
            [
                [np.cos(-theta), -np.sin(-theta)],
                [np.sin(-theta), np.cos(-theta)],
            ]
        )
        t = np.array(self.target_pos)
        t_new = rotation @ (t - np.array([dx, dy]))
        return t_new

    def _transform_points(self, pts, dtheta, dx, dy):
        rotation = np.array(
            [
                [np.cos(-dtheta), -np.sin(-dtheta)],
                [np.sin(-dtheta), np.cos(-dtheta)],
            ]
        )
        pts = np.asarray(pts, dtype=np.float64)
        return (rotation @ (pts - np.array([dx, dy])).T).T

    def _sample_obstacles(self, target_pos):
        """
        Coloca un único obstáculo (o hasta obs_max_obstacles) siempre entre el robot (0,0) y el target.
        El obstáculo se ubica sobre la recta robot->target con una distancia aleatoria respetando
        la clearance alrededor del robot, del target y del propio obstáculo.
        """
        if self.num_obstacles <= 0:
            return np.zeros((0, 3), dtype=np.float64)

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0
        min_r, max_r = self.obstacle_radius_range
        obstacles = []
        max_attempts = 200
        target_dist = float(np.linalg.norm(target_pos))

        if target_dist < 1e-6:
            return np.zeros((0, 3), dtype=np.float64)

        dir_vec = target_pos / target_dist
        # Limitamos la cantidad de slots para no generar más de lo que observa la red
        n_to_place = min(self.num_obstacles, self.obs_max_obstacles)

        for _ in range(n_to_place):
            placed = False
            for _ in range(max_attempts):
                r = float(np.random.uniform(min_r, max_r))
                buffer_robot = self.robot_radius + r + self.obstacle_clearance
                buffer_target = self.success_radius + r + self.obstacle_clearance
                usable_length = target_dist - (buffer_robot + buffer_target)

                if usable_length <= 0:
                    # El target está demasiado cerca para insertar un obstáculo seguro.
                    break

                # Elegimos una posición intermedia aleatoria entre el robot y el target.
                alpha = float(np.random.uniform(0.2, 0.8))
                d = buffer_robot + alpha * usable_length
                center = dir_vec * d

                # Aseguramos que el obstáculo queda dentro del mapa.
                if not (-half_x + r <= center[0] <= half_x - r and -half_y + r <= center[1] <= half_y - r):
                    continue

                obstacles.append([center[0], center[1], r])
                placed = True
                break

            if not placed:
                break

        if not obstacles:
            return np.zeros((0, 3), dtype=np.float64)

        return np.array(obstacles, dtype=np.float64)

    def _obstacle_obs(self):
        k = self.obs_max_obstacles
        if self.obstacles.size == 0 or k <= 0:
            coords = np.zeros(2 * k, dtype=np.float32)
            mask = np.zeros(k, dtype=np.float32)
            return coords, mask

        centers = self.obstacles[:, :2]
        dists = np.linalg.norm(centers, axis=1)

        idx_near = np.argsort(dists)[:k]
        centers_near = centers[idx_near]

        angles = np.arctan2(centers_near[:, 1], centers_near[:, 0])
        idx_sorted = np.argsort(angles)
        centers_sorted = centers_near[idx_sorted]

        take = len(centers_sorted)
        flat = centers_sorted.reshape(-1)
        if take < k:
            pad = np.zeros(2 * (k - take), dtype=np.float64)
            flat = np.concatenate([flat, pad])

        # Normaliza posiciones de obstáculos a [-1, 1] para estabilizar la red
        norm = self.half_extents.astype(np.float64)
        flat_norm = (flat / np.tile(norm, k)).astype(np.float32)

        mask = np.zeros(k, dtype=np.float32)
        mask[:take] = 1.0
        return flat_norm, mask

    def _collision_and_repulse(self):
        if self.obstacles.size == 0:
            return False, 0.0

        centers = self.obstacles[:, :2]
        radii = self.obstacles[:, 2] + self.robot_radius
        dists = np.linalg.norm(centers, axis=1)

        collision = bool(np.any(dists <= radii))
        clearance = dists - radii

        safe_clearance = max(self.obstacle_clearance, 1e-6)
        close_mask = clearance < safe_clearance
        repulse = 0.0
        if np.any(close_mask):
            closeness = 1.0 - np.clip(clearance[close_mask] / safe_clearance, 0.0, 1.0)
            repulse = float(np.sum(closeness**2))
        return collision, repulse

    def _apply_movement_noise(self, movement):
        movement = np.asarray(movement, dtype=np.float64)
        sigma = np.abs(movement) * self.noise_ratio
        noise = np.random.normal(loc=0.0, scale=sigma)
        return movement + noise

    def step(self, action):
        command, movement = self.commands[action]
        dx, dy, dtheta = self._apply_movement_noise(movement)
        self.target_pos = self.calc_new_target(dtheta, dx, dy)
        if self.obstacles.size:
            self.obstacles[:, :2] = self._transform_points(self.obstacles[:, :2], dtheta, dx, dy)
        self.step_count += 1

        dist = float(np.linalg.norm(self.target_pos))
        terminated = dist <= self.success_radius
        truncated = self.step_count >= self.max_steps

        new_angle = self.angle_misalignment(self.target_pos)
        ori_improvement = self.last_angle - new_angle
        ori_improvement /= (np.pi / 2.0)

        reward = 0.0
        distance_delta = self.last_distance - dist
        reward += distance_delta * self.distance_scale
        reward -= self.step_cost

        if distance_delta > 0:
            far_scale = min(1.0, dist / (self.world_size / 2.0)) if self.world_size > 0 else 0.0
            reward += self.orientation_weight * ori_improvement * far_scale

        if distance_delta < 0:
            reward -= self.backtrack_penalty

        collision, repulse = self._collision_and_repulse()
        repulse_scale = 1.0
        if dist <= (self.success_radius * 3.0):
            repulse_scale = dist / (self.success_radius * 3.0)
        reward -= self.repulse_weight * repulse * repulse_scale

        if collision:
            reward -= self.collision_penalty
            terminated = True

        if terminated and not collision:
            reward += self.success_bonus

        self.last_distance = dist
        self.last_angle = new_angle

        obs = self.get_obs()
        info = {"comando": command, "target": self.target_pos, "collision": collision}

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

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0
        top_left = self._world_to_screen(-half_x, half_y)
        bottom_right = self._world_to_screen(half_x, -half_y)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        pygame.draw.rect(
            self.screen,
            (80, 80, 80),
            (top_left[0], top_left[1], width, height),
            2,
        )

        robot_x, robot_y = self._world_to_screen(0.0, 0.0)
        pygame.draw.circle(self.screen, (0, 0, 255), (robot_x, robot_y), 8)

        radius_px = int(self.success_radius * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (robot_x, robot_y), radius_px, 1)

        for ox, oy, r in self.obstacles:
            obstacle_px = self._world_to_screen(ox, oy)
            radius = max(2, int(r * self.scale))
            pygame.draw.circle(self.screen, (200, 50, 50), obstacle_px, radius)
            pygame.draw.circle(self.screen, (120, 30, 30), obstacle_px, radius, 1)

        tx, ty = self.target_pos
        target_px = self._world_to_screen(tx, ty)
        pygame.draw.circle(self.screen, (255, 0, 0), target_px, 6)
        pygame.draw.line(self.screen, (200, 200, 0), (robot_x, robot_y), target_px, 1)

        font = pygame.font.SysFont(None, 20)
        dist = float(np.linalg.norm(self.target_pos))
        text_surface = font.render(f"Step: {self.step_count}  Dist: {dist:.2f} m", True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


__all__ = ["SpiderEnv"]
