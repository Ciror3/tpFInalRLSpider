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
        num_obstacles=6,
        obstacle_radius_range=(0.25, 0.35),
        obstacle_clearance=0.6,
        robot_radius=0.08,
        collision_penalty=5.0,
        repulse_weight=0.3,
        lidar_beams=16,
        lidar_max_range=None,
    ):
        self.world_size_x = float(map_shape_x)
        self.world_size_y = float(map_shape_y)
        self.world_size = max(self.world_size_x, self.world_size_y)

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0

        self.lidar_beams = int(lidar_beams)
        self.lidar_max_range = (
            float(lidar_max_range)
            if lidar_max_range is not None
            else max(self.world_size_x, self.world_size_y)
        )

        target_low = np.array([-half_x, -half_y], dtype=np.float32)
        target_high = np.array([half_x, half_y], dtype=np.float32)
        lidar_low = np.zeros(self.lidar_beams, dtype=np.float32)
        lidar_high = np.ones(self.lidar_beams, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([target_low, lidar_low]),
            high=np.concatenate([target_high, lidar_high]),
            shape=(2 + self.lidar_beams,),
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
        lidar = self._lidar()
        return np.concatenate([self.target_pos.astype(np.float32), lidar], axis=0)

    def _lidar(self):
        beams = self.lidar_beams
        max_range = self.lidar_max_range
        thetas = np.linspace(0.0, 2.0 * np.pi, beams, endpoint=False)
        dists = np.full(beams, max_range, dtype=np.float32)
        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0

        for i, theta in enumerate(thetas):
            dx = np.cos(theta)
            dy = np.sin(theta)

            if abs(dx) > 1e-6:
                for bound in (half_x, -half_x):
                    t = bound / dx
                    if 0 < t < dists[i]:
                        y_hit = t * dy
                        if -half_y <= y_hit <= half_y:
                            dists[i] = t
            if abs(dy) > 1e-6:
                for bound in (half_y, -half_y):
                    t = bound / dy
                    if 0 < t < dists[i]:
                        x_hit = t * dx
                        if -half_x <= x_hit <= half_x:
                            dists[i] = t

            if self.obstacles.size:
                for ox, oy, r in self.obstacles:
                    b = -2.0 * (ox * dx + oy * dy)
                    c = ox * ox + oy * oy - r * r
                    disc = b * b - 4.0 * c
                    if disc < 0:
                        continue
                    sqrt_disc = np.sqrt(disc)
                    t_hit = (-b - sqrt_disc) / 2.0
                    if 0 < t_hit < dists[i]:
                        dists[i] = t_hit

            dists[i] = min(dists[i], max_range)

        return dists / max_range

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
        if self.num_obstacles <= 0:
            return np.zeros((0, 3), dtype=np.float64)

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0
        min_r, max_r = self.obstacle_radius_range
        obstacles = []
        max_attempts = 200

        for _ in range(self.num_obstacles):
            placed = False
            for _ in range(max_attempts):
                r = float(np.random.uniform(min_r, max_r))
                x = float(np.random.uniform(-half_x + r, half_x - r))
                y = float(np.random.uniform(-half_y + r, half_y - r))
                center = np.array([x, y])

                if np.linalg.norm(center) < (self.robot_radius + r + self.obstacle_clearance):
                    continue
                if np.linalg.norm(center - target_pos) < (r + self.obstacle_clearance + self.success_radius):
                    continue

                too_close = False
                for ox, oy, orad in obstacles:
                    if np.linalg.norm(center - np.array([ox, oy])) < (r + orad + self.obstacle_clearance * 0.5):
                        too_close = True
                        break
                if too_close:
                    continue

                obstacles.append([x, y, r])
                placed = True
                break

            if not placed:
                break

        if not obstacles:
            return np.zeros((0, 3), dtype=np.float64)

        return np.array(obstacles, dtype=np.float64)

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
