import gymnasium as gym
import numpy as np
import pygame


INITIAL_PREVIOUS_MOTION = "__initial__"


ACTION_METADATA = {
    0: (227, "Pivot_Left"),
    1: (228, "Pivot_Right"),
    2: (251, "FwdSteer_Left"),
    3: (252, "Fwd"),
    4: (253, "FwdSteer_Right"),
    5: (256, "BwdSteer_Left"),
    6: (258, "BwdSteer_Right"),
    7: (262, "FastFwd"),
    8: (266, "Bwd"),
    9: (267, "FastBwd"),
    10: (261, "FastFwdSteer_Left"),
    11: (263, "FastFwdSteer_Right"),
}


COMMANDS_PREVIOUS_STEPS = {
    INITIAL_PREVIOUS_MOTION: {
        0: (227, [0.00589, -0.00468, -0.5445427266222308]),
        1: (228, [0.00002, 0.00490, 0.4764748857944515]),
        2: (251, [0.07877, -0.01080, -0.1785183249409813]),
        3: (252, [0.07160, -0.00625, 0.02825319337022122]),
        4: (253, [0.06613, 0.01076, 0.13631463916687662]),
        5: (256, [-0.05835, 0.00319, -0.07244984621479054]),
        6: (258, [-0.06480, 0.00246, 0.1791970342223921]),
        7: (262, [0.10923, -0.02321, 0.059248398268510336]),
        8: (266, [-0.05726, 0.00327, -0.06618263863747691]),
        9: (267, [-0.08296, -0.00177, -0.08489512493581532]),
        10: (261, [0.07163, -0.01131, -0.03777271096372554]),
        11: (263, [0.07658, -0.00776, 0.13609617837175253]),
    },
    "Pivot_Left": {
        0: (227, [-0.00484, -0.00808, 0.5503023131538122]),
        1: (228, [-0.00013, 0.00843, -0.530327019864737]),
        2: (251, [0.07034, -0.01135, 0.1772050242842363]),
        3: (252, [0.07637, -0.00849, 0.03548254369304472]),
        4: (253, [0.07384, 0.00517, -0.11359824502455493]),
        5: (256, [-0.04673, 0.00356, -0.05244190803467362]),
        6: (258, [-0.05904, -0.00037, 0.1664921933354951]),
        7: (262, [0.09694, -0.02516, -0.10545802939325337]),
        8: (266, [-0.04704, 0.00304, -0.05382071814374914]),
        9: (267, [-0.04135, -0.00178, 0.15777252839253142]),
        10: (261, [0.06701, -0.01952, -0.023394393293731993]),
        11: (263, [0.0413, -0.011, -0.13888981121520474]),
    },
    "Pivot_Right": {
        0: (227, [-0.00225, -0.00463, 0.5485203319875259]),
        1: (228, [0.00031, 0.00649, -0.517394130107459]),
        2: (251, [0.07214, -0.01231, 0.167748830396931]),
        3: (252, [0.07514, -0.00353, 0.02108706802259549]),
        4: (253, [0.0715, 0.00589, -0.12936904014557568]),
        5: (256, [-0.05289, 0.00131, -0.07130542659022833]),
        6: (258, [-0.06224, -0.00425, 0.15502712547914432]),
        7: (262, [0.10137, -0.03005, -0.11580434119907576]),
        8: (266, [-0.04916, 0.00278, -0.0398284135305106]),
        9: (267, [-0.03599, 6e-05, 0.20792281911933647]),
        10: (261, [0.06664, -0.02088, 0.028249899272780217]),
        11: (263, [0.04427, -0.0173, -0.11119667197381074]),
    },
    "FwdSteer_Left": {
        0: (227, [-0.00996, -0.0078, 0.4927169198135112]),
        1: (228, [-0.00308, 0.01549, -0.49679924493392597]),
        2: (251, [0.0759, -0.01155, 0.1883070636561722]),
        3: (252, [0.07346, -0.00713, 0.05337216852598659]),
        4: (253, [0.0689, 0.00587, -0.09438216996009736]),
        5: (256, [-0.04976, 0.00326, -0.05028293574995663]),
        6: (258, [-0.06079, 0.00151, 0.14361267217110144]),
        7: (262, [0.07342, -0.00857, -0.1430873280662511]),
        8: (266, [-0.05046, 0.00707, -0.06265208415884045]),
        9: (267, [-0.05446, 0.00616, 0.12036488653453695]),
        10: (261, [0.04816, -0.0063, 0.01907121273654204]),
        11: (263, [0.03565, -0.00827, -0.15083309928660193]),
    },
    "Fwd": {
        0: (227, [-0.01012, -0.00814, 0.5534910296972058]),
        1: (228, [-0.0019, 0.01307, -0.5137079947272469]),
        2: (251, [0.08169, -0.01001, 0.16019330006504753]),
        3: (252, [0.07936, -0.00454, 0.03298148687493684]),
        4: (253, [0.06069, 0.00814, -0.10314023214660491]),
        5: (256, [-0.05678, 0.00113, -0.10122386062791512]),
        6: (258, [-0.06335, -0.00272, 0.13471847430293832]),
        7: (262, [0.07924, -0.00932, -0.13984974230380165]),
        8: (266, [-0.05675, 0.00087, -0.1170522516142517]),
        9: (267, [-0.05218, -0.00165, 0.17333911999106882]),
        10: (261, [0.04232, -0.00839, 0.013409364643072434]),
        11: (263, [0.03695, -0.01262, -0.15062365977636263]),
    },
    "FwdSteer_Right": {
        0: (227, [-0.00507, -0.00565, 0.44733661393240665]),
        1: (228, [-0.00155, 0.01108, -0.5520441517473025]),
        2: (251, [0.07635, -0.00846, 0.1473319688071013]),
        3: (252, [0.07461, -0.0003, 0.015360642746802095]),
        4: (253, [0.06874, 0.01314, -0.13511640937239303]),
        5: (256, [-0.05564, 0.00217, -0.10816154440459258]),
        6: (258, [-0.06179, -0.00183, 0.12042597305835674]),
        7: (262, [0.09052, -0.02333, -0.07844731388938912]),
        8: (266, [-0.0558, 0.00176, -0.10771648544533405]),
        9: (267, [-0.05584, -0.00588, 0.13891250049548068]),
        10: (261, [0.06112, -0.0167, 0.052228977865930316]),
        11: (263, [0.04389, -0.01489, -0.12077329357950363]),
    },
    "BwdSteer_Left": {
        0: (227, [-0.00362, -0.0077, 0.5580009604843591]),
        1: (228, [-0.00488, 0.01529, -0.534641473775667]),
        2: (251, [0.07415, -0.00858, 0.18112852444271954]),
        3: (252, [0.07343, -0.00504, 0.04015828075913753]),
        4: (253, [0.06971, 0.01041, -0.1352979236146004]),
        5: (256, [-0.06118, 0.0021, -0.10044544378152566]),
        6: (258, [-0.06596, 0.00153, 0.1447209562461178]),
        7: (262, [0.07336, -0.01265, -0.11704876095574772]),
        8: (266, [-0.06222, 0.00156, -0.10761351101946637]),
        9: (267, [-0.05457, -0.00038, 0.15613366422490874]),
        10: (261, [0.05163, -0.01132, 0.03931354140117227]),
        11: (263, [0.03425, -0.00515, -0.18764034788191036]),
    },
    "BwdSteer_Right": {
        0: (227, [-0.00526, -0.00623, 0.5267490948981486]),
        1: (228, [-0.00245, 0.00976, -0.49658282410667864]),
        2: (251, [0.07701, -0.00828, 0.18016684802487068]),
        3: (252, [0.07564, -0.00423, 0.03400424981660552]),
        4: (253, [0.06585, 0.00917, -0.10595021224231578]),
        5: (256, [-0.05693, 0.00133, -0.09747314806537932]),
        6: (258, [-0.06418, 0.00308, 0.14364059743913332]),
        7: (262, [0.07821, -0.01225, -0.016779595428673483]),
        8: (266, [-0.05783, 0.00113, -0.10473197242442374]),
        9: (267, [-0.05307, -0.00165, 0.1735712487815841]),
        10: (261, [0.05568, -0.01317, 0.06717946823851374]),
        11: (263, [0.03841, -0.00419, -0.16169777388026665]),
    },
    "FastFwd": {
        0: (227, [0.04821, 0.00203, 0.6054669348215969]),
        1: (228, [0.05393, 0.01973, -0.45934273385687563]),
        2: (251, [0.07368, -0.01007, 0.13039354841649636]),
        3: (252, [0.06647, -0.00658, 0.02967408794240759]),
        4: (253, [0.05471, 0.00313, -0.06036221218022388]),
        5: (256, [-0.02704, 0.00541, -0.01193979741289321]),
        6: (258, [-0.03909, 0.00586, 0.11266972986249393]),
        7: (262, [0.12814, -0.01661, -0.054553756429586764]),
        8: (266, [-0.02793, 0.01031, -0.011145672603235788]),
        9: (267, [-0.0091, 0.01053, 0.31480329185296524]),
        10: (261, [0.09227, -0.00781, 0.05609837281760174]),
        11: (263, [0.08666, -0.00521, -0.08659102017919466]),
    },
    "Bwd": {
        0: (227, [-0.00544, -0.00678, 0.5113238749690228]),
        1: (228, [-0.00613, 0.01006, -0.5204135497134091]),
        2: (251, [0.07061, -0.01044, 0.164673560254917]),
        3: (252, [0.06894, -0.00204, 0.016018631874803957]),
        4: (253, [0.06742, 0.01105, -0.1398916302058495]),
        5: (256, [-0.06084, 0.00121, -0.11148465130038979]),
        6: (258, [-0.0636, -0.00102, 0.12198280675113568]),
        7: (262, [0.07429, -0.00782, -0.09509775495341503]),
        8: (266, [-0.05853, 0.00303, -0.09501746980782329]),
        9: (267, [-0.04761, 0.00115, 0.1557182758629341]),
        10: (261, [0.05167, -0.00646, 0.05806186822609537]),
        11: (263, [0.04011, -0.0003, -0.13963506680580634]),
    },
    "FastBwd": {
        0: (227, [-0.04597, -0.00249, 0.38416965764422784]),
        1: (228, [-0.06512, 0.01084, -0.6000302342016345]),
        2: (251, [0.06402, -0.01512, 0.19150450684582582]),
        3: (252, [0.06239, -0.00952, 0.08066737269792591]),
        4: (253, [0.05907, 0.00331, -0.04735950925286613]),
        5: (256, [-0.04153, 0.00162, 0.023123867259672873]),
        6: (258, [-0.05141, 0.00184, 0.08831191482166108]),
        7: (262, [0.04795, -0.0008, -0.20353855203832671]),
        8: (266, [-0.04089, 0.00117, 0.02734756404949915]),
        9: (267, [-0.077, 0.00055, 0.027447047816862826]),
        10: (261, [0.02337, -0.0016, -0.10528175113880194]),
        11: (263, [0.01594, 0.00106, -0.2809962642418351]),
    },
    "FastFwdSteer_Left": {
        0: (227, [0.0174, -0.00016, 0.5544911033585985]),
        1: (228, [0.0136, 0.01792, -0.5340637697932569]),
        2: (251, [0.07883, -0.02096, 0.21657616155072434]),
        3: (252, [0.07673, -0.01494, 0.07837051940230139]),
        4: (253, [0.07095, 0.00305, -0.07814188127029012]),
        5: (256, [-0.03701, 0.00085, -0.03659955441432109]),
        6: (258, [-0.04816, 0.00089, 0.15292749438899517]),
        7: (262, [0.08913, -0.01373, 0.0004014257279586958]),
        8: (266, [-0.03449, 0.00429, -0.03327819284777588]),
        9: (267, [-0.02714, 0.00559, 0.1934156423767596]),
        10: (261, [0.07702, -0.01137, 0.04989721798526589]),
        11: (263, [0.0548, -0.00651, -0.12384507306301364]),
    },
    "FastFwdSteer_Right": {
        0: (227, [0.04641, -0.00127, 0.6241367218301802]),
        1: (228, [0.05214, 0.02153, -0.4636624237555616]),
        2: (251, [0.07008, -0.0108, 0.13287889727133628]),
        3: (252, [0.0611, -0.0074, 0.04196644186420365]),
        4: (253, [0.04972, 0.00146, -0.05796063912947969]),
        5: (256, [-0.02472, 0.00785, -0.0031922072018976287]),
        6: (258, [-0.03675, 0.00495, 0.12761847490582537]),
        7: (262, [0.12737, -0.0195, 0.009866346261523945]),
        8: (266, [-0.02494, 0.00674, 0.003473205211468716]),
        9: (267, [-0.0099, 0.01109, 0.3014759576847365]),
        10: (261, [0.09701, -0.00734, 0.11262435130194208]),
        11: (263, [0.09152, -0.00369, -0.09660746475639012]),
    },
}


def build_previous_commands(calibration_path=None):
    return COMMANDS_PREVIOUS_STEPS


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
        calibration_path=None,
        include_previous_action=True,
    ):
        self.world_size_x = float(map_shape_x)
        self.world_size_y = float(map_shape_y)
        self.world_size = max(self.world_size_x, self.world_size_y)

        half_x = self.world_size_x / 2.0
        half_y = self.world_size_y / 2.0

        self.include_previous_action = bool(include_previous_action)
        self.previous_action_dim = len(ACTION_METADATA) if self.include_previous_action else 0
        obs_low = [np.array([-half_x, -half_y], dtype=np.float32)]
        obs_high = [np.array([half_x, half_y], dtype=np.float32)]

        if self.include_previous_action:
            obs_low.append(np.zeros(self.previous_action_dim, dtype=np.float32))
            obs_high.append(np.ones(self.previous_action_dim, dtype=np.float32))

        low = np.concatenate(obs_low)
        high = np.concatenate(obs_high)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )

        self.commands = build_previous_commands(calibration_path)
        self.motion_to_action = {motion_name: action for action, (_, motion_name) in ACTION_METADATA.items()}
        self.command_to_action = {command_id: action for action, (command_id, _) in ACTION_METADATA.items()}
        self.action_space = gym.spaces.Discrete(len(ACTION_METADATA))

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

        self.previous_motion_name = None
        self.previous_action = None
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

        previous_motion = None if options is None else options.get("previous_motion")
        previous_action = None if options is None else options.get("previous_action")
        self.previous_motion_name = self._resolve_previous_motion(previous_motion, previous_action)
        self.previous_action = (
            None if self.previous_motion_name is None else self.motion_to_action[self.previous_motion_name]
        )

        self.last_distance = float(np.linalg.norm(self.target_pos))
        self.last_angle = self.angle_misalignment(self.target_pos)
        obs = self.get_obs()
        info = {"previous_motion": self.previous_motion_name}
        return obs, info

    def _resolve_previous_motion(self, previous_motion, previous_action):
        if previous_motion is None and previous_action is None:
            return None

        if previous_motion is not None:
            if isinstance(previous_motion, str):
                if previous_motion not in self.motion_to_action:
                    valid = ", ".join(sorted(self.motion_to_action))
                    raise ValueError(f"Movimiento previo desconocido '{previous_motion}'. Opciones: {valid}")
                return previous_motion

            previous_motion = int(previous_motion)
            if previous_motion in self.command_to_action:
                return ACTION_METADATA[self.command_to_action[previous_motion]][1]
            if previous_motion in ACTION_METADATA:
                return ACTION_METADATA[previous_motion][1]

            raise ValueError(f"Movimiento previo desconocido: {previous_motion}")

        previous_action = int(previous_action)
        if previous_action not in ACTION_METADATA:
            raise ValueError(f"Accion previa fuera de rango: {previous_action}")
        return ACTION_METADATA[previous_action][1]

    def get_obs(self):
        target_obs = self.target_pos.astype(np.float32)
        if not self.include_previous_action:
            return target_obs

        previous_action_obs = np.zeros(self.previous_action_dim, dtype=np.float32)
        if self.previous_action is not None:
            previous_action_obs[self.previous_action] = 1.0
        return np.concatenate([target_obs, previous_action_obs])

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

    def _apply_movement_noise(self, movement):
        movement = np.asarray(movement, dtype=np.float64)
        sigma = np.abs(movement) * self.noise_ratio
        noise = np.random.normal(loc=0.0, scale=sigma)
        return movement + noise

    def step(self, action):
        action = int(action)
        if action not in ACTION_METADATA:
            raise ValueError(f"Accion fuera de rango: {action}")

        previous_motion_before_step = self.previous_motion_name
        previous_motion_key = previous_motion_before_step or INITIAL_PREVIOUS_MOTION
        command, movement = self.commands[previous_motion_key][action]
        _, target_motion = ACTION_METADATA[action]

        dx, dy, dtheta = self._apply_movement_noise(movement)
        self.target_pos = self.calc_new_target(dtheta, dx, dy)
        self.step_count += 1

        dist = float(np.linalg.norm(self.target_pos))
        terminated = dist <= self.success_radius
        truncated = self.step_count >= self.max_steps

        new_angle = self.angle_misalignment(self.target_pos)
        ori_improvement = self.last_angle - new_angle
        ori_improvement /= np.pi / 2.0

        reward = 0.0
        distance_delta = self.last_distance - dist
        reward += distance_delta * self.distance_scale
        reward -= self.step_cost

        if distance_delta > 0:
            far_scale = min(1.0, dist / (self.world_size / 2.0)) if self.world_size > 0 else 0.0
            reward += self.orientation_weight * ori_improvement * far_scale

        if distance_delta < 0:
            reward -= self.backtrack_penalty

        if terminated:
            reward += self.success_bonus

        self.last_distance = dist
        self.last_angle = new_angle
        self.previous_motion_name = target_motion
        self.previous_action = action

        obs = self.get_obs()
        info = {
            "comando": command,
            "target": self.target_pos,
            "previous_motion": previous_motion_before_step,
            "target_motion": target_motion,
            "movement": np.array([dx, dy, dtheta], dtype=np.float64),
        }

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
            pygame.display.set_caption("SpiderEnv - Previous steps")
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
        pygame.draw.rect(self.screen, (80, 80, 80), (top_left[0], top_left[1], width, height), 2)

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
            f"Step: {self.step_count}  Dist: {dist:.2f} m  Prev: {self.previous_motion_name}",
            True,
            (255, 255, 255),
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


__all__ = ["ACTION_METADATA", "COMMANDS_PREVIOUS_STEPS", "SpiderEnv", "build_previous_commands"]
