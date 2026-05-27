import sys
import types

import numpy as np


try:
    import pygame  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pygame"] = types.ModuleType("pygame")

try:
    from tpFInalRLSpider.spiderJAR.enviroment_previous_steps import (
        ACTION_METADATA,
        INITIAL_PREVIOUS_MOTION,
        SpiderEnv as PreviousStepsSpiderEnv,
        build_previous_commands,
    )
except ModuleNotFoundError:
    from enviroment_previous_steps import (
        ACTION_METADATA,
        INITIAL_PREVIOUS_MOTION,
        SpiderEnv as PreviousStepsSpiderEnv,
        build_previous_commands,
    )


class SpiderEnv(PreviousStepsSpiderEnv):
    """Previous-steps environment with softer reward for useful alignment moves."""

    def __init__(
        self,
        *args,
        orientation_backtrack_weight=0.04,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.orientation_backtrack_weight = float(orientation_backtrack_weight)

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
        ori_improvement = (self.last_angle - new_angle) / (np.pi / 2.0)
        distance_delta = self.last_distance - dist
        far_scale = min(1.0, dist / (self.world_size / 2.0)) if self.world_size > 0 else 0.0

        reward_distance = distance_delta * self.distance_scale
        reward_step = -self.step_cost
        reward_orientation = 0.0
        reward_backtrack = 0.0
        reward_success = self.success_bonus if terminated else 0.0

        if distance_delta > 0:
            reward_orientation = self.orientation_weight * ori_improvement * far_scale
        elif distance_delta < 0:
            reward_orientation = self.orientation_backtrack_weight * ori_improvement * far_scale
            reward_backtrack = -self.backtrack_penalty

        reward = (
            reward_distance
            + reward_step
            + reward_orientation
            + reward_backtrack
            + reward_success
        )

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
            "distance_delta": distance_delta,
            "orientation_improvement": ori_improvement,
            "reward_distance": reward_distance,
            "reward_step": reward_step,
            "reward_orientation": reward_orientation,
            "reward_backtrack": reward_backtrack,
            "reward_success": reward_success,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info


__all__ = ["ACTION_METADATA", "SpiderEnv", "build_previous_commands"]
