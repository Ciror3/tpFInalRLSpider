import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ZENV_DIR = os.path.join(CURRENT_DIR, "zenviroments")
if ZENV_DIR not in sys.path:
    sys.path.insert(0, ZENV_DIR)

from enviroment import ENV_VARIANTS, get_env_class


ACTION_NAMES = {
    0: "Pivot_Left",
    1: "Pivot_Right",
    2: "FwdSteer_Left",
    3: "Fwd",
    4: "FwdSteer_Right",
    5: "BwdSteer_Left",
    6: "BwdSteer_Right",
    7: "FastFwd",
    8: "Bwd",
    9: "FastBwd",
    10: "FastFwdSteer_Left",
    11: "FastFwdSteer_Right",
}


@dataclass
class PlanningState:
    target_pos: np.ndarray
    obstacles: np.ndarray


@dataclass
class PlanResult:
    action: int
    score: float
    sequence: tuple[int, ...]
    predicted_target: np.ndarray
    predicted_collision: bool


class DiscreteTreeSearchController:
    """Tree-search baseline over the same discrete commands used by PPO."""

    def __init__(
        self,
        horizon=3,
        beam_width=None,
        stop_on_success_depth=True,
    ):
        self.horizon = int(horizon)
        self.beam_width = None if beam_width is None else int(beam_width)
        self.stop_on_success_depth = bool(stop_on_success_depth)

    def plan(self, env):
        actions = tuple(range(env.action_space.n))
        initial = self._read_state(env)
        initial_distance = self._distance(initial)

        frontier = [(0.0, (), initial, False, initial_distance)]
        finished = []

        for _ in range(self.horizon):
            expanded = []
            found_success_at_depth = False

            for score, seq, state, collided, last_distance in frontier:
                if self._is_terminal(env, state, collided):
                    finished.append((score, seq, state, collided, last_distance))
                    found_success_at_depth = found_success_at_depth or self._is_success(env, state, collided)
                    continue

                for action in actions:
                    child = self._expand(env, score, seq, state, collided, last_distance, action)
                    _, _, child_state, child_collided, _ = child
                    if self._is_terminal(env, child_state, child_collided):
                        finished.append(child)
                        found_success_at_depth = (
                            found_success_at_depth
                            or self._is_success(env, child_state, child_collided)
                        )
                    else:
                        expanded.append(child)

            if found_success_at_depth and self.stop_on_success_depth:
                break

            if self.beam_width is not None:
                expanded.sort(key=lambda item: self._terminal_score(env, *item), reverse=True)
                expanded = expanded[: self.beam_width]

            frontier = expanded
            if not frontier:
                break

        candidates = finished + frontier
        successful = [
            item for item in candidates if self._is_success(env, item[2], item[3])
        ]
        selectable = successful if successful else candidates
        best = max(selectable, key=lambda item: self._terminal_score(env, *item))
        final_score = self._terminal_score(env, *best)
        _, sequence, state, collided, _ = best
        return PlanResult(
            action=sequence[0] if sequence else -1,
            score=final_score,
            sequence=sequence,
            predicted_target=state.target_pos.copy(),
            predicted_collision=collided,
        )

    def _expand(self, env, score, seq, state, collided, last_distance, action):
        next_state = self._simulate_action(env, state, action)
        dist = self._distance(next_state)
        next_collision, repulse = self._collision_and_repulse(env, next_state)
        next_collided = collided or next_collision
        reward = self._reward_like_env(env, state, next_state, next_collision, repulse)

        return score + reward, seq + (action,), next_state, next_collided, dist

    def _terminal_score(self, env, score, seq, state, collided, last_distance):
        return score

    def _reward_like_env(self, env, state, next_state, collision, repulse):
        last_distance = self._distance(state)
        dist = self._distance(next_state)
        terminated = dist <= env.success_radius

        new_angle = self._angle_misalignment(next_state.target_pos)
        last_angle = self._angle_misalignment(state.target_pos)
        ori_improvement = last_angle - new_angle
        ori_improvement /= np.pi / 2.0

        reward = 0.0
        distance_delta = last_distance - dist
        reward += distance_delta * env.distance_scale
        reward -= env.step_cost

        if distance_delta > 0:
            far_scale = min(1.0, dist / (env.world_size / 2.0)) if env.world_size > 0 else 0.0
            reward += env.orientation_weight * ori_improvement * far_scale

        if distance_delta < 0:
            reward -= env.backtrack_penalty

        repulse_weight = float(getattr(env, "repulse_weight", 0.0))
        if repulse_weight:
            repulse_scale = 1.0
            if dist <= (env.success_radius * 3.0):
                repulse_scale = dist / (env.success_radius * 3.0)
            reward -= repulse_weight * repulse * repulse_scale

        if collision:
            reward -= float(getattr(env, "collision_penalty", 0.0))

        if terminated and not collision:
            reward += env.success_bonus

        return reward

    def _is_terminal(self, env, state, collided):
        return collided or self._is_success(env, state, collided)

    def _is_success(self, env, state, collided):
        return self._distance(state) <= env.success_radius and not collided

    def _read_state(self, env):
        obstacles = getattr(env, "obstacles", np.zeros((0, 3), dtype=np.float64))
        return PlanningState(
            target_pos=np.asarray(env.target_pos, dtype=np.float64).copy(),
            obstacles=np.asarray(obstacles, dtype=np.float64).copy(),
        )

    def _simulate_action(self, env, state, action):
        _, movement = env.commands[action]
        dx, dy, dtheta = np.asarray(movement, dtype=np.float64)
        target = self._transform_points(state.target_pos.reshape(1, 2), dtheta, dx, dy)[0]

        obstacles = state.obstacles.copy()
        if obstacles.size:
            obstacles[:, :2] = self._transform_points(obstacles[:, :2], dtheta, dx, dy)

        return PlanningState(target_pos=target, obstacles=obstacles)

    @staticmethod
    def _transform_points(points, dtheta, dx, dy):
        rotation = np.array(
            [
                [np.cos(-dtheta), -np.sin(-dtheta)],
                [np.sin(-dtheta), np.cos(-dtheta)],
            ],
            dtype=np.float64,
        )
        points = np.asarray(points, dtype=np.float64)
        return (rotation @ (points - np.array([dx, dy], dtype=np.float64)).T).T

    @staticmethod
    def _distance(state):
        return float(np.linalg.norm(state.target_pos))

    @staticmethod
    def _angle_misalignment(target_pos):
        dist = float(np.linalg.norm(target_pos))
        if dist < 1e-6:
            return 0.0
        angle = float(np.arctan2(target_pos[1], target_pos[0]))
        abs_angle = abs(angle)
        return min(abs_angle, abs(np.pi - abs_angle))

    @staticmethod
    def _collision_and_repulse(env, state):
        if state.obstacles.size == 0:
            return False, 0.0

        robot_radius = float(getattr(env, "robot_radius", 0.0))
        obstacle_clearance = float(getattr(env, "obstacle_clearance", 0.0))
        centers = state.obstacles[:, :2]
        radii = state.obstacles[:, 2] + robot_radius
        dists = np.linalg.norm(centers, axis=1)
        clearances = dists - radii

        collision = bool(np.any(clearances <= 0.0))
        safe_clearance = max(obstacle_clearance, 1e-6)
        close_mask = clearances < safe_clearance
        if not np.any(close_mask):
            return collision, 0.0

        closeness = 1.0 - np.clip(clearances[close_mask] / safe_clearance, 0.0, 1.0)
        repulse = float(np.sum(closeness**2))
        return collision, repulse


def evaluate_controller(env_cls, args):
    controller = DiscreteTreeSearchController(
        horizon=args.horizon,
        beam_width=args.beam_width,
        stop_on_success_depth=not args.continue_after_first_success,
    )

    successes = 0
    collisions = 0
    returns = []
    steps = []

    for episode in range(args.episodes):
        env = env_cls(render_mode=args.render_mode, max_steps=args.max_steps)
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        info = {}
        episode_return = 0.0
        episode_actions = []

        while not done:
            plan = controller.plan(env)
            if not plan.sequence:
                break

            for action in plan.sequence:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_actions.append(action)
                done = terminated or truncated
                if done or args.replan_each_step:
                    break

        distance = float(np.linalg.norm(env.target_pos))
        success = distance <= env.success_radius and not info.get("collision", False)
        collision = bool(info.get("collision", False))
        successes += int(success)
        collisions += int(collision)
        returns.append(episode_return)
        steps.append(env.step_count)

        should_print_progress = (
            args.progress_every > 0
            and ((episode + 1) % args.progress_every == 0 or episode + 1 == args.episodes)
        )
        if should_print_progress:
            running_avg_reward = float(np.mean(returns))
            running_avg_steps = float(np.mean(steps))
            print(
                f"episode={episode + 1}/{args.episodes} "
                f"success={int(success)} collision={int(collision)} "
                f"steps={env.step_count} reward={episode_return:.3f} "
                f"avg_steps={running_avg_steps:.2f} avg_reward={running_avg_reward:.3f}"
            )

        if args.verbose:
            names = [ACTION_NAMES.get(action, str(action)) for action in episode_actions]
            print(
                f"episode={episode + 1} success={success} collision={collision} "
                f"steps={env.step_count} return={episode_return:.3f} "
                f"final_distance={distance:.3f} actions={names}"
            )

        env.close()

    n = max(args.episodes, 1)
    return {
        "episodes": args.episodes,
        "success_rate": successes / n,
        "collision_rate": collisions / n,
        "avg_reward": float(np.mean(returns)) if returns else 0.0,
        "std_reward": float(np.std(returns)) if returns else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "std_steps": float(np.std(steps)) if steps else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline de busqueda en arbol para SpiderEnv.")
    parser.add_argument("--env", choices=list(ENV_VARIANTS), default="sin_obstaculos")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--beam-width",
        type=int,
        default=128,
        help="cantidad de nodos que conserva por profundidad; más bajo es más rápido",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="expande todo el árbol completo; puede ser muy lento",
    )
    parser.add_argument(
        "--continue-after-first-success",
        action="store_true",
        help="sigue expandiendo hasta horizon aunque ya exista una ruta exitosa",
    )
    parser.add_argument(
        "--replan-each-step",
        action="store_true",
        help="ejecuta solo el primer comando del camino y vuelve a planificar",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="imprime progreso cada N episodios; usar 0 para desactivar",
    )
    parser.add_argument("--render-mode", choices=["human", "none"], default="none")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.render_mode == "none":
        args.render_mode = None
    if args.exhaustive:
        args.beam_width = None

    env_cls = get_env_class(args.env)
    metrics = evaluate_controller(env_cls, args)

    print("Tree-search baseline final")
    print(f"env={args.env} episodes={args.episodes} horizon={args.horizon} beam_width={args.beam_width}")
    print(f"avg_steps={metrics['avg_steps']:.2f} +- {metrics['std_steps']:.2f}")
    print(f"avg_reward={metrics['avg_reward']:.4f} +- {metrics['std_reward']:.4f}")
    print(f"success_rate={metrics['success_rate']:.4f}")
    print(f"collision_rate={metrics['collision_rate']:.4f}")


if __name__ == "__main__":
    main()
