import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from compare_walltime_steps import patch_numpy_pickle_aliases
from dwa_baseline import DiscreteTreeSearchController
from spiderJAR.enviroment_previous_steps import ACTION_METADATA, SpiderEnv


ACTION_NAMES = {action: name for action, (_, name) in ACTION_METADATA.items()}


@dataclass
class EpisodeStats:
    label: str
    episode: int
    target: np.ndarray
    success: bool
    collision: bool
    steps: int
    episode_return: float
    final_distance: float
    wall_time_sec: float
    actions: list[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza PPO previous_steps contra DWA usando los mismos targets "
            "iniciales y render_mode='human'."
        )
    )
    parser.add_argument(
        "--ppo-model",
        default=os.path.join(
            CURRENT_DIR,
            "spiderJAR",
            "models_previous_steps",
            "previous_steps",
            "ppo_spider_final_previous_steps.zip",
        ),
        help="ruta al modelo PPO previous_steps",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dwa-horizon", type=int, default=6)
    parser.add_argument("--dwa-beam-width", type=int, default=128)
    parser.add_argument(
        "--dwa-exhaustive",
        action="store_true",
        help="usa busqueda exhaustiva en vez de beam search",
    )
    parser.add_argument(
        "--dwa-replan-each-step",
        action="store_true",
        help="DWA ejecuta solo la primera accion del plan y replantea",
    )
    parser.add_argument(
        "--continue-after-first-success",
        action="store_true",
        help="DWA sigue expandiendo hasta el horizonte aunque ya haya exito",
    )
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument(
        "--pause-between",
        type=float,
        default=1.0,
        help="segundos de pausa entre PPO y DWA",
    )
    parser.add_argument(
        "--target",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="target fijo para todos los episodios; si no se pasa, se samplea por seed",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="usa policy deterministica para PPO",
    )
    return parser.parse_args()


def make_env(args):
    return SpiderEnv(
        render_mode="human",
        max_steps=args.max_steps,
        calibration_path=args.calibration_path,
    )


def sample_targets(args):
    if args.target is not None:
        target = np.array(args.target, dtype=np.float64)
        return [target.copy() for _ in range(args.episodes)]

    probe = make_env(args)
    rng = np.random.default_rng(args.seed)
    half_x = float(probe.world_size_x) / 2.0
    half_y = float(probe.world_size_y) / 2.0
    success_radius = float(probe.success_radius)
    targets = []

    while len(targets) < args.episodes:
        target = np.array(
            [rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)],
            dtype=np.float64,
        )
        if np.linalg.norm(target) > success_radius * 1.5:
            targets.append(target)

    probe.close()
    return targets


def run_ppo_episode(model, target, episode, args):
    env = make_env(args)
    np.random.seed(args.seed + episode)
    obs, _ = env.reset(
        seed=args.seed + episode,
        options={"target_init_pos": np.asarray(target, dtype=np.float32)},
    )
    env.render()
    done = False
    episode_return = 0.0
    actions = []
    info = {}
    start_time = time.perf_counter()

    while not done:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        actions.append(action)
        done = terminated or truncated

    stats = build_episode_stats("PPO", env, episode, target, episode_return, actions, info, start_time)
    print_episode_result(stats)
    env.close()
    return stats


def run_dwa_episode(controller, target, episode, args):
    env = make_env(args)
    np.random.seed(args.seed + episode)
    obs, _ = env.reset(
        seed=args.seed + episode,
        options={"target_init_pos": np.asarray(target, dtype=np.float32)},
    )
    env.render()
    done = False
    info = {}
    episode_return = 0.0
    actions = []
    start_time = time.perf_counter()

    while not done:
        plan = controller.plan(env)
        planned_actions = list(plan.sequence)
        if args.dwa_replan_each_step:
            planned_actions = planned_actions[:1]
        if not planned_actions:
            break

        for action in planned_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            actions.append(int(action))
            done = terminated or truncated
            if done or args.dwa_replan_each_step:
                break

    stats = build_episode_stats("DWA", env, episode, target, episode_return, actions, info, start_time)
    print_episode_result(stats)
    env.close()
    return stats


def build_episode_stats(label, env, episode, target, episode_return, actions, info, start_time):
    distance = float(np.linalg.norm(env.target_pos))
    collision = bool(info.get("collision", False))
    success = distance <= env.success_radius and not collision
    return EpisodeStats(
        label=label,
        episode=episode,
        target=np.asarray(target, dtype=np.float64).copy(),
        success=success,
        collision=collision,
        steps=int(env.step_count),
        episode_return=float(episode_return),
        final_distance=distance,
        wall_time_sec=time.perf_counter() - start_time,
        actions=list(actions),
    )


def print_episode_result(stats):
    action_names = [ACTION_NAMES.get(action, str(action)) for action in stats.actions]
    print(
        f"{stats.label}: success={int(stats.success)} steps={stats.steps} "
        f"return={stats.episode_return:.3f} final_dist={stats.final_distance:.3f} "
        f"actions={action_names}"
    )


def print_comparison(results):
    if not results:
        return

    print("\nComparativa final")
    print(
        "method episodes success_rate collision_rate avg_steps "
        "avg_return avg_final_dist avg_wall_time_sec"
    )
    for label in sorted({result.label for result in results}):
        rows = [result for result in results if result.label == label]
        n = len(rows)
        success_rate = sum(row.success for row in rows) / n
        collision_rate = sum(row.collision for row in rows) / n
        avg_steps = float(np.mean([row.steps for row in rows]))
        avg_return = float(np.mean([row.episode_return for row in rows]))
        avg_final_dist = float(np.mean([row.final_distance for row in rows]))
        avg_wall_time = float(np.mean([row.wall_time_sec for row in rows]))
        print(
            f"{label:>6} {n:8d} {success_rate:12.3f} {collision_rate:14.3f} "
            f"{avg_steps:9.2f} {avg_return:10.3f} {avg_final_dist:14.3f} "
            f"{avg_wall_time:17.4f}"
        )

    print("\nDiferencia por episodio (DWA - PPO)")
    print("episode target steps_delta return_delta final_dist_delta")
    for episode in sorted({result.episode for result in results}):
        ppo = next((result for result in results if result.episode == episode and result.label == "PPO"), None)
        dwa = next((result for result in results if result.episode == episode and result.label == "DWA"), None)
        if ppo is None or dwa is None:
            continue
        target = np.array2string(ppo.target, precision=3, separator=",")
        print(
            f"{episode + 1:7d} {target:>16} "
            f"{dwa.steps - ppo.steps:11d} "
            f"{dwa.episode_return - ppo.episode_return:12.3f} "
            f"{dwa.final_distance - ppo.final_distance:16.3f}"
        )


def main():
    args = parse_args()
    patch_numpy_pickle_aliases()

    from stable_baselines3 import PPO

    model = PPO.load(args.ppo_model)
    beam_width = None if args.dwa_exhaustive else args.dwa_beam_width
    controller = DiscreteTreeSearchController(
        horizon=args.dwa_horizon,
        beam_width=beam_width,
        stop_on_success_depth=not args.continue_after_first_success,
    )
    targets = sample_targets(args)
    results = []

    for episode, target in enumerate(targets):
        print(f"\nEpisode {episode + 1}/{len(targets)} target={target.tolist()}")
        results.append(run_ppo_episode(model, target, episode, args))
        time.sleep(args.pause_between)
        results.append(run_dwa_episode(controller, target, episode, args))
        time.sleep(args.pause_between)

    print_comparison(results)


if __name__ == "__main__":
    main()
