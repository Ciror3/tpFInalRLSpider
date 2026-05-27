import argparse
import os
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tpFInalRLSpider.spiderJAR.enviroment_previous_steps_alignment_reward import (
    ACTION_METADATA,
    SpiderEnv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena PPO con previous_steps y reward que premia alineacion util aun al alejarse."
    )
    parser.add_argument("-n", "--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--no-previous-action", action="store_true")
    parser.add_argument("--zero-previous-action", action="store_true")
    parser.add_argument(
        "--orientation-backtrack-weight",
        type=float,
        default=0.04,
        help="peso de mejora angular cuando el paso aumenta la distancia al objetivo",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--policy-map-freq", type=int, default=20_000)
    parser.add_argument("--policy-map-limit", type=float, default=2.0)
    parser.add_argument("--policy-map-step", type=float, default=0.1)
    return parser.parse_args()


def make_env(seed: int = 0, **env_kwargs):
    def _init():
        env = SpiderEnv(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def build_vec_env(n_envs: int, seed: int, **env_kwargs):
    env_fns = [make_env(seed=seed + i, **env_kwargs) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


class AlignmentRewardPolicyMapCallback(BaseCallback):
    def __init__(
        self,
        freq=20_000,
        save_path="policy_maps_previous_steps_alignment_reward",
        env_kwargs=None,
        map_limit=2.0,
        map_step=0.1,
        verbose=1,
    ):
        super().__init__(verbose)
        self.freq = int(freq)
        self.save_path = save_path
        self.env_kwargs = env_kwargs or {}
        self.map_limit = float(map_limit)
        self.map_step = float(map_step)
        self.eval_env = None
        self.last_saved_timesteps = 0

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.eval_env = SpiderEnv(**self.env_kwargs)

    def _on_step(self) -> bool:
        if self.freq <= 0 or self.num_timesteps - self.last_saved_timesteps < self.freq:
            return True
        self.last_saved_timesteps = self.num_timesteps

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        previous_motions = [None] + [motion_name for _, motion_name in ACTION_METADATA.values()]
        coords = np.arange(-self.map_limit, self.map_limit + self.map_step / 2.0, self.map_step)

        for previous_motion in previous_motions:
            policy_map = np.zeros((len(coords), len(coords)), dtype=int)
            for iy, y in enumerate(coords):
                for ix, x in enumerate(coords):
                    options = {"target_init_pos": np.array([x, y], dtype=np.float32)}
                    if previous_motion is not None:
                        options["previous_motion"] = previous_motion

                    obs, _ = self.eval_env.reset(options=options)
                    action, _ = self.model.predict(obs, deterministic=True)
                    policy_map[iy, ix] = int(action)

            previous_label = "initial" if previous_motion is None else previous_motion
            filename = os.path.join(
                self.save_path,
                f"policy_map_step_{self.num_timesteps}_{previous_label}.png",
            )

            plt.figure(figsize=(8, 8))
            plt.imshow(
                policy_map,
                extent=[-self.map_limit, self.map_limit, -self.map_limit, self.map_limit],
                origin="lower",
                cmap="turbo",
            )
            plt.colorbar(label="Accion PPO")
            plt.xlabel("x del target")
            plt.ylabel("y del target")
            plt.title(f"Policy map alignment reward - prev={previous_label} - step {self.num_timesteps}")
            plt.savefig(filename, dpi=150)
            plt.close()

        if self.verbose > 0:
            print(f"[Callback] Guardados mapas de politica en {self.save_path}")

        return True

    def _on_training_end(self):
        if self.eval_env is not None:
            self.eval_env.close()


def main():
    args = parse_args()

    if args.n_envs < 1:
        raise ValueError("--n-envs debe ser mayor o igual a 1")
    if args.no_previous_action and args.zero_previous_action:
        raise ValueError("Usar --no-previous-action o --zero-previous-action, no ambos.")

    run_name = args.run_name or datetime.now().strftime("alignment_reward_%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs_previous_steps_alignment_reward", run_name)
    models_dir = os.path.join("models_previous_steps_alignment_reward", run_name)
    policy_maps_dir = os.path.join("policy_maps_previous_steps_alignment_reward", run_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(policy_maps_dir, exist_ok=True)

    env_kwargs = dict(
        render_mode=None,
        calibration_path=args.calibration_path,
        include_previous_action=not args.no_previous_action,
        zero_previous_action=args.zero_previous_action,
        orientation_backtrack_weight=args.orientation_backtrack_weight,
    )

    env = build_vec_env(n_envs=args.n_envs, seed=args.seed, **env_kwargs)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.000355207825977822,
        n_steps=256,
        batch_size=8,
        n_epochs=20,
        gamma=0.9791381032304535,
        gae_lambda=0.9975914694455162,
        clip_range=0.1,
        ent_coef=1.440070886288556e-06,
        vf_coef=0.5,
        max_grad_norm=0.7456096284142648,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            activation_fn=torch.nn.Tanh,
        ),
        verbose=args.verbose,
        tensorboard_log=log_dir,
        device=args.device,
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=models_dir,
        name_prefix="ppo_spider_alignment_reward",
    )
    policy_map_cb = AlignmentRewardPolicyMapCallback(
        freq=args.policy_map_freq,
        save_path=policy_maps_dir,
        env_kwargs=env_kwargs,
        map_limit=args.policy_map_limit,
        map_step=args.policy_map_step,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=int(args.total_timesteps),
            callback=[checkpoint_cb, policy_map_cb],
            progress_bar=True,
        )

        final_model_path = os.path.join(models_dir, "ppo_spider_final_alignment_reward")
        model.save(final_model_path)
        print(f"Modelo final guardado en: {final_model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
