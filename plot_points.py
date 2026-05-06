import argparse
import numpy as np
from stable_baselines3 import PPO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enviroment import ENV_VARIANTS, get_env_class


def parse_args():
    parser = argparse.ArgumentParser(description="Genera un policy map para el modelo entrenado.")
    parser.add_argument(
        "--model-path",
        default="/home/facuvulcano/tpFInalRLSpider/models_spider/run_20251125_233759/ppo_spider_340000_steps.zip",
        help="ruta al modelo PPO a evaluar",
    )
    parser.add_argument(
        "--env",
        choices=list(ENV_VARIANTS),
        default="sin_obstaculos",
        help="elige el entorno: sin_obstaculos, obstaculos_sin_lidar o obstaculos_lidar",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_cls = get_env_class(args.env)

    model = PPO.load(args.model_path)
    env = env_cls(render_mode=None)
    mapX = np.arange(-4, 4.01, 0.02)
    mapY = np.arange(-4, 4.01, 0.02)
    policy_map = np.zeros((len(mapY), len(mapX)), dtype=int)

    for iy, y in enumerate(mapY):
        for ix, x in enumerate(mapX):
            obs, info = env.reset(options={"target_init_pos": np.array([x, y], dtype=np.float32)})
            action, _ = model.predict(obs, deterministic=True)
            policy_map[iy, ix] = action

    plt.figure(figsize=(8, 8))
    plt.imshow(policy_map, extent=[-4, 4, -4, 4], origin="lower", cmap="turbo")
    plt.colorbar(label="Acción PPO")
    plt.xlabel("x del target")
    plt.ylabel("y del target")
    plt.title("Policy map PPO")
    plt.savefig("policy_map.png", dpi=150)
    print("Guardado como policy_map.png")


if __name__ == "__main__":
    main()
