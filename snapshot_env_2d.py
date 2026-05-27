import argparse
import os
import sys
import tempfile
import types

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ZENVIROMENTS_DIR = os.path.join(CURRENT_DIR, "zenviroments")
if ZENVIROMENTS_DIR not in sys.path:
    sys.path.insert(0, ZENVIROMENTS_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Guarda una imagen PNG del entorno 2D.")
    parser.add_argument(
        "--env-mode",
        choices=["standard", "previous_steps"],
        default="previous_steps",
        help="standard usa zenviroments; previous_steps usa spiderJAR/enviroment_previous_steps.py",
    )
    parser.add_argument(
        "--env",
        choices=["sin_obstaculos", "obstaculos_sin_lidar", "obstaculos_lidar"],
        default="sin_obstaculos",
        help="sólo para --env-mode standard",
    )
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="env_2d_target.png")
    return parser.parse_args()


def make_env(args):
    ensure_pygame_importable()

    if args.env_mode == "previous_steps":
        from spiderJAR.enviroment_previous_steps import SpiderEnv

        return SpiderEnv(render_mode=None)

    from enviroment import get_env_class

    return get_env_class(args.env)(render_mode=None)


def ensure_pygame_importable():
    try:
        import pygame  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["pygame"] = types.ModuleType("pygame")


def reset_env(env, args):
    options = None
    if args.target_x is not None or args.target_y is not None:
        if args.target_x is None or args.target_y is None:
            raise ValueError("Pasá --target-x y --target-y juntos.")
        options = {"target_init_pos": np.array([args.target_x, args.target_y], dtype=np.float32)}

    env.reset(seed=args.seed, options=options)


def plot_snapshot(env, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    half_x = float(getattr(env, "world_size_x", env.world_size)) / 2.0
    half_y = float(getattr(env, "world_size_y", env.world_size)) / 2.0
    target = np.asarray(env.target_pos, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.add_patch(
        Rectangle(
            (-half_x, -half_y),
            2.0 * half_x,
            2.0 * half_y,
            fill=False,
            linewidth=2.0,
            edgecolor="#555555",
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            float(env.success_radius),
            fill=False,
            linewidth=1.8,
            edgecolor="#2ca02c",
            label="Radio de exito",
        )
    )

    obstacles = np.asarray(getattr(env, "obstacles", np.zeros((0, 3))), dtype=np.float64)
    for idx, (ox, oy, radius) in enumerate(obstacles):
        ax.add_patch(
            Circle(
                (ox, oy),
                radius,
                color="#d95f02",
                alpha=0.35,
                label="Obstaculo" if idx == 0 else None,
            )
        )

    ax.scatter([0.0], [0.0], s=150, color="#1f77b4", edgecolor="black", zorder=4, label="Robot")
    ax.scatter([target[0]], [target[1]], s=150, color="#d62728", marker="*", zorder=5, label="Target")
    ax.plot([0.0, target[0]], [0.0, target[1]], color="#b59f00", linewidth=1.5, alpha=0.9)

    distance = float(np.linalg.norm(target))
    ax.annotate(
        f"Target ({target[0]:.2f}, {target[1]:.2f}) m\nDist: {distance:.2f} m",
        xy=(target[0], target[1]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_title("Entorno 2D con target")
    ax.set_xlabel("x relativo [m]")
    ax.set_ylabel("y relativo [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half_x - 0.25, half_x + 0.25)
    ax.set_ylim(-half_y - 0.25, half_y + 0.25)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    env = make_env(args)
    try:
        reset_env(env, args)
        plot_snapshot(env, args.output)
    finally:
        env.close()
    print(f"Imagen guardada en: {args.output}")


if __name__ == "__main__":
    main()
