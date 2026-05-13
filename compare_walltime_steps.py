import argparse
import csv
import glob
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIRS = [
    os.path.join(CURRENT_DIR, "enviroments"),
    os.path.join(CURRENT_DIR, "zenviroments"),
]
for path in [CURRENT_DIR] + [env_dir for env_dir in ENV_DIRS if os.path.isdir(env_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dwa_baseline import DiscreteTreeSearchController
from enviroment import ENV_VARIANTS, get_env_class


@dataclass
class ModelSpec:
    label: str
    path: str
    training_steps: int | None


@dataclass
class DwaSpec:
    label: str
    horizon: int
    beam_width: int | None
    replan_each_step: bool
    continue_after_first_success: bool


@dataclass
class EpisodeResult:
    success: bool
    collision: bool
    steps: int
    episode_return: float
    wall_time_sec: float
    decision_times_sec: list[float]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evalua DWA/tree-search y PPO con los mismos targets, guarda CSV y "
            "grafica tiempo computacional de decision vs pasos promedio hasta el objetivo."
        )
    )
    parser.add_argument(
        "--env",
        choices=list(ENV_VARIANTS),
        default="sin_obstaculos",
        help="entorno estándar para PPO/DWA",
    )
    parser.add_argument(
        "--env-mode",
        choices=["standard", "previous_steps"],
        default="standard",
        help="standard usa zenviroments; previous_steps usa spiderJAR/enviroment_previous_steps.py",
    )
    parser.add_argument(
        "--ppo-model",
        action="append",
        default=[],
        help="ruta a un modelo PPO .zip; se puede pasar varias veces",
    )
    parser.add_argument(
        "--ppo-model-dir",
        default=None,
        help="directorio con checkpoints PPO .zip para evaluar",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50_000,
        help="si se usa --ppo-model-dir, evalua checkpoints múltiplos de este valor; 0 evalua todos",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="limita la cantidad de checkpoints PPO evaluados; 0 no limita",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="comparison_results")
    parser.add_argument(
        "--no-dwa",
        action="store_true",
        help="no evalua el baseline DWA/tree-search",
    )
    parser.add_argument("--dwa-horizon", type=int, default=6)
    parser.add_argument("--dwa-beam-width", type=int, default=128)
    parser.add_argument(
        "--dwa-horizons",
        default=None,
        help="lista separada por comas de horizontes DWA a comparar, por ejemplo: 3,6,9",
    )
    parser.add_argument(
        "--dwa-beam-widths",
        default=None,
        help=(
            "lista separada por comas de beam widths DWA a comparar; usar none, "
            "full o exhaustive para busqueda exhaustiva"
        ),
    )
    parser.add_argument(
        "--dwa-compare-replan",
        action="store_true",
        help="evalua cada configuracion DWA con y sin replanteo por step",
    )
    parser.add_argument(
        "--dwa-exhaustive",
        action="store_true",
        help="expande el árbol completo; puede ser muy lento",
    )
    parser.add_argument(
        "--dwa-replan-each-step",
        action="store_true",
        help="DWA ejecuta sólo la primera acción planificada y vuelve a planificar",
    )
    parser.add_argument(
        "--continue-after-first-success",
        action="store_true",
        help="DWA sigue expandiendo hasta horizon aunque encuentre una secuencia exitosa",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="sólo para env-mode=previous_steps",
    )
    parser.add_argument(
        "--no-previous-action",
        action="store_true",
        help="sólo para env-mode=previous_steps: no agrega one-hot del movimiento previo",
    )
    parser.add_argument(
        "--zero-previous-action",
        action="store_true",
        help="sólo para env-mode=previous_steps: agrega el one-hot pero lo fuerza siempre a cero",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="usa policy determinística para PPO",
    )
    parser.add_argument(
        "--warmup-decisions",
        type=int,
        default=3,
        help="inferencias PPO no medidas antes de evaluar, para evitar overhead de primera llamada",
    )
    return parser.parse_args()


def parse_training_steps(path):
    match = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
    return int(match.group(1)) if match else None


def discover_models(args):
    specs = []

    for path in args.ppo_model:
        specs.append(ModelSpec(label=ppo_label(path), path=path, training_steps=parse_training_steps(path)))

    if args.ppo_model_dir:
        paths = sorted(glob.glob(os.path.join(args.ppo_model_dir, "*.zip")))
        for path in paths:
            steps = parse_training_steps(path)
            is_final = "final" in os.path.basename(path)
            if args.checkpoint_every > 0 and steps is not None and steps % args.checkpoint_every != 0:
                continue
            if args.checkpoint_every > 0 and steps is None and not is_final:
                continue
            specs.append(ModelSpec(label=ppo_label(path), path=path, training_steps=steps))

    specs = sorted(
        specs,
        key=lambda spec: (
            math.inf if spec.training_steps is None else spec.training_steps,
            spec.label,
        ),
    )
    if args.max_models > 0 and len(specs) > args.max_models:
        idx = np.linspace(0, len(specs) - 1, args.max_models, dtype=int)
        specs = [specs[i] for i in np.unique(idx)]
    return specs


def ppo_label(path):
    steps = parse_training_steps(path)
    if steps is not None:
        return f"PPO {steps // 1000}k"
    return f"PPO {os.path.splitext(os.path.basename(path))[0]}"


def parse_int_list(value, option_name):
    values = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{option_name} debe contener enteros separados por coma: {value}") from exc
    if not values:
        raise ValueError(f"{option_name} no puede estar vacio")
    return values


def parse_beam_width_list(value):
    beams = []
    exhaustive_names = {"none", "full", "exhaustive", "all"}
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in exhaustive_names:
            beams.append(None)
            continue
        try:
            beams.append(int(item))
        except ValueError as exc:
            raise ValueError(
                "--dwa-beam-widths debe contener enteros o none/full/exhaustive"
            ) from exc
    if not beams:
        raise ValueError("--dwa-beam-widths no puede estar vacio")
    return beams


def discover_dwa_specs(args):
    if args.dwa_horizons is None and args.dwa_beam_widths is None and not args.dwa_compare_replan:
        beam_width = None if args.dwa_exhaustive else args.dwa_beam_width
        return [
            DwaSpec(
                label=dwa_label(args.dwa_horizon, beam_width, args.dwa_replan_each_step),
                horizon=args.dwa_horizon,
                beam_width=beam_width,
                replan_each_step=args.dwa_replan_each_step,
                continue_after_first_success=args.continue_after_first_success,
            )
        ]

    horizons = (
        parse_int_list(args.dwa_horizons, "--dwa-horizons")
        if args.dwa_horizons is not None
        else [args.dwa_horizon]
    )
    beam_widths = (
        parse_beam_width_list(args.dwa_beam_widths)
        if args.dwa_beam_widths is not None
        else [None if args.dwa_exhaustive else args.dwa_beam_width]
    )
    replan_modes = [False, True] if args.dwa_compare_replan else [args.dwa_replan_each_step]

    specs = []
    seen = set()
    for horizon in horizons:
        for beam_width in beam_widths:
            for replan_each_step in replan_modes:
                key = (horizon, beam_width, replan_each_step, args.continue_after_first_success)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(
                    DwaSpec(
                        label=dwa_label(horizon, beam_width, replan_each_step),
                        horizon=horizon,
                        beam_width=beam_width,
                        replan_each_step=replan_each_step,
                        continue_after_first_success=args.continue_after_first_success,
                    )
                )
    return specs


def dwa_label(horizon, beam_width, replan_each_step):
    beam = "full" if beam_width is None else str(beam_width)
    replan = "replan" if replan_each_step else "seq"
    return f"DWA H={horizon} B={beam} {replan}"


def build_env_config(args):
    if args.env_mode == "previous_steps":
        from spiderJAR.enviroment_previous_steps import SpiderEnv

        if args.no_previous_action and args.zero_previous_action:
            raise ValueError("Usar --no-previous-action o --zero-previous-action, no ambos.")

        env_kwargs = {
            "calibration_path": args.calibration_path,
            "include_previous_action": not args.no_previous_action,
            "zero_previous_action": args.zero_previous_action,
        }
        return SpiderEnv, env_kwargs

    return get_env_class(args.env), {}


def make_env(env_cls, env_kwargs, max_steps):
    return env_cls(render_mode=None, max_steps=max_steps, **env_kwargs)


def sample_targets(env_cls, env_kwargs, episodes, seed, max_steps):
    env = make_env(env_cls, env_kwargs, max_steps)
    rng = np.random.default_rng(seed)
    half_x = float(getattr(env, "world_size_x", env.world_size)) / 2.0
    half_y = float(getattr(env, "world_size_y", env.world_size)) / 2.0
    success_radius = float(env.success_radius)
    targets = []

    while len(targets) < episodes:
        target = np.array(
            [rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)],
            dtype=np.float64,
        )
        if np.linalg.norm(target) > success_radius * 1.5:
            targets.append(target)

    env.close()
    return targets


def reset_eval_env(env, target, seed):
    np.random.seed(seed)
    return env.reset(
        seed=seed,
        options={"target_init_pos": np.asarray(target, dtype=np.float32)},
    )


def evaluate_dwa(env_cls, env_kwargs, targets, args, spec):
    controller = DiscreteTreeSearchController(
        horizon=spec.horizon,
        beam_width=spec.beam_width,
        stop_on_success_depth=not spec.continue_after_first_success,
    )

    def policy(env, obs):
        plan = controller.plan(env)
        if not plan.sequence:
            return []
        if spec.replan_each_step:
            return [plan.sequence[0]]
        return list(plan.sequence)

    return evaluate_policy(
        spec.label,
        env_cls,
        env_kwargs,
        targets,
        args,
        policy,
        extra_fields={
            "controller_type": "DWA",
            "dwa_horizon": spec.horizon,
            "dwa_beam_width": "full" if spec.beam_width is None else spec.beam_width,
            "dwa_replan_each_step": int(spec.replan_each_step),
            "dwa_continue_after_first_success": int(spec.continue_after_first_success),
        },
    )


def evaluate_ppo(spec, env_cls, env_kwargs, targets, args):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 no está instalado, no puedo cargar PPO") from exc

    patch_numpy_pickle_aliases()
    probe_env = make_env(env_cls, env_kwargs, args.max_steps)
    custom_objects = {
        "observation_space": probe_env.observation_space,
        "action_space": probe_env.action_space,
    }
    probe_env.close()
    model = PPO.load(spec.path, custom_objects=custom_objects)
    warmup_ppo_model(model, env_cls, env_kwargs, args)

    def policy(env, obs):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        return [int(action)]

    return evaluate_policy(
        spec.label,
        env_cls,
        env_kwargs,
        targets,
        args,
        policy,
        model_path=spec.path,
        training_steps=spec.training_steps,
        extra_fields={"controller_type": "PPO"},
    )


def warmup_ppo_model(model, env_cls, env_kwargs, args):
    if args.warmup_decisions <= 0:
        return

    env = make_env(env_cls, env_kwargs, args.max_steps)
    try:
        obs, _ = env.reset(seed=args.seed)
        for _ in range(args.warmup_decisions):
            model.predict(obs, deterministic=args.deterministic)
    finally:
        env.close()


def patch_numpy_pickle_aliases():
    """Permite cargar modelos guardados con NumPy 2 en entornos con NumPy 1.x."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def evaluate_policy(
    label,
    env_cls,
    env_kwargs,
    targets,
    args,
    policy_fn,
    model_path="",
    training_steps=None,
    extra_fields=None,
):
    results = []

    for episode, target in enumerate(targets):
        env = make_env(env_cls, env_kwargs, args.max_steps)
        obs, _ = reset_eval_env(env, target, args.seed + episode)
        done = False
        info = {}
        episode_return = 0.0
        decision_times = []
        start_time = time.perf_counter()

        while not done:
            decision_start = time.perf_counter()
            actions = policy_fn(env, obs)
            decision_time = time.perf_counter() - decision_start
            decision_times.append(decision_time)
            if not actions:
                break

            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
                if done:
                    break

        wall_time = time.perf_counter() - start_time
        final_distance = float(np.linalg.norm(env.target_pos))
        collision = bool(info.get("collision", False))
        success = final_distance <= env.success_radius and not collision
        results.append(
            EpisodeResult(
                success=success,
                collision=collision,
                steps=int(env.step_count),
                episode_return=episode_return,
                wall_time_sec=wall_time,
                decision_times_sec=decision_times,
            )
        )
        env.close()

    return summarize(label, results, model_path, training_steps, extra_fields=extra_fields)


def summarize(label, results, model_path, training_steps, extra_fields=None):
    successes = [result for result in results if result.success]
    steps_all = np.array([result.steps for result in results], dtype=np.float64)
    steps_success = np.array([result.steps for result in successes], dtype=np.float64)
    wall_times = np.array([result.wall_time_sec for result in results], dtype=np.float64)
    decision_times = np.array(
        [
            decision_time
            for result in results
            for decision_time in result.decision_times_sec
        ],
        dtype=np.float64,
    )
    returns = np.array([result.episode_return for result in results], dtype=np.float64)
    total_steps = float(np.sum(steps_all))
    total_decision_time = float(np.sum(decision_times)) if len(decision_times) else 0.0
    n = max(len(results), 1)

    row = {
        "method": label,
        "controller_type": "",
        "dwa_horizon": "",
        "dwa_beam_width": "",
        "dwa_replan_each_step": "",
        "dwa_continue_after_first_success": "",
        "model_path": model_path,
        "training_steps": "" if training_steps is None else int(training_steps),
        "episodes": len(results),
        "successes": len(successes),
        "success_rate": len(successes) / n,
        "collision_rate": sum(result.collision for result in results) / n,
        "avg_steps_to_goal": float(np.mean(steps_success)) if len(steps_success) else "",
        "std_steps_to_goal": float(np.std(steps_success)) if len(steps_success) else "",
        "avg_steps_all": float(np.mean(steps_all)) if len(steps_all) else "",
        "std_steps_all": float(np.std(steps_all)) if len(steps_all) else "",
        "avg_wall_time_sec": float(np.mean(wall_times)) if len(wall_times) else "",
        "std_wall_time_sec": float(np.std(wall_times)) if len(wall_times) else "",
        "total_wall_time_sec": float(np.sum(wall_times)) if len(wall_times) else "",
        "decision_calls": int(len(decision_times)),
        "avg_decision_time_sec": float(np.mean(decision_times)) if len(decision_times) else "",
        "std_decision_time_sec": float(np.std(decision_times)) if len(decision_times) else "",
        "total_decision_time_sec": total_decision_time,
        "decision_time_per_step_sec": total_decision_time / total_steps if total_steps > 0 else "",
        "avg_reward": float(np.mean(returns)) if len(returns) else "",
        "std_reward": float(np.std(returns)) if len(returns) else "",
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(row):
    steps = row["avg_steps_to_goal"] if row["avg_steps_to_goal"] != "" else row["avg_steps_all"]
    steps_label = "avg_steps_to_goal" if row["avg_steps_to_goal"] != "" else "avg_steps_all"
    print(
        f"{row['method']}: "
        f"episodes={row['episodes']} "
        f"successes={row['successes']} "
        f"success_rate={row['success_rate']:.3f} "
        f"{steps_label}={float(steps):.2f} "
        f"avg_wall_time_sec={row['avg_wall_time_sec']:.4f} "
        f"decision_time_per_step_ms={float(row['decision_time_per_step_sec']) * 1000.0:.3f}"
    )


def plot_walltime_vs_steps(
    rows,
    output_dir,
    x_key="avg_wall_time_sec",
    filename="walltime_vs_steps.png",
    x_label="Wall time promedio por episodio [s]",
    title="Comparación wall time vs pasos",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        y = row["avg_steps_to_goal"] if row["avg_steps_to_goal"] != "" else row["avg_steps_all"]
        marker = "X" if row["method"].startswith("DWA") else "o"
        ax.scatter(row[x_key], y, s=80, marker=marker, label=row["method"])

    if len(rows) <= 15:
        for row in rows:
            y = row["avg_steps_to_goal"] if row["avg_steps_to_goal"] != "" else row["avg_steps_all"]
            ax.annotate(row["method"], (row[x_key], y), xytext=(5, 5), textcoords="offset points")

    x_values = [float(row[x_key]) for row in rows if row[x_key] != ""]
    if x_values and max(x_values) / max(min(x_values), 1e-9) > 20.0:
        ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Pasos promedio hasta el objetivo")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(rows) > 15:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=160)
    plt.close(fig)


def plot_ppo_checkpoints(rows, output_dir):
    ppo_rows = [
        row
        for row in rows
        if str(row["method"]).startswith("PPO") and row["training_steps"] != ""
    ]
    if len(ppo_rows) < 2:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ppo_rows.sort(key=lambda row: int(row["training_steps"]))
    x = [int(row["training_steps"]) for row in ppo_rows]
    y = [
        row["avg_steps_to_goal"] if row["avg_steps_to_goal"] != "" else row["avg_steps_all"]
        for row in ppo_rows
    ]
    success = [row["success_rate"] for row in ppo_rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, y, marker="o", label="Pasos hasta objetivo")
    ax1.set_xlabel("Timesteps de entrenamiento PPO")
    ax1.set_ylabel("Pasos promedio hasta el objetivo")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, success, color="tab:green", marker="s", alpha=0.7, label="Success rate")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0.0, 1.05)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("Evolución de PPO por checkpoint")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ppo_checkpoints_steps.png"), dpi=160)
    plt.close(fig)


def plot_ppo_progression_vs_dwa(rows, output_dir):
    ppo_rows = [
        row
        for row in rows
        if str(row["method"]).startswith("PPO") and row["training_steps"] != ""
    ]
    if len(ppo_rows) < 2:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ppo_rows.sort(key=lambda row: int(row["training_steps"]))
    x = np.array([int(row["training_steps"]) for row in ppo_rows], dtype=np.float64)
    ppo_steps = np.array(
        [
            row["avg_steps_to_goal"] if row["avg_steps_to_goal"] != "" else row["avg_steps_all"]
            for row in ppo_rows
        ],
        dtype=np.float64,
    )
    ppo_total_wall_time = np.array(
        [float(row["total_wall_time_sec"]) for row in ppo_rows],
        dtype=np.float64,
    )
    ppo_decision_time_per_step = np.array(
        [float(row["decision_time_per_step_sec"]) * 1000.0 for row in ppo_rows],
        dtype=np.float64,
    )

    dwa_row = best_dwa_row(rows)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(x, ppo_steps, color="tab:blue", marker="o", label="PPO")
    axes[0].set_ylabel("Pasos promedio")
    axes[0].set_title("Progresión de PPO contra DWA")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, ppo_total_wall_time, color="tab:green", marker="o", label="PPO")
    axes[1].set_ylabel("Wall time total [s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, ppo_decision_time_per_step, color="tab:purple", marker="o", label="PPO")
    axes[2].set_xlabel("Timesteps de entrenamiento PPO")
    axes[2].set_ylabel("Decisión por step [ms]")
    axes[2].grid(True, alpha=0.3)

    if dwa_row is not None:
        dwa_steps = dwa_row["avg_steps_to_goal"] if dwa_row["avg_steps_to_goal"] != "" else dwa_row["avg_steps_all"]
        dwa_total_wall_time = float(dwa_row["total_wall_time_sec"])
        dwa_decision_time_per_step = float(dwa_row["decision_time_per_step_sec"]) * 1000.0
        label = f"Mejor {dwa_row['method']}"
        axes[0].axhline(float(dwa_steps), color="tab:red", linestyle="--", label=label)
        axes[1].axhline(dwa_total_wall_time, color="tab:red", linestyle="--", label=label)
        axes[2].axhline(dwa_decision_time_per_step, color="tab:red", linestyle="--", label=label)

    for ax in axes:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ppo_progression_vs_dwa.png"), dpi=160)
    plt.close(fig)


def best_dwa_row(rows):
    dwa_rows = [row for row in rows if str(row["method"]).startswith("DWA")]
    if not dwa_rows:
        return None

    def score(row):
        steps = row["avg_steps_to_goal"] if row["avg_steps_to_goal"] != "" else row["avg_steps_all"]
        decision_time = row["decision_time_per_step_sec"]
        return (float(steps), float(decision_time))

    return min(dwa_rows, key=score)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    env_cls, env_kwargs = build_env_config(args)
    targets = sample_targets(env_cls, env_kwargs, args.episodes, args.seed, args.max_steps)
    model_specs = discover_models(args)
    dwa_specs = [] if args.no_dwa else discover_dwa_specs(args)

    rows = []
    for spec in dwa_specs:
        print(f"Evaluando {spec.label}...")
        row = evaluate_dwa(env_cls, env_kwargs, targets, args, spec)
        rows.append(row)
        print_summary(row)

    for spec in model_specs:
        print(f"Evaluando {spec.label}: {spec.path}")
        row = evaluate_ppo(spec, env_cls, env_kwargs, targets, args)
        rows.append(row)
        print_summary(row)

    if not rows:
        raise SystemExit("No hay nada para evaluar: pasá --ppo-model/--ppo-model-dir o quitá --no-dwa.")

    csv_path = os.path.join(args.output_dir, "walltime_steps_metrics.csv")
    write_csv(rows, csv_path)
    plot_walltime_vs_steps(
        rows,
        args.output_dir,
        x_key="decision_time_per_step_sec",
        filename="walltime_vs_steps.png",
        x_label="Tiempo computacional de decisión por step [s]",
        title="Comparación tiempo de decisión vs pasos",
    )
    plot_walltime_vs_steps(
        rows,
        args.output_dir,
        x_key="total_wall_time_sec",
        filename="total_walltime_vs_steps.png",
        x_label="Wall time total de evaluación [s]",
        title="Comparación wall time total vs pasos",
    )
    plot_walltime_vs_steps(
        rows,
        args.output_dir,
        x_key="decision_time_per_step_sec",
        filename="decision_time_per_step_vs_steps.png",
        x_label="Tiempo de decisión por step [s]",
        title="Comparación tiempo de decisión por step vs pasos",
    )
    plot_ppo_checkpoints(rows, args.output_dir)
    plot_ppo_progression_vs_dwa(rows, args.output_dir)

    print(f"CSV guardado en {csv_path}")
    print(f"Grafico guardado en {os.path.join(args.output_dir, 'walltime_vs_steps.png')}")
    print(f"Grafico total guardado en {os.path.join(args.output_dir, 'total_walltime_vs_steps.png')}")
    print(f"Grafico decision guardado en {os.path.join(args.output_dir, 'decision_time_per_step_vs_steps.png')}")
    if len([row for row in rows if str(row['method']).startswith('PPO')]) > 1:
        print(f"Grafico PPO guardado en {os.path.join(args.output_dir, 'ppo_checkpoints_steps.png')}")
        print(f"Grafico progresion guardado en {os.path.join(args.output_dir, 'ppo_progression_vs_dwa.png')}")


if __name__ == "__main__":
    main()
