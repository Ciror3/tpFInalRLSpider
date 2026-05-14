import argparse
import csv
import os
import tempfile


os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Une varios walltime_steps_metrics.csv generados por compare_walltime_steps.py "
            "y regenera graficos comparativos con todos los PPO y DWA juntos."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="archivos walltime_steps_metrics.csv a combinar",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_results/merged_walltime_steps",
        help="directorio donde guardar el CSV combinado y los graficos",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="conserva filas duplicadas exactas; por defecto se eliminan",
    )
    return parser.parse_args()


def read_rows(paths):
    rows = []
    fieldnames = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for fieldname in reader.fieldnames or []:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            for row in reader:
                row = dict(row)
                row["source_csv"] = path
                rows.append(row)

    if "source_csv" not in fieldnames:
        fieldnames.append("source_csv")
    return rows, fieldnames


def dedupe_rows(rows):
    unique = []
    seen = set()
    for row in rows:
        key = (
            row.get("method", ""),
            row.get("controller_type", ""),
            row.get("model_path", ""),
            row.get("training_steps", ""),
            row.get("dwa_horizon", ""),
            row.get("dwa_beam_width", ""),
            row.get("dwa_replan_each_step", ""),
            row.get("dwa_continue_after_first_success", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def steps_value(row):
    return numeric(row, "avg_steps_to_goal") or numeric(row, "avg_steps_all")


def plot_scatter(rows, output_dir, x_key, filename, x_label, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = []

    for row in rows:
        x = numeric(row, x_key)
        y = steps_value(row)
        if x is None or y is None:
            continue
        method = row.get("method", "")
        marker = "X" if method.startswith("H") else "o"
        ax.scatter(x, y, s=85, marker=marker, label=method)
        plotted.append((row, x, y))

    if len(plotted) <= 25:
        for row, x, y in plotted:
            ax.annotate(
                row.get("method", ""),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    x_values = [x for _, x, _ in plotted]
    if x_values and max(x_values) / max(min(x_values), 1e-9) > 20.0:
        ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if len(unique) <= 20:
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Pasos promedio hasta el objetivo")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows, fieldnames = read_rows(args.csv_files)
    if not args.keep_duplicates:
        rows = dedupe_rows(rows)

    combined_csv = os.path.join(args.output_dir, "walltime_steps_metrics_merged.csv")
    write_csv(rows, fieldnames, combined_csv)

    plot_scatter(
        rows,
        args.output_dir,
        x_key="decision_time_per_step_sec",
        filename="walltime_vs_steps_merged.png",
        x_label="Tiempo computacional de decisión por step [s]",
        title="Comparación combinada: tiempo de decisión vs pasos",
    )
    plot_scatter(
        rows,
        args.output_dir,
        x_key="total_wall_time_sec",
        filename="total_walltime_vs_steps_merged.png",
        x_label="Wall time total de evaluación [s]",
        title="Comparación combinada: wall time total vs pasos",
    )

    print(f"CSV combinado guardado en {combined_csv}")
    print(f"Grafico guardado en {os.path.join(args.output_dir, 'walltime_vs_steps_merged.png')}")
    print(f"Grafico total guardado en {os.path.join(args.output_dir, 'total_walltime_vs_steps_merged.png')}")


if __name__ == "__main__":
    main()
