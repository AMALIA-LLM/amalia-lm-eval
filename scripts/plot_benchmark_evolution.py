import argparse
import re
import os  # NEW

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_checkpoints(columns):
    checkpoint_data = []
    for idx, col in enumerate(columns):
        match = re.search(r"checkpoint-(\d+)", str(col))
        if match:
            checkpoint_data.append((int(match.group(1)), idx))  # (checkpoint_num, 0-based col idx in checkpoint_cols)
    checkpoint_data.sort(key=lambda x: x[0])
    ckpt_nums = [x[0] for x in checkpoint_data]
    ckpt_col_idxs = [x[1] for x in checkpoint_data]
    return ckpt_nums, ckpt_col_idxs


def _load_model_csv(path):
    df = pd.read_csv(path)
    metric_names = df.iloc[:, 0].astype(str).values
    checkpoint_cols = df.columns[1:]  # skip first column (metric names)
    ckpt_nums, ckpt_col_idxs = _extract_checkpoints(checkpoint_cols)

    if not ckpt_nums:
        raise ValueError(f"No checkpoint-<N> columns found in {path}")

    # Map metric -> values aligned to ckpt_nums
    by_metric = {}
    for row_idx, metric in enumerate(metric_names):
        row_vals = df.iloc[row_idx, 1:].values.astype(float)  # values for checkpoint_cols in their current order
        sorted_vals = row_vals[np.array(ckpt_col_idxs)]
        by_metric[metric] = sorted_vals

    return {
        "path": path,
        "df": df,
        "metrics": list(metric_names),
        "checkpoint_numbers": ckpt_nums,
        "values_by_metric": by_metric,
    }


def _default_model_name_from_csv_path(path: str) -> str:
    # Expect: .../<model_name>/<csv_file>.csv
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.basename(path).rsplit(".", 1)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark evolution over checkpoints; overlay one line per model (CSV)."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more CSV files, one per model (same format as performance_per_epoch.csv).",
    )
    parser.add_argument(
        "--name",
        nargs="*",
        default=None,
        help="Optional model names (same order as --csv). Defaults to file stem.",
    )
    parser.add_argument(
        "--out",
        default="metrics_evolution.png",
        help="Output image path (default: metrics_evolution.png).",
    )
    parser.add_argument(
        "--no-trend",
        action="store_true",
        help="Disable trend lines.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=18,  # bigger + more noticeable
        help="Global legend font size (default: 12).",
    )
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=4,
        help="Number of columns in the global legend (default: 4).",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Main line width (default: 2.0).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=4.5,
        help="Marker size (default: 4.5).",
    )
    args = parser.parse_args()

    models = [_load_model_csv(p) for p in args.csv]

    # Resolve display names
    if args.name is not None and len(args.name) not in (0, len(models)):
        raise ValueError("--name must be omitted or have the same count as --csv")
    if args.name and len(args.name) == len(models):
        model_names = list(args.name)
    else:
        model_names = [_default_model_name_from_csv_path(m["path"]) for m in models]

    # Union of metrics across models (preserve order from first model, then append unseen)
    metrics = []
    seen = set()
    for metric in models[0]["metrics"]:
        if metric not in seen:
            metrics.append(metric)
            seen.add(metric)
    for m in models[1:]:
        for metric in m["metrics"]:
            if metric not in seen:
                metrics.append(metric)
                seen.add(metric)

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    # Collect legend entries across all subplots
    legend_handles = {}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Collect y-range across models for this metric
        y_all = []

        for mi, m in enumerate(models):
            if metric not in m["values_by_metric"]:
                continue  # metric absent in this model's CSV

            ckpt_nums = m["checkpoint_numbers"]
            values = m["values_by_metric"][metric]
            y_all.append(values)

            color = None
            if colours:
                color = colours[mi % len(colours)]

            (line,) = ax.plot(
                ckpt_nums,
                values,
                marker="o",
                markersize=args.marker_size,
                linewidth=args.line_width,
                label=model_names[mi],  # used for global legend
                color=color,
            )
            # Ensure one handle per model in the global legend
            legend_handles.setdefault(model_names[mi], line)

            if not args.no_trend and len(ckpt_nums) >= 2:
                z = np.polyfit(ckpt_nums, values, 1)
                p = np.poly1d(z)
                ax.plot(
                    ckpt_nums,
                    p(ckpt_nums),
                    linestyle="--",
                    alpha=0.35,
                    linewidth=max(1.0, args.line_width * 0.6),
                    color=color,
                    label="_nolegend_",
                )

        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Score")
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Dynamic y-limits (per metric) so lines are easier to see
        if y_all:
            y_concat = np.concatenate([np.asarray(v, dtype=float) for v in y_all])
            y_min = float(np.nanmin(y_concat))
            y_max = float(np.nanmax(y_concat))

            if np.isfinite(y_min) and np.isfinite(y_max):
                if y_min == y_max:
                    pad = 0.05 if y_min == 0 else 0.02 * abs(y_min)
                else:
                    pad = 0.08 * (y_max - y_min)

                lo = y_min - pad
                hi = y_max + pad

                # If scores are in [0,1], keep within that range; otherwise allow full range.
                if y_min >= 0.0 and y_max <= 1.0:
                    lo = max(0.0, lo)
                    hi = min(1.0, hi)

                ax.set_ylim(lo, hi)

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    # Add a single global legend (unique labels) from collected handles
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=args.legend_ncol,
            frameon=True,          # make it stand out more
            fancybox=True,
            framealpha=0.9,
            fontsize=args.legend_fontsize,
            bbox_to_anchor=(0.5, 0.995),
            borderpad=0.8,
            labelspacing=0.6,
            handlelength=2.2,
            handletextpad=0.8,
        )
        plt.tight_layout(rect=[0, 0.0, 1, 0.90])  # reserve top room for bigger legend
    else:
        plt.tight_layout()

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved plot to '{args.out}'")
    plt.show()

    print("\n" + "=" * 80)
    print("METRIC SUMMARY (per model)")
    print("=" * 80)
    for mi, m in enumerate(models):
        print("\n" + "-" * 80)
        print(f"MODEL: {model_names[mi]}")
        print("-" * 80)

        for metric in m["metrics"]:
            values = m["values_by_metric"][metric]
            ckpt_nums = m["checkpoint_numbers"]

            best_idx = int(np.argmax(values))
            worst_idx = int(np.argmin(values))

            denom = values[0] if values[0] != 0 else np.nan
            improvement = ((values[-1] - values[0]) / denom * 100.0) if np.isfinite(denom) else np.nan

            print(f"\n{metric}:")
            print(f"  Best:  {values[best_idx]:.4f} at checkpoint-{ckpt_nums[best_idx]}")
            print(f"  Worst: {values[worst_idx]:.4f} at checkpoint-{ckpt_nums[worst_idx]}")
            print(f"  Final: {values[-1]:.4f}")
            if np.isfinite(improvement):
                print(f"  Improvement: {improvement:+.2f}%")
            else:
                print("  Improvement: n/a (initial value is 0)")


if __name__ == "__main__":
    main()
