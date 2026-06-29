"""
Benchmark Results Visualizer — 3 figures
  Figure 1 : one subplot per raw task metric, models as lines
  Figure 2 : one subplot per aggregate category + one overall subplot, models as lines
  Figure 3 : score drop 8K→32K for every category, grouped bars per model

Usage:
  python plot_benchmarks.py                      # dark mode (default)
  python plot_benchmarks.py --light              # light mode
  python plot_benchmarks.py data.csv             # custom CSV, dark mode
  python plot_benchmarks.py data.csv --light     # custom CSV, light mode
"""

import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from model_task_pretty_names import turn_model_to_pretty_name


# ── Inline data ───────────────────────────────────────────────────────────────
DEFAULT_CSV = "results.csv"


# ── Column definitions ────────────────────────────────────────────────────────
TASK_COLS = {
    "ruler_niah_mk_2 ruler_recall":  "NIAH-MK2",
    "ruler_niah_mk_3 ruler_recall":  "NIAH-MK3",
    "ruler_niah_mv ruler_recall":    "NIAH-MV",
    "json_kv substring_exact_match": "JSON-KV",
    "nq substring_exact_match":      "NQ",
    "triviaqa substring_exact_match":"TriviaQA",
    "hotpotqa substring_exact_match":"HotpotQA",
    "popqa substring_exact_match":   "PopQA",
    "infbench_qa rougeL_f1":         "InfBench-QA",
    "infbench_choice exact_match":   "InfBench-Choice",
    "msmarco_rerank_psg NDCG@10":    "MSMARCO-Rerank",
    "trec_coarse exact_match":       "TREC-Coarse",
    "trec_fine exact_match":         "TREC-Fine",
    "banking77 exact_match":         "Banking77",
    "clinic150 exact_match":         "Clinic150",
    "nlu exact_match":               "NLU",
    "alce_asqa str_em":              "ASQA-StrEM",
    "alce_asqa citation_rec":        "ASQA-CitRec",
    "alce_asqa citation_prec":       "ASQA-CitPrec",
    "alce_qampari qampari_rec_top5": "QAMPARI-Rec",
    "alce_qampari citation_rec":     "QAMPARI-CitRec",
    "alce_qampari citation_prec":    "QAMPARI-CitPrec",
}

AGG_COLS   = ["Recall", "RAG", "ICL", "Cite", "Re-rank"]
CTX_ORDER  = [8192, 16384, 32768]
CTX_LABELS = {8192: "8K", 16384: "16K", 32768: "32K"}

# ── Themes ────────────────────────────────────────────────────────────────────
PALETTE = [
    "#00a878", "#e8315b", "#4a7c8e", "#d4870a", "#0097c4",
    "#b830d4", "#7aab1e", "#c9a000", "#3d68c4", "#d45880", "#7060d4",
]

THEMES = {
    "dark": {
        "fig_bg":      "#0b0c10",
        "ax_bg":       "#13151a",
        "ax_edge":     "#2a2d3a",
        "ax_label":    "#9ca3af",
        "tick":        "#6b7280",
        "text":        "#e8eaf0",
        "grid":        "#1e2130",
        "muted":       "#9ca3af",
        "legend_edge": "#2a2d3a",
        "suptitle":    "#e8eaf0",
        "zero_line":   "#3a3d4a",
    },
    "light": {
        "fig_bg":      "#f5f6fa",
        "ax_bg":       "#ffffff",
        "ax_edge":     "#d1d5db",
        "ax_label":    "#374151",
        "tick":        "#6b7280",
        "text":        "#111827",
        "grid":        "#e5e7eb",
        "muted":       "#6b7280",
        "legend_edge": "#d1d5db",
        "suptitle":    "#111827",
        "zero_line":   "#d1d5db",
    },
}


def apply_theme(theme):
    plt.rcParams.update({
        "figure.facecolor":  theme["fig_bg"],
        "axes.facecolor":    theme["ax_bg"],
        "axes.edgecolor":    theme["ax_edge"],
        "axes.labelcolor":   theme["ax_label"],
        "xtick.color":       theme["tick"],
        "ytick.color":       theme["tick"],
        "text.color":        theme["text"],
        "grid.color":        theme["grid"],
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlepad":     10,
        "axes.titlesize":    9,
        "axes.titleweight":  "bold",
        "legend.facecolor":  theme["ax_bg"],
        "legend.edgecolor":  theme["legend_edge"],
        "legend.labelcolor": theme["text"],
        "legend.fontsize":   7,
    })


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(DEFAULT_CSV)
    df.columns = df.columns.str.strip()
    df["input_max_length"] = df["input_max_length"].astype(int)
    # short model names for display
    df["model_short"] = df["model"].apply(turn_model_to_pretty_name)
    # mean aggregate score
    df["agg_score"] = df[AGG_COLS].mean(axis=1)
    return df


def model_color(model, all_models):
    return PALETTE[list(all_models).index(model) % len(PALETTE)]


def shared_legend_handles(df, all_models):
    return [
        plt.Line2D([0], [0], color=model_color(m, all_models),
                   linewidth=2, marker="o", markersize=4,
                   label=df.loc[df["model"] == m, "model_short"].iloc[0])
        for m in all_models
    ]


# ── Core: draw one line-per-model into an axes ────────────────────────────────
def _line_subplot(ax, df, col, title, theme, all_models):
    ctx_labels = [CTX_LABELS[c] for c in CTX_ORDER]  # ['8K', '16K', '32K']
    has_data = False
    for m in all_models:
        sub = df[df["model"] == m].sort_values("input_max_length")
        indices = []  # Plot against 0, 1, 2
        ys = []
        for i, c in enumerate(CTX_ORDER):
            row = sub[sub["input_max_length"] == c]
            if row.empty or pd.isna(row[col].values[0]):
                continue
            indices.append(i)  # Use the integer index from CTX_ORDER
            ys.append(row[col].values[0])
        if not ys:
            continue
        has_data = True
        # Plot using numeric indices instead of string labels
        ax.plot(indices, ys, marker="o", markersize=4, linewidth=1.6,
                color=model_color(m, all_models),
                label=df.loc[df["model"] == m, "model_short"].iloc[0])
    ax.set_title(title)
    ax.set_ylim(0, 105)
    # Explicitly set the ticks and labels based on the fixed order
    ax.set_xticks(range(len(ctx_labels)))
    ax.set_xticklabels(ctx_labels)
    ax.grid(True, axis="y")
    if not has_data:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color=theme["muted"], fontsize=9)


# ── Figure 1: raw task metrics ────────────────────────────────────────────────
def figure_raw_tasks(df, theme, theme_name, output_path):
    all_models = df["model"].unique()
    task_items = [(col, lbl) for col, lbl in TASK_COLS.items() if col in df.columns]
    n = len(task_items)
    ncols, nrows = 4, math.ceil(n / 4)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5),
                             facecolor=theme["fig_bg"])
    fig.suptitle("Figure 1 · Raw Task Metrics vs Context Length",
                 fontsize=14, fontweight="bold", color=theme["suptitle"], y=1.01)

    for i, (col, lbl) in enumerate(task_items):
        _line_subplot(axes.flat[i], df, col, lbl, theme, all_models)
    for j in range(n, len(axes.flat)):
        axes.flat[j].set_visible(False)

    fig.legend(handles=shared_legend_handles(df, all_models),
               loc="lower center", ncol=min(len(all_models), 5),
               framealpha=0.5, edgecolor=theme["legend_edge"],
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    out = f"{output_path}/fig1_raw_tasks_{theme_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    return fig


# ── Figure 2: aggregate categories + overall ─────────────────────────────────
def figure_categories(df, theme, theme_name, output_path):
    all_models = df["model"].unique()
    df = df.copy()
    df["Overall"] = df[AGG_COLS].mean(axis=1)
    cats = AGG_COLS + ["Overall"]

    ncols, nrows = 3, math.ceil(len(cats) / 3)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5),
                             facecolor=theme["fig_bg"])
    fig.suptitle("Figure 2 · Aggregate Category Scores vs Context Length",
                 fontsize=14, fontweight="bold", color=theme["suptitle"], y=1.01)

    for i, cat in enumerate(cats):
        _line_subplot(axes.flat[i], df, cat, cat, theme, all_models)
    for j in range(len(cats), len(axes.flat)):
        axes.flat[j].set_visible(False)

    fig.legend(handles=shared_legend_handles(df, all_models),
               loc="lower center", ncol=min(len(all_models), 5),
               framealpha=0.5, edgecolor=theme["legend_edge"],
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    out = f"{output_path}/fig2_categories_{theme_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    return fig


# ── Figure 3: degradation across all categories ───────────────────────────────
def figure_degradation(df, theme, theme_name, output_path):
    all_models = df["model"].unique()
    d8  = df[df["input_max_length"] == 8192].set_index("model")
    d32 = df[df["input_max_length"] == 32768].set_index("model")
    common = [m for m in all_models if m in d8.index and m in d32.index]

    avail_cats = [c for c in AGG_COLS
                  if d8.loc[common, c].notna().any() and d32.loc[common, c].notna().any()]

    n_cats, n_models = len(avail_cats), len(common)
    x     = np.arange(n_cats)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 2.2), 6),
                           facecolor=theme["fig_bg"])
    fig.suptitle("Figure 3 · Score Change 32K → 8K per Category  "
                 "(positive = improved, negative = degraded)",
                 fontsize=13, fontweight="bold", color=theme["suptitle"])

    # x positions = categories, grouped bars = models
    x      = np.arange(n_cats)
    width  = 0.8 / n_models

    for mi, m in enumerate(common):
        drops = []
        for cat in avail_cats:
            v8 = d8.loc[m, cat] if not pd.isna(d8.loc[m, cat]) else np.nan
            v32 = d32.loc[m, cat] if not pd.isna(d32.loc[m, cat]) else np.nan
            drops.append(v32 - v8 if not (np.isnan(v8) or np.isnan(v32)) else np.nan)

        offsets = x + mi * width - (n_models * width) / 2 + width / 2
        short   = df.loc[df["model"] == m, "model_short"].iloc[0]
        bars    = ax.bar(offsets, drops, width=width * 0.9,
                         color=model_color(m, all_models), label=short, alpha=0.88)

        for bar, val in zip(bars, drops):
            if np.isnan(val):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if val >= 0 else -1.8),
                    f"{val:+.1f}",
                    ha="center", va="bottom", fontsize=6,
                    color=theme["muted"], rotation=90)

    ax.axhline(0, color=theme["zero_line"], linewidth=1.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(avail_cats, fontsize=9)
    ax.set_ylabel("Score change  (32K − 8K)")
    ax.grid(True, axis="y")
    ax.legend(title="Model", framealpha=0.5, edgecolor=theme["legend_edge"],
              ncol=2)
    fig.tight_layout()

    out = f"{output_path}/fig3_degradation_{theme_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark Results Visualizer")
    parser.add_argument("csv",     nargs="?", default=None, help="Path to CSV file")
    parser.add_argument("--dark", action="store_true",     help="Use dark mode theme")
    args = parser.parse_args()

    theme_name = "dark" if args.dark else "light"
    theme      = THEMES[theme_name]
    apply_theme(theme)

    output_path = os.path.dirname(args.csv)

    df = load_data(args.csv)

    figure_raw_tasks(df, theme, theme_name, output_path)
    figure_categories(df, theme, theme_name, output_path)
    figure_degradation(df, theme, theme_name, output_path)

    plt.show()


if __name__ == "__main__":
    main()