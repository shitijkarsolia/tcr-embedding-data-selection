"""Regenerate the four summary plots from the CV results CSVs.

Run:
    python bap_model/make_plots.py

Reads:
    results/catELMo_tcr_results.csv
    results/catELMo_epi_results.csv

Writes:
    results/plots/catELMo_split_comparison.png
    results/plots/catELMo_split_difference.png
    results/plots/catELMo_detailed_comparison.png
    results/plots/catELMo_performance_table.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ["auc", "accuracy", "precision", "recall", "f1_macro"]
METRIC_LABELS = {
    "auc": "AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_macro": "F1 (Macro)",
}

TCR_COLOR = "#4C9BE6"
EPI_COLOR = "#E9736A"
DIFF_COLOR = "#4CAF50"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


def load():
    tcr = pd.read_csv(RESULTS / "catELMo_tcr_results.csv", index_col=0)
    epi = pd.read_csv(RESULTS / "catELMo_epi_results.csv", index_col=0)
    return tcr, epi


def plot_split_comparison(tcr, epi):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        means = [tcr.loc[metric, "mean"], epi.loc[metric, "mean"]]
        stds = [tcr.loc[metric, "std"], epi.loc[metric, "std"]]
        bars = ax.bar(
            ["TCR Split", "EPI Split"],
            means,
            yerr=stds,
            color=[TCR_COLOR, EPI_COLOR],
            edgecolor="black",
            capsize=5,
            width=0.6,
        )
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{METRIC_LABELS[metric]} - catELMo", pad=12)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    axes[-1].axis("off")

    fig.suptitle(
        "catELMo: TCR Split vs Epitope Split (per metric)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "catELMo_split_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_difference(tcr, epi):
    diffs = [tcr.loc[m, "mean"] - epi.loc[m, "mean"] for m in METRICS]
    labels = [METRIC_LABELS[m] for m in METRICS]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        labels,
        diffs,
        color=DIFF_COLOR,
        edgecolor="black",
        width=0.65,
    )
    top = max(diffs)
    ax.set_ylim(0, top * 1.35)
    ax.bar_label(bars, fmt="+%.4f", padding=5, fontsize=11, fontweight="bold")
    ax.set_ylabel("Difference (TCR - Epitope)")
    ax.set_title(
        "Performance difference: TCR split vs epitope split\n(positive = TCR better)",
        pad=18,
        fontsize=14,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(OUT / "catELMo_split_difference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detailed_comparison(tcr, epi):
    x = np.arange(len(METRICS))
    width = 0.38

    tcr_means = [tcr.loc[m, "mean"] for m in METRICS]
    tcr_stds = [tcr.loc[m, "std"] for m in METRICS]
    epi_means = [epi.loc[m, "mean"] for m in METRICS]
    epi_stds = [epi.loc[m, "std"] for m in METRICS]

    fig, ax = plt.subplots(figsize=(14, 7))
    b1 = ax.bar(
        x - width / 2, tcr_means, width,
        yerr=tcr_stds, color=TCR_COLOR, edgecolor="black",
        label="TCR Split", capsize=4,
    )
    b2 = ax.bar(
        x + width / 2, epi_means, width,
        yerr=epi_stds, color=EPI_COLOR, edgecolor="black",
        label="Epitope Split", capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_xlabel("Metrics")
    ax.set_ylim(0, 1.12)
    ax.set_title("catELMo: TCR split vs epitope split performance", pad=14, fontsize=14)
    ax.bar_label(b1, fmt="%.3f", padding=4, fontsize=10)
    ax.bar_label(b2, fmt="%.3f", padding=4, fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "catELMo_detailed_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_performance_table(tcr, epi):
    headers = [METRIC_LABELS[m] for m in METRICS]
    rows = ["TCR Split", "Epitope Split"]

    def fmt(df, m):
        return f"{df.loc[m, 'mean']:.4f}\n\u00b1{df.loc[m, 'std']:.4f}"

    cells = [
        [fmt(tcr, m) for m in METRICS],
        [fmt(epi, m) for m in METRICS],
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_title(
        "catELMo performance summary: mean \u00b1 std (5 repeats \u00d7 5 folds)",
        pad=20,
        fontsize=15,
    )

    table = ax.table(
        cellText=cells,
        colLabels=headers,
        rowLabels=rows,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.4)

    header_color = "#4C9BE6"
    row_label_color = "#EEEEEE"
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor(row_label_color)
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "catELMo_performance_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    tcr, epi = load()
    plot_split_comparison(tcr, epi)
    plot_split_difference(tcr, epi)
    plot_detailed_comparison(tcr, epi)
    plot_performance_table(tcr, epi)
    print(f"wrote 4 plots to {OUT}")


if __name__ == "__main__":
    main()
