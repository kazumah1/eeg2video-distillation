"""Go/no-go figures from one mi_verdict.py output (run with --size-matched).

  python scripts/plot_erasure_summary.py results/verdicts_sm_10sub.json --out results/figures

crossover.png       extra top-1 damage vs size-matched shuffles, by repetition budget
                    (absolute, and as a fraction of baseline top-1)
control_ladder.png  the same effects measured against random, shuffled and
                    size-matched shuffled controls
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {"coarse_category": ("Category (semantic)", "#2a78d6"),
          "lowlevel": ("Low-level (edges + contrast + SF)", "#eb6834")}
CONTROLS = [("excess_top1", "Random subspace (L2-matched)"),
            ("excess_vs_shuffled_top1", "Concept-shuffled"),
            ("excess_vs_size_matched_shuffled_top1", "Concept-shuffled, size-matched")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def stat(verdicts, budget, prop, key):
    s = verdicts[budget][prop][key]
    return s["mean"], s["ci95"]


def crossover(verdicts, budgets, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    key = "excess_vs_size_matched_shuffled_top1"
    xs = range(len(budgets))
    for panel, ax in enumerate(axes):
        style(ax)
        for i, (prop, (label, color)) in enumerate(SERIES.items()):
            base = [verdicts[b][prop]["mean_top1"]["baseline"] for b in budgets]
            scale = base if panel else [1] * len(budgets)
            means, lo, hi = [], [], []
            for b, s in zip(budgets, scale):
                m, (l, h) = stat(verdicts, b, prop, key)
                means.append(m / s), lo.append((m - l) / s), hi.append((h - m) / s)
            x = [v + (i - 0.5) * 0.08 for v in xs]
            ax.errorbar(x, means, yerr=[lo, hi], color=color, linewidth=2, marker="o",
                        markersize=8, capsize=0, elinewidth=2,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=label)
            ax.annotate(f"{means[-1]:.3f}" if not panel else f"{means[-1]:.0%}",
                        (x[-1], means[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        if not panel:
            ax.axhline(0.02, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(-0.25, 0.022, "smallest effect of interest (0.02)", fontsize=8, color=MUTED)
        ax.set_xticks(list(xs), [f"{b} rep{'s' if b != '1' else ''}\nbaseline "
                                 f"{verdicts[b]['coarse_category']['mean_top1']['baseline']:.2f}"
                                 for b in budgets])
        ax.set_xlim(-0.35, len(budgets) - 0.5)
        ax.set_title(["Extra top-1 damage (absolute)", "As a fraction of baseline top-1"][panel],
                     fontsize=11, color=INK, loc="left")
        if panel:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle("Low-level matches semantic reliance when EEG is noisy; semantic leads when clean",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Erasure vs concept-shuffled controls rescaled to the same per-image edit size. "
             "Mean and 95% bootstrap CI over 10 subjects.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "crossover.png", dpi=200, facecolor=SURFACE)


def control_ladder(verdicts, budgets, out):
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 3.6),
                             sharey=True, facecolor=SURFACE)
    for ax, b in zip(axes, budgets):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for i, (prop, (label, color)) in enumerate(SERIES.items()):
            for j, (key, _) in enumerate(CONTROLS):
                m, (lo, hi) = stat(verdicts, b, prop, key)
                y = j + (i - 0.5) * 0.3
                ax.plot([lo, hi], [y, y], color=color, linewidth=2)
                ax.plot(m, y, "o", color=color, markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=2, label=label if j == 0 else None)
                ax.annotate(f"{m:+.3f}", (hi, y), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0.02, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yticks(range(len(CONTROLS)), [c for _, c in CONTROLS])
        ax.set_title(f"{b} rep{'s' if b != '1' else ''} (baseline top-1 "
                     f"{verdicts[b]['coarse_category']['mean_top1']['baseline']:.2f})",
                     fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Extra top-1 damage beyond control", fontsize=9, color=MUTED)
        ax.set_xlim(left=min(0, ax.get_xlim()[0]), right=ax.get_xlim()[1] * 1.15)
    axes[0].invert_yaxis()  # shared y: invert once; weakest control on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2,
               labelcolor=INK)
    fig.suptitle("Stricter controls shrink the semantic effect about 5x; the low-level effect survives",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out / "control_ladder.png", dpi=200, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    verdicts = json.loads(args.verdicts.read_text())["verdicts"]
    budgets = sorted(verdicts, key=int)
    args.out.mkdir(parents=True, exist_ok=True)
    crossover(verdicts, budgets, args.out)
    control_ladder(verdicts, [b for b in budgets if b != "1"], args.out)
    print(f"wrote {args.out}/crossover.png and {args.out}/control_ladder.png")


if __name__ == "__main__":
    main()
