"""Charts for the experiments. Every function saves one labelled PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rps import MOVE_NAMES  # noqa: E402

CHANCE = 1.0 / 3.0
PREDICTOR_COLORS = {
    "Random": "#9e9e9e",
    "Frequency": "#4c78a8",
    "Markov-1": "#f58518",
    "Markov-2": "#e45756",
    "Markov-1J": "#b279a2",
    "HMM": "#54a24b",
}
MOVE_COLORS = ("#8c564b", "#1f77b4", "#d62728")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_win_rate_by_bot(summary: pd.DataFrame, path: Path) -> Path:
    """Grouped bars: mean AI win rate (± std) for each predictor against each bot."""
    bots = list(dict.fromkeys(summary["bot"]))
    predictors = list(dict.fromkeys(summary["predictor"]))
    x = np.arange(len(bots))
    width = 0.8 / len(predictors)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(predictors):
        rows = summary[summary["predictor"] == name].set_index("bot").loc[bots]
        ax.bar(x + (j - (len(predictors) - 1) / 2) * width, rows["win_mean"], width,
               yerr=rows["win_std"], capsize=2, label=name,
               color=PREDICTOR_COLORS.get(name))
    ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="Chance (1/3)")
    ax.set_xticks(x, bots)
    ax.set_xlabel("Opponent bot")
    ax.set_ylabel("AI win rate (mean ± std over games)")
    ax.set_ylim(0, 1)
    ax.set_title("AI win rate by predictor and opponent")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_learning_curves(curves: pd.DataFrame, path: Path) -> Path:
    """Cumulative win rate over rounds, one panel per bot."""
    bots = list(dict.fromkeys(curves["bot"]))
    fig, axes = plt.subplots(1, len(bots), figsize=(4 * len(bots), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bot in zip(axes, bots):
        sub = curves[curves["bot"] == bot]
        for name, grp in sub.groupby("predictor", sort=False):
            ax.plot(grp["round"], grp["win_rate"], label=name, color=PREDICTOR_COLORS.get(name))
        ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"vs {bot}")
        ax.set_xlabel("Round")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative AI win rate (mean over games)")
    axes[0].legend(fontsize=8, loc="upper left")  # the Random panel has empty space
    fig.suptitle("How quickly each predictor learns (dashed line = chance)")
    return _save(fig, path)


def plot_tournament_heatmap(matrix: pd.DataFrame, path: Path) -> Path:
    """Round-robin heatmap: cell (row, col) = win rate of row predictor vs column predictor."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.2, vmax=0.6)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Opponent predictor")
    ax.set_ylabel("Predictor")
    ax.set_title("Bot-vs-bot tournament: win rate of row vs column")
    fig.colorbar(im, ax=ax, label="Win rate")
    return _save(fig, path)


def plot_hmm_belief(trace: pd.DataFrame, path: Path) -> Path:
    """HMM filtered belief vs the true hidden strategy and the raw move signal."""
    rounds = trace["round"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 2, 2]})
    truth = trace["true_strategy"].to_numpy()
    axes[0].scatter(rounds, truth, c=[MOVE_COLORS[s] for s in truth], s=8, marker="s")
    axes[0].set_yticks(range(3), [f"favours {m}" for m in MOVE_NAMES], fontsize=8)
    axes[0].set_title("True hidden strategy of the Switching bot (never shown to the AI)")
    for m, name in enumerate(MOVE_NAMES):
        axes[1].plot(rounds, trace[f"belief_{name.lower()}"], color=MOVE_COLORS[m],
                     label=f"P(favours {name})")
        axes[2].plot(rounds, trace[f"raw_{name.lower()}"], color=MOVE_COLORS[m],
                     label=f"share of {name}", alpha=0.8)
    axes[1].set_ylabel("Filtered belief")
    axes[1].set_title("HMM forward-filter belief over the hidden strategy")
    axes[2].set_ylabel("Raw signal")
    axes[2].set_title("Raw signal: share of each move in the last 10 rounds")
    axes[2].set_xlabel("Round")
    for ax in axes[1:]:
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
        ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_markov_tables(tables: dict[str, np.ndarray], path: Path) -> Path:
    """Learned order-1 transition matrices P(next move | last move), one per bot."""
    fig, axes = plt.subplots(1, len(tables), figsize=(3.6 * len(tables), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (bot, table) in zip(axes, tables.items()):
        ax.imshow(table, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), MOVE_NAMES, fontsize=8)
        ax.set_yticks(range(3), MOVE_NAMES, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center",
                        color="white" if table[i, j] > 0.6 else "black", fontsize=9)
        ax.set_title(f"vs {bot}")
        ax.set_xlabel("Next move")
    axes[0].set_ylabel("Previous move")
    fig.suptitle("Markov-1 transition probabilities learned after one 200-round game")
    return _save(fig, path)


def plot_hmm_ablation(ablation: pd.DataFrame, path: Path) -> Path:
    """Win rate of the HMM predictor as the number of hidden states changes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bot, grp in ablation.groupby("bot", sort=False):
        ax.errorbar(grp["n_states"], grp["win_mean"], yerr=grp["win_std"], marker="o",
                    capsize=3, label=f"vs {bot}")
    ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="Chance (1/3)")
    ax.set_xticks(sorted(ablation["n_states"].unique()))
    ax.set_xlabel("Number of hidden states K")
    ax.set_ylabel("HMM win rate (mean ± std)")
    ax.set_ylim(0, 1)
    ax.set_title("Effect of the number of hidden strategies on the HMM")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, path)
