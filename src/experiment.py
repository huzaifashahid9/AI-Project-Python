"""Reproduce every number and chart in the report with one command.

    python src/experiment.py --seed 42

Writes CSV tables to ``outputs/results/`` and PNG charts to ``outputs/figures/``.
Use ``--quick`` for a fast smoke run (10 games of 100 rounds).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import plots
from agent import PREDICTOR_NAMES
from bots import BOT_NAMES
from tournament import (
    evaluate_vs_bots,
    hmm_belief_trace,
    hmm_state_ablation,
    markov_table_after_game,
    round_robin,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Run the Rock-Paper-Scissors Mind Reader experiments.")
    parser.add_argument("--seed", type=int, default=42, help="base seed; game i uses seed + i (default 42)")
    parser.add_argument("--games", type=int, default=100, help="games per pairing (default 100)")
    parser.add_argument("--rounds", type=int, default=200, help="rounds per game (default 200)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs",
                        help="output folder (default: outputs/)")
    parser.add_argument("--quick", action="store_true", help="smoke run: 10 games of 100 rounds")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="parallel worker processes (results do not depend on this)")
    args = parser.parse_args(argv)
    if args.quick:
        args.games, args.rounds = 10, 100
    if args.games < 2:
        parser.error("--games must be at least 2 (needed for a standard deviation)")
    if args.rounds < 30:
        parser.error("--rounds must be at least 30 (the HMM needs 20 moves before its first fit)")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def results_table_markdown(summary: pd.DataFrame) -> str:
    """Pivot the summary into a 'predictor x bot' table of 'mean ± std' strings."""
    cell = summary.assign(
        cell=lambda d: (d["win_mean"] * 100).map("{:.1f}".format) + " ± "
        + (d["win_std"] * 100).map("{:.1f}".format)
    )
    pivot = cell.pivot(index="predictor", columns="bot", values="cell")
    pivot = pivot.loc[list(dict.fromkeys(summary["predictor"])), list(dict.fromkeys(summary["bot"]))]
    overall = summary.groupby("predictor", sort=False)["win_mean"].mean() * 100
    pivot["Mean"] = overall.map("{:.1f}".format)
    header = "| Predictor | " + " | ".join(pivot.columns) + " |"
    rule = "|---|" + "---|" * len(pivot.columns)
    lines = [header, rule]
    for name, row in pivot.iterrows():
        lines.append(f"| {name} | " + " | ".join(row.values) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, prepare folders and run all experiments."""
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    results_dir = args.out / "results"
    figures_dir = args.out / "figures"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Cannot create output folders under {args.out}: {exc}", file=sys.stderr)
        return 1
    if args.jobs == 1:
        run_all(args, results_dir, figures_dir, pool=None)
        return 0
    # One worker per core, each single-threaded: the matrices are tiny, so
    # BLAS threads would only fight each other. Workers inherit this env.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        run_all(args, results_dir, figures_dir, pool=pool)
    return 0


def run_all(args: argparse.Namespace, results_dir: Path, figures_dir: Path,
            pool: Executor | None) -> None:
    """Run the five experiments and write every table and figure."""
    predictors, bots = list(PREDICTOR_NAMES), list(BOT_NAMES)
    start = time.perf_counter()
    print(f"Seed {args.seed}: {args.games} games x {args.rounds} rounds per pairing, {args.jobs} job(s)\n")

    print("[1/5] Predictors vs scripted bots")
    evaluation = evaluate_vs_bots(predictors, bots, args.games, args.rounds, args.seed, progress=print,
                                  pool=pool)
    evaluation.games.to_csv(results_dir / "predictor_vs_bot_games.csv", index=False)
    evaluation.summary.to_csv(results_dir / "predictor_vs_bot_summary.csv", index=False, float_format="%.4f")
    evaluation.curves.to_csv(results_dir / "learning_curves.csv", index=False, float_format="%.4f")
    table = results_table_markdown(evaluation.summary)
    (results_dir / "results_table.md").write_text(
        "AI win rate (%) — mean ± std over "
        f"{args.games} games of {args.rounds} rounds, seed {args.seed}\n\n{table}\n",
        encoding="utf-8",
    )
    plots.plot_win_rate_by_bot(evaluation.summary, figures_dir / "fig1_win_rate_by_bot.png")
    plots.plot_learning_curves(evaluation.curves, figures_dir / "fig2_learning_curves.png")

    print("\n[2/5] Round-robin tournament between predictors")
    matrix, rr_games = round_robin(predictors, args.games, args.rounds, args.seed,
                                   progress=print, pool=pool)
    matrix.to_csv(results_dir / "tournament_matrix.csv", float_format="%.4f")
    rr_games.to_csv(results_dir / "tournament_games.csv", index=False)
    plots.plot_tournament_heatmap(matrix, figures_dir / "fig3_tournament_heatmap.png")

    print("\n[3/5] HMM belief vs the Switching bot's true strategy")
    trace = hmm_belief_trace(args.seed, args.rounds)
    trace.to_csv(results_dir / "hmm_belief_trace.csv", index=False, float_format="%.4f")
    filtered_acc = float(np.mean(trace["filtered_guess"] == trace["true_strategy"]))
    raw_acc = float(np.mean(trace["raw_guess"] == trace["true_strategy"]))
    plots.plot_hmm_belief(trace, figures_dir / "fig4_hmm_belief.png")
    np.set_printoptions(precision=3, suppress=True)
    print(f"  strategy identified correctly: filtered belief {filtered_acc:.1%}, raw signal {raw_acc:.1%}")
    print(f"  learned transition matrix A:\n{trace.attrs['transmat']}")
    print(f"  learned emission matrix B (rows = states, cols = R/P/S):\n{trace.attrs['emission']}")

    print("\n[4/5] Markov-1 transition tables learned in one game")
    tables = {bot: markov_table_after_game(bot, args.seed, args.rounds) for bot in ("Biased", "Cyclic", "WSLS")}
    pd.concat(
        {bot: pd.DataFrame(t, index=["Rock", "Paper", "Scissors"], columns=["Rock", "Paper", "Scissors"])
         for bot, t in tables.items()}
    ).to_csv(results_dir / "markov1_transition_tables.csv", float_format="%.4f")
    plots.plot_markov_tables(tables, figures_dir / "fig5_markov_transitions.png")

    print("\n[5/5] HMM ablation: number of hidden states")
    ablation = hmm_state_ablation([2, 3, 4, 5], ["Cyclic", "Switching"],
                                  n_games=max(2, args.games // 3), n_rounds=args.rounds, base_seed=args.seed,
                                  pool=pool)
    ablation.to_csv(results_dir / "hmm_states_ablation.csv", index=False, float_format="%.4f")
    plots.plot_hmm_ablation(ablation, figures_dir / "fig6_hmm_states_ablation.png")

    with open(results_dir / "hmm_belief_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Seed {args.seed}, {args.rounds} rounds vs Switching bot\n")
        fh.write(f"HMM win rate in this game: {trace.attrs['win_rate']:.3f}\n")
        fh.write(f"Hidden strategy identified by filtered belief: {filtered_acc:.3f}\n")
        fh.write(f"Hidden strategy identified by raw 10-round signal: {raw_acc:.3f}\n")

    print(f"\n{table}\n")
    print(f"Done in {time.perf_counter() - start:.0f}s. Tables: {results_dir}  Figures: {figures_dir}")


if __name__ == "__main__":
    sys.exit(main())
