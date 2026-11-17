"""Running many seeded matches: predictors vs bots, and a round-robin between predictors.

Seeding scheme (from the proposal): game ``i`` uses seed ``base_seed + i`` for
the bot, and ``base_seed + i + AGENT_SEED_OFFSET`` for the AI agent. Every
predictor therefore faces exactly the same opponent move streams, so
differences in win rate come from the predictors, not from luck.

Games are independent and fully seeded, so they can run in parallel processes
(pass a ``concurrent.futures`` executor as ``pool``) without changing any result.
"""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from agent import make_agent
from bots import SwitchingBot, make_bot
from game import Game, MatchResult

AGENT_SEED_OFFSET = 10_000
OPPONENT_SEED_OFFSET = 20_000

ProgressFn = Callable[[str], None]


@dataclass
class BotEvaluation:
    """Output of ``evaluate_vs_bots``."""

    games: pd.DataFrame          # one row per (predictor, bot, game)
    summary: pd.DataFrame        # mean/std per (predictor, bot)
    curves: pd.DataFrame         # mean cumulative win rate per round


def play_match(predictor_name: str, bot_name: str, seed: int, n_rounds: int,
               **predictor_kwargs) -> MatchResult:
    """One game of ``n_rounds`` between an AI agent and a scripted bot."""
    agent = make_agent(predictor_name, seed=seed + AGENT_SEED_OFFSET, **predictor_kwargs)
    bot = make_bot(bot_name, seed=seed)
    result = Game(agent, bot, n_rounds).play()
    result.extra["agent"] = agent
    result.extra["bot"] = bot
    return result


# ---------------------------------------------------------------- workers
# Top-level functions so they can be sent to worker processes.

def _bot_game(task: tuple) -> tuple[float, float, float, np.ndarray]:
    predictor_name, bot_name, seed, n_rounds, kwargs = task
    result = play_match(predictor_name, bot_name, seed, n_rounds, **kwargs)
    return result.win_rate, result.tie_rate, result.loss_rate, result.cumulative_win_rate()


def _pair_game(task: tuple) -> tuple[float, float, float]:
    name_a, name_b, seed, n_rounds = task
    agent_a = make_agent(name_a, seed=seed + AGENT_SEED_OFFSET)
    agent_b = make_agent(name_b, seed=seed + OPPONENT_SEED_OFFSET)
    result = Game(agent_a, agent_b, n_rounds).play()
    return result.win_rate, result.tie_rate, result.loss_rate


def _run(worker: Callable, tasks: list, pool: Executor | None) -> list:
    """Run ``worker`` over ``tasks`` serially, or on ``pool`` if given; keeps task order."""
    if pool is None:
        return [worker(t) for t in tasks]
    return list(pool.map(worker, tasks, chunksize=4))


# ------------------------------------------------------------ experiments

def evaluate_vs_bots(
    predictor_names: Iterable[str],
    bot_names: Iterable[str],
    n_games: int = 100,
    n_rounds: int = 200,
    base_seed: int = 42,
    progress: ProgressFn | None = None,
    pool: Executor | None = None,
) -> BotEvaluation:
    """Play every predictor against every bot ``n_games`` times."""
    rows, curve_rows = [], []
    for bot_name in bot_names:
        for predictor_name in predictor_names:
            tasks = [(predictor_name, bot_name, base_seed + i, n_rounds, {}) for i in range(n_games)]
            outputs = _run(_bot_game, tasks, pool)
            cumulative = np.zeros(n_rounds)
            for i, (win, tie, loss, curve) in enumerate(outputs):
                rows.append({"predictor": predictor_name, "bot": bot_name, "game": i,
                             "seed": base_seed + i, "win_rate": win, "tie_rate": tie,
                             "loss_rate": loss})
                cumulative += curve
            cumulative /= n_games
            curve_rows.extend(
                {"predictor": predictor_name, "bot": bot_name, "round": r + 1, "win_rate": w}
                for r, w in enumerate(cumulative)
            )
            if progress:
                progress(f"  {predictor_name:>9} vs {bot_name:<9} done")
    games = pd.DataFrame(rows)
    return BotEvaluation(games=games, summary=summarise(games, ["predictor", "bot"]),
                         curves=pd.DataFrame(curve_rows))


def summarise(games: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and standard deviation of win/tie/loss rates per group."""
    grouped = games.groupby(keys, sort=False)
    summary = grouped.agg(
        games=("win_rate", "size"),
        win_mean=("win_rate", "mean"),
        win_std=("win_rate", "std"),
        tie_mean=("tie_rate", "mean"),
        loss_mean=("loss_rate", "mean"),
    ).reset_index()
    summary["net_score"] = summary["win_mean"] - summary["loss_mean"]
    return summary


def round_robin(
    predictor_names: list[str],
    n_games: int = 100,
    n_rounds: int = 200,
    base_seed: int = 42,
    progress: ProgressFn | None = None,
    pool: Executor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bot-vs-bot tournament: every pair of predictors (including self-play).

    Returns:
        ``(win_matrix, games)`` where ``win_matrix.loc[a, b]`` is the mean
        win rate of predictor ``a`` against predictor ``b``.
    """
    matrix = pd.DataFrame(np.nan, index=predictor_names, columns=predictor_names)
    rows = []
    for name_a, name_b in combinations_with_replacement(predictor_names, 2):
        tasks = [(name_a, name_b, base_seed + i, n_rounds) for i in range(n_games)]
        outputs = _run(_pair_game, tasks, pool)
        for i, (win, tie, loss) in enumerate(outputs):
            rows.append({"a": name_a, "b": name_b, "game": i, "seed": base_seed + i,
                         "win_rate_a": win, "tie_rate": tie, "win_rate_b": loss})
        wins_a = np.mean([o[0] for o in outputs])
        wins_b = np.mean([o[2] for o in outputs])
        matrix.loc[name_a, name_b] = wins_a
        if name_a != name_b:
            matrix.loc[name_b, name_a] = wins_b
        if progress:
            progress(f"  {name_a:>9} vs {name_b:<9} done")
    return matrix, pd.DataFrame(rows)


def hmm_belief_trace(seed: int, n_rounds: int = 200, n_states: int = 3) -> pd.DataFrame:
    """Play the HMM agent against the switching bot and return, per round,
    the true hidden strategy, the HMM's filtered belief and the raw signal.

    HMM state labels are arbitrary, so each learned state is matched to the
    true strategy whose favoured move it emits most often.
    """
    agent = make_agent("HMM", seed=seed + AGENT_SEED_OFFSET, n_states=n_states)
    bot = SwitchingBot(seed=seed)
    result = Game(agent, bot, n_rounds).play()
    hmm = agent.predictor
    beliefs = hmm.filtered_beliefs()                   # (T, K)
    state_to_move = hmm.emission.argmax(axis=1)        # favoured move of each learned state
    # Belief that the player favours move m = total belief of the states favouring m.
    belief_by_move = np.zeros((n_rounds, 3))
    for state, move in enumerate(state_to_move):
        belief_by_move[:, move] += beliefs[:, state]
    moves = result.moves_b
    window = 10
    raw = np.zeros((n_rounds, 3))
    for t in range(n_rounds):
        recent = moves[max(0, t - window + 1): t + 1]
        raw[t] = np.bincount(recent, minlength=3) / recent.size
    frame = pd.DataFrame({
        "round": np.arange(1, n_rounds + 1),
        "move": moves,
        "true_strategy": bot.strategy_history,
        "belief_rock": belief_by_move[:, 0],
        "belief_paper": belief_by_move[:, 1],
        "belief_scissors": belief_by_move[:, 2],
        "raw_rock": raw[:, 0],
        "raw_paper": raw[:, 1],
        "raw_scissors": raw[:, 2],
    })
    frame["filtered_guess"] = belief_by_move.argmax(axis=1)
    frame["raw_guess"] = raw.argmax(axis=1)
    frame.attrs["win_rate"] = result.win_rate
    frame.attrs["transmat"] = hmm.transmat
    frame.attrs["emission"] = hmm.emission
    return frame


def hmm_state_ablation(
    state_counts: list[int],
    bot_names: list[str],
    n_games: int = 30,
    n_rounds: int = 200,
    base_seed: int = 42,
    pool: Executor | None = None,
) -> pd.DataFrame:
    """Second configuration: how the number of hidden states K affects the HMM."""
    rows = []
    for bot_name in bot_names:
        for k in state_counts:
            tasks = [("HMM", bot_name, base_seed + i, n_rounds, {"n_states": k}) for i in range(n_games)]
            for i, (win, tie, loss, _) in enumerate(_run(_bot_game, tasks, pool)):
                rows.append({"n_states": k, "bot": bot_name, "game": i,
                             "win_rate": win, "tie_rate": tie, "loss_rate": loss})
    return summarise(pd.DataFrame(rows), ["n_states", "bot"])


def markov_table_after_game(bot_name: str, seed: int, n_rounds: int = 200, order: int = 1) -> np.ndarray:
    """The transition table a Markov predictor learned in one game against ``bot_name``."""
    result = play_match(f"Markov-{order}", bot_name, seed, n_rounds)
    return result.extra["agent"].predictor.transition_table()
