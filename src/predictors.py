"""Predictors: models of the opponent that output P(opponent's next move).

Every predictor shares one interface (``Predictor``):

* ``update(opponent_move, my_move)`` — learn from the round just played;
* ``predict_proba()`` — a probability vector over Rock, Paper, Scissors;
* ``predict()`` — the single most likely move (ties broken at random).

The HMM predictor lives in ``hmm_predictor.py``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

from rps import N_MOVES, validate_move


class Predictor(ABC):
    """Base class for opponent models."""

    name: str = "Predictor"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self.rng = random.Random(seed)
        self.history: list[int] = []      # opponent's moves
        self.my_history: list[int] = []   # our own moves

    def reset(self) -> None:
        """Forget all observations and restart the random stream."""
        self.rng = random.Random(self._seed)
        self.history = []
        self.my_history = []

    def update(self, opponent_move: int, my_move: int | None = None) -> None:
        """Record one round and let the model learn from it."""
        opponent_move = validate_move(opponent_move)
        self.history.append(opponent_move)
        if my_move is not None:
            self.my_history.append(validate_move(my_move))
        self._learn(opponent_move)

    def _learn(self, opponent_move: int) -> None:
        """Hook for subclasses to update their model. Default: nothing."""

    @abstractmethod
    def predict_proba(self) -> np.ndarray:
        """Return P(next opponent move) as a length-3 array summing to 1."""

    def predict(self) -> int:
        """Return the most likely next opponent move, breaking ties at random."""
        proba = self.predict_proba()
        best = np.flatnonzero(np.isclose(proba, proba.max()))
        return int(self.rng.choice(list(best)))

    def describe(self) -> str:
        """One-line description used in tables and the app."""
        return self.name


def _smoothed_frequency(moves: list[int], alpha: float) -> np.ndarray:
    """Laplace-smoothed relative frequency of each move."""
    counts = np.bincount(np.asarray(moves, dtype=int), minlength=N_MOVES).astype(float)
    counts += alpha
    return counts / counts.sum()


class RandomPredictor(Predictor):
    """Baseline: assumes every move is equally likely, so the agent plays randomly.

    Expected win rate against any opponent is exactly 1/3.
    """

    name = "Random"

    def predict_proba(self) -> np.ndarray:
        """Always the uniform distribution (1/3, 1/3, 1/3)."""
        return np.full(N_MOVES, 1.0 / N_MOVES)


class FrequencyPredictor(Predictor):
    """Predicts the opponent's next move from how often each move was played.

    P(m) = (count(m) + alpha) / (n + 3 * alpha). It ignores order, so it
    catches a biased player but not a cyclic one.
    """

    name = "Frequency"

    def __init__(self, alpha: float = 1.0, seed: int | None = None) -> None:
        super().__init__(seed)
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha

    def predict_proba(self) -> np.ndarray:
        """Laplace-smoothed share of each move so far."""
        return _smoothed_frequency(self.history, self.alpha)


class MarkovPredictor(Predictor):
    """Order-k Markov chain over the opponent's moves.

    The state (context) is the opponent's last ``order`` moves. ``counts[ctx][m]``
    counts how often move ``m`` followed context ``ctx``. The transition
    probability is the smoothed count:

        P(m | ctx) = (counts[ctx][m] + alpha) / (sum(counts[ctx]) + 3 * alpha)

    ``decay`` < 1 multiplies all counts by ``decay`` before each update, so old
    rounds fade and the model can follow a player who changes habits.

    ``context="joint"`` makes each context symbol a whole round — the pair
    (opponent move, our move), 9 symbols — so the model can learn players who
    react to *our* moves (e.g. win-stay / lose-shift). The default
    ``"opponent"`` uses only the opponent's own moves, as in the proposal.
    """

    CONTEXTS = ("opponent", "joint")

    def __init__(
        self,
        order: int = 1,
        alpha: float = 1.0,
        decay: float = 1.0,
        context: str = "opponent",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        if order < 1:
            raise ValueError("order must be at least 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if context not in self.CONTEXTS:
            raise ValueError(f"context must be one of {self.CONTEXTS}")
        self.order = order
        self.alpha = alpha
        self.decay = decay
        self.context = context
        self.n_symbols = N_MOVES if context == "opponent" else N_MOVES * N_MOVES
        self.name = f"Markov-{order}" + ("J" if context == "joint" else "")
        self.counts = self._empty_counts()

    def _empty_counts(self) -> np.ndarray:
        return np.zeros((self.n_symbols,) * self.order + (N_MOVES,))

    def reset(self) -> None:
        """Forget all observations and empty the count table."""
        super().reset()
        self.counts = self._empty_counts()

    def update(self, opponent_move: int, my_move: int | None = None) -> None:
        """As ``Predictor.update``, but a joint context also needs our own move."""
        if self.context == "joint" and my_move is None:
            raise ValueError("A joint-context Markov predictor needs my_move in every update()")
        super().update(opponent_move, my_move)

    def _symbols(self) -> list[int]:
        """The sequence the context is read from: opponent moves, or round codes ``3*opp + mine``."""
        if self.context == "opponent":
            return self.history
        if len(self.my_history) != len(self.history):
            raise ValueError("A joint-context Markov predictor needs my_move in every update()")
        return [N_MOVES * o + m for o, m in zip(self.history, self.my_history)]

    def _context(self, symbols: list[int]) -> tuple[int, ...] | None:
        """The last ``order`` symbols, or ``None`` if not enough history yet."""
        if len(symbols) < self.order:
            return None
        return tuple(symbols[-self.order:])

    def _learn(self, opponent_move: int) -> None:
        # Context = the rounds *before* the move just observed.
        context = self._context(self._symbols()[:-1])
        if context is None:
            return
        if self.decay < 1.0:
            self.counts *= self.decay
        self.counts[context + (opponent_move,)] += 1.0

    def predict_proba(self) -> np.ndarray:
        """Smoothed P(next move | current context); frequency fallback early on."""
        context = self._context(self._symbols())
        if context is None:
            # Not enough history for a context: fall back to plain frequency.
            return _smoothed_frequency(self.history, self.alpha)
        row = self.counts[context] + self.alpha
        total = row.sum()
        if total <= 0:  # alpha == 0 and an unseen context
            return np.full(N_MOVES, 1.0 / N_MOVES)
        return row / total

    def transition_table(self) -> np.ndarray:
        """Smoothed transition matrix with one row per context.

        Shape ``(n_symbols ** order, 3)``; for order 2 the rows are ordered
        RR, RP, RS, PR, ...
        """
        flat = self.counts.reshape(-1, N_MOVES) + self.alpha
        sums = flat.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return flat / sums

    def describe(self) -> str:
        """Name plus smoothing (and decay, if used)."""
        extra = f", decay={self.decay}" if self.decay < 1.0 else ""
        return f"{self.name} (alpha={self.alpha}{extra})"
