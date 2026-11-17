"""Game engine: the ``Player`` interface, the ``Game`` environment and match results.

The environment is a repeated two-player game. Each round both players commit
to a move without seeing the other's, the moves are revealed, and each player
is told what the other played (its *percept*). The match ends after a fixed
number of rounds (the terminal test).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rps import LOSS, TIE, WIN, outcome, validate_move


class Player(ABC):
    """Anything that can take part in a match: a scripted bot, the AI or a human.

    Each player owns a private, seeded ``random.Random`` so that experiments are
    reproducible and two players never share a random stream.
    """

    def __init__(self, name: str, seed: int | None = None) -> None:
        self.name = name
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def seed(self) -> int | None:
        """Seed the player's random generator was created with."""
        return self._seed

    def reset(self) -> None:
        """Forget everything learned and restart the random stream."""
        self._rng = random.Random(self._seed)

    @abstractmethod
    def next_move(self) -> int:
        """Choose the move for the coming round (0, 1 or 2)."""

    def observe(self, my_move: int, opponent_move: int) -> None:
        """Receive the percept after a round. Default: ignore it."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, seed={self._seed})"


class HumanPlayer(Player):
    """A player whose move is supplied from outside (the Streamlit buttons)."""

    def __init__(self, name: str = "Human") -> None:
        super().__init__(name, seed=None)
        self._pending: int | None = None

    def set_move(self, move: int) -> None:
        """Queue the move the human just chose."""
        self._pending = validate_move(move)

    def next_move(self) -> int:
        """Return the move queued by ``set_move`` (once)."""
        if self._pending is None:
            raise RuntimeError("HumanPlayer.next_move called before set_move")
        move, self._pending = self._pending, None
        return move


@dataclass
class RoundRecord:
    """What happened in one round, from player A's point of view."""

    index: int
    move_a: int
    move_b: int
    result: int  # WIN / TIE / LOSS for player A


@dataclass
class MatchResult:
    """The full record of a match, with summary statistics for player A."""

    name_a: str
    name_b: str
    moves_a: np.ndarray
    moves_b: np.ndarray
    results: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def n_rounds(self) -> int:
        """Number of rounds played."""
        return int(self.results.size)

    def _rate(self, value: int) -> float:
        return float(np.mean(self.results == value)) if self.n_rounds else 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of rounds won by player A (the performance measure)."""
        return self._rate(WIN)

    @property
    def tie_rate(self) -> float:
        """Fraction of tied rounds."""
        return self._rate(TIE)

    @property
    def loss_rate(self) -> float:
        """Fraction of rounds lost by player A."""
        return self._rate(LOSS)

    def cumulative_win_rate(self) -> np.ndarray:
        """Win rate of player A after each round (round 1 .. n)."""
        wins = np.cumsum(self.results == WIN)
        return wins / np.arange(1, self.n_rounds + 1)

    def rolling_win_rate(self, window: int = 20) -> np.ndarray:
        """Win rate over the last ``window`` rounds (shorter at the start)."""
        if window < 1:
            raise ValueError("window must be >= 1")
        wins = (self.results == WIN).astype(float)
        csum = np.concatenate(([0.0], np.cumsum(wins)))
        idx = np.arange(1, self.n_rounds + 1)
        start = np.maximum(0, idx - window)
        return (csum[idx] - csum[start]) / (idx - start)


class Game:
    """The environment: a match of ``n_rounds`` rounds between two players.

    State: the move history and the score. Terminal test: ``rounds_played ==
    n_rounds``. Utility for A: +1 win, 0 tie, -1 loss per round.
    """

    def __init__(self, player_a: Player, player_b: Player, n_rounds: int = 200) -> None:
        if n_rounds < 1:
            raise ValueError("n_rounds must be at least 1")
        self.player_a = player_a
        self.player_b = player_b
        self.n_rounds = n_rounds
        self.history: list[RoundRecord] = []

    @property
    def rounds_played(self) -> int:
        """Number of rounds completed so far."""
        return len(self.history)

    def is_over(self) -> bool:
        """Terminal test: has the match reached its round limit?"""
        return self.rounds_played >= self.n_rounds

    @property
    def score(self) -> tuple[int, int, int]:
        """``(wins_a, ties, wins_b)`` so far."""
        results = [r.result for r in self.history]
        return results.count(WIN), results.count(TIE), results.count(LOSS)

    def play_round(self) -> RoundRecord:
        """Play one round: both players commit, moves are revealed, both observe."""
        if self.is_over():
            raise RuntimeError("The match is over; start a new Game")
        # Player A commits first, but its choice depends only on past rounds,
        # so the order does not leak information between players.
        move_a = validate_move(self.player_a.next_move())
        move_b = validate_move(self.player_b.next_move())
        result = outcome(move_a, move_b)
        self.player_a.observe(move_a, move_b)
        self.player_b.observe(move_b, move_a)
        record = RoundRecord(self.rounds_played, move_a, move_b, result)
        self.history.append(record)
        return record

    def play(self) -> MatchResult:
        """Play the remaining rounds and return the match record."""
        while not self.is_over():
            self.play_round()
        return self.result()

    def result(self) -> MatchResult:
        """Snapshot of the match so far as a ``MatchResult``."""
        return MatchResult(
            name_a=self.player_a.name,
            name_b=self.player_b.name,
            moves_a=np.array([r.move_a for r in self.history], dtype=int),
            moves_b=np.array([r.move_b for r in self.history], dtype=int),
            results=np.array([r.result for r in self.history], dtype=int),
        )
