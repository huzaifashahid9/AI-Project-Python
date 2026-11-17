"""The AI agent: a learning, model-based agent built around a ``Predictor``.

Each round the agent

1. asks its predictor for P(opponent's next move)  — its internal model,
2. picks the move with the highest expected utility under that distribution,
   ``EU(a) = sum_b P(b) * PAYOFF[a, b]``,
3. after the round, feeds the revealed moves back to the predictor (learning).
"""

from __future__ import annotations

import numpy as np

from game import Player
from hmm_predictor import HMMPredictor
from predictors import FrequencyPredictor, MarkovPredictor, Predictor, RandomPredictor
from rps import PAYOFF

PREDICTOR_NAMES: tuple[str, ...] = ("Random", "Frequency", "Markov-1", "Markov-2", "Markov-1J", "HMM")


def make_predictor(name: str, seed: int | None = None, **kwargs) -> Predictor:
    """Build a predictor from its display name.

    ``Markov-1J`` is an order-1 Markov chain whose context is the whole last
    round (opponent move and our move) instead of the opponent move alone.

    Raises:
        ValueError: if ``name`` is not one of ``PREDICTOR_NAMES``.
    """
    factories = {
        "Random": lambda: RandomPredictor(seed=seed),
        "Frequency": lambda: FrequencyPredictor(seed=seed, **kwargs),
        "Markov-1": lambda: MarkovPredictor(order=1, seed=seed, **kwargs),
        "Markov-2": lambda: MarkovPredictor(order=2, seed=seed, **kwargs),
        "Markov-1J": lambda: MarkovPredictor(order=1, context="joint", seed=seed, **kwargs),
        "HMM": lambda: HMMPredictor(seed=seed, **kwargs),
    }
    try:
        factory = factories[name]
    except KeyError:
        raise ValueError(f"Unknown predictor {name!r}; choose from {PREDICTOR_NAMES}") from None
    return factory()


class AIAgent(Player):
    """Plays the move that maximises expected utility against its predictor's forecast."""

    def __init__(self, predictor: Predictor, name: str | None = None, seed: int | None = None) -> None:
        super().__init__(name or f"AI[{predictor.name}]", seed)
        self.predictor = predictor
        self.last_prediction: np.ndarray | None = None
        self.last_expected_utility: np.ndarray | None = None

    def reset(self) -> None:
        """Reset the agent's random stream and its predictor."""
        super().reset()
        self.predictor.reset()
        self.last_prediction = None
        self.last_expected_utility = None

    def next_move(self) -> int:
        """Play the expected-utility best response to the predictor's forecast."""
        proba = self.predictor.predict_proba()
        expected_utility = PAYOFF @ proba
        best = np.flatnonzero(np.isclose(expected_utility, expected_utility.max()))
        self.last_prediction = proba
        self.last_expected_utility = expected_utility
        return int(self._rng.choice(list(best)))

    def observe(self, my_move: int, opponent_move: int) -> None:
        """Learning step: pass the round to the predictor."""
        self.predictor.update(opponent_move, my_move)


def make_agent(predictor_name: str, seed: int | None = None, **kwargs) -> AIAgent:
    """Convenience: an ``AIAgent`` around ``make_predictor(predictor_name, ...)``."""
    predictor = make_predictor(predictor_name, seed=seed, **kwargs)
    return AIAgent(predictor, name=predictor.name, seed=seed)
