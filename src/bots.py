"""Scripted opponents with hidden strategies.

These generate the data for the experiments: every bot is seeded, so game
``i`` played with seed ``42 + i`` produces the same move sequence every time.
"""

from __future__ import annotations

from abc import abstractmethod

from game import Player
from rps import MOVES, N_MOVES, PAPER, ROCK, SCISSORS, counter_move, outcome, validate_move

BOT_NAMES: tuple[str, ...] = ("Random", "Biased", "Cyclic", "WSLS", "Switching")


class Opponent(Player):
    """Base class for scripted bots.

    ``next_move`` plays the bot's strategy, except that with probability
    ``noise`` it plays a uniformly random move instead (humans slip too).
    Subclasses implement ``strategy_move``.
    """

    def __init__(self, name: str, seed: int | None = None, noise: float = 0.0) -> None:
        super().__init__(name, seed)
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be between 0 and 1")
        self.noise = noise
        self.last_move: int | None = None
        self.last_opponent_move: int | None = None

    def reset(self) -> None:
        """Restart the random stream and forget the last round."""
        super().reset()
        self.last_move = None
        self.last_opponent_move = None

    def next_move(self) -> int:
        """Play the strategy's move, or a random move with probability ``noise``."""
        if self.noise > 0.0 and self._rng.random() < self.noise:
            return self._rng.randrange(N_MOVES)
        return validate_move(self.strategy_move())

    @abstractmethod
    def strategy_move(self) -> int:
        """The move this bot's strategy calls for."""

    def observe(self, my_move: int, opponent_move: int) -> None:
        """Remember both moves of the last round (used by reactive bots)."""
        self.last_move = my_move
        self.last_opponent_move = opponent_move


class RandomBot(Opponent):
    """Uniformly random: unbeatable in expectation (every predictor gets 1/3)."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__("Random", seed)

    def strategy_move(self) -> int:
        """A uniformly random move."""
        return self._rng.randrange(N_MOVES)


class BiasedBot(Opponent):
    """Plays moves with fixed, unequal probabilities (default 50% Rock)."""

    def __init__(self, weights: tuple[float, float, float] = (0.5, 0.3, 0.2), seed: int | None = None) -> None:
        super().__init__("Biased", seed)
        if len(weights) != N_MOVES or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("weights must be three non-negative numbers with a positive sum")
        self.weights = tuple(w / sum(weights) for w in weights)

    def strategy_move(self) -> int:
        """Sample a move with probabilities ``weights``."""
        return self._rng.choices(MOVES, weights=self.weights)[0]


class CyclicBot(Opponent):
    """Repeats a fixed cycle (default Rock -> Paper -> Scissors) with some noise."""

    def __init__(self, cycle: tuple[int, ...] = (ROCK, PAPER, SCISSORS), noise: float = 0.1,
                 seed: int | None = None) -> None:
        super().__init__("Cyclic", seed, noise)
        if not cycle:
            raise ValueError("cycle must contain at least one move")
        self.cycle = tuple(validate_move(m) for m in cycle)
        self._position = 0

    def reset(self) -> None:
        """Restart the random stream and go back to the start of the cycle."""
        super().reset()
        self._position = 0

    def strategy_move(self) -> int:
        """The next move of the cycle."""
        move = self.cycle[self._position % len(self.cycle)]
        self._position += 1
        return move


class WinStayLoseShiftBot(Opponent):
    """Win-stay / lose-shift: repeat a winning move; after a loss or tie, play
    the move that would have beaten the opponent's last move."""

    def __init__(self, noise: float = 0.1, seed: int | None = None) -> None:
        super().__init__("WSLS", seed, noise)

    def strategy_move(self) -> int:
        """Repeat after a win; otherwise play what beats the opponent's last move."""
        if self.last_move is None or self.last_opponent_move is None:
            return self._rng.randrange(N_MOVES)
        if outcome(self.last_move, self.last_opponent_move) == 1:
            return self.last_move
        return counter_move(self.last_opponent_move)


class SwitchingBot(Opponent):
    """A player whose hidden strategy changes over time — an HMM by construction.

    Strategies: "favours Rock", "favours Paper", "favours Scissors"; under
    strategy ``s`` the favoured move is played with probability ``strength``.
    After every round the bot switches to a different strategy with
    probability ``switch_prob``. The true strategy sequence is kept in
    ``strategy_history`` so the HMM's belief can be checked against it.
    """

    STRATEGIES: tuple[str, ...] = ("favours Rock", "favours Paper", "favours Scissors")

    def __init__(self, switch_prob: float = 0.05, strength: float = 0.8, seed: int | None = None) -> None:
        super().__init__("Switching", seed)
        if not 0.0 <= switch_prob <= 1.0 or not 0.0 <= strength <= 1.0:
            raise ValueError("switch_prob and strength must be between 0 and 1")
        self.switch_prob = switch_prob
        self.strength = strength
        self._init_strategy()

    def _init_strategy(self) -> None:
        self.strategy = self._rng.randrange(len(self.STRATEGIES))
        self.strategy_history: list[int] = []

    def reset(self) -> None:
        """Restart the random stream and draw a new initial strategy."""
        super().reset()
        self._init_strategy()

    @property
    def strategy_name(self) -> str:
        """Name of the strategy currently in use (hidden from the AI)."""
        return self.STRATEGIES[self.strategy]

    def strategy_move(self) -> int:
        """Play the current strategy's favoured move with probability ``strength``."""
        self.strategy_history.append(self.strategy)
        favoured = self.strategy  # strategy i favours move i
        if self._rng.random() < self.strength:
            return favoured
        return self._rng.choice([m for m in MOVES if m != favoured])

    def next_move(self) -> int:
        """Play a move, then switch strategy with probability ``switch_prob``."""
        move = super().next_move()
        if self._rng.random() < self.switch_prob:
            self.strategy = self._rng.choice(
                [s for s in range(len(self.STRATEGIES)) if s != self.strategy]
            )
        return move


def make_bot(name: str, seed: int | None = None) -> Opponent:
    """Build a bot from its display name.

    Raises:
        ValueError: if ``name`` is not one of ``BOT_NAMES``.
    """
    factories = {
        "Random": lambda: RandomBot(seed=seed),
        "Biased": lambda: BiasedBot(seed=seed),
        "Cyclic": lambda: CyclicBot(seed=seed),
        "WSLS": lambda: WinStayLoseShiftBot(seed=seed),
        "Switching": lambda: SwitchingBot(seed=seed),
    }
    try:
        return factories[name]()
    except KeyError:
        raise ValueError(f"Unknown bot {name!r}; choose from {BOT_NAMES}") from None
