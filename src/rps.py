"""Rules of Rock-Paper-Scissors shared by every other module.

Moves are encoded as integers so they can index NumPy arrays directly:
``0 = Rock``, ``1 = Paper``, ``2 = Scissors``. With this encoding the move
that beats ``m`` is simply ``(m + 1) % 3`` and the result of a round is a
function of ``(a - b) % 3``.
"""

from __future__ import annotations

import numpy as np

ROCK, PAPER, SCISSORS = 0, 1, 2
MOVES: tuple[int, ...] = (ROCK, PAPER, SCISSORS)
N_MOVES: int = 3
MOVE_NAMES: tuple[str, ...] = ("Rock", "Paper", "Scissors")
MOVE_EMOJI: tuple[str, ...] = ("✊", "✋", "✌️")

WIN, TIE, LOSS = 1, 0, -1

# PAYOFF[a, b] is the utility for playing ``a`` against ``b``:
# +1 win, 0 tie, -1 loss (the utility function in the proposal).
PAYOFF: np.ndarray = np.array(
    [[0, -1, 1],
     [1, 0, -1],
     [-1, 1, 0]],
    dtype=float,
)

_ALIASES = {
    "r": ROCK, "rock": ROCK,
    "p": PAPER, "paper": PAPER,
    "s": SCISSORS, "scissors": SCISSORS,
}


def validate_move(move: int) -> int:
    """Return ``move`` unchanged if it is a legal move, else raise ``ValueError``."""
    if isinstance(move, bool) or not isinstance(move, (int, np.integer)):
        raise ValueError(f"A move must be an int in {MOVES}, got {move!r}")
    if move not in MOVES:
        raise ValueError(f"A move must be one of {MOVES}, got {move}")
    return int(move)


def parse_move(text: str | int) -> int:
    """Convert user input such as ``"rock"``, ``"R"`` or ``0`` into a move code.

    Raises:
        ValueError: if the input does not name a move.
    """
    if isinstance(text, (int, np.integer)) and not isinstance(text, bool):
        return validate_move(int(text))
    key = str(text).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if key.isdigit():
        return validate_move(int(key))
    raise ValueError(f"Unknown move {text!r}; use rock/paper/scissors or r/p/s")


def counter_move(move: int) -> int:
    """Return the move that beats ``move`` (Paper beats Rock, and so on)."""
    return (validate_move(move) + 1) % N_MOVES


def outcome(move_a: int, move_b: int) -> int:
    """Return the result for player A: ``WIN`` (+1), ``TIE`` (0) or ``LOSS`` (-1)."""
    diff = (validate_move(move_a) - validate_move(move_b)) % N_MOVES
    if diff == 0:
        return TIE
    return WIN if diff == 1 else LOSS


def move_name(move: int) -> str:
    """Human-readable name of a move code."""
    return MOVE_NAMES[validate_move(move)]
