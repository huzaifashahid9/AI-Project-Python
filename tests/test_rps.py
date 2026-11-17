"""Tests for the game rules."""

import pytest

from rps import LOSS, PAPER, PAYOFF, ROCK, SCISSORS, TIE, WIN, counter_move, outcome, parse_move


def test_counter_move_beats_every_move():
    for move in (ROCK, PAPER, SCISSORS):
        assert outcome(counter_move(move), move) == WIN


def test_outcome_table():
    assert outcome(ROCK, SCISSORS) == WIN
    assert outcome(ROCK, PAPER) == LOSS
    assert outcome(PAPER, PAPER) == TIE


def test_payoff_matrix_matches_outcome():
    for a in range(3):
        for b in range(3):
            assert PAYOFF[a, b] == outcome(a, b)


@pytest.mark.parametrize("text,expected", [("rock", ROCK), ("P", PAPER), (" scissors ", SCISSORS), ("2", SCISSORS), (1, PAPER)])
def test_parse_move_accepts_names_letters_and_codes(text, expected):
    assert parse_move(text) == expected


@pytest.mark.parametrize("bad", ["lizard", "", 3, -1, True, 1.0, None])
def test_invalid_moves_raise_value_error(bad):
    with pytest.raises(ValueError):
        parse_move(bad) if not isinstance(bad, (float, type(None))) else outcome(bad, ROCK)
