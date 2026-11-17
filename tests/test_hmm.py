"""Tests for the HMM predictor and the forward filter."""

import numpy as np
import pytest

from hmm_predictor import HMMPredictor, forward_filter, forward_step
from rps import PAPER, ROCK, SCISSORS
from tournament import hmm_belief_trace, play_match


def test_forward_step_is_bayes_rule():
    belief = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    # predict: (0.5, 0.5); correct with Rock: (0.4, 0.05) -> normalise
    assert np.allclose(forward_step(belief, A, B, ROCK), [0.4 / 0.45, 0.05 / 0.45])


def test_forward_filter_tracks_a_switch():
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    B = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    obs = [ROCK] * 10 + [SCISSORS] * 10
    beliefs = forward_filter(obs, np.array([0.5, 0.5]), A, B)
    assert beliefs[9, 0] > 0.9 and beliefs[-1, 1] > 0.9
    assert np.allclose(beliefs.sum(axis=1), 1.0)


def test_hmm_falls_back_before_min_history():
    hmm = HMMPredictor(min_history=20, seed=0)
    for _ in range(5):
        hmm.update(ROCK)
    assert not hmm.is_fitted
    assert hmm.predict() == ROCK


def test_hmm_learns_a_cycle():
    hmm = HMMPredictor(seed=0)
    for move in [ROCK, PAPER, SCISSORS] * 20:
        hmm.update(move)
    assert hmm.is_fitted
    assert hmm.predict() == ROCK
    assert np.allclose(hmm.transmat.sum(axis=1), 1.0)
    assert np.allclose(hmm.emission.sum(axis=1), 1.0)


def test_hmm_is_reproducible():
    a = play_match("HMM", "Switching", seed=5, n_rounds=80)
    b = play_match("HMM", "Switching", seed=5, n_rounds=80)
    assert np.array_equal(a.moves_a, b.moves_a)


def test_filtered_belief_beats_chance_at_finding_the_strategy():
    trace = hmm_belief_trace(seed=42, n_rounds=200)
    accuracy = np.mean(trace["filtered_guess"] == trace["true_strategy"])
    assert accuracy > 0.6


def test_bad_hmm_parameters():
    with pytest.raises(ValueError):
        HMMPredictor(n_states=0)
    with pytest.raises(ValueError):
        HMMPredictor(window=5, min_history=20)
