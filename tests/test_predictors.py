
import numpy as np
import pytest

from agent import AIAgent, make_agent, make_predictor, PREDICTOR_NAMES
from predictors import FrequencyPredictor, MarkovPredictor, RandomPredictor
from rps import PAPER, ROCK, SCISSORS


def test_every_predictor_outputs_a_distribution():
    for name in PREDICTOR_NAMES:
        predictor = make_predictor(name, seed=0)
        for move in [ROCK, PAPER, SCISSORS] * 10:
            predictor.update(move, ROCK)
            proba = predictor.predict_proba()
            assert proba.shape == (3,)
            assert np.isclose(proba.sum(), 1.0)
            assert (proba >= 0).all()


def test_random_predictor_is_uniform():
    assert np.allclose(RandomPredictor().predict_proba(), 1 / 3)


def test_frequency_predictor_uses_smoothed_counts():
    predictor = FrequencyPredictor(alpha=1.0)
    for move in [ROCK, ROCK, PAPER]:
        predictor.update(move)
    # counts (2, 1, 0) + alpha 1 -> (3, 2, 1) / 6
    assert np.allclose(predictor.predict_proba(), [3 / 6, 2 / 6, 1 / 6])
    assert predictor.predict() == ROCK


def test_markov1_counts_transitions():
    predictor = MarkovPredictor(order=1, alpha=0.0)
    for move in [ROCK, PAPER, ROCK, PAPER, ROCK]:
        predictor.update(move)
    # After Rock the opponent always played Paper.
    assert np.allclose(predictor.predict_proba(), [0, 1, 0])
    assert predictor.counts[ROCK, PAPER] == 2
    assert predictor.counts[PAPER, ROCK] == 2


def test_markov2_learns_a_cycle_of_length_three():
    predictor = MarkovPredictor(order=2, alpha=0.5)
    for move in [ROCK, PAPER, SCISSORS] * 10:
        predictor.update(move)
    assert predictor.predict() == ROCK  # context (Paper, Scissors) -> Rock


def test_markov_decay_forgets_old_habits():
    slow, fast = MarkovPredictor(order=1, decay=1.0), MarkovPredictor(order=1, decay=0.8)
    old, new = [ROCK, PAPER] * 30, [ROCK, SCISSORS] * 5
    for move in old + new:
        slow.update(move)
        fast.update(move)
    slow.update(ROCK)
    fast.update(ROCK)
    assert slow.predict() == PAPER      # still dominated by the old habit
    assert fast.predict() == SCISSORS   # followed the change


def test_joint_markov_requires_own_moves():
    predictor = MarkovPredictor(order=1, context="joint")
    with pytest.raises(ValueError):
        predictor.update(ROCK)
    assert predictor.history == []  # rejected before any state changed
    predictor.update(ROCK, PAPER)
    assert predictor.history == [ROCK]


def test_markov_rejects_bad_parameters():
    with pytest.raises(ValueError):
        MarkovPredictor(order=0)
    with pytest.raises(ValueError):
        MarkovPredictor(decay=1.5)
    with pytest.raises(ValueError):
        MarkovPredictor(context="nonsense")


def test_unknown_predictor_name():
    with pytest.raises(ValueError):
        make_predictor("Oracle")


def test_agent_plays_expected_utility_best_response():
    agent = AIAgent(FrequencyPredictor(alpha=0.0), seed=0)
    for _ in range(5):
        agent.observe(PAPER, ROCK)  # the opponent keeps playing Rock
    assert agent.next_move() == PAPER


def test_agent_reset_clears_learning():
    agent = make_agent("Markov-1", seed=1)
    for _ in range(10):
        agent.observe(PAPER, ROCK)
    agent.reset()
    assert agent.predictor.history == []
    assert agent.predictor.counts.sum() == 0
