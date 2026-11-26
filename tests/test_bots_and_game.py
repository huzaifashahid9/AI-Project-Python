import numpy as np
import pytest

from bots import BOT_NAMES, BiasedBot, CyclicBot, SwitchingBot, WinStayLoseShiftBot, make_bot
from game import Game, HumanPlayer
from agent import make_agent
from rps import PAPER, ROCK, SCISSORS
from tournament import play_match


def test_cyclic_bot_without_noise_cycles():
    bot = CyclicBot(noise=0.0, seed=0)
    assert [bot.next_move() for _ in range(6)] == [ROCK, PAPER, SCISSORS] * 2


def test_biased_bot_frequencies():
    bot = BiasedBot(weights=(0.7, 0.2, 0.1), seed=0)
    moves = np.array([bot.next_move() for _ in range(5000)])
    assert abs(np.mean(moves == ROCK) - 0.7) < 0.03


def test_wsls_stays_after_win_and_shifts_after_loss():
    bot = WinStayLoseShiftBot(noise=0.0, seed=0)
    bot.observe(ROCK, SCISSORS)      # bot won with Rock
    assert bot.next_move() == ROCK
    bot.observe(ROCK, PAPER)         # bot lost to Paper -> play what beats Paper
    assert bot.next_move() == SCISSORS


def test_switching_bot_records_its_hidden_strategy():
    bot = SwitchingBot(seed=3)
    for _ in range(300):
        bot.next_move()
    assert len(bot.strategy_history) == 300
    assert len(set(bot.strategy_history)) > 1


def test_same_seed_same_match():
    first = play_match("Markov-1", "Biased", seed=7, n_rounds=100)
    second = play_match("Markov-1", "Biased", seed=7, n_rounds=100)
    assert np.array_equal(first.moves_a, second.moves_a)
    assert np.array_equal(first.moves_b, second.moves_b)


def test_reset_restarts_a_bot_exactly():
    bot = make_bot("Switching", seed=11)
    first = [bot.next_move() for _ in range(50)]
    bot.reset()
    assert [bot.next_move() for _ in range(50)] == first


def test_game_terminal_test_and_rates():
    game = Game(make_agent("Random", seed=0), make_bot("Random", seed=0), n_rounds=60)
    result = game.play()
    assert game.is_over()
    assert result.n_rounds == 60
    assert np.isclose(result.win_rate + result.tie_rate + result.loss_rate, 1.0)
    with pytest.raises(RuntimeError):
        game.play_round()


def test_human_player_must_choose_before_round():
    game = Game(make_agent("Random", seed=0), HumanPlayer(), n_rounds=5)
    with pytest.raises(RuntimeError):
        game.play_round()
    game.player_b.set_move(ROCK)
    assert game.play_round().move_b == ROCK


def test_rolling_win_rate_window():
    result = play_match("Random", "Random", seed=0, n_rounds=50)
    rolling = result.rolling_win_rate(10)
    assert rolling.shape == (50,)
    with pytest.raises(ValueError):
        result.rolling_win_rate(0)


@pytest.mark.parametrize("name", BOT_NAMES)
def test_markov_beats_chance_or_ties_it(name):
    rate = np.mean([play_match("Markov-1J", name, seed=s, n_rounds=200).win_rate for s in range(5)])
    assert rate > 0.28


def test_unknown_bot_name():
    with pytest.raises(ValueError):
        make_bot("Terminator")
