"""Streamlit front end: play against the Mind Reader, watch bots fight, view the tournament.

Run with:

    streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agent import PREDICTOR_NAMES, AIAgent, make_agent
from bots import BOT_NAMES, SwitchingBot, make_bot
from game import Game, HumanPlayer, MatchResult
from hmm_predictor import HMMPredictor
from predictors import MarkovPredictor
from rps import LOSS, MOVE_EMOJI, MOVE_NAMES, TIE, WIN
from tournament import round_robin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
HUMAN_ROUND_LIMIT = 1000
CHANCE = 1.0 / 3.0


# ------------------------------------------------------------------ helpers

def predictor_options(name: str, key: str) -> dict:
    """Sidebar/inline widgets for the parameters of the chosen predictor."""
    if name == "HMM":
        return {
            "n_states": st.slider("Hidden strategies K", 2, 5, 3, key=f"{key}_k",
                                  help="Number of hidden states in the HMM"),
            "refit_every": st.slider("Refit every n rounds", 1, 30, 10, key=f"{key}_refit",
                                     help="How often Baum-Welch re-learns A and B"),
        }
    if name.startswith("Markov") or name == "Frequency":
        options = {"alpha": st.slider("Smoothing alpha", 0.0, 5.0, 1.0, 0.5, key=f"{key}_alpha",
                                      help="Laplace prior added to every count")}
        if name.startswith("Markov"):
            options["decay"] = st.slider("Memory decay", 0.80, 1.0, 1.0, 0.01, key=f"{key}_decay",
                                         help="1.0 = remember everything; lower = forget old rounds")
        return options
    return {}


def win_rate_frame(result: MatchResult, window: int = 20) -> pd.DataFrame:
    """Cumulative and rolling win rate with a chance line, indexed by round."""
    rounds = np.arange(1, result.n_rounds + 1)
    return pd.DataFrame({
        "AI win rate (cumulative)": result.cumulative_win_rate(),
        f"AI win rate (last {window})": result.rolling_win_rate(window),
        "Chance (1/3)": np.full(result.n_rounds, CHANCE),
    }, index=pd.Index(rounds, name="Round"))


def show_model_internals(agent: AIAgent) -> None:
    """Explain what the agent currently believes about its opponent."""
    predictor = agent.predictor
    proba = predictor.predict_proba()
    st.markdown("**Predicted next move of the opponent**")
    st.bar_chart(pd.Series(proba, index=MOVE_NAMES, name="Probability"))
    if isinstance(predictor, MarkovPredictor):
        table = predictor.transition_table()
        if predictor.context == "joint":
            labels = [f"{MOVE_NAMES[o][0]}/{MOVE_NAMES[m][0]}" for o in range(3) for m in range(3)]
            caption = "Rows: last round as opponent/AI move. Columns: opponent's next move."
        elif predictor.order == 1:
            labels = list(MOVE_NAMES)
            caption = "Rows: opponent's last move. Columns: opponent's next move."
        else:
            labels = ["".join(MOVE_NAMES[i][0] for i in np.unravel_index(r, (3,) * predictor.order))
                      for r in range(table.shape[0])]
            caption = f"Rows: opponent's last {predictor.order} moves. Columns: next move."
        if predictor.order <= 2:
            st.markdown("**Learned transition probabilities**")
            st.caption(caption)
            st.dataframe(pd.DataFrame(table, index=labels, columns=MOVE_NAMES).style.format("{:.2f}")
                         .background_gradient(cmap="Blues", vmin=0, vmax=1))
    elif isinstance(predictor, HMMPredictor):
        if not predictor.is_fitted:
            st.info(f"The HMM fits its first model after {predictor.min_history} rounds "
                    f"({len(predictor.history)} so far). Until then it uses move frequencies.")
            return
        states = [f"State {i}" for i in range(predictor.n_states)]
        st.markdown("**Belief over the hidden strategy** — P(state | moves so far)")
        st.bar_chart(pd.Series(predictor.belief, index=states, name="Belief"))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transition matrix A** (strategy switching)")
            st.dataframe(pd.DataFrame(predictor.transmat, index=states, columns=states)
                         .style.format("{:.2f}"))
        with col2:
            st.markdown("**Emission matrix B** (moves under each strategy)")
            st.dataframe(pd.DataFrame(predictor.emission, index=states, columns=MOVE_NAMES)
                         .style.format("{:.2f}"))
        st.caption(f"Baum-Welch fits so far: {predictor.n_fits}; "
                   f"log-likelihood of last fit: {predictor.log_likelihood:.1f}")


# --------------------------------------------------------------- play tab

def new_human_game(predictor_name: str, seed: int, options: dict) -> None:
    """Start a fresh human-vs-AI match and store it in the session."""
    agent = make_agent(predictor_name, seed=seed, **options)
    human = HumanPlayer("You")
    st.session_state.game = Game(agent, human, n_rounds=HUMAN_ROUND_LIMIT)
    st.session_state.agent = agent
    st.session_state.last_round = None
    st.session_state.last_prediction = None


def play_tab(predictor_name: str, seed: int, options: dict) -> None:
    """Human vs AI."""
    config = (predictor_name, seed, tuple(sorted(options.items())))
    if st.session_state.get("config") != config or "game" not in st.session_state:
        st.session_state.config = config
        new_human_game(predictor_name, seed, options)
    game: Game = st.session_state.game
    agent: AIAgent = st.session_state.agent

    st.write("Pick a move. The AI commits to its move **before** reading yours — "
             "it only uses your previous moves. Try to be random!")
    cols = st.columns(3)
    for move, col in enumerate(cols):
        if col.button(f"{MOVE_EMOJI[move]}  {MOVE_NAMES[move]}", key=f"move_{move}",
                      disabled=game.is_over()):
            game.player_b.set_move(move)
            st.session_state.last_round = game.play_round()
            st.session_state.last_prediction = agent.last_prediction

    record = st.session_state.get("last_round")
    if record is not None:
        verdict = {WIN: "🤖 The AI wins this round", TIE: "🤝 Tie", LOSS: "🎉 You win this round"}[record.result]
        guess = int(np.argmax(st.session_state.last_prediction))
        st.subheader(f"You: {MOVE_EMOJI[record.move_b]} {MOVE_NAMES[record.move_b]}  ·  "
                     f"AI: {MOVE_EMOJI[record.move_a]} {MOVE_NAMES[record.move_a]}  —  {verdict}")
        st.caption(f"The AI expected {MOVE_NAMES[guess]} "
                   f"(p = {st.session_state.last_prediction[guess]:.2f}).")

    ai_wins, ties, human_wins = game.score
    played = game.rounds_played
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rounds", played)
    m2.metric("AI wins", ai_wins)
    m3.metric("Ties", ties)
    m4.metric("AI win rate", f"{ai_wins / played:.1%}" if played else "—",
              delta=f"{ai_wins / played - CHANCE:+.1%} vs chance" if played else None)

    if played:
        result = game.result()
        st.line_chart(win_rate_frame(result))
        with st.expander("Inside the AI's head", expanded=True):
            show_model_internals(agent)
        moves = pd.DataFrame({
            "round": np.arange(1, played + 1),
            "your_move": [MOVE_NAMES[m] for m in result.moves_b],
            "ai_move": [MOVE_NAMES[m] for m in result.moves_a],
            "ai_result": result.results,
        })
        st.download_button("Download this session's moves (CSV, no personal data)",
                           moves.to_csv(index=False), file_name="rps_session.csv", mime="text/csv")
    if st.button("Restart match"):
        new_human_game(predictor_name, seed, options)
        st.rerun()


# ------------------------------------------------------------ bot-vs-bot tab

def bot_tab() -> None:
    """AI predictor vs a scripted bot or another predictor."""
    col1, col2, col3 = st.columns(3)
    with col1:
        ai_name = st.selectbox("AI predictor", PREDICTOR_NAMES, index=len(PREDICTOR_NAMES) - 1, key="bb_ai")
    with col2:
        opponents = [f"Bot: {b}" for b in BOT_NAMES] + [f"AI: {p}" for p in PREDICTOR_NAMES]
        opponent = st.selectbox("Opponent", opponents, index=BOT_NAMES.index("Switching"), key="bb_opp")
    with col3:
        n_rounds = st.slider("Rounds", 50, 500, 200, 50, key="bb_rounds")
        seed = int(st.number_input("Seed", value=42, step=1, key="bb_seed"))
    if not st.button("Play match", type="primary"):
        return
    agent = make_agent(ai_name, seed=seed + 10_000)
    kind, name = opponent.split(": ", 1)
    other = make_bot(name, seed=seed) if kind == "Bot" else make_agent(name, seed=seed + 20_000)
    with st.spinner("Playing..."):
        result = Game(agent, other, n_rounds).play()
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{ai_name} win rate", f"{result.win_rate:.1%}", delta=f"{result.win_rate - CHANCE:+.1%} vs chance")
    m2.metric("Tie rate", f"{result.tie_rate:.1%}")
    m3.metric(f"{other.name} win rate", f"{result.loss_rate:.1%}")
    st.line_chart(win_rate_frame(result))
    if isinstance(other, SwitchingBot) and isinstance(agent.predictor, HMMPredictor) and agent.predictor.is_fitted:
        st.markdown("**HMM filtered belief vs the bot's true hidden strategy**")
        beliefs = agent.predictor.filtered_beliefs()
        favoured = agent.predictor.emission.argmax(axis=1)
        frame = pd.DataFrame(index=pd.Index(np.arange(1, n_rounds + 1), name="Round"))
        for move, label in enumerate(MOVE_NAMES):
            frame[f"belief: favours {label}"] = beliefs[:, favoured == move].sum(axis=1)
        frame["true strategy (0=R,1=P,2=S) / 2"] = np.array(other.strategy_history) / 2
        st.line_chart(frame)
    with st.expander("Inside the AI's head after the match"):
        show_model_internals(agent)


# ------------------------------------------------------------ tournament tab

def tournament_tab() -> None:
    """Show saved experiment results, or run a small round-robin live."""
    matrix_path = RESULTS_DIR / "tournament_matrix.csv"
    table_path = RESULTS_DIR / "results_table.md"
    try:
        st.markdown("#### Saved results (`python src/experiment.py --seed 42`)")
        st.markdown(table_path.read_text(encoding="utf-8"))
        matrix = pd.read_csv(matrix_path, index_col=0)
        st.markdown("**Round-robin: win rate of row predictor vs column predictor**")
        st.dataframe(matrix.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0.2, vmax=0.6))
        for figure in sorted(FIGURES_DIR.glob("fig*.png")):
            st.image(str(figure), caption=figure.stem)
    except FileNotFoundError:
        st.info("No saved results yet. Run `python src/experiment.py --seed 42` to create them, "
                "or run a small tournament below.")

    st.markdown("#### Run a quick tournament now")
    col1, col2 = st.columns(2)
    n_games = col1.slider("Games per pairing", 2, 20, 5)
    n_rounds = col2.slider("Rounds per game", 50, 300, 100, 50)
    if st.button("Run round-robin"):
        with st.spinner("Running..."):
            live, _ = round_robin(list(PREDICTOR_NAMES), n_games, n_rounds, base_seed=42)
        st.dataframe(live.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0.2, vmax=0.6))


# ------------------------------------------------------------------- main

def main() -> None:
    """Page layout."""
    st.set_page_config(page_title="RPS Mind Reader", page_icon="✊", layout="wide")
    st.title("✊✋✌️ Rock-Paper-Scissors Mind Reader")
    st.caption("Humans are bad at being random. The AI learns your habits and plays to beat them.")
    with st.sidebar:
        st.header("Your opponent")
        predictor_name = st.selectbox("AI predictor", PREDICTOR_NAMES, index=PREDICTOR_NAMES.index("HMM"))
        options = predictor_options(predictor_name, "play")
        seed = int(st.number_input("Seed", value=42, step=1))
        st.caption("Changing any setting starts a new match.")
    play, bots_vs, tourney = st.tabs(["Play vs AI", "Bot vs Bot", "Tournament"])
    with play:
        play_tab(predictor_name, seed, options)
    with bots_vs:
        bot_tab()
    with tourney:
        tournament_tab()


main()
