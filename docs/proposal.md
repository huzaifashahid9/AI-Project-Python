# AL-2002 Project Proposal: Rock-Paper-Scissors Mind Reader

Oct 20, 2026

## 1. Group and Project

| Item | Details |
|---|---|
| Course | AL-2002 — Artificial Intelligence (Laboratory), Fall 2026, FAST-NUCES Karachi |
| Project | Rock-Paper-Scissors Mind Reader (Idea 8 from the list, Part A — Game) |
| Section | A |
| Group leader | Muneeb Mustafa (24K-3102) |
| Member 2 | Huzaifa Shahid (24K-3069) |

## 2. Problem Formulation

People are poor random-number generators, so a player's next Rock-Paper-Scissors move often depends on the moves before it. Our system predicts the player's next move from the history of moves and plays the move that beats it. The difficulty is that the AI sees only moves, never the player's strategy, and has a few hundred observations to learn from: a model that is too simple misses patterns and one that is too rich overfits noise.

| PEAS | Description |
|---|---|
| Performance measure | AI win rate over 200 rounds (wins ÷ rounds); above 1/3 beats chance |
| Environment | Repeated two-player Rock-Paper-Scissors against a human or a scripted bot with a hidden strategy |
| Actuators | Choose Rock, Paper or Scissors |
| Sensors | The opponent's revealed move and the running score |

The environment is partially observable (the strategy is hidden), stochastic, static, discrete and multi-agent. The agent is a learning, model-based agent.

| Element | Representation |
|---|---|
| State | The last k opponent moves (k = 1 or 2) encoded 0, 1, 2; for the HMM, the full move sequence and a belief over hidden strategies |
| Actions | Rock, Paper, Scissors |
| Initial state | Empty history, score 0–0 |
| Terminal test | The match ends after 200 rounds |
| Utility | +1 win, 0 tie, −1 loss; judged by win rate |

## 3. Chosen AI Techniques

| Technique | Lab | Role | Own code or library |
|---|---|---|---|
| Markov chain predictor, order 1 and 2 | 09 | Core | Own code (NumPy) |
| Hidden Markov Model predictor (the player's hidden strategy is the state) | 09 | Core | hmmlearn, declared in the report |
| Frequency predictor | 09 | Comparison | Own code |
| Random predictor | — | Baseline (expected win rate 1/3) | Own code |

The Markov model states its transition probabilities as counts of what follows each context. The HMM states its transition matrix (how likely the player is to switch strategy) and emission matrix (how likely each move is under a strategy), and the belief over strategies is shown changing round by round.

**How the lab requirements are met**

- **Python and OOP:** at least three classes (Predictor, Opponent, Game and AIAgent) with inheritance and overridden `predict()` and `next_move()` methods; the random module is seeded.
- **Agents and environments:** PEAS and the environment classification above; the agent is implemented as a learning, model-based agent.
- **Core and comparison:** Markov and HMM predictors compared with the frequency predictor and the random baseline.
- **Practice:** modular `src/` layout, docstrings, specific except blocks, README, requirements.txt and a limitations and ethics section.

## 4. Data or Environment

No public dataset is needed: the data are generated. Four scripted bots (biased, cyclic, win-stay/lose-shift and uniformly random) produce the opponent moves, each from a seeded generator (seed 42, with game i using seed 42 + i), so every run is reproducible. If human demo sessions are logged, only move choices are stored, with the players' consent and no personal data. Results are saved as CSV files in `outputs/results/`.

## 5. Evaluation Plan

| Item | Plan |
|---|---|
| What is compared | Random baseline, Frequency, Markov-1, Markov-2 and HMM predictors against each bot, plus a round-robin tournament between the predictors |
| Metrics | AI win rate (mean ± standard deviation), tie and loss rate, win rate over rounds |
| Baseline | Random predictor (expected win rate 1/3) |
| Instances | 100 games of 200 rounds per pairing, with the same seeds for every predictor |
| Outputs | A results table and at least three charts: win rate per predictor and bot, rolling win rate over rounds, tournament heatmap (and the learned Markov transition table) |
| Reproduction | One command: `python src/experiment.py --seed 42` |

## 6. Work Division

| Member | Responsibility |
|---|---|
| Muneeb Mustafa (24K-3102) | Game engine and AIAgent, frequency and Markov predictors, Streamlit app |
| Huzaifa Shahid (24K-3069) | HMM predictor, opponent bots, tournament and experiments, charts |
| Both | README, requirements.txt, report sections on their own modules; commits from each member's own GitHub account |

Each member will be able to explain, trace and modify the code they wrote, as the viva requires.

## 7. Milestones

| # | Milestone | Owner | Target |
|---|---|---|---|
| 1 | Repository set up, proposal submitted | Both | [TO FILL — date] |
| 2 | Game engine, bots and random and frequency predictors working | Muneeb / Huzaifa | [TO FILL — date] |
| 3 | Markov predictors and HMM predictor working | Muneeb / Huzaifa | [TO FILL — date] |
| 4 | Streamlit app and tournament runner | Muneeb / Huzaifa | [TO FILL — date] |
| 5 | Experiments, results table and charts | Huzaifa | [TO FILL — date] |
| 6 | Report (PDF and Markdown) and README | Both | [TO FILL — date] |
| 7 | Clean-clone test, viva preparation, submission to Google Classroom | Both | Week 15 |
