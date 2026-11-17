"""Hidden Markov Model predictor: the player's hidden *strategy* is the state.

Model
-----
* Hidden states ``z_t in {0..K-1}``: the strategy the player is following at
  round t (for example "favours Rock" or "cycling"). The AI never sees it.
* Observations ``x_t in {Rock, Paper, Scissors}``: the move actually played.
* Transition model ``A[i, j] = P(z_{t+1} = j | z_t = i)``: how likely the player
  is to switch strategy.
* Emission (sensor) model ``B[i, m] = P(x_t = m | z_t = i)``: how likely each
  move is under a strategy.
* Initial distribution ``pi[i] = P(z_1 = i)``.

Learning: ``hmmlearn.hmm.CategoricalHMM.fit`` runs Baum-Welch (EM) on the move
history every ``refit_every`` rounds to estimate ``pi``, ``A`` and ``B``.

Inference (own code, NumPy): the forward filter keeps the belief
``b_t(i) = P(z_t = i | x_1..x_t)`` and updates it after each move with

    b_t  ∝  (b_{t-1} @ A) * B[:, x_t]

Prediction: ``P(x_{t+1} = m) = (b_t @ A) @ B[:, m]`` — push the belief one step
through the transition model, then through the emission model.
"""

from __future__ import annotations

import logging

import numpy as np

from predictors import Predictor, _smoothed_frequency
from rps import N_MOVES

try:  # hmmlearn is declared in requirements.txt; fail with a clear message if missing
    from hmmlearn.hmm import CategoricalHMM
except ImportError as exc:  # pragma: no cover - depends on the environment
    CategoricalHMM = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

# hmmlearn logs "model is not converging" warnings for short sequences; they are
# expected while the history is small and would flood the experiment output.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def forward_step(belief: np.ndarray, transmat: np.ndarray, emission: np.ndarray, obs: int) -> np.ndarray:
    """One step of the forward filter: predict with ``A``, correct with ``B[:, obs]``."""
    predicted = belief @ transmat
    unnormalised = predicted * emission[:, obs]
    total = unnormalised.sum()
    if total <= 0.0 or not np.isfinite(total):
        # The observation was impossible under every state: restart from the prior.
        return np.full_like(belief, 1.0 / belief.size)
    return unnormalised / total


def forward_filter(
    observations: list[int] | np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emission: np.ndarray,
) -> np.ndarray:
    """Filtered beliefs ``P(z_t | x_1..x_t)`` for every t. Shape ``(T, K)``."""
    obs = np.asarray(observations, dtype=int)
    n_states = startprob.size
    beliefs = np.zeros((obs.size, n_states))
    if obs.size == 0:
        return beliefs
    first = startprob * emission[:, obs[0]]
    beliefs[0] = first / first.sum() if first.sum() > 0 else np.full(n_states, 1.0 / n_states)
    for t in range(1, obs.size):
        beliefs[t] = forward_step(beliefs[t - 1], transmat, emission, obs[t])
    return beliefs


class HMMPredictor(Predictor):
    """Predicts the next move by tracking a belief over the player's hidden strategy.

    Args:
        n_states: number of hidden strategies K.
        refit_every: re-run Baum-Welch after this many new moves.
        min_history: moves needed before the first fit; before that the
            predictor falls back to smoothed move frequencies.
        n_iter: maximum EM iterations per fit.
        window: fit only on the last ``window`` moves (``None`` = all).
        emission_floor: minimum emission probability, so one noisy move can
            never drive the belief in every state to zero.
        n_restarts: random initialisations tried at each fit (EM is local).
        seed: seeds both the tie-breaking RNG and hmmlearn's initialisation.
    """

    name = "HMM"

    def __init__(
        self,
        n_states: int = 3,
        refit_every: int = 10,
        min_history: int = 20,
        n_iter: int = 30,
        window: int | None = None,
        emission_floor: float = 1e-3,
        n_restarts: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        if CategoricalHMM is None:
            raise ImportError(
                "hmmlearn is required for HMMPredictor; run `pip install -r requirements.txt`"
            ) from _IMPORT_ERROR
        if n_states < 1:
            raise ValueError("n_states must be at least 1")
        if refit_every < 1 or min_history < 2 or n_iter < 1:
            raise ValueError("refit_every >= 1, min_history >= 2 and n_iter >= 1 are required")
        if window is not None and window < min_history:
            raise ValueError("window must be None or >= min_history")
        if n_restarts < 1:
            raise ValueError("n_restarts must be at least 1")
        self.n_restarts = n_restarts
        self.n_states = n_states
        self.refit_every = refit_every
        self.min_history = min_history
        self.n_iter = n_iter
        self.window = window
        self.emission_floor = emission_floor
        self.name = f"HMM-{n_states}" if n_states != 3 else "HMM"
        self._clear_model()

    # ----------------------------------------------------------------- state
    def _clear_model(self) -> None:
        self.startprob: np.ndarray | None = None
        self.transmat: np.ndarray | None = None
        self.emission: np.ndarray | None = None
        self.belief: np.ndarray | None = None
        self.belief_history: list[np.ndarray] = []
        self.log_likelihood: float | None = None
        self.n_fits = 0
        self.n_failed_fits = 0
        self._last_fit_len = 0

    def reset(self) -> None:
        """Forget all observations and the fitted model."""
        super().reset()
        self._clear_model()

    @property
    def is_fitted(self) -> bool:
        """True once Baum-Welch has produced a model."""
        return self.transmat is not None

    # -------------------------------------------------------------- learning
    def _learn(self, opponent_move: int) -> None:
        n = len(self.history)
        needs_fit = n >= self.min_history and (
            not self.is_fitted or n - self._last_fit_len >= self.refit_every
        )
        if needs_fit:
            self._fit()
        elif self.is_fitted:
            self.belief = forward_step(self.belief, self.transmat, self.emission, opponent_move)
        if self.belief is not None:
            self.belief_history.append(self.belief.copy())
        else:
            self.belief_history.append(np.full(self.n_states, np.nan))

    def _new_model(self, random_state: int | None, warm: bool) -> "CategoricalHMM":
        """A CategoricalHMM either randomly initialised or warm-started from our last fit."""
        model = CategoricalHMM(
            n_components=self.n_states,
            n_features=N_MOVES,
            n_iter=self.n_iter,
            tol=1e-3,
            random_state=random_state,
            init_params="" if warm else "ste",
            params="ste",
        )
        if warm:
            model.startprob_ = self.startprob.copy()
            model.transmat_ = self.transmat.copy()
            model.emissionprob_ = self.emission.copy()
        return model

    def _fit(self) -> None:
        """Run Baum-Welch on the history and refresh the filtered belief.

        EM only finds a local optimum, so several candidates are fitted — the
        previous model as a warm start plus ``n_restarts`` random starts — and
        the one with the highest log-likelihood is kept.
        """
        data = self.history if self.window is None else self.history[-self.window:]
        X = np.asarray(data, dtype=int).reshape(-1, 1)
        base = 0 if self._seed is None else self._seed
        candidates = [self._new_model(base + 7919 * r + self.n_fits, warm=False)
                      for r in range(self.n_restarts)]
        if self.is_fitted:
            candidates.insert(0, self._new_model(base, warm=True))
        best, best_score = None, -np.inf
        for model in candidates:
            try:
                model.fit(X)
                score = model.score(X)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue  # this start failed; try the others
            if np.isfinite(score) and score > best_score:
                best, best_score = model, score
        if best is None:
            # Keep the previous model (if any); try again after refit_every moves.
            self.n_failed_fits += 1
            self._last_fit_len = len(self.history)
            return
        model = best
        self.log_likelihood = float(best_score)
        emission = model.emissionprob_ + self.emission_floor
        self.emission = emission / emission.sum(axis=1, keepdims=True)
        self.transmat = np.asarray(model.transmat_, dtype=float)
        self.startprob = np.asarray(model.startprob_, dtype=float)
        self.n_fits += 1
        self._last_fit_len = len(self.history)
        # Re-filter the fitted window with the new parameters to get the current belief.
        self.belief = forward_filter(data, self.startprob, self.transmat, self.emission)[-1]

    # ------------------------------------------------------------ prediction
    def predict_proba(self) -> np.ndarray:
        """P(next move) = (belief @ A) @ B; smoothed frequencies before the first fit."""
        if not self.is_fitted:
            return _smoothed_frequency(self.history, 1.0)
        next_state = self.belief @ self.transmat
        proba = next_state @ self.emission
        return proba / proba.sum()

    def filtered_beliefs(self) -> np.ndarray:
        """Forward-filter the whole history with the *current* parameters.

        Unlike ``belief_history`` (whose state labels can permute between
        refits), this gives a consistent labelling for plotting.
        """
        if not self.is_fitted:
            raise RuntimeError("The HMM has not been fitted yet")
        return forward_filter(self.history, self.startprob, self.transmat, self.emission)

    def describe(self) -> str:
        """Name plus K and the refit interval."""
        return f"{self.name} (K={self.n_states}, refit every {self.refit_every})"
