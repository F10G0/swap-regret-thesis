# Algorithms

Regret-minimizing learners used by the experiment runners. Every learner exposes `strategy()`, `sample_action()`, `update(feedback)`, and `reset()`. Full-information learners receive a reward vector; bandit learners receive only the selected reward. Experiment runners create one deterministically seeded learner per player.

The computation core trusts the validated experiment and environment. `update()` does not revalidate feedback or probabilities each round. `strategy()` returns the current array, which callers must not mutate; updates replace it. Configuration is checked at construction, and tests verify valid-run feedback, normalization, and finite state. The stationary solver trusts its stochastic input but retains numerical postconditions, clipping/normalization, residual verification, and direct-solve/pseudoinverse fallback.

The dashboard and figures use concise thesis-facing names while registry and result IDs distinguish every reduction variant. Full-information reductions are BM-Hedge (`bm_hedge`, implemented by `FullBM` with Hedge), BM-OptHedge (`bm_optimistic_hedge`, implemented by `BMOptimisticHedge` with Optimistic Hedge), and Ito-Hedge (`ito_hedge`, implemented by `FullIto` with Hedge). Bandit reductions are BM-EXP3 (`bm_exp3`, implemented by `BanditBM` with EXP3) and Ito-Tsallis (`ito_tsallis`, implemented by `BanditIto` with Tsallis-INF); the shorter display name intentionally omits the `-INF` suffix. Class names remain unchanged. OptHedge is `OptimisticHedge`, and EXP3 is the Auer et al. `AuerExp3` implementation.

## Horizon contracts

The experiment runner always stops at a preset positive horizon `T`. The common `Algorithm` and `StationaryReduction` bases have no horizon parameter, and `ExponentialWeightsAlgorithm` leaves the rate entirely to its subclasses.

| Construction | Horizon contract |
|---|---|
| `Hedge(K, horizon=T, seed=...)` | Positive integer `T`; fixed rate, including after `T` updates |
| `Hedge(K, horizon=None, seed=...)` | Local-time rate; `None` is the default and is specific to Hedge |
| `OptimisticHedge(K, horizon=T, seed=...)` | Required positive integer `T`; fixed standalone or explicitly supplied rate |
| `AuerExp3(K, horizon=T, seed=...)`, `Exp3IX(K, horizon=T, seed=...)` | Required positive integer `T`; fixed parameters |
| `FullBM(K, horizon=T, seed=...)`, `BMOptimisticHedge(K, horizon=T, seed=...)`, `BanditBM(K, horizon=T, seed=...)` | Pass the actual positive experiment horizon to every inner learner |
| `TsallisINF(K, seed=...)`, `LCEIXInner(K, seed=...)` | No horizon argument; each implements its own local-time schedule |
| `RegretMatching`, `StationaryRegretMatching`, `FullIto`, `BanditIto`, `LCEIX` | No horizon argument |

Zero is rejected by horizon-taking learners. Fixed rates never switch automatically to local-time rates. FullIto explicitly constructs Hedge with `horizon=None`; BanditIto constructs TsallisINF without a horizon. Custom inner factories receive `seed` by keyword and, where applicable, `horizon` by keyword. Registries explicitly declare whether to pass the experiment horizon. With `t` completed local updates, a local-time learner computes its next strategy using the paper's round index `t+1`.

## External-regret learners

| Class | Feedback and update | Parameter schedule | Paper and guarantee |
|---|---|---|---|
| `Hedge` | Full reward vector; cumulative-gain exponential weights | Fixed `eta=sqrt(8 log(K)/T)` for positive `horizon=T`; local `eta_s=sqrt(8 log(K)/s)` for `horizon=None` | Hedge-style external regret `O(sqrt(T log K))` |
| `OptimisticHedge` | Full reward vector; Chen–Peng Eq. (3) in reward form: `x_(t+1)(j) ∝ x_t(j) exp(eta (2 r_t(j) - r_(t-1)(j)))`, with `r_0=0` | Chen–Peng Theorem 3.1: fixed `eta=(log(K)/T)^(1/6)`, unless an explicit fixed rate is supplied | Full-information repeated-game tuning |
| `Exp3IX` | Selected reward is converted to an implicit-exploration loss estimate | [Neu (2015)](https://proceedings.neurips.cc/paper_files/paper/2015/file/1c64ee92596e8ea5050fc435a1d57459-Paper.pdf): fixed `eta=sqrt(2 log(K)/(K T))` and `gamma=eta/2` | Neu-style high-probability adversarial bandit control |
| `AuerExp3` | Auer et al. (2002), Figure 1: `x_hat[k]=r_k/p_k`; probabilities explicitly mix normalized weights with uniform exploration | [Auer et al. (2002), Corollary 3.2](https://www.schapire.net/papers/AuerCeFrSc01.pdf): `gamma=min(1,sqrt(K log(K)/((e-1)T)))`, `eta=gamma/K`, and `w_k <- w_k exp(eta x_hat[k])` | Standalone known-horizon external-regret baseline |
| `TsallisINF` | Standard importance-weighted bandit losses and the `1/2`-Tsallis FTRL/OMD distribution | [Zimmert and Seldin (2019)](https://proceedings.mlr.press/v89/zimmert19a/zimmert19a.pdf): `eta_t=1/sqrt(t)` with local update time; `w_i=1/(eta_t^2 (L_hat_i+lambda)^2)` and `lambda` normalizes `w` | Anytime adversarial pseudo-regret at most `4 sqrt(KT)+1`; canonical inner learner for Bandit Ito |

`Exp3IX`, `AuerExp3`, and `TsallisINF` have distinct exploration mechanisms, estimators, schedules, and associated proofs. For all anytime learners, after local update `t` the newly computed strategy uses the round-`t+1` parameter.

Fixed-horizon Hedge, AuerExp3, and Exp3IX cache their constant parameters at construction. TsallisINF solves the same KKT equation in scaled coordinates: `a = eta * (L - min(L))`, `sum((a + z)^(-2)) = 1`, with `z` bracketed by `1` and `sqrt(K)`. Safeguarded Newton steps use bisection when a step leaves the bracket, stopping at a `1e-14` residual or relative bracket width before the existing final normalization. A 64-step safety ceiling allows endpoint cases to converge by bisection; ordinary solves terminate much earlier. The loss estimator and local-time schedule are unchanged.

The bandit experiment registry exposes `auer_exp3`, `exp3_ix`, `bm_exp3`, `ito_tsallis`, and `lce_ix`. `auer_exp3`, displayed as EXP3, is the selectable standalone AuerExp3 external baseline in both web experiment modes and uses the Auer et al. rate with the `(e-1)` factor. There is no generic `Exp3` learner or `exp3` run option: BM-EXP3 uses the same `AuerExp3` implementation with its Blum–Mansour tuning, Ito-Tsallis uses `TsallisINF`, and LCE-IX uses its dedicated IX learner. `ImplicitExplorationAlgorithm` shares only the IX estimator, not a horizon convention or learning-rate schedule.

`AuerExp3` defaults to the parameter choice from [Auer et al. (2002), Corollary 3.2](https://www.schapire.net/papers/AuerCeFrSc01.pdf). `BanditBM` explicitly selects the alternative parameter choice from the proof of Blum–Mansour Theorem 11: `gamma=min(1,sqrt(K log(K)/T))` and `eta=gamma/K`. Its `B_max=T` is valid because `B_{i,j}=E[sum_t p_i^t b_j^t] <= T` for rewards in `[0,1]`.

## Swap- and internal-regret reductions

- `RegretMatching` implements Hart–Mas-Colell positive action-replacement regret. `StationaryRegretMatching` uses the stationary distribution of its regret transition matrix; its solver supports `solve`, `pinv`, and `iteration`.
- `FullBM` is the full-information [Blum–Mansour (2007)](https://www.jmlr.org/papers/volume8/blum07a/blum07a.pdf) stationary reduction with one known-horizon `Hedge` learner per outer action.
- `BMOptimisticHedge` is the Chen–Peng BM-Optimistic-Hedge algorithm: the standard `FullBM` reduction with one `OptimisticHedge` learner per outer action. Inner learner `i` receives weighted reward `x_t(i) r_t` and uses the fixed Theorem 5.1 rate `eta=(K log(K)/(m^2 T))^(1/4)` for `m` players. This and the standalone Optimistic Hedge rate are repeated-game tunings from Chen and Peng, not universal optimal choices for arbitrary environments.
- `BanditBM` is the partial-information Blum–Mansour reduction with one BM-tuned `AuerExp3` learner per outer action. When outer action `k` is played, inner learner `i` receives the paper's observed gain `g_{i,k}=p_i q_{i,k} r_k/p_k`; its Auer learner then importance-weights by `q_{i,k}`. All inner learners update every round. Theorem 11 bounds swap pseudo-regret (the maximum over swap functions outside the expectation) by `O(K sqrt(K T log K))` with `B_max=T`.
- `FullIto` follows the efficient reduction of [Ito (2020)](https://proceedings.neurips.cc/paper/2020/file/d79c8788088c2193f0244d8f1f36d2db-Paper.pdf): sample one inner learner from the stationary outer distribution, sample its action, and update only that learner. Its inner learners are anytime `Hedge` instances.
- `BanditIto` uses the same Ito reduction with independent anytime `TsallisINF` inner learners. Each instance advances only when selected, so its time index is its own random local update count. Combining Ito's reduction with the minimax `O(sqrt(KT))` bandit learner yields the `O(K sqrt(T))` bandit swap-regret order.
- `LCEIX` implements [Huang–Pan (2023)](https://proceedings.mlr.press/v216/huang23b/huang23b.pdf). Its dedicated `LCEIXInner` uses `eta_t=sqrt(log(K)/t)` and `gamma_t=eta_t/2` directly from local time. It shares the IX loss estimator with standalone `Exp3IX`, while owning its schedule independently. Its proportional implicit-exploration loss estimator is the paper's LCE-IX construction, corresponding to expected swap regret `O(K sqrt(T log K))` and the stated high-probability instantaneous bound.

Games and feedback live in `environments/`, orchestration in `experiments/`, and evaluation in `metrics/`.
