# Performance cleanup report — 2026-09-08

Historical benchmark evidence: this report describes the implementation measured at that time, not the current API. Current runs use a single action-regret tracker and replicate-mean figures. References below to former evaluation modes and probability floors describe the old benchmark baseline, not supported configuration options today.

## Scope and working-tree preservation

Implemented against the current local tree, not GitHub main. Git sees this audit directory as untracked inside `/home/florian`; therefore a normal Git diff cannot isolate these changes. Review used a source archive captured before editing, without restoring it into the working tree.

## Changes by file

| Files | Changes |
|---|---|
| `algorithms/base.py` | Removed `_validate_feedback` and `_validate_strategy` and their per-update calls. `strategy()` returns the current array. |
| `algorithms/external_regret/exp3.py` | Cached fixed Auer gamma/eta and fixed Exp3-IX eta. Kept the simplified learner hierarchy and all estimators. |
| `algorithms/external_regret/hedge.py` | Cached the fixed-horizon rate; the anytime branch still computes its local-time rate. |
| `algorithms/external_regret/tsallis_inf.py` | Replaced fixed 100-step bisection with the scaled safeguarded Newton solver described below. |
| `algorithms/swap_regret/base.py` | Builds the stationary transition matrix directly from inner strategy arrays, without individual copies or `vstack`. |
| `algorithms/swap_regret/blum_mansour.py` | BM/LCE feedback updates read each learner's pre-update `q[i,a]` directly. No feedback-only full matrix construction. |
| `algorithms/stationary.py` | Removed stochastic-input checks. Constructs solve/pseudoinverse systems without temporary identity/ones arrays. Numerical postconditions and solver selection/fallback remain. |
| `environments/base.py`, `environments/repeated_game.py` | Removed per-step action/player validators. Payoff vectors are views; externally supplied tensors are still validated and copied at construction. |
| `environments/adversarial.py` | Removed random-walk exhaustion/step-before-feedback guards. Constructor, precomputation, transitions, and RNG sequence are unchanged. |
| `metrics/regret.py` | Removed per-update evaluator validators and the positive-summary-time guard present in the measured implementation. |
| `metrics/empirical_distribution.py` | Removed per-profile integer conversion, player-count checking, and range checking. Shape/checkpoint configuration checks remain. |
| `experiments/results.py` | Validates constant file metadata once, then compares raw constant fields on every row. Dynamic trajectory checks remain. |
| `experiments/runtime_environment.py` | Fingerprints already-canonical JSON without parsing it again. Callers validate at construction/file input. |
| `experiments/result_trajectories.py` | Loads validated cumulative joint-action histogram checkpoints for equilibrium analysis. |
| `Makefile` | Defaults numerical-library thread counts to one before launching Python, honoring explicit environment/Make overrides. |
| `algorithms/README.md`, `environments/README.md`, `metrics/README.md`, `experiments/README.md` | Documented the trusted core, borrowed-array contract, solver, metadata validation, and launch settings. |
| `tests/algorithms/test_numerical_safety.py`, `tests/algorithms/test_stationary.py`, `tests/environments/test_environments.py`, `tests/experiments/test_parallel.py` | Updated tests for the trust model and expanded solver/parallel coverage. |
| New `tests/algorithms/test_trusted_core.py`, `tests/algorithms/test_tsallis_solver.py`, `tests/experiments/test_result_validation.py`, `tests/experiments/test_numerical_threads.py` | Added valid-run, reference-solver, persistence-boundary, and launch-time regression tests. |
| `experiments/PERFORMANCE_CLEANUP.md` | This report. |

Production callers of strategies and feedback were audited: none mutates the borrowed arrays. Valid-run tests additionally verify that learner updates leave both the supplied feedback and the previous strategy array unchanged.

## Tsallis solver and mathematical invariants

The new solver uses exactly the scaled KKT equation:

```text
a = eta * (L - min(L))
f(z) = sum_i 1 / (a_i + z)^2 - 1
f'(z) = -2 * sum_i 1 / (a_i + z)^3
1 <= z <= sqrt(K)
```

It starts at the bracket midpoint, updates the bracket using the residual sign, accepts a finite Newton candidate strictly inside the bracket, and otherwise bisects. Reciprocal/probability work arrays are reused. Stopping requires absolute residual at most `1e-14` or relative bracket width at most `1e-14`. Final probability normalization remains; K=1 returns `[1]`.

The 64-iteration safety ceiling accommodates near-endpoint roots that fall back to bisection. It is not a fixed iteration count: final profiles averaged **6.12 iterations at K=3** and **5.79 at K=9**. Failure to converge raises a numerical error instead of silently returning a loose approximation.

Reference tests reproduce the old 100-step solver, covering K=1/3/5/9/100, six local times up to `10^12`, equal, moderate, extreme, multiple-minimum, and randomized losses, plus near-endpoint cases. Probability comparisons use `atol=2e-14, rtol=2e-13`; reconstructed scaled-KKT residuals must be below `1e-13`. The original KKT test remains.

No learner update formula, loss/gain estimator, learning-rate formula or indexing, regret definition, seed derivation, sampling primitive, stationary equation, or selected solver was changed. In particular:

- In the measured historical implementation, standalone Auer and BM both used `gamma=min(1,sqrt(K log(K)/T))`, with **no e−1 factor**, and `eta=gamma/K`. Current standalone `AuerExp3` uses the Auer et al. `(e−1)` factor, while BM explicitly retains the measured tuning.
- Fixed Hedge and Exp3-IX retain their original formulas, including behavior after T updates.
- Anytime Hedge, Tsallis-INF, and LCE-IX retain local update counts and schedules.
- BM still updates every inner learner with `p_i*q[i,a]*reward/p_a`, preserving arithmetic order. LCE retains its original observed-loss expression and IX estimator.
- Ito still samples one inner learner, samples that learner's action, and updates only that learner. Its source file is unchanged.
- Hart–Mas-Colell transitions, selected stationary method, solve-to-pseudoinverse fallback, clipping, normalization, and residual checks remain unchanged.
- Logit shifting, probability floors, and normalization in exponential weights remain unchanged.

## Persistence boundaries and existing optimizations

Fixed-game CSV metadata parsing, profile JSON validation, payoff-digest validation, regret-column validation, and runtime canonicalization/hashing now occur once per file. Subsequent rows must have identical raw constant fields. Runtime metadata is required; legacy profile columns remain supported.

The adversarial loader already validated metadata/derived seeds once per file and was left unchanged. Fixed-game files contain base seeds rather than separately recorded derived seeds; their seed schedule and run identity were not changed. Both loaders retain dynamic checkpoint and current-schema validation.

Current related invariants are:

- At most 500 regret checkpoints: every round for short horizons and geometric spacing for longer runs, including the first and final rounds.
- Per-round learner/regret-state updates with checkpoint-only summaries.
- Cumulative joint-action histograms using the regret checkpoint sampling rule with a separate 20-point budget.
- Spawn-based replicate processes, bounded workers/tasks, deterministic output order, cancellation, and no nested pools.
- Figure Builder artifact caching, including ordinary one-player action-space comparisons.
- Run identity fields, seed derivation, CSV schemas, and sampling schedules.
- Custom-game tensor/file validation, `allow_pickle=False`, path restrictions, atomic publication, and no-overwrite behavior.

No checked/unchecked API, validation switch, new learner variant, mathematical stationary method, inner-learner parallelism, or new architecture was introduced.

## Regression results and floating-point caveat

Before editing, the current code produced 40 historical-adversary state/action traces at T=300, K=3/9, seeds 7/17, covering all ten registry entries across the two feedback modes. Snapshots include all sampled actions, per-round strategies, cumulative learner and regret state, and RNG state. Also captured: 40 adversarial and 20 fixed-game CSVs containing the persistence variants and regret summaries used by that benchmark.

After comparison:

- All **12,000 captured actions** matched.
- All **36 non-Tsallis traces** matched exactly, including strategies, cumulative state, regret, and RNG state.
- All **54 non-Tsallis CSVs** were byte-identical with identical filenames/run identities.
- The six short bandit-Ito CSVs differed only in evaluator floating-point values, by at most `3.65e-12`. Their actions, action blocks, payoffs, other recorded metrics, and identities matched exactly.
- In the four T=300 Ito state traces, maximum strategy difference was `7.42e-14`, cumulative learner-state difference `3.15e-12`, and regret-state/summary difference `7.23e-13`.
- All non-Ito T=20,000 benchmark CSVs also matched byte-for-byte.

**Long adaptive Ito trajectories are not guaranteed to remain close to the old implementation.** The T=20,000 historical-frequency benchmarks with base seed 23 diverged in sampled actions at round 3,650 for K=3 and 18,688 for K=9. A paired replay using the old root solver inside the otherwise cleaned-up implementation found that same-input probability differences remained at most `1.00e-15` and `2.34e-15`, respectively, before the first differing sample. These tiny differences accumulate through importance-weighted loss updates, eventually changing action choices and the adaptive adversary's subsequent history.

Consequently, final swap action regret in those particular old/new Ito benchmark runs was 175/614 at K=3 and 830/801 at K=9. It would be incorrect to describe the complete long-run trajectory differences as tiny, or to claim byte-identical old/new Ito results. The tiny-difference guarantee is for the numerical KKT solution on the same input; long adaptive pathwise identity is a separate property. This is the only observed old/new execution exception. Serial/parallel execution of the **new** implementation remains byte-identical in the tested configurations, including Ito.

## Timings and final profiles

Same machine and process configuration on both sides: Python 3.10.12, NumPy 2.2.6, CPU affinity 0–15, and OMP/OpenBLAS/MKL/NumExpr thread counts all one. Each cell is the mean of two serial single-replicate passes, T=20,000, historical-frequency adversary, bandit feedback, action regret, base seed 23, and the benchmark's sparse recording policy. Timings include simulation and CSV output, not plotting or a process pool.

| Learner | K | Before (s) | After (s) | Speedup |
|---|---:|---:|---:|---:|
| Exp3-IX | 3 | 1.814 | 0.886 | 2.05× |
| Bandit BM | 3 | 6.844 | 3.064 | 2.23× |
| LCE-IX | 3 | 6.497 | 3.238 | 2.01× |
| Bandit Ito | 3 | 14.511 | 3.345 | 4.34× |
| Exp3-IX | 9 | 1.813 | 0.894 | 2.03× |
| Bandit BM | 9 | 12.680 | 4.914 | 2.58× |
| LCE-IX | 9 | 12.028 | 5.311 | 2.26× |
| Bandit Ito | 9 | 14.816 | 3.379 | 4.38× |

Saved paired timings remain applicable to the final computation code; the subsequent correction affects only invalid result-file loading. Absolute times depend on machine load. The Ito timings compare the same configured experiment but, as noted above, not an identical long action path.

Fresh T=5,000 profiles covered BM/LCE/Ito at both K values. At K=9, stationary solving accounted for approximately 24% of BM, 23% of LCE, and 35% of Ito runtime. Exponential-weight strategy computation accounted for approximately 42%/44% of BM/LCE. Tsallis strategy computation accounted for approximately 28% of Ito. Removed input validators no longer appear as hotspots; the remaining stationary postconditions are intentionally retained. No further algorithmic optimization was attempted based on these percentages.

## Tests and launch settings

Removed tests that only demanded graceful failures for non-finite internal learner feedback, invalid internal environment actions/player ids, or invalid stationary input matrices. Updated payoff-copy and random-walk call-order tests to check supported behavior. Retained mathematical tests and external-input/configuration safety tests.

New/expanded tests cover:

- Finite/nonnegative/normalized outer and inner strategies, stationary residuals, generated feedback range, finite regret state, borrowed-array immutability, and local update counts across fixed, historical-frequency, and random-walk runs.
- Old-bisection equivalence, endpoint roots, unchanged learning-rate schedules, and no repeated fixed-parameter sqrt/log calculation.
- Exactly one BM/LCE transition construction per outer update.
- Once-per-file metadata work, corruption on subsequent rows, current-schema validation, checkpoint action bounds, grouping, and completeness.
- Serial/process byte identity for every bandit algorithm and recorded configuration, fixed cross-play, out-of-order replicate inputs, and bounded/nonnested execution.
- Launch-time numerical thread defaults and explicit overrides.

The historical cleanup's full-suite result was **880 passed in 228.60 seconds**, including both additional runtime-metadata boundary cases. No failures or skips were reported.

The full-suite command uses:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
TMPDIR=/tmp/codex-lceix-tests PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/codex-lceix-tests/matplotlib \
python3 -m pytest -q -p no:cacheprovider
```

`make web` now supplies the same numerical thread defaults before Python/NumPy starts; existing explicit environment values take precedence. No late worker environment mutation was added. Restart an existing dashboard to load the updated code and launch settings; no running dashboard was stopped or restarted by this task.
