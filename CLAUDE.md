# decijax

JAX library for sequential model-based decision making (currently Bayesian
optimisation; active-learning acquisitions planned). Built on GPJax for the
default GP surrogates, but the core is deliberately backend-agnostic.

## Commands

- Run tests: `uv run pytest tests`
- Tests enable float64 via `jax.config.update("jax_enable_x64", True)` at the
  top of each test module, before other imports.

## Architecture

The decision loop (`decision_maker.py`) is an ask/tell loop over three
abstractions:

1. **Models** (`decijax/models/`) — the surrogate layer.
   - `ProbabilisticModel` is the root interface. It carries *no* predictive
     capability; it only obliges a model to expose `training_inputs` and
     `observations` (the data it was conditioned on).
   - Predictive capabilities are separate mix-in interfaces, one per thing an
     acquisition can require: `SupportsGaussianPrediction` (EI, PI, UCB),
     `SupportsSamplePaths` (Thompson sampling), `SupportsJointPrediction` and
     `SupportsPosteriorSamples` (reserved for q-family / Monte-Carlo batch
     acquisitions). A model implements only the capabilities it can honestly
     provide; acquisitions check capabilities with `isinstance` and raise
     `ValueError` otherwise.
   - `AbstractModelBuilder.build(dataset, key)` fits a fresh model to the full
     accumulated dataset. The decision maker refits from scratch on every
     `tell`; warm-starting is a possible future opt-in, not the default.
   - GPJax coupling is confined to `models/gps.py` and `models/builder.py`.
     Nothing outside `models/` (and the generic `Dataset` container) should
     import from `gpjax`.

2. **Utility (acquisition) functions** (`decijax/utility_functions/`) —
   builders take `Mapping[str, ProbabilisticModel]` plus a PRNG key and return
   a closure `[N, D] -> [N, 1]` to be *maximised*. Models carry their own
   training data, so builders take no separate `datasets` argument.

3. **Decision makers** (`decision_maker.py`) — hold the canonical
   original-space `datasets`, the `model_builders`, and the fitted `models`,
   keyed by tag (`OBJECTIVE`, `CONSTRAINT`, ...). Tags must be consistent
   across the three dicts.

## Core contracts (do not break these silently)

- **Leading sample axis `S`**: predictive distributions
  (`models/distributions.py`) expose `mean`/`variance`/`stddev` of shape
  `[S, N]`. `S == 1` for point-estimate models; `S` = number of hyperparameter
  samples for fully Bayesian models; `S` = ensemble size for ensembles.
  Acquisitions compute their value per sample and reduce over `S` (mean for
  expectation-based acquisitions — the correct marginalisation
  `E_theta[alpha_theta(x)]`). This is what lets fully Bayesian BO reuse the
  analytic acquisitions unchanged.
- **Prediction space**: a model reports its training data and makes
  predictions in the *same* space. Observation transforms (e.g. the default
  `standardize_observations`) are applied by the builder at fit time; the
  model carries the transformed dataset so its incumbent and predictions are
  automatically consistent. Inputs `X` are never transformed (the search space
  and maximiser operate in original coordinates). The decision maker's
  `datasets` remain the untransformed source of truth for reporting.
- **Sample paths**: `SupportsSamplePaths.draw_sample_paths` returns one
  callable `[N, D] -> [N, num_samples]` that is differentiable and returns the
  *same* function consistently for a fixed key (this is what Thompson sampling
  optimises). Backend details like the number of RFF features belong to the
  model, not to the acquisition.
- Acquisition closures must be pure JAX functions of `x` (jit/grad-safe);
  anything constant (e.g. the incumbent) is computed once in
  `build_utility_function`, not inside the closure.

## Design principles

- New surrogate backends (e.g. Laplace-approximated BNNs via Laplax, NumPyro
  fully Bayesian GPs) are added as new `ProbabilisticModel` adapters + a
  builder — never by widening acquisition signatures or branching on model
  type inside acquisitions.
- New acquisitions state their model requirements by checking capability
  interfaces, never concrete classes like `ConjugatePosterior`.
- Prefer plain `jax.scipy` / `jax.numpy` in acquisition math over pulling in
  distribution libraries.
- Docstrings use the models/builders vocabulary ("model", "refit"), not the
  legacy GPJax vocabulary ("posterior handler").

## Roadmap context

Planned: q-family (batch) acquisitions via `SupportsJointPrediction` /
`SupportsPosteriorSamples`; fully Bayesian BO with GPJax + NumPyro (leading
`S` axis, see `GPJaxFullyBayesianModel` sketch); active-learning acquisitions;
third-party surrogates via the capability interfaces.
