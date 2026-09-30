# Change log

`decijax` is in alpha development: the public API may change in any release.

## Unreleased

- **Breaking Changes**:
  - `AbstractSinglePointAcquisitionFunctionBuilder` is renamed to
    `AbstractSingleModelAcquisitionFunctionBuilder`, and its
    `build_acquisition_function` now takes a single `model` rather than a
    `models` dictionary keyed by tag. The `check_objective_present` helper is
    removed.
- **Changes**:
  - `AcquisitionDrivenDecisionMaker` now raises a `ValueError` at construction if
    no `OBJECTIVE`-tagged model builder and dataset are provided, and passes the
    `OBJECTIVE` model to the acquisition function builder.

## decijax 0.0.1 (August 9, 2026)

- Initial release.
