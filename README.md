# ModelDiff-ML

[![PyPI](https://img.shields.io/pypi/v/modeldiff-ml.svg)](https://pypi.org/project/modeldiff-ml/)
[![Python](https://img.shields.io/pypi/pyversions/modeldiff-ml.svg)](https://pypi.org/project/modeldiff-ml/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ecklein/modeldiff-ml/blob/main/LICENSE)

Your candidate model improved AUC from 0.84 to 0.85. But did it get better for everyone?

ModelDiff-ML compares two machine learning models one observation at a time. It finds readable
regions where a candidate behaves differently from a baseline, then validates those regions on
independent data. The result is a practical answer that an average metric cannot provide:

```text
The candidate improved overall, but regressed for customers with high utilization and low income.
```

The library is deliberately small. It works with fitted models or precomputed predictions, depends
only on NumPy, pandas, and SciPy, and produces plain Python objects, JSON, and standalone HTML. It is
not an MLOps platform and does not require a model registry, service, or database.

## Installation

ModelDiff-ML requires Python 3.11 or newer.

```bash
pip install modeldiff-ml
```

The package is published as `modeldiff-ml` and imported as `modeldiff`.

## Quick start

Compare two fitted models on a shared evaluation set:

```python
import modeldiff

report = modeldiff.compare(
    baseline=old_model,
    candidate=new_model,
    X=X_test,
    y=y_test,
    task="binary_classification",
    metric="log_loss",
    random_state=42,
)

print(report.summary())
report.to_json("modeldiff-report.json")
report.to_html("modeldiff-report.html")
```

Binary classifiers must provide `predict_proba(X)` with two columns. ModelDiff-ML does not turn an
uncalibrated `decision_function` into a probability. Regression estimators only need `predict(X)`.

If predictions already exist, use the model-independent API:

```python
from modeldiff import ModelDiffConfig, compare_predictions

config = ModelDiffConfig(
    max_depth=3,
    min_region_fraction=0.02,
    n_bootstrap=1000,
    n_permutations=2000,
    random_state=42,
)

report = compare_predictions(
    baseline_predictions=old_probabilities,
    candidate_predictions=new_probabilities,
    X=X_test,
    y=y_test,
    task="binary_classification",
    metric="log_loss",
    config=config,
)

for region in report.regressions(significant_only=True):
    print(region.description, region.validation_delta)
```

`X` can be a pandas DataFrame or a two-dimensional NumPy array. DataFrames produce more useful
condition names. Array columns are named `x0`, `x1`, and so on.

## What ModelDiff-ML reports

A comparison includes the overall metric change, prediction disagreement diagnostics, validated
regression and improvement regions, confidence intervals, p-values, q-values adjusted for the false
discovery rate, and warnings about small or unstable samples. Reports can be viewed in the terminal,
serialized to JSON, or saved as a self-contained HTML file.

Every delta follows the same convention:

```text
delta = candidate loss - baseline loss
```

Positive values mean the candidate is worse. Negative values mean it is better.

| Task | Supported metrics | Region loss unit |
| --- | --- | --- |
| Binary classification | `log_loss`, `brier`, `accuracy` | Mean log loss, Brier loss, or misclassification rate |
| Regression | `squared_error`, `mse`, `rmse`, `absolute_error`, `mae` | Mean squared error or mean absolute error |

For `rmse`, the overall metric is square rooted after aggregation. Region discovery and saved checks
continue to operate on per-observation squared error, so their deltas are expressed as mean squared
error rather than a difference between aggregate RMSE values.

## Find regressions that survive resampling

One discovery split can still produce a slice that depends on a lucky threshold or a handful of
rows. The stability API repeats region discovery on subsamples of the discovery partition, groups
similar regions by their membership, and keeps only those selected often enough. It then evaluates
the surviving conditions once on the untouched validation partition.

```python
from modeldiff import ModelDiffConfig, StabilityConfig, compare_stable_predictions

report = compare_stable_predictions(
    old_predictions,
    new_predictions,
    X,
    y,
    task="regression",
    metric="mae",
    config=ModelDiffConfig(random_state=42),
    stability_config=StabilityConfig(
        n_resamples=30,
        subsample_fraction=0.75,
        min_selection_frequency=0.50,
        random_state=42,
    ),
)

for region in report.regressions(significant_only=True):
    print(region.description, region.selection_frequency, region.validation_delta)
```

Each stable region records its selection frequency, sign consistency, similarity to other members
of its cluster, and the ranges of discovery effects and support across resamples. Random and
explicit splits resample rows, group splits resample whole groups, and time splits use contiguous
blocks. In every case, resampling stays entirely inside discovery data.

Pass an existing `CheckSuite` through `known_checks=` to distinguish newly discovered regressions
from behavior the project already tracks. A stable regression is labeled `known` when saved checks
cover at least 80% of its validation membership, `partially_covered` when coverage is lower but
nonzero, and `new` when there is no overlap. Without a suite, triage remains `unclassified`.

## Choosing a validation split

ModelDiff-ML searches for regions on one partition and evaluates them unchanged on another. A random
split is the default, while three additional strategies help prevent leakage in structured data.

| Strategy | Use it when | Required argument |
| --- | --- | --- |
| `random` | Rows are independent | None |
| `group` | Several rows belong to the same customer, patient, device, or account | `groups=...` |
| `time` | Validation should represent later observations | `time_values=...` |
| `explicit` | A development and holdout partition already exists | `discovery_indices=...`, `validation_indices=...` |

For example, a group split keeps every row for a customer in one partition:

```python
report = compare_predictions(
    old_predictions,
    new_predictions,
    X,
    y,
    task="regression",
    metric="mae",
    validation_strategy="group",
    groups=customer_ids,
)
```

A chronological comparison uses earlier observations for discovery and later observations for
validation:

```python
report = compare_predictions(
    old_predictions,
    new_predictions,
    X,
    y,
    task="regression",
    metric="mae",
    validation_strategy="time",
    time_values=event_timestamps,
)
```

Group and time boundaries stay as close as possible to `discovery_fraction` without splitting a
group or equal timestamp. Explicit partitions must assign every observation exactly once. Split
metadata is available through `report.split` and is included in serialized reports.

## Turn discoveries into reusable checks

A discovered region can become a fixed regression check for future model releases. The saved check
keeps the exact structured conditions, metric settings, allowed loss increase, minimum sample count,
and enough provenance to trace it back to a comparison report.

```python
checks = report.export_checks(
    regions=report.regressions(significant_only=True),
    max_loss_increase=0.01,
    min_samples=100,
    names=["High utilization applicants"],
)
checks.to_json("model-checks.json")

result = modeldiff.evaluate_checks(
    checks="model-checks.json",
    baseline_predictions=production_predictions,
    candidate_predictions=new_predictions,
    X=X_fresh,
    y=y_fresh,
)

print(result.summary())
result.to_json("check-results.json")
result.to_html("check-results.html")
```

Omitting `regions` exports significant regression regions by default. Passing an empty list creates
a valid empty suite. When `names` is provided, it must contain one name for each selected region.
Check IDs are deterministic, and policy settings can be refined without mutating the original suite:

```python
check_id = checks.checks[0].id
checks = checks.with_check_settings(
    check_id,
    name="Priority production segment",
    max_loss_increase=0.005,
    min_samples=250,
)
```

Observed checks receive `pass`, `fail`, or `insufficient_data`. A loss increase equal to the
tolerance passes. A slice with fewer than `min_samples`, including a slice with zero matches, is
never allowed to pass silently.

| Overall status | Meaning |
| --- | --- |
| `pass` | The suite is not empty and every check passes |
| `fail` | At least one check exceeds its tolerance |
| `inconclusive` | No check fails, but a confidence interval crosses its tolerance |
| `insufficient_data` | No check fails, but at least one lacks enough support |
| `no_checks` | The suite is empty |

Check tolerances use the region loss units shown above. An accuracy tolerance is an increase in
misclassification rate. An `rmse` tolerance is an increase in mean squared error. Classification
thresholds and the numerical epsilon used for log loss are preserved in the saved suite.

Saved checks are release policies, not causal or hypothesis-test conclusions. Fresh evaluation data
is strongly recommended. Reusing discovery data is descriptive, while repeatedly tuning against
the same evaluation set can overfit that set. Exported checks also represent selected findings
rather than complete coverage of every regression a future model might introduce.

For uncertainty-aware release gates, use a confidence decision policy. A check passes only when its
entire interval is within tolerance, fails only when the entire interval exceeds tolerance, and is
otherwise inconclusive. Bonferroni correction is the default so the requested confidence level
applies to the suite rather than to each check in isolation.

```python
from modeldiff import CheckDecisionPolicy

result = modeldiff.evaluate_checks(
    checks,
    production_predictions,
    new_predictions,
    X_fresh,
    y_fresh,
    decision_policy=CheckDecisionPolicy(
        method="confidence",
        confidence_level=0.95,
        correction="bonferroni",
    ),
)

result.to_junit_xml("modeldiff-checks.xml")
```

## Evaluate checks repeatedly without statistical peeking

Fixed-sample confidence intervals are not designed to be recomputed after every new batch and
stopped whenever the result first looks favorable. ModelDiff-ML can instead maintain time-uniform
confidence sequences for saved checks. These intervals remain valid whenever they are inspected,
provided observations are independent and stationary and every observed loss delta stays inside
the configured bounds.

```python
from modeldiff import CheckMonitor, SequentialCheckPolicy

monitor = CheckMonitor(
    checks,
    SequentialCheckPolicy(
        confidence_level=0.95,
        correction="bonferroni",
        loss_delta_bounds=(-2.0, 2.0),
    ),
)

result = monitor.update(
    baseline_predictions=production_predictions_batch_1,
    candidate_predictions=new_predictions_batch_1,
    X=X_batch_1,
    y=y_batch_1,
)
monitor.to_json("monitor-state.json")

# Resume later without resetting the statistical history.
monitor = CheckMonitor.load("monitor-state.json")
result = monitor.update(
    production_predictions_batch_2,
    new_predictions_batch_2,
    X_batch_2,
    y_batch_2,
)
```

Each check receives one of four statuses:

| Status | Meaning |
| --- | --- |
| `pass` | The time-uniform upper bound is within tolerance |
| `fail` | The time-uniform lower bound exceeds tolerance |
| `continue` | The confidence sequence still crosses the tolerance |
| `insufficient_data` | The region has not reached its configured minimum support |

Accuracy and Brier loss deltas have automatic bounds of `[-1, 1]`. Log-loss bounds are derived from
the suite's numerical epsilon. Regression losses are not intrinsically bounded, so their monitors
require explicit finite `loss_delta_bounds`. A batch is rejected atomically if a matching
observation falls outside those bounds; no partial state is retained.

The implementation uses a two-sided Hoeffding confidence sequence with a summable alpha-spending
schedule and optional Bonferroni correction across the complete suite. It is intentionally
conservative and requires no parametric distribution assumption. The monitor state contains the
suite, policy, sufficient statistics, and a suite fingerprint. Updates are chunk-invariant and the
state file is replaced atomically after a successful CLI update.

Time-uniform validity does not make arbitrary production data safe to reuse. Do not submit duplicate
observations, change a saved region after monitoring starts, or use the guarantee across an
unmodeled distribution change. The method follows the confidence-sequence framework described by
[Howard et al.](https://doi.org/10.1214/20-AOS1991).

The complete workflow is available in
[`examples/saved_checks_example.py`](https://github.com/ecklein/modeldiff-ml/blob/main/examples/saved_checks_example.py).

## Command line and CI

The CLI works with CSV files containing features, the target, and both prediction columns:

```bash
modeldiff-ml compare \
  --data validation.csv \
  --target y \
  --baseline-pred old_prediction \
  --candidate-pred new_prediction \
  --task binary_classification \
  --metric log_loss \
  --html report.html \
  --json report.json
```

Columns other than the target and prediction columns are treated as features. Repeat
`--exclude COLUMN` to omit metadata. Group, time, and explicit partitions are also available:

```bash
modeldiff-ml compare ... --validation-strategy group --group-column customer_id
modeldiff-ml compare ... --validation-strategy time --time-column event_time
modeldiff-ml compare ... --validation-strategy explicit --split-column partition
```

An explicit partition column must contain only `discovery` and `validation`. Columns used to control
a split are automatically excluded from region search.

The existing comparison command can fail a CI job for a significant region or excessive overall
regression:

```bash
modeldiff-ml compare ... --fail-on-significant-regression
modeldiff-ml compare ... --max-overall-regression 0.01
```

Stable discovery has its own command and accepts the same split controls:

```bash
modeldiff-ml stability \
  --data validation.csv \
  --target y \
  --baseline-pred old \
  --candidate-pred new \
  --task regression \
  --metric mae \
  --n-resamples 30 \
  --min-selection-frequency 0.5 \
  --known-checks model-checks.json \
  --json stability-report.json \
  --html stability-report.html
```

To save discovered regions and evaluate them later:

```bash
modeldiff-ml compare \
  --data discovery.csv \
  --target y \
  --baseline-pred old \
  --candidate-pred new \
  --task regression \
  --metric mae \
  --export-checks model-checks.json \
  --check-max-loss-increase 0.01 \
  --check-min-samples 100

modeldiff-ml check \
  --checks model-checks.json \
  --data fresh-predictions.csv \
  --target y \
  --baseline-pred production_prediction \
  --candidate-pred new_prediction \
  --decision confidence \
  --correction bonferroni \
  --junit check-results.xml \
  --json check-results.json \
  --html check-results.html
```

To create a monitor from a saved suite and add its first labeled batch:

```bash
modeldiff-ml monitor \
  --checks model-checks.json \
  --state monitor-state.json \
  --data labeled-batch-1.csv \
  --target y \
  --baseline-pred production_prediction \
  --candidate-pred new_prediction \
  --loss-delta-lower -2 \
  --loss-delta-upper 2 \
  --json monitor-result.json \
  --junit monitor-result.xml
```

Later updates only need the state and new batch:

```bash
modeldiff-ml monitor \
  --state monitor-state.json \
  --data labeled-batch-2.csv \
  --target y \
  --baseline-pred production_prediction \
  --candidate-pred new_prediction
```

Monitor exit code `0` means every nonempty check passed. Exit code `1` covers `fail`, `continue`,
`insufficient_data`, or an empty suite. Exit code `2` indicates invalid state, bounds, inputs, or
usage. Policy and bounds are fixed when state is created and cannot be changed when it is resumed.

The `check` command reads the task, metric, classification threshold, and epsilon from the suite.
Exit code `0` means every check in a nonempty suite passed. Exit code `1` covers a failed or
inconclusive check, insufficient data, or an empty suite. Exit code `2` indicates invalid usage,
inputs, or suite data. Requested reports are written before a valid evaluation returns exit code
`1`.

## How discovery works

ModelDiff-ML keeps selection and evaluation separate.

1. It computes a loss difference for every observation.
2. It creates independent discovery and validation partitions.
3. Numeric conditions come from empirical quantiles. Categorical conditions use frequent values.
4. A bounded beam search builds short conjunctions and ranks them by effect and support.
5. Highly overlapping regions are removed using Jaccard similarity.
6. Selected conditions are applied unchanged to the validation partition.
7. Validation effects receive percentile bootstrap intervals, paired sign flip p-values, and
   Benjamini–Hochberg adjusted q-values.

This separation matters. Choosing a threshold and reporting a p-value from the same rows would
understate selection bias. Independent validation is a practical first defense, although it does
not turn exploratory subgroup discovery into a causal conclusion.

## Configuration and output schemas

`ModelDiffConfig` controls partition size, beam width, search depth, support limits, overlap
deduplication, bootstrap and permutation counts, confidence and significance levels, classification
threshold, numerical epsilon, and random seed. Defaults favor readable regions and avoid tiny
slices. Missing feature values are never imputed; a missing value does not match a condition.

Standard comparison reports retain schema `1.0`. Stability reports, newly saved check suites, and
check evaluation results use schema `1.1`. Anytime-valid monitor state and result snapshots each use
their own schema `1.0`. Version `1.0` check suites remain loadable and use the observed decision
policy. Serialized artifacts contain no model objects, raw observations, or original sample IDs.
Unavailable numeric results are represented as `null`, never `NaN` or infinity.

## Examples

The
[`classification example`](https://github.com/ecklein/modeldiff-ml/blob/main/examples/classification_example.py)
plants a local classification regression inside a candidate that improves overall. The
[`regression example`](https://github.com/ecklein/modeldiff-ml/blob/main/examples/regression_example.py)
does the same for continuous predictions. The
[`saved checks example`](https://github.com/ecklein/modeldiff-ml/blob/main/examples/saved_checks_example.py)
discovers a region, saves it, and evaluates both a failing and a passing candidate on fresh data.
The
[`anytime-valid checks example`](https://github.com/ecklein/modeldiff-ml/blob/main/examples/anytime_checks_example.py)
resumes a saved monitor as labeled batches arrive.

## Scope and limitations

ModelDiff-ML currently supports binary classification and regression. Discovered subgroups remain
exploratory. Independent validation and false discovery rate correction reduce false findings but
cannot eliminate them, especially with small data, distribution shift, correlated observations, or
repeated human analysis. A fresh final test set is appropriate before high-stakes deployment.

The project does not make causal claims or provide fairness certification. It also does not include
multiclass evaluation, distributed execution, hosted monitoring infrastructure, model registries,
dashboards, or automated remediation. Anytime-valid monitors are local, resumable statistical
objects rather than a scheduling, storage, or alerting service.

## Development

```bash
git clone https://github.com/ecklein/modeldiff-ml.git
cd modeldiff-ml
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
mypy src
```

Bug reports and focused contributions are welcome through
[GitHub Issues](https://github.com/ecklein/modeldiff-ml/issues).

## License

ModelDiff-ML is available under the
[MIT License](https://github.com/ecklein/modeldiff-ml/blob/main/LICENSE).
