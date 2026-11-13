from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeldiff import (
    CheckMonitor,
    CheckSuite,
    SavedCheck,
    SequentialCheckPolicy,
)
from modeldiff.analysis.regions import Condition


def _suite(*, tolerance: float = 0.5, minimum: int = 1, metric: str = "mae") -> CheckSuite:
    return CheckSuite(
        task="regression" if metric == "mae" else "binary_classification",
        metric=metric,
        checks=(
            SavedCheck(
                check_id="all-x",
                name="All observations",
                conditions=(Condition("x", ">", -1),),
                max_loss_increase=tolerance,
                min_samples=minimum,
                provenance={"source": "test"},
            ),
        ),
    )


def _batch(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    n = len(candidate)
    return np.zeros(n), candidate, pd.DataFrame({"x": np.arange(n)}), np.zeros(n)


def _regression_monitor(
    *, tolerance: float = 0.5, minimum: int = 1
) -> CheckMonitor:
    return CheckMonitor(
        _suite(tolerance=tolerance, minimum=minimum),
        SequentialCheckPolicy(loss_delta_bounds=(-1, 1)),
    )


@pytest.mark.parametrize(
    ("candidate_value", "expected"),
    [(0.25, "pass"), (0.75, "fail"), (0.50, "continue")],
)
def test_monitor_reaches_time_uniform_decisions(
    candidate_value: float, expected: str
) -> None:
    monitor = _regression_monitor()
    baseline, candidate, X, y = _batch(np.full(5000, candidate_value))

    result = monitor.update(baseline, candidate, X, y)

    assert result.overall_status == expected
    assert result.outcomes[0].status == expected
    assert result.outcomes[0].loss_delta == pytest.approx(candidate_value)
    assert result.outcomes[0].confidence_sequence is not None


def test_monitor_is_chunk_invariant_and_supports_single_row_batches() -> None:
    values = np.r_[np.zeros(2500), np.full(2500, 0.25)]
    one_batch = _regression_monitor()
    many_batches = _regression_monitor()

    baseline, candidate, X, y = _batch(values)
    one_result = one_batch.update(baseline, candidate, X, y)
    for indices in np.array_split(np.arange(len(values)), 100):
        many_batches.update(
            baseline[indices],
            candidate[indices],
            X.iloc[indices].reset_index(drop=True),
            y[indices],
        )
    many_result = many_batches.result()

    assert many_result.to_dict() == one_result.to_dict()
    assert many_batches.to_dict() == one_batch.to_dict()


def test_monitor_state_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    monitor = _regression_monitor()
    baseline, candidate, X, y = _batch(np.full(100, 0.25))
    monitor.update(baseline, candidate, X, y)
    path = tmp_path / "state.json"
    monitor.to_json(path)

    loaded = CheckMonitor.load(path)

    assert loaded.to_dict() == monitor.to_dict()
    assert loaded.result().to_dict() == monitor.result().to_dict()
    tampered = json.loads(path.read_text())
    tampered["suite"]["checks"][0]["max_loss_increase"] = 0.9
    with pytest.raises(ValueError, match="fingerprint"):
        CheckMonitor.from_dict(tampered)

    missing_policy = monitor.to_dict()
    del missing_policy["policy"]["correction"]
    with pytest.raises(ValueError, match="sequential policy is missing"):
        CheckMonitor.from_dict(missing_policy)

    invalid_sum = monitor.to_dict()
    invalid_sum["accumulators"][0]["candidate_loss_sum"] = 500.0
    invalid_sum["accumulators"][0]["loss_delta_sum"] = 500.0
    with pytest.raises(ValueError, match="outside configured bounds"):
        CheckMonitor.from_dict(invalid_sum)


def test_bound_violation_is_atomic_and_only_checks_matching_rows() -> None:
    monitor = _regression_monitor()
    before = monitor.to_json()
    baseline, candidate, X, y = _batch(np.full(10, 2.0))
    with pytest.raises(ValueError, match="outside configured bounds"):
        monitor.update(baseline, candidate, X, y)
    assert monitor.to_json() == before

    outside_region = CheckSuite(
        task="regression",
        metric="mae",
        checks=(
            SavedCheck(
                check_id="none",
                name="No rows",
                conditions=(Condition("x", ">", 100),),
                max_loss_increase=0,
                min_samples=1,
                provenance={},
            ),
        ),
    )
    monitor = CheckMonitor(
        outside_region, SequentialCheckPolicy(loss_delta_bounds=(-1, 1))
    )
    result = monitor.update(baseline, candidate, X, y)
    assert result.overall_status == "insufficient_data"


def test_regression_requires_bounds_and_classification_derives_them() -> None:
    with pytest.raises(ValueError, match="regression monitoring requires"):
        CheckMonitor(_suite())

    classification = CheckMonitor(_suite(metric="accuracy"))
    assert classification.loss_delta_bounds == (-1, 1)

    log_loss = CheckMonitor(_suite(metric="log_loss"))
    maximum = -math.log(log_loss.suite.epsilon)
    assert log_loss.loss_delta_bounds == (-maximum, maximum)


def test_minimum_support_and_empty_suite_do_not_pass() -> None:
    monitor = _regression_monitor(minimum=20)
    baseline, candidate, X, y = _batch(np.zeros(10))
    result = monitor.update(baseline, candidate, X, y)
    assert result.overall_status == "insufficient_data"

    empty = CheckMonitor(
        CheckSuite(task="regression", metric="mae"),
        SequentialCheckPolicy(loss_delta_bounds=(-1, 1)),
    )
    assert empty.result().overall_status == "no_checks"


def test_confidence_sequence_matches_documented_alpha_spending_formula() -> None:
    monitor = _regression_monitor()
    baseline, candidate, X, y = _batch(np.full(100, 0.25))
    outcome = monitor.update(baseline, candidate, X, y).outcomes[0]
    alpha_n = 0.05 / (100 * 101)
    radius = 2 * math.sqrt(math.log(2 / alpha_n) / 200)

    assert outcome.confidence_sequence == pytest.approx(
        (max(-1, 0.25 - radius), min(1, 0.25 + radius))
    )


def test_bonferroni_produces_simultaneous_wider_sequences() -> None:
    base_check = _suite().checks[0]
    suite = CheckSuite(
        task="regression",
        metric="mae",
        checks=(
            base_check,
            SavedCheck(
                check_id="second",
                name="Second check",
                conditions=base_check.conditions,
                max_loss_increase=base_check.max_loss_increase,
                min_samples=base_check.min_samples,
                provenance={},
            ),
        ),
    )
    corrected = CheckMonitor(
        suite,
        SequentialCheckPolicy(correction="bonferroni", loss_delta_bounds=(-1, 1)),
    )
    uncorrected = CheckMonitor(
        suite,
        SequentialCheckPolicy(correction="none", loss_delta_bounds=(-1, 1)),
    )
    baseline, candidate, X, y = _batch(np.full(100, 0.25))

    corrected_interval = corrected.update(baseline, candidate, X, y).outcomes[
        0
    ].confidence_sequence
    uncorrected_interval = uncorrected.update(baseline, candidate, X, y).outcomes[
        0
    ].confidence_sequence

    assert corrected_interval is not None
    assert uncorrected_interval is not None
    assert corrected_interval[0] < uncorrected_interval[0]
    assert corrected_interval[1] > uncorrected_interval[1]


def test_repeated_inspection_does_not_false_fail_at_the_tolerance() -> None:
    rng = np.random.default_rng(19)
    for _ in range(25):
        monitor = _regression_monitor(tolerance=0.5)
        for _ in range(10):
            values = rng.choice([0.0, 1.0], size=100)
            baseline, candidate, X, y = _batch(values)
            result = monitor.update(baseline, candidate, X, y)
            assert result.overall_status != "fail"


def test_monitor_outputs_escape_html_and_emit_junit(tmp_path: Path) -> None:
    suite = CheckSuite(
        task="regression",
        metric="mae",
        checks=(
            SavedCheck(
                check_id="html",
                name="<script>alert(1)</script>",
                conditions=(Condition("<x>", ">", -1),),
                max_loss_increase=0.5,
                min_samples=1,
                provenance={},
            ),
        ),
    )
    monitor = CheckMonitor(suite, SequentialCheckPolicy(loss_delta_bounds=(-1, 1)))
    result = monitor.update(
        np.zeros(5),
        np.full(5, 0.5),
        pd.DataFrame({"<x>": np.arange(5)}),
        np.zeros(5),
    )
    html = result.to_html(tmp_path / "result.html")
    junit = result.to_junit_xml(tmp_path / "result.xml")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'name="modeldiff-ml anytime-valid checks"' in junit
