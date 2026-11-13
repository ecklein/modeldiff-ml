"""Anytime-valid, resumable monitoring for saved regression checks."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal, cast
from xml.etree import ElementTree

import numpy as np

from modeldiff.analysis.regions import Condition
from modeldiff.checks import (
    CheckSuite,
    _check_mask,
    _finite_or_none,
    _validate_feature_schema,
    loss_unit,
)
from modeldiff.metrics.base import observation_losses
from modeldiff.utils import validate_inputs

MONITOR_STATE_SCHEMA_VERSION = "1.0"
MONITOR_RESULT_SCHEMA_VERSION = "1.0"

SequentialCheckStatus = Literal["pass", "fail", "continue", "insufficient_data"]
SequentialOverallStatus = Literal[
    "pass", "fail", "continue", "insufficient_data", "no_checks"
]


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class SequentialCheckPolicy:
    """Configuration for time-uniform saved-check decisions.

    The confidence sequence uses two-sided Hoeffding bounds with an alpha-spending
    schedule ``alpha / (n * (n + 1))``. This is conservative but valid at every
    observation count under independent bounded observations.
    """

    confidence_level: float = 0.95
    correction: Literal["bonferroni", "none"] = "bonferroni"
    loss_delta_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.correction not in {"bonferroni", "none"}:
            raise ValueError("correction must be 'bonferroni' or 'none'")
        if self.loss_delta_bounds is not None:
            if len(self.loss_delta_bounds) != 2:
                raise ValueError("loss_delta_bounds must contain exactly two values")
            lower = _finite_number(self.loss_delta_bounds[0], "loss delta lower bound")
            upper = _finite_number(self.loss_delta_bounds[1], "loss delta upper bound")
            if lower >= upper:
                raise ValueError("loss_delta_bounds must satisfy lower < upper")
            object.__setattr__(self, "loss_delta_bounds", (lower, upper))

    def resolve_bounds(self, suite: CheckSuite) -> tuple[float, float]:
        """Return explicit or metric-derived bounds for one-observation deltas."""
        if self.loss_delta_bounds is not None:
            return self.loss_delta_bounds
        if suite.task == "binary_classification":
            if suite.metric in {"accuracy", "brier"}:
                return (-1.0, 1.0)
            maximum_loss = -math.log(suite.epsilon)
            return (-maximum_loss, maximum_loss)
        raise ValueError(
            "regression monitoring requires explicit finite loss_delta_bounds"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "correction": self.correction,
            "loss_delta_bounds": (
                list(self.loss_delta_bounds) if self.loss_delta_bounds is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Any) -> SequentialCheckPolicy:
        if not isinstance(data, dict):
            raise ValueError("sequential policy must be an object")
        required = {"confidence_level", "correction", "loss_delta_bounds"}
        missing = required - set(data)
        if missing:
            raise ValueError(
                f"sequential policy is missing fields: {', '.join(sorted(missing))}"
            )
        bounds = data.get("loss_delta_bounds")
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2):
            raise ValueError("loss_delta_bounds must be null or a two-value array")
        return cls(
            confidence_level=data["confidence_level"],
            correction=data["correction"],
            loss_delta_bounds=tuple(bounds) if bounds is not None else None,
        )


@dataclass
class _Accumulator:
    support_count: int = 0
    baseline_loss_sum: float = 0.0
    candidate_loss_sum: float = 0.0
    loss_delta_sum: float = 0.0

    def to_dict(self, check_id: str) -> dict[str, Any]:
        return {
            "id": check_id,
            "support_count": self.support_count,
            "baseline_loss_sum": self.baseline_loss_sum,
            "candidate_loss_sum": self.candidate_loss_sum,
            "loss_delta_sum": self.loss_delta_sum,
        }


@dataclass(frozen=True)
class SequentialCheckOutcome:
    """Current anytime-valid conclusion for one fixed saved check."""

    check_id: str
    name: str
    conditions: tuple[Condition, ...]
    support_count: int
    support_fraction: float
    baseline_mean_loss: float | None
    candidate_mean_loss: float | None
    loss_delta: float | None
    confidence_sequence: tuple[float, float] | None
    max_loss_increase: float
    min_samples: int
    status: SequentialCheckStatus
    explanation: str

    @property
    def description(self) -> str:
        return " AND ".join(str(condition) for condition in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "name": self.name,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "description": self.description,
            "support_count": self.support_count,
            "support_fraction": self.support_fraction,
            "baseline_mean_loss": self.baseline_mean_loss,
            "candidate_mean_loss": self.candidate_mean_loss,
            "loss_delta": self.loss_delta,
            "confidence_sequence": (
                list(self.confidence_sequence)
                if self.confidence_sequence is not None
                else None
            ),
            "max_loss_increase": self.max_loss_increase,
            "min_samples": self.min_samples,
            "status": self.status,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class SequentialCheckResult:
    """Snapshot of an anytime-valid saved-check monitor."""

    task: str
    metric: str
    sample_count: int
    outcomes: tuple[SequentialCheckOutcome, ...]
    suite_schema_version: str
    policy: SequentialCheckPolicy
    loss_delta_bounds: tuple[float, float]
    schema_version: str = MONITOR_RESULT_SCHEMA_VERSION

    @property
    def passed_count(self) -> int:
        return sum(outcome.status == "pass" for outcome in self.outcomes)

    @property
    def failed_count(self) -> int:
        return sum(outcome.status == "fail" for outcome in self.outcomes)

    @property
    def continuing_count(self) -> int:
        return sum(outcome.status == "continue" for outcome in self.outcomes)

    @property
    def insufficient_data_count(self) -> int:
        return sum(outcome.status == "insufficient_data" for outcome in self.outcomes)

    @property
    def overall_status(self) -> SequentialOverallStatus:
        if not self.outcomes:
            return "no_checks"
        if self.failed_count:
            return "fail"
        if self.continuing_count:
            return "continue"
        if self.insufficient_data_count:
            return "insufficient_data"
        return "pass"

    @property
    def loss_unit(self) -> str:
        return loss_unit(self.task, self.metric)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "suite_schema_version": self.suite_schema_version,
            "method": "time_uniform_hoeffding_alpha_spending",
            "task": self.task,
            "metric": self.metric,
            "loss_unit": self.loss_unit,
            "sample_count": self.sample_count,
            "overall_status": self.overall_status,
            "policy": self.policy.to_dict(),
            "loss_delta_bounds": list(self.loss_delta_bounds),
            "counts": {
                "total": len(self.outcomes),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "continuing": self.continuing_count,
                "insufficient_data": self.insufficient_data_count,
            },
            "checks": [outcome.to_dict() for outcome in self.outcomes],
        }
        return cast(dict[str, Any], _finite_or_none(data))

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        rendered = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            target = Path(path)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.{id(self)}.tmp")
            try:
                temporary.write_text(rendered + "\n", encoding="utf-8")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return rendered

    def summary(self) -> str:
        lines = [
            "ModelDiff-ML Anytime-Valid Checks",
            "==================================",
            "",
            f"Overall status: {self.overall_status}",
            f"Samples observed: {self.sample_count:,}",
            f"Task / metric: {self.task} / {self.metric}",
            f"Confidence level: {self.policy.confidence_level:.1%} "
            f"({self.policy.correction})",
            f"Delta bounds: [{self.loss_delta_bounds[0]:.6g}, "
            f"{self.loss_delta_bounds[1]:.6g}]",
            "",
        ]
        if not self.outcomes:
            lines.append("No checks monitored: the saved suite is empty.")
            return "\n".join(lines)
        for outcome in self.outcomes:
            interval = (
                "n/a"
                if outcome.confidence_sequence is None
                else (
                    f"[{outcome.confidence_sequence[0]:+.6g}, "
                    f"{outcome.confidence_sequence[1]:+.6g}]"
                )
            )
            delta = "n/a" if outcome.loss_delta is None else f"{outcome.loss_delta:+.6g}"
            lines.extend(
                [
                    f"[{outcome.status.upper()}] {outcome.name} ({outcome.check_id})",
                    f"  Region: {outcome.description}",
                    f"  Support: {outcome.support_count:,}/{self.sample_count:,}",
                    f"  Delta: {delta}; confidence sequence: {interval}; "
                    f"tolerance: {outcome.max_loss_increase:.6g}",
                    f"  {outcome.explanation}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()

    def to_junit_xml(self, path: str | Path | None = None) -> str:
        suite = ElementTree.Element(
            "testsuite",
            name="modeldiff-ml anytime-valid checks",
            tests=str(len(self.outcomes)),
            failures=str(self.failed_count),
            skipped=str(self.continuing_count + self.insufficient_data_count),
        )
        for outcome in self.outcomes:
            case = ElementTree.SubElement(
                suite,
                "testcase",
                classname="modeldiff.anytime_checks",
                name=f"{outcome.name} ({outcome.check_id})",
            )
            if outcome.status == "fail":
                node = ElementTree.SubElement(case, "failure", message=outcome.explanation)
                node.text = outcome.description
            elif outcome.status in {"continue", "insufficient_data"}:
                ElementTree.SubElement(case, "skipped", message=outcome.explanation)
        rendered = ElementTree.tostring(suite, encoding="unicode")
        if path is not None:
            Path(path).write_text(rendered + "\n", encoding="utf-8")
        return rendered

    def to_html(self, path: str | Path | None = None) -> str:
        rows = "".join(
            "<tr>"
            f"<td><strong>{escape(outcome.status)}</strong></td>"
            f"<td>{escape(outcome.name)}<br><small>{escape(outcome.check_id)}</small></td>"
            f"<td>{escape(outcome.description)}</td>"
            f"<td>{outcome.support_count:,}</td>"
            f"<td>{'n/a' if outcome.loss_delta is None else f'{outcome.loss_delta:+.6g}'}</td>"
            f"<td>{'n/a' if outcome.confidence_sequence is None else f'[{outcome.confidence_sequence[0]:+.6g}, {outcome.confidence_sequence[1]:+.6g}]'}</td>"
            f"<td>{outcome.max_loss_increase:.6g}</td>"
            f"<td>{escape(outcome.explanation)}</td>"
            "</tr>"
            for outcome in self.outcomes
        )
        rendered = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ModelDiff-ML Anytime-Valid Checks</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#17202a}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}th{{font-size:12px;text-transform:uppercase;color:#667085}}small{{color:#667085}}
</style></head><body><h1>ModelDiff-ML Anytime-Valid Checks</h1>
<p>Overall status: <strong>{escape(self.overall_status)}</strong> · Samples: {self.sample_count:,} · Confidence: {self.policy.confidence_level:.1%}</p>
<table><thead><tr><th>Status</th><th>Check</th><th>Region</th><th>Support</th><th>Delta</th><th>Confidence sequence</th><th>Tolerance</th><th>Explanation</th></tr></thead><tbody>{rows}</tbody></table>
<p>Time-uniform Hoeffding confidence sequences remain valid under repeated inspection, assuming independent observations and valid fixed bounds.</p>
</body></html>"""
        if path is not None:
            Path(path).write_text(rendered, encoding="utf-8")
        return rendered


def _suite_fingerprint(suite: CheckSuite) -> str:
    payload = json.dumps(
        suite.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CheckMonitor:
    """Accumulate labeled batches and expose anytime-valid saved-check decisions."""

    suite: CheckSuite
    policy: SequentialCheckPolicy = field(default_factory=SequentialCheckPolicy)
    sample_count: int = 0
    _accumulators: dict[str, _Accumulator] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.suite, CheckSuite):
            raise ValueError("suite must be a CheckSuite")
        if not isinstance(self.policy, SequentialCheckPolicy):
            raise ValueError("policy must be a SequentialCheckPolicy")
        self.policy.resolve_bounds(self.suite)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, numbers.Integral)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be a nonnegative integer")
        self.sample_count = int(self.sample_count)
        expected = {check.check_id for check in self.suite.checks}
        if not self._accumulators:
            self._accumulators = {check_id: _Accumulator() for check_id in expected}
        if set(self._accumulators) != expected:
            raise ValueError("monitor accumulators must match saved check IDs")

    @property
    def loss_delta_bounds(self) -> tuple[float, float]:
        return self.policy.resolve_bounds(self.suite)

    def update(
        self,
        baseline_predictions: Any,
        candidate_predictions: Any,
        X: Any,
        y: Any,
    ) -> SequentialCheckResult:
        """Atomically incorporate one labeled batch and return the new snapshot."""
        frame, target, baseline, candidate = validate_inputs(
            X,
            y,
            baseline_predictions,
            candidate_predictions,
            self.suite.task,
            self.suite.metric,
            min_samples=1,
        )
        baseline_loss = observation_losses(
            self.suite.task,
            self.suite.metric,
            target,
            baseline,
            threshold=self.suite.threshold,
            epsilon=self.suite.epsilon,
        )
        candidate_loss = observation_losses(
            self.suite.task,
            self.suite.metric,
            target,
            candidate,
            threshold=self.suite.threshold,
            epsilon=self.suite.epsilon,
        )
        delta = candidate_loss - baseline_loss
        lower, upper = self.loss_delta_bounds
        increments: dict[str, _Accumulator] = {}
        for check in self.suite.checks:
            _validate_feature_schema(frame, check)
            mask = _check_mask(frame, check)
            region_delta = delta[mask]
            outside = (region_delta < lower) | (region_delta > upper)
            if outside.any():
                observed_min = float(np.min(region_delta))
                observed_max = float(np.max(region_delta))
                raise ValueError(
                    f"check {check.check_id!r} observed a loss delta outside configured bounds: "
                    f"observed [{observed_min:.6g}, {observed_max:.6g}], "
                    f"configured [{lower:.6g}, {upper:.6g}]"
                )
            increments[check.check_id] = _Accumulator(
                support_count=int(mask.sum()),
                baseline_loss_sum=float(np.sum(baseline_loss[mask])),
                candidate_loss_sum=float(np.sum(candidate_loss[mask])),
                loss_delta_sum=float(np.sum(region_delta)),
            )
        for check_id, increment in increments.items():
            accumulator = self._accumulators[check_id]
            accumulator.support_count += increment.support_count
            accumulator.baseline_loss_sum += increment.baseline_loss_sum
            accumulator.candidate_loss_sum += increment.candidate_loss_sum
            accumulator.loss_delta_sum += increment.loss_delta_sum
        self.sample_count += len(frame)
        return self.result()

    def _confidence_sequence(
        self, mean: float, count: int, alpha: float
    ) -> tuple[float, float]:
        lower, upper = self.loss_delta_bounds
        alpha_at_count = alpha / (count * (count + 1))
        radius = (upper - lower) * math.sqrt(
            math.log(2 / alpha_at_count) / (2 * count)
        )
        return max(lower, mean - radius), min(upper, mean + radius)

    def result(self) -> SequentialCheckResult:
        """Return the current time-uniform monitoring snapshot without mutation."""
        check_count = len(self.suite.checks)
        alpha = 1 - self.policy.confidence_level
        if self.policy.correction == "bonferroni" and check_count:
            alpha /= check_count
        outcomes: list[SequentialCheckOutcome] = []
        for check in self.suite.checks:
            accumulator = self._accumulators[check.check_id]
            count = accumulator.support_count
            fraction = count / self.sample_count if self.sample_count else 0.0
            baseline_mean: float | None
            candidate_mean: float | None
            mean_delta: float | None
            interval: tuple[float, float] | None
            if count:
                baseline_mean = accumulator.baseline_loss_sum / count
                candidate_mean = accumulator.candidate_loss_sum / count
                mean_delta = accumulator.loss_delta_sum / count
                interval = self._confidence_sequence(mean_delta, count, alpha)
            else:
                baseline_mean = candidate_mean = mean_delta = None
                interval = None
            if count < check.min_samples:
                status: SequentialCheckStatus = "insufficient_data"
                explanation = (
                    f"Observed {count} matching samples; at least "
                    f"{check.min_samples} are required."
                )
            elif interval is not None and interval[1] <= check.max_loss_increase:
                status = "pass"
                explanation = (
                    f"The time-uniform upper bound {interval[1]:.6g} is within "
                    f"tolerance {check.max_loss_increase:.6g}."
                )
            elif interval is not None and interval[0] > check.max_loss_increase:
                status = "fail"
                explanation = (
                    f"The time-uniform lower bound {interval[0]:.6g} exceeds "
                    f"tolerance {check.max_loss_increase:.6g}."
                )
            else:
                status = "continue"
                explanation = (
                    "The confidence sequence still crosses the tolerance; "
                    "continue collecting labeled observations."
                )
            outcomes.append(
                SequentialCheckOutcome(
                    check_id=check.check_id,
                    name=check.name,
                    conditions=check.conditions,
                    support_count=count,
                    support_fraction=fraction,
                    baseline_mean_loss=baseline_mean,
                    candidate_mean_loss=candidate_mean,
                    loss_delta=mean_delta,
                    confidence_sequence=interval,
                    max_loss_increase=check.max_loss_increase,
                    min_samples=check.min_samples,
                    status=status,
                    explanation=explanation,
                )
            )
        return SequentialCheckResult(
            task=self.suite.task,
            metric=self.suite.metric,
            sample_count=self.sample_count,
            outcomes=tuple(outcomes),
            suite_schema_version=self.suite.schema_version,
            policy=self.policy,
            loss_delta_bounds=self.loss_delta_bounds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete, resumable monitor-state mapping."""
        return {
            "schema_version": MONITOR_STATE_SCHEMA_VERSION,
            "suite_fingerprint": _suite_fingerprint(self.suite),
            "suite": self.suite.to_dict(),
            "policy": self.policy.to_dict(),
            "sample_count": self.sample_count,
            "accumulators": [
                self._accumulators[check.check_id].to_dict(check.check_id)
                for check in self.suite.checks
            ],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        rendered = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            target = Path(path)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.{id(self)}.tmp")
            try:
                temporary.write_text(rendered + "\n", encoding="utf-8")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return rendered

    @classmethod
    def from_dict(cls, data: Any) -> CheckMonitor:
        if not isinstance(data, dict):
            raise ValueError("monitor state must be a JSON object")
        required = {
            "schema_version",
            "suite_fingerprint",
            "suite",
            "policy",
            "sample_count",
            "accumulators",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"monitor state is missing fields: {', '.join(sorted(missing))}")
        if data["schema_version"] != MONITOR_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported monitor-state schema version: {data['schema_version']!r}"
            )
        suite = CheckSuite.from_dict(data["suite"])
        if data["suite_fingerprint"] != _suite_fingerprint(suite):
            raise ValueError("monitor state suite fingerprint does not match its suite")
        if (
            isinstance(data["sample_count"], bool)
            or not isinstance(data["sample_count"], numbers.Integral)
            or int(data["sample_count"]) < 0
        ):
            raise ValueError("monitor sample_count must be a nonnegative integer")
        policy = SequentialCheckPolicy.from_dict(data["policy"])
        lower_bound, upper_bound = policy.resolve_bounds(suite)
        if not isinstance(data["accumulators"], list):
            raise ValueError("monitor accumulators must be an array")
        accumulators: dict[str, _Accumulator] = {}
        for item in data["accumulators"]:
            if not isinstance(item, dict):
                raise ValueError("each monitor accumulator must be an object")
            item_required = {
                "id",
                "support_count",
                "baseline_loss_sum",
                "candidate_loss_sum",
                "loss_delta_sum",
            }
            if not item_required.issubset(item):
                raise ValueError("monitor accumulator is missing required fields")
            check_id = item["id"]
            if not isinstance(check_id, str) or check_id in accumulators:
                raise ValueError("monitor accumulator IDs must be unique strings")
            count = item["support_count"]
            if (
                isinstance(count, bool)
                or not isinstance(count, numbers.Integral)
                or not 0 <= int(count) <= int(data["sample_count"])
            ):
                raise ValueError("monitor support_count is invalid")
            baseline_sum = _finite_number(item["baseline_loss_sum"], "baseline_loss_sum")
            candidate_sum = _finite_number(item["candidate_loss_sum"], "candidate_loss_sum")
            delta_sum = _finite_number(item["loss_delta_sum"], "loss_delta_sum")
            if baseline_sum < 0 or candidate_sum < 0:
                raise ValueError("monitor loss sums must be nonnegative")
            if not math.isclose(
                candidate_sum - baseline_sum,
                delta_sum,
                rel_tol=1e-10,
                abs_tol=1e-10 * max(1, int(count)),
            ):
                raise ValueError("monitor loss sums are inconsistent")
            tolerance = 1e-10 * max(1, int(count))
            if not (
                lower_bound * int(count) - tolerance
                <= delta_sum
                <= upper_bound * int(count) + tolerance
            ):
                raise ValueError("monitor delta sum is outside configured bounds")
            accumulators[check_id] = _Accumulator(
                support_count=int(count),
                baseline_loss_sum=baseline_sum,
                candidate_loss_sum=candidate_sum,
                loss_delta_sum=delta_sum,
            )
        return cls(
            suite=suite,
            policy=policy,
            sample_count=int(data["sample_count"]),
            _accumulators=accumulators,
        )

    @classmethod
    def from_json(cls, text: str) -> CheckMonitor:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid monitor-state JSON: {error.msg}") from error
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> CheckMonitor:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
