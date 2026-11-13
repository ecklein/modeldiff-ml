from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeldiff import CheckSuite, ModelDiffConfig, SavedCheck
from modeldiff.analysis.regions import Condition
from modeldiff.cli import build_parser, main
from modeldiff.result import ModelDiffReport, RegionResult


def _region() -> RegionResult:
    return RegionResult(
        conditions=(Condition("x", ">", 2),),
        direction="regression",
        discovery_support=3,
        discovery_fraction=0.5,
        validation_support=3,
        validation_fraction=0.5,
        discovery_delta=0.5,
        validation_delta=0.5,
        confidence_interval=(0.4, 0.6),
        p_value=0.01,
        q_value=0.01,
        score=0.2,
        baseline_mean_loss=0.0,
        candidate_mean_loss=0.5,
        significant=True,
    )


def _report() -> ModelDiffReport:
    return ModelDiffReport(
        task="regression",
        metric="mae",
        sample_count=6,
        discovery_count=3,
        validation_count=3,
        overall={
            "baseline": 0.0,
            "candidate": 0.25,
            "absolute_delta": 0.25,
            "relative_delta": None,
        },
        aggregate_metrics={"baseline": {}, "candidate": {}},
        disagreement={},
        regression_regions=[_region()],
        configuration=ModelDiffConfig(n_bootstrap=10, n_permutations=10),
    )


def _suite(*, minimum: int = 3, empty: bool = False) -> CheckSuite:
    checks = ()
    if not empty:
        checks = (
            SavedCheck(
                check_id="high-x",
                name="High x",
                conditions=(Condition("x", ">", 2),),
                max_loss_increase=0.1,
                min_samples=minimum,
                provenance={"source": "test"},
            ),
        )
    return CheckSuite(task="regression", metric="mae", checks=checks)


def _write_data(path: Path, candidate: np.ndarray) -> None:
    pd.DataFrame(
        {
            "x": np.arange(6),
            "y": np.zeros(6),
            "old": np.zeros(6),
            "new": candidate,
        }
    ).to_csv(path, index=False)


def _check_args(checks: Path, data: Path, *extra: str) -> list[str]:
    return [
        "check",
        "--checks",
        str(checks),
        "--data",
        str(data),
        "--target",
        "y",
        "--baseline-pred",
        "old",
        "--candidate-pred",
        "new",
        *extra,
    ]


def test_compare_cli_exports_significant_checks(
    tmp_path: Path, monkeypatch: object
) -> None:
    import modeldiff.cli as cli

    data = tmp_path / "data.csv"
    checks = tmp_path / "checks.json"
    _write_data(data, np.full(6, 0.5))
    monkeypatch.setattr(cli, "compare_predictions", lambda *args, **kwargs: _report())  # type: ignore[attr-defined]

    exit_code = main(
        [
            "compare",
            "--data",
            str(data),
            "--target",
            "y",
            "--baseline-pred",
            "old",
            "--candidate-pred",
            "new",
            "--task",
            "regression",
            "--metric",
            "mae",
            "--export-checks",
            str(checks),
            "--check-max-loss-increase",
            "0.02",
            "--check-min-samples",
            "2",
        ]
    )

    exported = CheckSuite.load(checks)
    assert exit_code == 0
    assert len(exported.checks) == 1
    assert exported.checks[0].max_loss_increase == 0.02
    assert exported.checks[0].min_samples == 2


def test_check_cli_pass_fail_and_writes_failure_reports(tmp_path: Path) -> None:
    checks = tmp_path / "checks.json"
    _suite().to_json(checks)
    passing = tmp_path / "passing.csv"
    failing = tmp_path / "failing.csv"
    _write_data(passing, np.array([0, 0, 0, 0.05, 0.05, 0.05]))
    _write_data(failing, np.array([0, 0, 0, 0.5, 0.5, 0.5]))

    assert main(_check_args(checks, passing)) == 0
    result_json = tmp_path / "failed.json"
    result_html = tmp_path / "failed.html"
    assert (
        main(
            _check_args(
                checks,
                failing,
                "--json",
                str(result_json),
                "--html",
                str(result_html),
            )
        )
        == 1
    )
    assert json.loads(result_json.read_text())["overall_status"] == "fail"
    assert "<!doctype html>" in result_html.read_text()


def test_check_cli_returns_one_for_insufficient_data_and_empty_suite(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    _write_data(data, np.zeros(6))
    insufficient = tmp_path / "insufficient.json"
    empty = tmp_path / "empty.json"
    _suite(minimum=10).to_json(insufficient)
    _suite(empty=True).to_json(empty)

    assert main(_check_args(insufficient, data)) == 1
    assert main(_check_args(empty, data)) == 1


def test_check_cli_returns_two_for_invalid_suite_or_inputs(
    tmp_path: Path, capsys: object
) -> None:
    data = tmp_path / "data.csv"
    _write_data(data, np.zeros(6))
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json")
    assert main(_check_args(invalid, data)) == 2

    checks = tmp_path / "checks.json"
    _suite().to_json(checks)
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"x": range(6), "y": np.zeros(6), "old": np.zeros(6)}).to_csv(
        missing, index=False
    )
    assert main(_check_args(checks, missing)) == 2
    assert "error:" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_check_help_documents_exit_code_contract() -> None:
    help_text = build_parser()._subparsers._group_actions[0].choices["check"].format_help()
    assert "Exit 0: all pass" in help_text
    assert "exit 2: invalid suite" in help_text


def test_check_cli_writes_junit_for_confidence_gate(tmp_path: Path) -> None:
    checks = tmp_path / "checks.json"
    data = tmp_path / "data.csv"
    junit = tmp_path / "checks.xml"
    _suite().to_json(checks)
    _write_data(data, np.full(6, 0.5))

    exit_code = main(
        _check_args(
            checks,
            data,
            "--decision",
            "confidence",
            "--bootstrap",
            "50",
            "--junit",
            str(junit),
        )
    )

    assert exit_code == 1
    assert '<testsuite name="modeldiff-ml saved checks"' in junit.read_text()


def test_stability_cli_runs_stable_comparison(tmp_path: Path, monkeypatch: object) -> None:
    import modeldiff.cli as cli

    data = tmp_path / "data.csv"
    _write_data(data, np.full(6, 0.5))
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "compare_stable_predictions", lambda *args, **kwargs: _report()
    )

    exit_code = main(
        [
            "stability",
            "--data",
            str(data),
            "--target",
            "y",
            "--baseline-pred",
            "old",
            "--candidate-pred",
            "new",
            "--task",
            "regression",
            "--metric",
            "mae",
            "--n-resamples",
            "2",
        ]
    )

    assert exit_code == 0


def _write_monitor_data(path: Path, candidate: float, n: int = 5000) -> None:
    pd.DataFrame(
        {
            "x": np.arange(n),
            "y": np.zeros(n),
            "old": np.zeros(n),
            "new": np.full(n, candidate),
        }
    ).to_csv(path, index=False)


def test_monitor_cli_creates_resumes_and_reports_state(tmp_path: Path) -> None:
    checks = tmp_path / "checks.json"
    state = tmp_path / "monitor.json"
    data = tmp_path / "batch.csv"
    result_json = tmp_path / "result.json"
    junit = tmp_path / "result.xml"
    _suite(minimum=1).to_json(checks)
    _write_monitor_data(data, 0.0)
    common = [
        "--state",
        str(state),
        "--data",
        str(data),
        "--target",
        "y",
        "--baseline-pred",
        "old",
        "--candidate-pred",
        "new",
    ]

    first = main(
        [
            "monitor",
            "--checks",
            str(checks),
            *common,
            "--loss-delta-lower",
            "-1",
            "--loss-delta-upper",
            "1",
        ]
    )
    second = main(
        [
            "monitor",
            *common,
            "--json",
            str(result_json),
            "--junit",
            str(junit),
        ]
    )

    assert first == second == 0
    assert json.loads(state.read_text())["sample_count"] == 10_000
    assert json.loads(result_json.read_text())["overall_status"] == "pass"
    assert "anytime-valid checks" in junit.read_text()


def test_monitor_cli_rejects_missing_initial_suite_and_policy_changes(
    tmp_path: Path, capsys: object
) -> None:
    state = tmp_path / "monitor.json"
    data = tmp_path / "batch.csv"
    _write_monitor_data(data, 0.05, n=10)
    common = [
        "monitor",
        "--state",
        str(state),
        "--data",
        str(data),
        "--target",
        "y",
        "--baseline-pred",
        "old",
        "--candidate-pred",
        "new",
    ]

    assert main(common) == 2
    assert "--checks is required" in capsys.readouterr().err  # type: ignore[attr-defined]

    checks = tmp_path / "checks.json"
    _suite(minimum=1).to_json(checks)
    assert (
        main(
            [
                *common,
                "--checks",
                str(checks),
                "--loss-delta-lower",
                "-1",
                "--loss-delta-upper",
                "1",
            ]
        )
        == 1
    )
    assert main([*common, "--confidence-level", "0.9"]) == 2
    assert "only be set when creating" in capsys.readouterr().err  # type: ignore[attr-defined]
