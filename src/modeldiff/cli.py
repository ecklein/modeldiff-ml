"""Command-line interface for precomputed-prediction comparisons."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from modeldiff import (
    CheckDecisionPolicy,
    CheckMonitor,
    CheckSuite,
    ModelDiffConfig,
    SequentialCheckPolicy,
    StabilityConfig,
    compare_predictions,
    compare_stable_predictions,
    evaluate_checks,
)


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(prog="modeldiff-ml", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare", help="compare predictions in a CSV file")
    compare_parser.add_argument("--data", required=True, type=Path)
    compare_parser.add_argument("--target", required=True)
    compare_parser.add_argument("--baseline-pred", required=True)
    compare_parser.add_argument("--candidate-pred", required=True)
    compare_parser.add_argument(
        "--task", required=True, choices=["binary_classification", "regression"]
    )
    compare_parser.add_argument("--metric", required=True)
    compare_parser.add_argument("--exclude", action="append", default=[])
    compare_parser.add_argument("--html", type=Path)
    compare_parser.add_argument("--json", type=Path)
    compare_parser.add_argument("--random-state", type=int, default=42)
    compare_parser.add_argument(
        "--validation-strategy",
        choices=["random", "group", "time", "explicit"],
        default="random",
        help="how to partition discovery and validation rows (default: random)",
    )
    compare_parser.add_argument("--group-column", help="group identifier for group splitting")
    compare_parser.add_argument("--time-column", help="ordering column for time splitting")
    compare_parser.add_argument(
        "--split-column",
        help="column containing 'discovery'/'validation' labels for explicit splitting",
    )
    compare_parser.add_argument("--fail-on-significant-regression", action="store_true")
    compare_parser.add_argument("--max-overall-regression", type=float)
    compare_parser.add_argument(
        "--export-checks",
        type=Path,
        help="export significant regression regions as a reusable check suite",
    )
    compare_parser.add_argument(
        "--check-max-loss-increase",
        type=_nonnegative_float,
        default=0.0,
        help="allowed mean loss increase for exported checks (default: 0)",
    )
    compare_parser.add_argument(
        "--check-min-samples",
        type=_positive_int,
        default=30,
        help="minimum fresh-data support for exported checks (default: 30)",
    )

    stability_parser = subparsers.add_parser(
        "stability", help="discover regressions that survive repeated resampling"
    )
    stability_parser.add_argument("--data", required=True, type=Path)
    stability_parser.add_argument("--target", required=True)
    stability_parser.add_argument("--baseline-pred", required=True)
    stability_parser.add_argument("--candidate-pred", required=True)
    stability_parser.add_argument(
        "--task", required=True, choices=["binary_classification", "regression"]
    )
    stability_parser.add_argument("--metric", required=True)
    stability_parser.add_argument("--exclude", action="append", default=[])
    stability_parser.add_argument("--html", type=Path)
    stability_parser.add_argument("--json", type=Path)
    stability_parser.add_argument("--known-checks", type=Path)
    stability_parser.add_argument("--random-state", type=int, default=42)
    stability_parser.add_argument("--n-resamples", type=_positive_int, default=30)
    stability_parser.add_argument("--subsample-fraction", type=float, default=0.75)
    stability_parser.add_argument("--min-selection-frequency", type=float, default=0.5)
    stability_parser.add_argument(
        "--validation-strategy", choices=["random", "group", "time", "explicit"], default="random"
    )
    stability_parser.add_argument("--group-column")
    stability_parser.add_argument("--time-column")
    stability_parser.add_argument("--split-column")
    stability_parser.add_argument("--fail-on-significant-regression", action="store_true")

    check_parser = subparsers.add_parser(
        "check",
        help="evaluate a saved check suite (exit 0 pass, 1 fail/incomplete/empty, 2 error)",
        description=(
            "Evaluate fixed saved checks. Exit 0: all pass; exit 1: failure, insufficient "
            "data, or empty suite; exit 2: invalid suite, inputs, or usage."
        ),
    )
    check_parser.add_argument("--checks", required=True, type=Path)
    check_parser.add_argument("--data", required=True, type=Path)
    check_parser.add_argument("--target", required=True)
    check_parser.add_argument("--baseline-pred", required=True)
    check_parser.add_argument("--candidate-pred", required=True)
    check_parser.add_argument("--exclude", action="append", default=[])
    check_parser.add_argument("--html", type=Path)
    check_parser.add_argument("--json", type=Path)
    check_parser.add_argument("--junit", type=Path)
    check_parser.add_argument("--decision", choices=["observed", "confidence"])
    check_parser.add_argument("--confidence-level", type=float, default=0.95)
    check_parser.add_argument("--correction", choices=["bonferroni", "none"], default="bonferroni")
    check_parser.add_argument("--bootstrap", type=_positive_int, default=2000)
    check_parser.add_argument("--random-state", type=int, default=42)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="update resumable, anytime-valid saved checks with one labeled batch",
        description=(
            "Update a monitor state with one labeled batch. Exit 0: every check passes; "
            "exit 1: fail, continue, insufficient data, or empty suite; exit 2: invalid state, "
            "bounds, inputs, or usage."
        ),
    )
    monitor_parser.add_argument("--state", required=True, type=Path)
    monitor_parser.add_argument(
        "--checks", type=Path, help="saved suite; required only when creating monitor state"
    )
    monitor_parser.add_argument("--data", required=True, type=Path)
    monitor_parser.add_argument("--target", required=True)
    monitor_parser.add_argument("--baseline-pred", required=True)
    monitor_parser.add_argument("--candidate-pred", required=True)
    monitor_parser.add_argument("--exclude", action="append", default=[])
    monitor_parser.add_argument("--confidence-level", type=float)
    monitor_parser.add_argument("--correction", choices=["bonferroni", "none"])
    monitor_parser.add_argument("--loss-delta-lower", type=float)
    monitor_parser.add_argument("--loss-delta-upper", type=float)
    monitor_parser.add_argument("--html", type=Path)
    monitor_parser.add_argument("--json", type=Path)
    monitor_parser.add_argument("--junit", type=Path)
    return parser


def _run_compare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    strategy_columns = {
        "group": ("group_column", args.group_column),
        "time": ("time_column", args.time_column),
        "explicit": ("split_column", args.split_column),
    }
    split_options: dict[str, Any] = {"validation_strategy": args.validation_strategy}
    control_columns: set[str] = set()
    split_column: str | None = None
    if args.validation_strategy in strategy_columns:
        option_name, column = strategy_columns[args.validation_strategy]
        if column is None:
            parser.error(
                f"--{option_name.replace('_', '-')} is required for "
                f"--validation-strategy {args.validation_strategy}"
            )
        required.add(column)
        control_columns.add(column)
        split_column = column
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"missing columns: {', '.join(sorted(missing))}")
    if split_column is not None:
        if args.validation_strategy == "group":
            split_options["groups"] = data[split_column]
        elif args.validation_strategy == "time":
            split_options["time_values"] = data[split_column]
        else:
            labels = data[split_column].astype(str).str.lower()
            unexpected = set(labels) - {"discovery", "validation"}
            if unexpected:
                parser.error(
                    "--split-column values must be 'discovery' or 'validation'; found: "
                    + ", ".join(sorted(unexpected))
                )
            split_options["discovery_indices"] = labels.eq("discovery")
            split_options["validation_indices"] = labels.eq("validation")
    excluded = (
        {args.target, args.baseline_pred, args.candidate_pred} | control_columns | set(args.exclude)
    )
    features = data.drop(columns=list(excluded), errors="ignore")
    if features.shape[1] == 0:
        raise SystemExit("no feature columns remain after exclusions")
    config = ModelDiffConfig(random_state=args.random_state)
    report = compare_predictions(
        data[args.baseline_pred],
        data[args.candidate_pred],
        features,
        data[args.target],
        args.task,
        args.metric,
        config=config,
        **split_options,
    )
    print(report.summary())
    if args.html:
        report.to_html(args.html)
    if args.json:
        report.to_json(args.json)
    if args.export_checks:
        suite = report.export_checks(
            max_loss_increase=args.check_max_loss_increase,
            min_samples=args.check_min_samples,
        )
        suite.to_json(args.export_checks)
        print(f"Exported {len(suite.checks)} saved checks to {args.export_checks}.")
    failed = args.fail_on_significant_regression and bool(report.regressions(significant_only=True))
    if (
        args.max_overall_regression is not None
        and report.overall["absolute_delta"] > args.max_overall_regression
    ):
        failed = True
    return 1 if failed else 0


def _run_check(args: argparse.Namespace) -> int:
    suite = CheckSuite.load(args.checks)
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    excluded = required | set(args.exclude)
    features = data.drop(columns=list(excluded), errors="ignore")
    policy = (
        CheckDecisionPolicy(
            method=args.decision,
            confidence_level=args.confidence_level,
            correction=args.correction,
            n_bootstrap=args.bootstrap,
            random_state=args.random_state,
        )
        if args.decision is not None
        else None
    )
    result = evaluate_checks(
        suite,
        data[args.baseline_pred],
        data[args.candidate_pred],
        features,
        data[args.target],
        decision_policy=policy,
    )
    print(result.summary())
    if args.json:
        result.to_json(args.json)
    if args.html:
        result.to_html(args.html)
    if args.junit:
        result.to_junit_xml(args.junit)
    return 0 if result.overall_status == "pass" else 1


def _run_stability(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    control_columns: set[str] = set()
    split_options: dict[str, Any] = {"validation_strategy": args.validation_strategy}
    option = {
        "group": ("group_column", args.group_column),
        "time": ("time_column", args.time_column),
        "explicit": ("split_column", args.split_column),
    }.get(args.validation_strategy)
    split_column: str | None = None
    if option is not None:
        option_name, column = option
        if column is None:
            parser.error(f"--{option_name.replace('_', '-')} is required")
        required.add(column)
        control_columns.add(column)
        split_column = column
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"missing columns: {', '.join(sorted(missing))}")
    if split_column is not None:
        if args.validation_strategy == "group":
            split_options["groups"] = data[split_column]
        elif args.validation_strategy == "time":
            split_options["time_values"] = data[split_column]
        else:
            labels = data[split_column].astype(str).str.lower()
            unexpected = set(labels) - {"discovery", "validation"}
            if unexpected:
                parser.error("--split-column values must be 'discovery' or 'validation'")
            split_options["discovery_indices"] = labels.eq("discovery")
            split_options["validation_indices"] = labels.eq("validation")
    excluded = required | control_columns | set(args.exclude)
    features = data.drop(columns=list(excluded), errors="ignore")
    if features.shape[1] == 0:
        raise SystemExit("no feature columns remain after exclusions")
    report = compare_stable_predictions(
        data[args.baseline_pred],
        data[args.candidate_pred],
        features,
        data[args.target],
        args.task,
        args.metric,
        config=ModelDiffConfig(random_state=args.random_state),
        stability_config=StabilityConfig(
            n_resamples=args.n_resamples,
            subsample_fraction=args.subsample_fraction,
            min_selection_frequency=args.min_selection_frequency,
            random_state=args.random_state,
        ),
        known_checks=args.known_checks,
        **split_options,
    )
    print(report.summary())
    if args.html:
        report.to_html(args.html)
    if args.json:
        report.to_json(args.json)
    failed = args.fail_on_significant_regression and bool(report.regressions(significant_only=True))
    return 1 if failed else 0


def _run_monitor(args: argparse.Namespace) -> int:
    supplied_policy = any(
        value is not None
        for value in (
            args.confidence_level,
            args.correction,
            args.loss_delta_lower,
            args.loss_delta_upper,
        )
    )
    if (args.loss_delta_lower is None) != (args.loss_delta_upper is None):
        raise ValueError("--loss-delta-lower and --loss-delta-upper must be provided together")
    if args.state.exists():
        monitor = CheckMonitor.load(args.state)
        if supplied_policy:
            raise ValueError("monitor policy options can only be set when creating state")
        if args.checks is not None and CheckSuite.load(args.checks) != monitor.suite:
            raise ValueError("--checks does not match the suite stored in monitor state")
    else:
        if args.checks is None:
            raise ValueError("--checks is required when creating monitor state")
        bounds = (
            (args.loss_delta_lower, args.loss_delta_upper)
            if args.loss_delta_lower is not None and args.loss_delta_upper is not None
            else None
        )
        monitor = CheckMonitor(
            CheckSuite.load(args.checks),
            SequentialCheckPolicy(
                confidence_level=(
                    0.95 if args.confidence_level is None else args.confidence_level
                ),
                correction="bonferroni" if args.correction is None else args.correction,
                loss_delta_bounds=bounds,
            ),
        )
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    features = data.drop(columns=list(required | set(args.exclude)), errors="ignore")
    result = monitor.update(
        data[args.baseline_pred],
        data[args.candidate_pred],
        features,
        data[args.target],
    )
    monitor.to_json(args.state)
    print(result.summary())
    if args.json:
        result.to_json(args.json)
    if args.html:
        result.to_html(args.html)
    if args.junit:
        result.to_junit_xml(args.junit)
    return 0 if result.overall_status == "pass" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ModelDiff-ML CLI and return a CI-friendly exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        return _run_compare(args, parser)
    if args.command == "stability":
        return _run_stability(args, parser)
    try:
        return _run_monitor(args) if args.command == "monitor" else _run_check(args)
    except (ValueError, TypeError, OSError, UnicodeError, pd.errors.ParserError) as error:
        print(f"modeldiff-ml {args.command}: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
