"""Resume an anytime-valid release gate as labeled batches arrive."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modeldiff import CheckMonitor, CheckSuite, SavedCheck, SequentialCheckPolicy
from modeldiff.analysis.regions import Condition


def make_batch(
    seed: int, loss_increase: float
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"utilization": rng.uniform(size=1500)})
    y = np.zeros(len(X))
    baseline = np.zeros(len(X))
    candidate = np.where(X["utilization"] > 0.7, loss_increase, 0.0)
    return baseline, candidate, X, y


state_path = Path("monitor-state.json")
if state_path.exists():
    monitor = CheckMonitor.load(state_path)
else:
    suite = CheckSuite(
        task="regression",
        metric="mae",
        checks=(
            SavedCheck(
                check_id="high-utilization",
                name="High-utilization customers",
                conditions=(Condition("utilization", ">", 0.7),),
                max_loss_increase=0.05,
                min_samples=200,
                provenance={"source": "example"},
            ),
        ),
    )
    monitor = CheckMonitor(
        suite,
        SequentialCheckPolicy(loss_delta_bounds=(-1.0, 1.0)),
    )

for batch_number in range(1, 6):
    result = monitor.update(*make_batch(batch_number, loss_increase=0.02))
    monitor.to_json(state_path)
    print(f"Batch {batch_number}: {result.overall_status}")

print(result.summary())
