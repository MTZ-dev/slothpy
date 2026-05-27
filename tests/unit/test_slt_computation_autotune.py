from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import xarray as xr

from slothpy.compute.autotune import AutotuneResult
from slothpy.core.slt_computation import (
    SltComputation,
    SltComputationResources,
    SltComputationStatus,
)
from slothpy.core.slt_results import SltResults
from slothpy.core.slt_session import SltAllocation, SltSession
from slothpy.specs.magnetisation import MAGNETISATION_SLT_TYPE, SltMagnetisationResult


@patch("slothpy.compute.autotune.autotune_mpi_threading")
def test_computation_autotune_updates_resources(mock_autotune) -> None:
    mock_autotune.return_value = AutotuneResult(
        num_processes=4,
        num_threads=2,
        estimated_seconds=12.5,
        num_cpu=8,
        trials=(),
        nodes=2,
    )

    computation = _EchoComputation(
        source=object(),
        options={"value": 1, "n_tasks": 24},
        resources=SltComputationResources(nodes=2),
    )

    result = computation.autotune(cores=16, verbose=False)

    mock_autotune.assert_called_once()
    call_kwargs = mock_autotune.call_args.kwargs
    assert call_kwargs["n_parallel_tasks"] == 24
    assert call_kwargs["nodes"] == 2
    assert call_kwargs["cores_per_node"] == 16

    assert result.num_processes == 4
    assert computation.resources.num_processes == 4
    assert computation.resources.num_threads == 2
    assert computation.resources.nodes == 2


@patch("slothpy.compute.autotune.autotune_mpi_threading")
def test_run_with_autotune_invokes_autotune_first(mock_autotune) -> None:
    mock_autotune.return_value = AutotuneResult(
        num_processes=2,
        num_threads=2,
        estimated_seconds=1.0,
        num_cpu=4,
        trials=(),
        nodes=1,
    )

    computation = _EchoComputation(
        source=object(),
        options={"value": 2, "n_tasks": 8},
        resources=SltComputationResources(num_processes=2, num_threads=2),
    )

    view = computation.run(autotune=True, save=False)

    mock_autotune.assert_called_once()
    assert int(view.results.dataset["value"].values[0]) == 2 * 2 * 2


def test_resolve_autotune_num_cpu_multi_node_session() -> None:
    from slothpy.compute.autotune import resolve_autotune_num_cpu

    session = SltSession.from_nodes(
        [("node-a", 16), ("node-b", 16)],
        control_node_name="node-a",
        parent_reserved_cores=1,
    )
    try:
        total = resolve_autotune_num_cpu(
            session=session,
            nodes=2,
            parent_reserved_cores=1,
        )
        assert total == 31
    finally:
        session.shutdown(wait=False)


@patch("slothpy.compute.autotune.autotune_mpi_threading")
def test_autotune_skips_when_finished_unless_forced(mock_autotune) -> None:
    computation = _EchoComputation(
        source=object(),
        options={"value": 1, "n_tasks": 4},
    )
    computation._status = SltComputationStatus.FINISHED

    computation.autotune(verbose=False)
    mock_autotune.assert_not_called()

    computation.autotune(force=True, verbose=False)
    mock_autotune.assert_called_once()


@dataclass(slots=True)
class _EchoComputation(SltComputation[dict[str, int], SltMagnetisationResult]):
    computation_name: ClassVar[str] = "EchoComputation"

    def _task_count(self) -> int:
        return int(self.options["n_tasks"])

    def _compute(self, *, allocation: SltAllocation) -> SltResults:
        value = int(self.options["value"])
        processes = allocation.num_processes
        threads = allocation.num_threads

        dataset = xr.Dataset(
            data_vars={
                "value": (
                    "x",
                    np.array([value * processes * threads], dtype=np.int64),
                )
            }
        )

        return SltResults(
            dataset=dataset,
            slt_type=MAGNETISATION_SLT_TYPE,
            primary="value",
        )
