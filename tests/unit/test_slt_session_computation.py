from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from slothpy.core.slt_computation import (
    SltComputation,
    SltComputationResources,
)
from slothpy.core.slt_results import SltResults
from slothpy.core.slt_session import SltAllocation, SltSession
from slothpy.specs.magnetisation import MAGNETISATION_SLT_TYPE, SltMagnetisationResult


@dataclass(slots=True)
class _EchoComputation(SltComputation[dict[str, int], SltMagnetisationResult]):
    computation_name: ClassVar[str] = "EchoComputation"

    def _compute(self, *, allocation: SltAllocation) -> SltResults:
        value = int(self.options["value"])
        processes = allocation.num_processes
        threads = allocation.num_threads

        dataset = xr.Dataset(
            data_vars={
                "value": ("x", np.array([value * processes * threads], dtype=np.int64))
            }
        )

        return SltResults(
            dataset=dataset,
            slt_type=MAGNETISATION_SLT_TYPE,
            primary="value",
        )


def test_computation_resource_request_resolves_zeros() -> None:
    computation = _EchoComputation(
        source=object(),
        options={"value": 1},
        resources=SltComputationResources(num_processes=0, num_threads=0),
    )

    request = computation.resource_request

    assert request.num_processes >= 1
    assert request.num_threads >= 1
    assert request.num_nodes == 1


def test_session_submit_computation() -> None:
    session = SltSession.local(
        cores=4,
        max_running_jobs=2,
        install_signal_handlers=False,
    )

    try:
        computation = _EchoComputation(
            source=object(),
            options={"value": 3},
            resources=SltComputationResources(num_processes=1, num_threads=1),
        )

        job = computation.submit(session)
        result = job.result(timeout=30.0)

        assert int(result.results.dataset["value"].values[0]) == 3
        assert job.status.value == "finished"
    finally:
        session.shutdown(wait=True)
