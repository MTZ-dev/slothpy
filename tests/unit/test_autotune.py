from __future__ import annotations

import os

import numpy as np
import pytest

from slothpy.compute.autotune import (
    AutotuneConfig,
    _estimate_runtime_seconds,
    iter_mpi_thread_configs,
    max_tasks_per_process,
    normalize_process_thread_pair,
    resolve_autotune_num_cpu,
)


def test_iter_mpi_thread_configs_deduplicates_process_counts() -> None:
    configs = list(
        iter_mpi_thread_configs(16, n_parallel_tasks=100, max_threads=64)
    )
    process_counts = [processes for processes, _ in configs]
    assert len(process_counts) == len(set(process_counts))
    assert all(processes * threads <= 16 for processes, threads in configs)


def test_normalize_process_thread_pair_caps_by_tasks() -> None:
    processes, threads = normalize_process_thread_pair(
        16,
        32,
        1,
        n_parallel_tasks=4,
    )
    assert processes == 4
    assert processes * threads <= 16


def test_max_tasks_per_process_distributes_remainder() -> None:
    counts = max_tasks_per_process(10, 4)
    assert counts.tolist() == [3, 3, 2, 2]
    assert int(counts.sum()) == 10


def test_estimate_runtime_seconds_uses_upper_median() -> None:
    elapsed_ns = int(1e9)
    progress_delta = np.array([5, 10], dtype=np.int64)
    max_steps = np.array([100, 100], dtype=np.int64)

    estimate = _estimate_runtime_seconds(
        elapsed_ns=elapsed_ns,
        progress_delta=progress_delta,
        max_steps_per_process=max_steps,
    )

    # Upper-half median of per-rank projections: 20 s and 10 s -> 20 s.
    assert estimate == pytest.approx(20.0)


def test_resolve_autotune_num_cpu_subtracts_parent_reserve() -> None:
    count = resolve_autotune_num_cpu(parent_reserved_cores=2)
    assert count >= 1


def test_resolve_autotune_num_cpu_nodes_times_cores_per_node() -> None:
    count = resolve_autotune_num_cpu(
        nodes=3,
        cores_per_node=10,
        parent_reserved_cores=1,
    )
    assert count == 29


@pytest.mark.skipif(
    os.environ.get("SLOTHPY_MPI_TESTS", "") == "",
    reason="Set SLOTHPY_MPI_TESTS=1 to run MPI autotune integration test.",
)
def test_autotune_mpi_threading_short_run() -> None:
    pytest.importorskip("mpi4py")

    from slothpy.compute.autotune import autotune_mpi_threading

    result = autotune_mpi_threading(
        n_parallel_tasks=24,
        config=AutotuneConfig(
            num_cpu=4,
            steps_per_task=8,
            sleep_seconds=0.002,
            min_tasks_per_process=2,
            measurement_progress=2,
            worse_stop_count=1,
            timeout_seconds=60.0,
            verbose=False,
        ),
    )

    assert result.num_processes >= 1
    assert result.num_threads >= 1
    assert result.logical_cpus <= 4
