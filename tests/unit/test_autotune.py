from __future__ import annotations

import os

import numpy as np
import pytest

from slothpy.compute.autotune import (
    AutotuneConfig,
    AutotuneResult,
    AutotuneSearchSnapshot,
    AutotuneTrial,
    _estimate_runtime_seconds,
    _trial_timeout_seconds,
    iter_mpi_thread_configs,
    max_tasks_per_process,
    normalize_process_thread_pair,
    resolve_autotune_display_mode,
    resolve_autotune_num_cpu,
)
from slothpy.compute.autotune_dashboard import (
    autotune_result_to_html,
    autotune_result_to_rich,
    autotune_search_snapshot_to_html,
)
from slothpy.compute.autotune_display import create_autotune_display_driver
from slothpy.types.composite import AutotuneDisplay


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


def test_resolve_autotune_display_mode() -> None:
    assert (
        resolve_autotune_display_mode(AutotuneConfig(verbose=True))
        is AutotuneDisplay.PRINT
    )
    assert (
        resolve_autotune_display_mode(AutotuneConfig(verbose=False))
        is AutotuneDisplay.NONE
    )
    assert (
        resolve_autotune_display_mode(
            AutotuneConfig(verbose=False, display=AutotuneDisplay.RICH)
        )
        is AutotuneDisplay.RICH
    )


def test_autotune_notebook_output_returns_result_outside_notebook() -> None:
    from slothpy.compute.autotune_display import autotune_notebook_output

    result = _sample_autotune_result()
    assert autotune_notebook_output(result) is result


def test_computation_autotune_note_for_session_dashboard() -> None:
    from slothpy.core.slt_session import _computation_autotune_note

    class _Comp:
        autotune_result = AutotuneResult(
            num_processes=2,
            num_threads=4,
            estimated_seconds=3.5,
            num_cpu=8,
            trials=(),
            nodes=1,
        )

    note = _computation_autotune_note(_Comp())
    assert note is not None
    assert "autotuned 2×4" in note
    assert "3.50s" in note


def test_live_output_backend_selects_marimo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types
    import sys

    from slothpy.compute.autotune_display import _live_output_backend

    fake = types.SimpleNamespace(runtime_context_installed=lambda: True)
    monkeypatch.setitem(sys.modules, "marimo._runtime.context.types", fake)
    assert _live_output_backend() == "marimo"


def test_create_autotune_display_driver_modes() -> None:
    assert (
        create_autotune_display_driver(
            AutotuneConfig(display=AutotuneDisplay.NONE)
        ).__class__.__name__
        == "NullAutotuneDisplayDriver"
    )
    assert (
        create_autotune_display_driver(
            AutotuneConfig(display=AutotuneDisplay.RICH)
        ).__class__.__name__
        == "RichLiveAutotuneDisplayDriver"
    )


def test_autotune_search_snapshot_html_shows_running_trial() -> None:
    snapshot = AutotuneSearchSnapshot(
        num_cpu=4,
        nodes=1,
        trials=(),
        best_processes=1,
        best_threads=4,
        best_time=float("inf"),
        status="searching",
        current_processes=2,
        current_threads=2,
    )
    html = autotune_search_snapshot_to_html(snapshot)
    assert "running" in html
    assert "MPI benchmark in progress" in html


def test_trial_timeout_seconds_uses_finite_fallback() -> None:
    assert _trial_timeout_seconds(AutotuneConfig(timeout_seconds=45.0)) == 45.0
    assert _trial_timeout_seconds(AutotuneConfig()) == pytest.approx(120.0)
    assert _trial_timeout_seconds(
        AutotuneConfig(timeout_seconds=float("inf"))
    ) == pytest.approx(120.0)


def _sample_autotune_result() -> AutotuneResult:
    return AutotuneResult(
        num_processes=2,
        num_threads=4,
        estimated_seconds=3.5,
        num_cpu=8,
        nodes=1,
        trials=(
            AutotuneTrial(
                num_processes=4,
                num_threads=2,
                estimated_seconds=5.0,
                improved=False,
                skipped=True,
                skip_reason="too few tasks per rank",
            ),
            AutotuneTrial(
                num_processes=2,
                num_threads=4,
                estimated_seconds=3.5,
                improved=True,
            ),
        ),
    )


def test_autotune_result_html_summary() -> None:
    html = autotune_result_to_html(_sample_autotune_result())

    assert "SlothPy autotune" in html
    assert "2×4" in html or "2&#215;4" in html or "2×4" in html
    assert "3.50 s" in html
    assert "too few tasks per rank" in html
    assert "slt-dashboard" in html


def test_autotune_result_rich_summary() -> None:
    panel = autotune_result_to_rich(_sample_autotune_result())

    assert panel.title == "SlothPy autotune"


def test_autotune_result_repr_html_method() -> None:
    result = _sample_autotune_result()
    assert "Best so far" in result.dashboard_html()


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
