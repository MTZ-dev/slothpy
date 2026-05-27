"""
Local MPI + per-rank thread autotuning.

Searches for a good ``(num_processes, num_threads)`` pair on the control node,
assuming other cluster nodes are homogeneous. Each trial launches a short MPI
benchmark with BLAS/OMP/Numba thread environment variables set from
``MPIJobRunner.environment()``.

The search strategy follows the legacy multiprocessing autotuner: sweep
thread counts downward, deduplicate by process count, skip configurations
with too little work per rank, sample progress for a short window, estimate
runtime from the median of per-rank projections, and stop after several
worsening trials.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy import median

from slothpy.compute.mpi_job import MPIJobResources, MPIJobRunner, MPIJobSpec
from slothpy.compute.mpi_progress import THREAD_PROGRESS_KEY, thread_progress_shape
from slothpy.core.slt_computation import SltComputationResources
from slothpy.core.slt_progress import SltProgressStatus, SltProgressTracker
from slothpy.io.shared_memory import SharedArrayBundle
from slothpy.types.composite import AutotuneDisplay

if TYPE_CHECKING:
    from slothpy.core.slt_session import SltSession

_AUTOTUNE_WORKER_MODULE = "slothpy.compute.workers.autotune_benchmark"
_DEFAULT_MIN_TASKS_PER_PROCESS = 6
_DEFAULT_MEASUREMENT_PROGRESS = 5
_DEFAULT_WORSE_STOP_COUNT = 3
_DEFAULT_MAX_THREADS = 64
_DEFAULT_MAX_PROCESSES = 4096
_DEFAULT_TRIAL_TIMEOUT_SECONDS = 120.0

AutotuneDisplayMode = AutotuneDisplay
AutotuneSearchStatus = Literal["searching", "done", "aborted"]

AUTOTUNE_CONFIG_FIELD_NAMES = frozenset(
    {
        "num_cpu",
        "parent_reserved_cores",
        "n_parallel_tasks",
        "total_work_units",
        "steps_per_task",
        "sleep_seconds",
        "compute_size",
        "progress_interval_steps",
        "publish_interval_steps",
        "min_tasks_per_process",
        "measurement_progress",
        "worse_stop_count",
        "max_threads",
        "max_processes",
        "timeout_seconds",
        "mpi_executable",
        "python_executable",
        "extra_mpi_args",
        "mpi_bind_to",
        "verbose",
        "display",
    }
)


def autotune_keyword_overrides(**keywords: Any) -> dict[str, Any]:
    """
    Collect explicit autotune keyword arguments for :class:`AutotuneConfig`.

    ``None`` means "leave default"; ``False`` and ``0`` are forwarded as overrides.
    """
    return {
        key: value
        for key, value in keywords.items()
        if key in AUTOTUNE_CONFIG_FIELD_NAMES and value is not None
    }


def merge_autotune_config(
    *,
    base: AutotuneConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
    option_defaults: Mapping[str, Any] | None = None,
) -> AutotuneConfig:
    """Build an :class:`AutotuneConfig` from optional base, keywords, and computation options."""
    explicit = dict(overrides or {})
    if base is not None:
        return replace(base, **explicit) if explicit else base

    hints = dict(explicit)
    if option_defaults is not None:
        for name in ("steps_per_task", "sleep_seconds", "progress_interval_steps"):
            if name not in hints and name in option_defaults:
                hints[name] = option_defaults[name]
    return AutotuneConfig(**hints)


@dataclass(frozen=True, slots=True)
class AutotuneConfig:
    """
    Parameters controlling the MPI/thread search.

    Progress reporting is controlled by ``display`` and ``verbose``:

    * ``display="print"`` — plain trial lines;
    * ``display="rich"`` — live Rich panel (session-dashboard style);
    * ``display="html"`` — live HTML in Jupyter/marimo;
    * ``display="none"`` — silent.

    When ``display`` is unset, ``verbose=True`` selects ``print`` and
    ``verbose=False`` selects ``none``.
    """

    num_cpu: int | None = None
    parent_reserved_cores: int = 1
    n_parallel_tasks: int | None = None
    total_work_units: int | None = None
    steps_per_task: int = 32
    sleep_seconds: float = 0.001
    compute_size: int = 64
    progress_interval_steps: int = 1
    publish_interval_steps: int = 4
    min_tasks_per_process: int = _DEFAULT_MIN_TASKS_PER_PROCESS
    measurement_progress: int = _DEFAULT_MEASUREMENT_PROGRESS
    worse_stop_count: int = _DEFAULT_WORSE_STOP_COUNT
    max_threads: int = _DEFAULT_MAX_THREADS
    max_processes: int = _DEFAULT_MAX_PROCESSES
    timeout_seconds: float = float("inf")
    mpi_executable: str = "mpirun"
    python_executable: str | None = None
    extra_mpi_args: tuple[str, ...] = ()
    mpi_bind_to: str | None = "none"
    verbose: bool = True
    display: AutotuneDisplay | None = None


@dataclass(frozen=True, slots=True)
class AutotuneTrial:
    """Outcome of one benchmark configuration."""

    num_processes: int
    num_threads: int
    estimated_seconds: float
    improved: bool
    skipped: bool = False
    skip_reason: str | None = None
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class AutotuneResult:
    """Best configuration found for the requested cluster shape."""

    num_processes: int
    num_threads: int
    estimated_seconds: float
    num_cpu: int
    trials: tuple[AutotuneTrial, ...]
    nodes: int = 1

    @property
    def logical_cpus(self) -> int:
        return self.num_processes * self.num_threads

    def to_computation_resources(
        self,
        *,
        exclusive_nodes: bool = False,
    ) -> SltComputationResources:
        num_processes = max(self.num_processes, self.nodes)
        return SltComputationResources(
            num_processes=num_processes,
            num_threads=self.num_threads,
            nodes=self.nodes,
            exclusive_nodes=exclusive_nodes,
        )

    def to_rich(self, *, width: int | None = None) -> Any:
        from slothpy.compute.autotune_dashboard import autotune_result_to_rich

        return autotune_result_to_rich(self, width=width)

    def print_rich(
        self,
        *,
        console: Any | None = None,
        width: int | None = None,
    ) -> None:
        from slothpy.core.slt_common import print_rich_renderable

        print_rich_renderable(self.to_rich(width=width), console=console)

    def show(self, *, console: Any | None = None, width: int | None = None) -> None:
        """Alias for :meth:`print_rich`."""
        self.print_rich(console=console, width=width)

    def dashboard_html(self) -> str:
        from slothpy.compute.autotune_dashboard import autotune_result_to_html

        return autotune_result_to_html(self)

    def dashboard(self) -> str:
        """HTML summary for notebooks (marimo/Jupyter)."""
        return self.dashboard_html()

    def dashboard_text(self, *, width: int | None = None) -> str:
        from slothpy.compute.autotune_dashboard import autotune_result_to_text

        return autotune_result_to_text(self, width=width)

    def _repr_html_(self) -> str:
        return self.dashboard_html()

    def __rich__(self) -> Any:
        return self.to_rich()

    def notebook_output(
        self,
        *,
        title_md: str | None = None,
        summary_md: str | None = None,
    ) -> Any:
        """
        Marimo/Jupyter cell value that keeps the autotune dashboard visible.

        In marimo, the cell's **last expression** becomes its output; a trailing
        ``mo.md(...)`` replaces live ``mo.output.replace`` updates. Return this
        helper (or put it last) after :meth:`~slothpy.core.slt_computation.SltComputation.autotune`::

            tune_result = comp.autotune(..., display="rich")
            tune_result.notebook_output(summary_md="...")
        """
        from slothpy.compute.autotune_display import autotune_notebook_output

        return autotune_notebook_output(
            self,
            title_md=title_md,
            summary_md=summary_md,
        )


@dataclass(frozen=True, slots=True)
class AutotuneSearchSnapshot:
    """Live or final view of an MPI/thread autotune search."""

    num_cpu: int
    nodes: int
    trials: tuple[AutotuneTrial, ...]
    best_processes: int
    best_threads: int
    best_time: float
    status: AutotuneSearchStatus = "searching"
    current_processes: int | None = None
    current_threads: int | None = None
    message: str | None = None

    def to_result(self) -> AutotuneResult:
        return AutotuneResult(
            num_processes=self.best_processes,
            num_threads=self.best_threads,
            estimated_seconds=self.best_time,
            num_cpu=self.num_cpu,
            trials=self.trials,
            nodes=self.nodes,
        )


def resolve_autotune_display_mode(config: AutotuneConfig) -> AutotuneDisplay:
    """
    Choose how autotune reports progress.

    When ``display`` is unset, plain ``print`` logging follows ``verbose``;
    otherwise ``display`` takes precedence.
    """
    if config.display is not None:
        display = config.display
        if not isinstance(display, AutotuneDisplay):
            display = AutotuneDisplay(display)
        return display
    return AutotuneDisplay.PRINT if config.verbose else AutotuneDisplay.NONE


def _ordered_pool_nodes(session: SltSession) -> list[Any]:
    pool = session.resource_pool
    if pool is None:
        return []

    control_name = session.control_node_name
    control = [node for node in pool.nodes if node.name == control_name]
    others = [node for node in pool.nodes if node.name != control_name]
    return [*control, *others]


def resolve_autotune_num_cpu(
    *,
    session: SltSession | None = None,
    parent_reserved_cores: int = 1,
    nodes: int | None = None,
    cores_per_node: int | None = None,
    total_cores: int | None = None,
) -> int:
    """
    Logical CPUs available for the MPI × thread search.

    Rank 0 / parent work on the control node is excluded once via
    ``parent_reserved_cores``.

    Parameters
    ----------
    nodes, cores_per_node
        Homogeneous cluster assumption: ``nodes * cores_per_node`` total cores.
    total_cores
        Explicit total core budget (parent reserve subtracted once).
    session
        When ``nodes`` is set without ``cores_per_node``, cores are summed from
        the first ``nodes`` entries in the session pool (control node first).
    """
    if total_cores is not None:
        return max(1, int(total_cores) - int(parent_reserved_cores))

    if nodes is not None and cores_per_node is not None:
        cluster_total = int(nodes) * int(cores_per_node)
        return max(1, cluster_total - int(parent_reserved_cores))

    if session is not None and session.resource_pool is not None:
        pool_nodes = _ordered_pool_nodes(session)
        if nodes is not None:
            if nodes < 1:
                raise ValueError("nodes must be >= 1.")
            if nodes > len(pool_nodes):
                raise ValueError(
                    f"Requested {nodes} nodes but the session pool has "
                    f"{len(pool_nodes)}."
                )
            selected = pool_nodes[: int(nodes)]
            cluster_total = sum(int(node.cores) for node in selected)
            return max(1, cluster_total - int(parent_reserved_cores))

        control_name = session.control_node_name
        for node in pool_nodes:
            if node.name == control_name:
                return max(1, int(node.cores) - int(parent_reserved_cores))

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count is not None:
            return max(1, int(count) - int(parent_reserved_cores))

    count = os.cpu_count()
    if count is None:
        return 1
    return max(1, int(count) - int(parent_reserved_cores))


def normalize_process_thread_pair(
    num_cpu: int,
    num_processes: int,
    num_threads: int,
    *,
    n_parallel_tasks: int,
) -> tuple[int, int]:
    """
    Fit ``(num_processes, num_threads)`` to available CPUs and task count.
    """
    if num_cpu < 1:
        raise ValueError("num_cpu must be >= 1.")
    if num_threads < 1:
        num_threads = num_cpu
    if num_processes < 1:
        num_processes = max(1, num_cpu // num_threads)

    num_processes = min(num_processes, num_cpu // num_threads, n_parallel_tasks)
    num_processes = max(1, num_processes)
    num_threads = max(1, num_cpu // num_processes)
    return num_processes, num_threads


def iter_mpi_thread_configs(
    num_cpu: int,
    *,
    n_parallel_tasks: int,
    max_threads: int = _DEFAULT_MAX_THREADS,
    max_processes: int = _DEFAULT_MAX_PROCESSES,
) -> Iterator[tuple[int, int]]:
    """
    Yield distinct ``(num_processes, num_threads)`` pairs for benchmarking.

      Thread counts are swept from high to low (legacy behaviour).
    """
    old_processes = 0
    for num_threads in range(min(max_threads, num_cpu), 0, -1):
        num_processes = num_cpu // num_threads
        if num_processes > max_processes:
            break
        if num_processes >= n_parallel_tasks:
            num_processes = n_parallel_tasks
            num_threads = max(1, num_cpu // num_processes)
        if num_processes != old_processes:
            old_processes = num_processes
            yield num_processes, max(1, num_threads)


def _trial_timeout_seconds(config: AutotuneConfig) -> float:
    """
    Wall-clock budget for one MPI benchmark trial (warmup + measurement).

    ``AutotuneConfig.timeout_seconds`` defaults to infinity; comparisons against
    ``perf_counter() + inf`` never succeed, so a finite fallback is required.
    """
    timeout = float(config.timeout_seconds)
    if timeout > 0 and np.isfinite(timeout):
        return timeout
    return _DEFAULT_TRIAL_TIMEOUT_SECONDS


def max_tasks_per_process(
    n_parallel_tasks: int,
    num_processes: int,
) -> np.ndarray:
    chunk_size = n_parallel_tasks // num_processes
    remainder = n_parallel_tasks % num_processes
    return np.array(
        [
            chunk_size + (1 if index < remainder else 0)
            for index in range(num_processes)
        ],
        dtype=np.int64,
    )


def _estimate_runtime_seconds(
    *,
    elapsed_ns: int,
    progress_delta: np.ndarray,
    max_steps_per_process: np.ndarray,
) -> float:
    estimates: list[float] = []
    for delta, cap in zip(progress_delta, max_steps_per_process, strict=True):
        delta_value = int(delta)
        cap_value = int(cap)
        if delta_value <= 0 or cap_value <= 0:
            continue
        estimates.append(elapsed_ns * (cap_value / delta_value))

    if not estimates:
        return float("inf")

    estimates.sort()
    upper_half = estimates[len(estimates) // 2 :]
    if not upper_half:
        return estimates[-1] / 1e9
    return float(median(upper_half)) / 1e9


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    from slothpy.core.process_tree import terminate_subprocess

    terminate_subprocess(process, grace_seconds=5.0)


@contextmanager
def _optional_threadpool_limits(num_threads: int):
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except ImportError:
        yield
        return

    with threadpool_limits(limits=num_threads, user_api="blas"):
        yield


def _run_benchmark_trial(
    *,
    num_processes: int,
    num_threads: int,
    n_tasks: int,
    steps_per_task: int,
    total_work_units: int,
    config: AutotuneConfig,
) -> AutotuneTrial:
    shape = thread_progress_shape(
        num_processes=num_processes,
        num_threads=num_threads,
    )
    max_tasks = max_tasks_per_process(n_tasks, num_processes)
    max_steps = max_tasks * steps_per_task

    if np.any(max_tasks < config.min_tasks_per_process):
        return AutotuneTrial(
            num_processes=num_processes,
            num_threads=num_threads,
            estimated_seconds=float("inf"),
            improved=False,
            skipped=True,
            skip_reason=(
                f"Fewer than {config.min_tasks_per_process} tasks per MPI rank."
            ),
        )

    with TemporaryDirectory(prefix="slothpy-autotune-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "shared_memory.json"
        job_spec_path = tmpdir_path / "mpi_job.json"

        from slothpy.compute.autotune_runtime import begin_autotune_trial, end_autotune_trial

        trial_scope = begin_autotune_trial()
        progress = SltProgressTracker.create(
            total=total_work_units,
            status=SltProgressStatus.QUEUED,
            track=False,
        )
        bundle = SharedArrayBundle()
        bundle.add_empty(
            THREAD_PROGRESS_KEY,
            shape,
            dtype=np.int64,
            readonly=False,
            track=False,
        )
        trial_scope.register_release(bundle.release)
        trial_scope.register_release(progress.release)

        try:
            bundle.write_manifest(manifest_path)
            payload: dict[str, Any] = {
                "n_tasks": n_tasks,
                "num_threads": num_threads,
                "steps_per_task": steps_per_task,
                "sleep_seconds": config.sleep_seconds,
                "compute_size": config.compute_size,
                "progress_interval_steps": config.progress_interval_steps,
                "publish_interval_steps": config.publish_interval_steps,
                "total_work": total_work_units,
                "progress": progress.spec.to_json_dict(),
            }
            spec = MPIJobSpec(
                worker_module=_AUTOTUNE_WORKER_MODULE,
                shared_memory_manifest=str(manifest_path),
                payload=payload,
            )
            spec.write_json(job_spec_path)

            runner = MPIJobRunner(
                resources=MPIJobResources(
                    num_processes=num_processes,
                    num_threads=num_threads,
                    mpi_executable=config.mpi_executable,
                    python_executable=config.python_executable or sys.executable,
                    extra_mpi_args=config.extra_mpi_args,
                    mpi_bind_to=config.mpi_bind_to,
                ),
                capture_output=True,
            )
            command = runner.command(spec, job_spec_path)
            thread_board = bundle[THREAD_PROGRESS_KEY].array

            with _optional_threadpool_limits(num_threads):
                process = subprocess.Popen(
                    command,
                    cwd=tmpdir_path,
                    env=runner.environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
            from slothpy.compute.autotune_runtime import bind_autotune_process

            bind_autotune_process(process)

            try:
                trial_deadline = time.perf_counter() + _trial_timeout_seconds(config)
                while (
                    int(thread_board.sum()) <= num_processes
                    and progress.snapshot().done <= num_processes
                ):
                    if process.poll() is not None:
                        return AutotuneTrial(
                            num_processes=num_processes,
                            num_threads=num_threads,
                            estimated_seconds=float("inf"),
                            improved=False,
                            skip_reason="MPI benchmark exited before reporting progress.",
                        )
                    if time.perf_counter() >= trial_deadline:
                        _terminate_process_group(process)
                        return AutotuneTrial(
                            num_processes=num_processes,
                            num_threads=num_threads,
                            estimated_seconds=float("inf"),
                            improved=False,
                            timed_out=True,
                        )
                    time.sleep(0.001)

                start_time = time.perf_counter_ns()
                start_board = np.array(thread_board, copy=True)
                start_done = progress.snapshot().done
                stop_time = start_time

                start_rank = start_board.sum(axis=1)
                rank_progress = start_rank.copy()
                aggregate_done = start_done

                while np.any(
                    rank_progress - start_rank <= config.measurement_progress
                ) and np.all(rank_progress < max_steps):
                    if process.poll() is not None:
                        break
                    if time.perf_counter() >= trial_deadline:
                        _terminate_process_group(process)
                        return AutotuneTrial(
                            num_processes=num_processes,
                            num_threads=num_threads,
                            estimated_seconds=float("inf"),
                            improved=False,
                            timed_out=True,
                        )
                    stop_time = time.perf_counter_ns()
                    rank_progress = np.array(thread_board, copy=True).sum(axis=1)
                    aggregate_done = progress.snapshot().done
                    time.sleep(0.01)

                _terminate_process_group(process)

                overall_time = stop_time - start_time
                progress_delta = rank_progress - start_rank
                aggregate_delta = aggregate_done - start_done

                if (
                    np.all(progress_delta <= 1) and aggregate_delta <= 1
                ) or overall_time == 0:
                    return AutotuneTrial(
                        num_processes=num_processes,
                        num_threads=num_threads,
                        estimated_seconds=float("inf"),
                        improved=False,
                        skip_reason="Progress moved too little to measure reliably.",
                    )

                if np.any(progress_delta > 1):
                    estimated = _estimate_runtime_seconds(
                        elapsed_ns=overall_time,
                        progress_delta=progress_delta,
                        max_steps_per_process=max_steps,
                    )
                else:
                    estimated = (
                        overall_time * (total_work_units / aggregate_delta) / 1e9
                        if aggregate_delta > 0
                        else float("inf")
                    )
                return AutotuneTrial(
                    num_processes=num_processes,
                    num_threads=num_threads,
                    estimated_seconds=estimated,
                    improved=False,
                )

            finally:
                _terminate_process_group(process)
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    stdout, stderr = b"", b""
                if process.returncode not in (0, -signal.SIGTERM, -15):
                    if (
                        resolve_autotune_display_mode(config) is AutotuneDisplay.PRINT
                        and stderr
                    ):
                        print(stderr.decode(errors="replace"))

        finally:
            end_autotune_trial(trial_scope)
            bundle.release()
            progress.release()


def autotune_mpi_threading(
    *,
    n_parallel_tasks: int,
    total_work_units: int | None = None,
    config: AutotuneConfig | None = None,
    session: SltSession | None = None,
    nodes: int | None = None,
    cores_per_node: int | None = None,
    total_cores: int | None = None,
) -> AutotuneResult:
    """
    Search for a good MPI rank / per-rank thread configuration.

    Parameters
    ----------
    n_parallel_tasks
        Number of independent tasks distributed across MPI ranks (for example
        magnetisation field/orientation count).
    total_work_units
        Total progress steps for estimation. Defaults to
        ``n_parallel_tasks * config.steps_per_task``.
    config
        Search and benchmark parameters.
    session
        Optional session used to resolve control-node CPU count.
    """
    if n_parallel_tasks < 1:
        raise ValueError("n_parallel_tasks must be >= 1.")

    cfg = config or AutotuneConfig()
    resolved_nodes = nodes if nodes is not None else 1
    if resolved_nodes < 1:
        raise ValueError("nodes must be >= 1.")

    num_cpu = cfg.num_cpu
    if num_cpu is None:
        num_cpu = resolve_autotune_num_cpu(
            session=session,
            parent_reserved_cores=cfg.parent_reserved_cores,
            nodes=nodes,
            cores_per_node=cores_per_node,
            total_cores=total_cores,
        )

    steps_per_task = max(1, cfg.steps_per_task)
    total_work = total_work_units
    if total_work is None:
        total_work = n_parallel_tasks * steps_per_task

    from slothpy.compute.autotune_display import run_autotune_search_with_display

    finished = run_autotune_search_with_display(
        config=cfg,
        num_cpu=num_cpu,
        nodes=resolved_nodes,
        n_parallel_tasks=n_parallel_tasks,
        steps_per_task=steps_per_task,
        total_work=total_work,
        session=session,
    )
    return finished.to_result()


def apply_autotune_result(
    result: AutotuneResult,
    *,
    permanent: bool = False,
) -> None:
    """Write autotune result into global SlothPy settings."""
    from slothpy.config.settings import _configure

    _configure(
        num_processes=result.num_processes,
        num_threads=result.num_threads,
        permanent=permanent,
    )


def autotune_computation_resources(
    *,
    n_parallel_tasks: int,
    total_work_units: int | None = None,
    config: AutotuneConfig | None = None,
    session: SltSession | None = None,
    nodes: int | None = None,
    cores_per_node: int | None = None,
    total_cores: int | None = None,
    apply: bool = False,
    permanent: bool = False,
    exclusive_nodes: bool = False,
) -> SltComputationResources:
    """
    Run autotune and return resources suitable for :class:`~slothpy.core.slt_computation.SltComputation`.
    """
    result = autotune_mpi_threading(
        n_parallel_tasks=n_parallel_tasks,
        total_work_units=total_work_units,
        config=config,
        session=session,
        nodes=nodes,
        cores_per_node=cores_per_node,
        total_cores=total_cores,
    )
    if apply:
        apply_autotune_result(result, permanent=permanent)
    return result.to_computation_resources(exclusive_nodes=exclusive_nodes)


def default_autotune_config_from_settings() -> AutotuneConfig:
    """Build an :class:`AutotuneConfig` using current global settings as hints."""
    return AutotuneConfig(
        num_cpu=None,
        parent_reserved_cores=1,
    )


__all__ = [
    "AutotuneConfig",
    "AutotuneResult",
    "AutotuneTrial",
    "AUTOTUNE_CONFIG_FIELD_NAMES",
    "apply_autotune_result",
    "autotune_computation_resources",
    "autotune_keyword_overrides",
    "autotune_mpi_threading",
    "default_autotune_config_from_settings",
    "merge_autotune_config",
    "iter_mpi_thread_configs",
    "max_tasks_per_process",
    "normalize_process_thread_pair",
    "resolve_autotune_num_cpu",
]

# Re-export dashboard helpers for convenience.
from slothpy.compute.autotune_dashboard import (  # noqa: E402
    autotune_result_to_html,
    autotune_result_to_rich,
    autotune_result_to_text,
)

__all__ += [
    "autotune_result_to_html",
    "autotune_result_to_rich",
    "autotune_result_to_text",
]
