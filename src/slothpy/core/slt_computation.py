from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from traceback import format_exception
from typing import TYPE_CHECKING, Any, ClassVar

import xarray as xr
from pydantic import ConfigDict, validate_call

from slothpy.compute.mpi_job import MPIJobSpec
from slothpy.core.slt import (
    SltFile,
    create_slt_file,
    open_slt_file,
)
from slothpy.core.slt_group import SltGroup
from slothpy.core.slt_results import SltResults, SltResultView
from slothpy.types.aliases import PathLike
from slothpy.types.composite import NUM_PROCESSES_ADAPTER, NUM_THREADS_ADAPTER

if TYPE_CHECKING:
    from slothpy.core.slt_session import (
        MPIProcessHandle,
        SltAllocation,
        SltJob,
        SltProgressSnapshot,
        SltProgressTracker,
        SltResourceRequest,
        SltSession,
    )

_VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


class SltComputationStatus(StrEnum):
    """
    Runtime status of a SlothPy computation object.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class SltComputationResources:
    """
    Resource request attached to a computation.

    ``0`` means: resolve from SlothPy settings.
    The base class converts this to :class:`~slothpy.core.slt_session.SltResourceRequest`
    for :class:`~slothpy.core.slt_session.SltSession` scheduling.
    """

    num_processes: int = 0
    num_threads: int = 0
    nodes: int = 1
    exclusive_nodes: bool = False

    def __post_init__(self) -> None:
        if self.num_processes < 0:
            raise ValueError("num_processes must be >= 0.")
        if self.num_threads < 0:
            raise ValueError("num_threads must be >= 0.")
        if self.nodes < 1:
            raise ValueError("nodes must be >= 1.")


@dataclass(frozen=True, slots=True)
class SltComputationFailure:
    """
    Stored information about a failed computation.
    """

    exception_type: str
    message: str
    traceback: str


@dataclass(slots=True)
class SltComputation[OptionsT, ResultViewT: SltResultView](ABC):
    """
    Base class for SlothPy computations.

    A computation:

    1. holds input source(s), options, and resource request;
    2. computes an in-memory ``SltResults`` object;
    3. can optionally save that result as a SlothPy semantic group;
    4. implements :class:`~slothpy.core.slt_session.SltRunnableComputation` and can
       be submitted to :class:`~slothpy.core.slt_session.SltSession`.

    Subclasses should usually implement only ``_compute`` and optionally
    ``_before_run`` / ``_after_success``.
    """

    source: Any
    options: OptionsT
    resources: SltComputationResources = field(default_factory=SltComputationResources)

    save_as: str | None = None
    output_slt: SltFile | PathLike | None = None
    overwrite: bool = False
    encoding: dict[str, Any] | None = None

    _status: SltComputationStatus = field(
        default=SltComputationStatus.PENDING,
        init=False,
    )
    _result: SltResults | None = field(default=None, init=False)
    _wrapped_result: ResultViewT | None = field(default=None, init=False)
    _saved_group: SltGroup | None = field(default=None, init=False)
    _failure: SltComputationFailure | None = field(default=None, init=False)

    _progress: float | None = field(default=None, init=False)
    _message: str | None = field(default=None, init=False)
    _started_perf_counter: float | None = field(default=None, init=False)
    _finished_perf_counter: float | None = field(default=None, init=False)

    _cancel_requested: bool = field(default=False, init=False)
    _progress_tracker: SltProgressTracker | None = field(default=None, init=False)
    _mpi_handle: MPIProcessHandle | None = field(default=None, init=False)

    computation_name: ClassVar[str] = "SltComputation"

    @property
    def status(self) -> SltComputationStatus:
        return self._status

    @property
    def resource_request(self) -> SltResourceRequest:
        """
        Normalized resource request consumed by :class:`~slothpy.core.slt_session.SltSession`.
        """
        from slothpy.core.slt_session import SltResourceRequest

        num_processes = NUM_PROCESSES_ADAPTER.validate_python(self.resources.num_processes)
        num_threads = NUM_THREADS_ADAPTER.validate_python(self.resources.num_threads)

        return SltResourceRequest(
            num_processes=num_processes,
            num_threads=num_threads,
            num_nodes=self.resources.nodes,
            exclusive_nodes=self.resources.exclusive_nodes,
        )

    @property
    def progress_tracker(self) -> SltProgressTracker | None:
        """
        Optional shared-memory progress block for MPI workers.
        """
        return self._progress_tracker

    @property
    def is_pending(self) -> bool:
        return self._status == SltComputationStatus.PENDING

    @property
    def is_running(self) -> bool:
        return self._status == SltComputationStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        return self._status == SltComputationStatus.FINISHED

    @property
    def is_failed(self) -> bool:
        return self._status == SltComputationStatus.FAILED

    @property
    def is_cancelled(self) -> bool:
        return self._status == SltComputationStatus.CANCELLED

    @property
    def progress(self) -> float | None:
        """
        Progress in the inclusive range [0.0, 1.0], if known.
        """
        snapshot = self.progress_snapshot()
        if snapshot is not None and snapshot.fraction is not None:
            return snapshot.fraction
        return self._progress

    @property
    def message(self) -> str | None:
        """
        Human-readable status message, if available.
        """
        return self._message

    @property
    def failure(self) -> SltComputationFailure | None:
        """
        Failure information after a failed run.
        """
        return self._failure

    @property
    def saved_group(self) -> SltGroup | None:
        """
        Saved group handle, if this computation has been persisted.
        """
        return self._saved_group

    @property
    def elapsed_seconds(self) -> float | None:
        """
        Runtime duration in seconds, if the computation has started.
        """
        if self._started_perf_counter is None:
            return None

        end = (
            perf_counter()
            if self._finished_perf_counter is None
            else self._finished_perf_counter
        )
        return end - self._started_perf_counter

    @property
    def result(self) -> SltResults:
        """
        Return computed in-memory results.

        Raises
        ------
        RuntimeError
            If the computation has not finished successfully.
        """
        if self._result is None:
            raise RuntimeError(
                f"{self.computation_name} has no result yet. "
                "Call run() first or submit it to a session."
            )
        return self._result

    def progress_snapshot(self) -> SltProgressSnapshot | None:
        """
        Snapshot for :class:`~slothpy.core.slt_session.SltSession` dashboards.
        """
        from slothpy.core.slt_session import SltProgressSnapshot, SltProgressStatus

        if self._progress_tracker is not None:
            return self._progress_tracker.snapshot()

        if self._progress is None:
            return None

        total = 100
        done = int(round(max(0.0, min(1.0, self._progress)) * total))

        status = SltProgressStatus.UNKNOWN
        if self._status == SltComputationStatus.RUNNING:
            status = SltProgressStatus.RUNNING
        elif self._status == SltComputationStatus.FINISHED:
            status = SltProgressStatus.FINISHED
        elif self._status == SltComputationStatus.FAILED:
            status = SltProgressStatus.FAILED
        elif self._status == SltComputationStatus.CANCELLED:
            status = SltProgressStatus.CANCELLED

        return SltProgressSnapshot(
            done=done,
            total=total,
            status=status,
            cancel_requested=self._cancel_requested,
            heartbeat_ns=0,
        )

    @validate_call(config=_VALIDATE_CONFIG)
    def run(self, *, force: bool = False, save: bool | None = None) -> ResultViewT:
        """
        Execute the computation synchronously on the local machine.

        Parameters
        ----------
        force
            Re-run even if a previous result exists.
        save
            If ``True``, save after computing. If ``None``, save automatically
            when ``save_as`` was provided.
        """
        return self._execute(
            allocation=self._local_allocation(),
            force=force,
            save=save,
        )

    def run_with_allocation(
        self,
        allocation: SltAllocation | None = None,
    ) -> ResultViewT:
        """
        Execute using resources allocated by :class:`~slothpy.core.slt_session.SltSession`.

        This is the entry point used by the asynchronous session scheduler.
        """
        if self._cancel_requested:
            raise CancelledError()

        if allocation is None:
            allocation = self._local_allocation()

        return self._execute(allocation=allocation, force=True, save=None)

    @validate_call(config=_VALIDATE_CONFIG)
    def save(
        self,
        slt: SltFile | PathLike | None = None,
        group_name: str | None = None,
        *,
        overwrite: bool | None = None,
        encoding: dict[str, Any] | None = None,
    ) -> SltGroup:
        """
        Save computed results as a SlothPy semantic group.

        If the computation has not been run yet, it is run first with
        ``save=False`` to avoid recursive saving.
        """
        if self._result is None:
            self.run(save=False)

        results = self.result
        target_slt = self._resolve_output_slt(slt)

        final_group_name = group_name or self.save_as
        if final_group_name is None:
            raise ValueError(
                "No output group name was provided. Pass group_name=... "
                "or construct the computation with save_as=..."
            )

        final_overwrite = self.overwrite if overwrite is None else overwrite
        final_encoding = self.encoding if encoding is None else encoding

        group = target_slt._write_slothpy_group(
            final_group_name,
            results,
            overwrite=final_overwrite,
            encoding=final_encoding,
        )

        self._saved_group = group
        return group

    def submit(self, session: SltSession[ResultViewT]) -> SltJob[ResultViewT]:
        """
        Submit this computation to an asynchronous :class:`~slothpy.core.slt_session.SltSession`.
        """
        return session.submit(self)

    def request_cancel(self) -> None:
        """
        Request cooperative cancellation.

        For queued computations this marks them cancelled. For running MPI jobs
        this sets the shared progress cancellation flag.
        """
        self._cancel_requested = True

        if self._progress_tracker is not None:
            self._progress_tracker.request_cancel()

        if self._status == SltComputationStatus.PENDING:
            self._status = SltComputationStatus.CANCELLED
            self._set_progress(None, "Computation cancelled.")

    def cancel(self, *, grace_seconds: float = 5.0) -> None:
        """
        Cancel the computation, terminating a running MPI worker when needed.
        """
        self._cancel_requested = True

        if self._mpi_handle is not None and self._mpi_handle.running:
            self._mpi_handle.terminate(grace_seconds=grace_seconds)
            return

        if self._status == SltComputationStatus.PENDING:
            self._status = SltComputationStatus.CANCELLED
            self._set_progress(None, "Computation cancelled.")
            return

        if self._status == SltComputationStatus.RUNNING:
            raise RuntimeError(
                "Cannot cancel a running computation without an MPI handle. "
                "Cancel it through the SltSession job handle."
            )

        if self._status == SltComputationStatus.FINISHED:
            raise RuntimeError("Cannot cancel a finished computation.")

    def kill(self) -> None:
        """
        Forcibly terminate a running MPI worker.
        """
        self._cancel_requested = True

        if self._mpi_handle is not None and self._mpi_handle.running:
            self._mpi_handle.kill()

    def to_dataset(self, *, copy: bool = False) -> xr.Dataset | xr.DataArray:
        """
        Return the computed xarray Dataset.
        """
        dataset = self.result.dataset
        return dataset.copy(deep=True) if copy else dataset

    def to_xarray(self) -> xr.Dataset | xr.DataArray:
        """
        Return the primary xarray object from the computed result.
        """
        results = self.result
        dataset = results.dataset
        primary = results.primary

        if primary is None or primary == "__dataset__":
            return dataset

        if primary in dataset.data_vars:
            return dataset[primary]

        if primary in dataset.coords:
            return dataset.coords[primary]

        raise KeyError(
            f"Computed result declares primary={primary!r}, "
            "but this variable or coordinate is missing from the dataset."
        )

    def _execute(
        self,
        *,
        allocation: SltAllocation,
        force: bool,
        save: bool | None,
    ) -> ResultViewT:
        if self._status == SltComputationStatus.RUNNING:
            raise RuntimeError(f"{self.computation_name} is already running.")

        if self._status == SltComputationStatus.FINISHED and not force:
            if save is True and self._saved_group is None:
                self.save()
            return self._wrap_results(self.result)

        if self._status == SltComputationStatus.CANCELLED and not force:
            raise RuntimeError(
                f"{self.computation_name} was cancelled. "
                "Use run(force=True) to run it anyway."
            )

        if self._cancel_requested and not force:
            raise CancelledError()

        self._reset_runtime_state()
        self._status = SltComputationStatus.RUNNING
        self._started_perf_counter = perf_counter()

        try:
            self._set_progress(0.0, "Starting computation.")
            self._before_run()

            if self._cancel_requested:
                raise CancelledError()

            result = self._compute(allocation=allocation)

            if self._cancel_requested:
                raise CancelledError()

            self._result = result
            self._status = SltComputationStatus.FINISHED
            self._set_progress(1.0, "Computation finished.")

            if self._progress_tracker is not None:
                self._progress_tracker.set_finished()

            self._after_success(result)

            should_save = self.save_as is not None if save is None else save
            if should_save:
                self.save()

            return self._wrap_results(result)

        except CancelledError:
            self._status = SltComputationStatus.CANCELLED
            self._set_progress(None, "Computation cancelled.")
            if self._progress_tracker is not None:
                self._progress_tracker.set_cancelled()
            raise

        except BaseException as exc:
            self._status = SltComputationStatus.FAILED
            self._failure = SltComputationFailure(
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback="".join(format_exception(type(exc), exc, exc.__traceback__)),
            )
            self._set_progress(None, "Computation failed.")
            if self._progress_tracker is not None:
                self._progress_tracker.set_failed()
            raise

        finally:
            self._finished_perf_counter = perf_counter()
            self._release_progress_tracker()
            self._mpi_handle = None

    def _local_allocation(self) -> SltAllocation:
        """
        Build a localhost allocation from the resolved resource request.
        """
        from slothpy.core.slt_session import SltAllocation, SltNodeAllocation

        request = self.resource_request
        return SltAllocation(
            request=request,
            nodes=(
                SltNodeAllocation(
                    node_name="localhost",
                    ranks=request.num_processes,
                    threads_per_rank=request.num_threads,
                    reserved_cores=request.total_cores,
                    exclusive=request.exclusive_nodes,
                ),
            ),
        )

    def _ensure_progress_tracker(self, *, total: int) -> SltProgressTracker:
        from slothpy.core.slt_session import SltProgressStatus, SltProgressTracker

        if self._progress_tracker is None:
            self._progress_tracker = SltProgressTracker.create(
                total=total,
                status=SltProgressStatus.QUEUED,
            )
        else:
            self._progress_tracker.set_total(total)

        return self._progress_tracker

    def _run_mpi_process(
        self,
        *,
        spec: MPIJobSpec,
        job_spec_path: PathLike,
        allocation: SltAllocation,
    ) -> None:
        """
        Launch an MPI worker and wait for completion.

        Subclasses call this from ``_compute`` after writing the job spec.
        """
        from slothpy.core.slt_session import start_mpi_process

        extra_mpi_args: tuple[str, ...] = ()
        hostfile_dir: TemporaryDirectory[str] | None = None

        if not allocation.is_localhost_only:
            hostfile_dir = TemporaryDirectory(prefix="slothpy-mpi-hostfile-")
            hostfile = allocation.write_openmpi_hostfile(
                Path(hostfile_dir.name) / "hostfile"
            )
            extra_mpi_args = allocation.openmpi_extra_args(hostfile)

        self._set_progress(0.0, "Launching MPI worker.")
        if self._progress_tracker is not None:
            self._progress_tracker.set_running()

        handle = start_mpi_process(
            worker_module=spec.worker_module,
            job_spec_path=job_spec_path,
            allocation=allocation,
            extra_mpi_args=extra_mpi_args,
        )
        self._mpi_handle = handle

        try:
            if self._cancel_requested:
                handle.terminate(grace_seconds=0.0)
                raise CancelledError()

            returncode = handle.wait()

            if self._cancel_requested:
                raise CancelledError()

            if returncode != 0:
                stdout, stderr = handle.communicate()
                raise RuntimeError(
                    "MPI worker failed with return code "
                    f"{returncode}.\n\n"
                    f"Command:\n{' '.join(handle.command)}\n\n"
                    f"STDOUT:\n{stdout}\n\n"
                    f"STDERR:\n{stderr}"
                )
        finally:
            self._mpi_handle = None
            if hostfile_dir is not None:
                hostfile_dir.cleanup()

    def _reset_runtime_state(self) -> None:
        self._result = None
        self._wrapped_result = None
        self._saved_group = None
        self._failure = None
        self._progress = None
        self._message = None
        self._started_perf_counter = None
        self._finished_perf_counter = None
        self._cancel_requested = False

    def _release_progress_tracker(self) -> None:
        if self._progress_tracker is None:
            return

        if self._progress_tracker.owns_memory:
            self._progress_tracker.release()
        else:
            self._progress_tracker.close()

        self._progress_tracker = None

    def _set_progress(self, progress: float | None, message: str | None = None) -> None:
        if progress is not None and not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0.0 and 1.0.")

        self._progress = progress
        self._message = message

        if self._progress_tracker is not None and progress is not None:
            total = int(self._progress_tracker.snapshot().total)
            if total > 0:
                self._progress_tracker.set_done(int(round(progress * total)))

    def _before_run(self) -> None:
        """
        Optional subclass hook before computation starts.
        """
        return None

    def _after_success(self, result: SltResults) -> None:
        """
        Optional subclass hook after successful computation.
        """
        return None

    @abstractmethod
    def _compute(self, *, allocation: SltAllocation) -> SltResults:
        """
        Perform the actual computation and return SltResults.
        """

    def _wrap_results(self, results: SltResults) -> ResultViewT:
        """
        Wrap the raw results into a view.
        """
        if self._wrapped_result is not None:
            return self._wrapped_result

        wrapped = results.to_typed_slt_results()
        self._wrapped_result = wrapped  # type: ignore[assignment]
        return wrapped  # type: ignore[return-value]

    def _resolve_output_slt(self, slt: SltFile | PathLike | None) -> SltFile:
        if slt is not None:
            return self._coerce_slt_file(slt)

        if self.output_slt is not None:
            return self._coerce_slt_file(self.output_slt)

        inferred = self._infer_output_slt_from_source()
        if inferred is not None:
            return inferred

        raise ValueError(
            "Could not infer output .slt file. Pass output_slt=... when creating "
            "the computation or pass slt=... to save()."
        )

    @staticmethod
    def _coerce_slt_file(value: SltFile | PathLike) -> SltFile:
        if isinstance(value, SltFile):
            return value

        try:
            return open_slt_file(value)
        except FileNotFoundError:
            return create_slt_file(value)

    def _infer_output_slt_from_source(self) -> SltFile | None:
        file_paths = self._source_file_paths(self.source)

        if not file_paths:
            return None

        if len(file_paths) > 1:
            raise ValueError(
                "Computation uses groups from multiple .slt files. "
                "Pass output_slt=... explicitly."
            )

        return open_slt_file(next(iter(file_paths)))

    @classmethod
    def _source_file_paths(cls, source: Any) -> set[Path]:
        paths: set[Path] = set()

        if isinstance(source, SltGroup):
            paths.add(source.file_path)
            return paths

        if isinstance(source, Mapping):
            for value in source.values():
                paths.update(cls._source_file_paths(value))
            return paths

        if isinstance(source, Sequence) and not isinstance(source, str | bytes):
            for value in source:
                paths.update(cls._source_file_paths(value))
            return paths

        return paths

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"status={self.status.value!r}, "
            f"save_as={self.save_as!r}, "
            f"resources={self.resources!r})"
        )


__all__ = [
    "SltComputation",
    "SltComputationFailure",
    "SltComputationResources",
    "SltComputationStatus",
]
